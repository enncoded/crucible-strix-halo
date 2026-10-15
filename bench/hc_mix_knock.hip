// Diagnostic knockouts of the prefill HC mix (PrefillHcMixWideKernel<true,true>):
// how much of one site's time is the epilogue's H read? Variant 0 is the
// engine kernel; variant 1 is a copy that must match it bit for bit; variant 2
// reads H from an 8-row window (cache resident, wrong values); variant 3 reads
// no H at all. Exact candidates (same WMMA chains and epilogue, must be
// bitwise): "lds up" stages the block's 40 KB up slice in LDS once instead of
// eight wave loads; "nt2" gives each wave two 16-column tiles, so each t
// fragment serves eight WMMAs and the H reads cover 128-byte lines.
// Usage: hc_mix_knock [R]
#include "prefill_common.h"
#include "../engine/layers/prefill_hc.h"
using namespace crucible;
using namespace prefill_bench;

namespace crucible {
template <int V>
__global__ void __launch_bounds__(256) MixKnockKernel(const __half* up, const float* w1, const __half* thi,
    const __half* tlo, const float* ss, const float* H, float* mixed, int R, __half* xh, int ldh) {
  const int lane = threadIdx.x & 31, wave = threadIdx.x >> 5;
  const int r0 = blockIdx.y * 128 + wave * 16, d = blockIdx.x * 16 + (lane & 15);
  if (r0 >= R) return;
  const int ra = r0 + (lane & 15);
  prefill::F8 hi[4] = {}, lo[4] = {};
#pragma unroll 2
  for (int k = 0; k < kRank; k += 16) {
    const prefill::H16 ah = ra < R ? PrefillLoad16(thi + (size_t)ra * kRank + k) : prefill::H16{};
    const prefill::H16 al = ra < R ? PrefillLoad16(tlo + (size_t)ra * kRank + k) : prefill::H16{};
#pragma unroll
    for (int s = 0; s < 4; ++s) {
      const prefill::H16 b = PrefillLoad16(up + (((size_t)s * (kHidden / 16) + (d >> 4)) * (kRank / 16) + (k >> 4)) * 256 + (lane & 15) * 16);
      hi[s] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(ah, b, hi[s]);
      lo[s] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(al, b, lo[s]);
    }
  }
  float w[4];
#pragma unroll
  for (int s = 0; s < 4; ++s) w[s] = w1[s * kHidden + d];
#pragma unroll
  for (int z = 0; z < 8; ++z) {
    const int r = r0 + prefill::FragmentRow(z);
    if (r >= R) continue;
    float v[4];
#pragma unroll
    for (int s = 0; s < 4; ++s) {
      const float g = hi[s][z] + lo[s][z] * (1.f / prefill::kResidualScale);
      const float inv = rsqrtf(ss[r * 4 + s] / kHidden + kEps);
      const float h = V == 3 ? 1.f : H[(size_t)(V == 2 ? (r & 7) : r) * ldh + s * kHidden + d];
      v[s] = Sigmoid(g) * h * inv * w[s];
    }
    const float x = RoundH(((v[0] + v[1]) + (v[2] + v[3])) * .25f);
    if (mixed) mixed[(size_t)r * kHidden + d] = x;
    if (xh) xh[(size_t)r * kHidden + d] = __float2half(x);
  }
}

// Shared epilogue of the engine kernel for one 16x16 tile.
__device__ __forceinline__ void MixEpilogue(const prefill::F8* hi, const prefill::F8* lo, int r0, int d,
    const float* w1, const float* ss, const float* H, int R, __half* xh, int ldh) {
  float w[4];
#pragma unroll
  for (int s = 0; s < 4; ++s) w[s] = w1[s * kHidden + d];
#pragma unroll
  for (int z = 0; z < 8; ++z) {
    const int r = r0 + prefill::FragmentRow(z);
    if (r >= R) continue;
    float v[4];
#pragma unroll
    for (int s = 0; s < 4; ++s) {
      const float g = hi[s][z] + lo[s][z] * (1.f / prefill::kResidualScale);
      const float inv = rsqrtf(ss[r * 4 + s] / kHidden + kEps);
      v[s] = Sigmoid(g) * H[(size_t)r * ldh + s * kHidden + d] * inv * w[s];
    }
    xh[(size_t)r * kHidden + d] = __float2half(RoundH(((v[0] + v[1]) + (v[2] + v[3])) * .25f));
  }
}

__global__ void __launch_bounds__(256) MixLdsKernel(const __half* up, const float* w1, const __half* thi,
    const __half* tlo, const float* ss, const float* H, int R, __half* xh, int ldh) {
  constexpr int kKt = kRank / 16;
  __shared__ __attribute__((aligned(32))) __half lu[4 * kKt * 256];
  const int lane = threadIdx.x & 31, wave = threadIdx.x >> 5;
  const int r0 = blockIdx.y * 128 + wave * 16, d = blockIdx.x * 16 + (lane & 15);
  for (int i = threadIdx.x; i < 4 * kKt * 32; i += 256) {
    const int s = i / (kKt * 32), o = i % (kKt * 32);
    reinterpret_cast<uint4*>(lu)[i] =
        reinterpret_cast<const uint4*>(up + ((size_t)s * (kHidden / 16) + blockIdx.x) * kKt * 256)[o];
  }
  __syncthreads();
  if (r0 >= R) return;
  const int ra = r0 + (lane & 15);
  prefill::F8 hi[4] = {}, lo[4] = {};
#pragma unroll 2
  for (int k = 0; k < kRank; k += 16) {
    const prefill::H16 ah = ra < R ? PrefillLoad16(thi + (size_t)ra * kRank + k) : prefill::H16{};
    const prefill::H16 al = ra < R ? PrefillLoad16(tlo + (size_t)ra * kRank + k) : prefill::H16{};
#pragma unroll
    for (int s = 0; s < 4; ++s) {
      const prefill::H16 b = *reinterpret_cast<const prefill::H16*>(lu + (s * kKt + (k >> 4)) * 256 + (lane & 15) * 16);
      hi[s] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(ah, b, hi[s]);
      lo[s] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(al, b, lo[s]);
    }
  }
  MixEpilogue(hi, lo, r0, d, w1, ss, H, R, xh, ldh);
}

__global__ void __launch_bounds__(256) MixNt2Kernel(const __half* up, const float* w1, const __half* thi,
    const __half* tlo, const float* ss, const float* H, int R, __half* xh, int ldh) {
  const int lane = threadIdx.x & 31, wave = threadIdx.x >> 5;
  const int r0 = blockIdx.y * 128 + wave * 16, dt = blockIdx.x * 2;
  if (r0 >= R) return;
  const int ra = r0 + (lane & 15);
  prefill::F8 hi[2][4] = {}, lo[2][4] = {};
#pragma unroll 2
  for (int k = 0; k < kRank; k += 16) {
    const prefill::H16 ah = ra < R ? PrefillLoad16(thi + (size_t)ra * kRank + k) : prefill::H16{};
    const prefill::H16 al = ra < R ? PrefillLoad16(tlo + (size_t)ra * kRank + k) : prefill::H16{};
#pragma unroll
    for (int n = 0; n < 2; ++n)
#pragma unroll
      for (int s = 0; s < 4; ++s) {
        const prefill::H16 b = PrefillLoad16(up + (((size_t)s * (kHidden / 16) + dt + n) * (kRank / 16) + (k >> 4)) * 256 + (lane & 15) * 16);
        hi[n][s] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(ah, b, hi[n][s]);
        lo[n][s] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(al, b, lo[n][s]);
      }
  }
#pragma unroll
  for (int n = 0; n < 2; ++n) MixEpilogue(hi[n], lo[n], r0, (dt + n) * 16 + (lane & 15), w1, ss, H, R, xh, ldh);
}
}  // namespace crucible

int main(int argc, char** argv) {
  const int R = argc > 1 ? std::atoi(argv[1]) : 4096, ldh = kWide;
  if (R < 128 || R > 8192 || R % 128) return 2;
  Buffer<float> H(Random((size_t)R * ldh, .4f, 311));
  Buffer<__half> up(Half(Random((size_t)kWide * kRank, .02f, 317))), uf((size_t)kWide * kRank);
  Buffer<__half> thi(Half(Random((size_t)R * kRank, .3f, 321))), tlo(Half(Random((size_t)R * kRank, .3f, 323)));
  auto ws = Random(kWide, .05f, 319);
  for (auto& v : ws) v += 1.f;
  auto sv = Random(R * 4, .5f, 325);
  for (auto& v : sv) v = std::fabs(v) * kHidden + 1.f;
  Buffer<float> w1(ws), ss(sv), mixed((size_t)R * kHidden);
  Buffer<__half> xh((size_t)R * kHidden);
  HcFragUpKernel<<<(kWide * kRank + 255) / 256, 256>>>(up.p, uf.p);
  CheckLaunch();
  const dim3 grid(kHidden / 16, R / 128);
  auto eng = [&] { PrefillHcMixWideKernel<true, true><<<grid, 256>>>(uf.p, w1.p, thi.p, tlo.p, ss.p, H.p, nullptr, R, xh.p, ldh); };
  eng(); CheckLaunch();
  const auto x0 = xh.Get();
  bool ok = true;
  auto run = [&](const char* name, auto launch, bool check) {
    xh.Zero(); launch(); CheckLaunch();
    const bool same = xh.Get() == x0;
    if (check) ok = ok && same;
    std::vector<double> t;
    for (int i = 0; i < 3; ++i) t.push_back(Time(launch));
    std::printf("R=%d mix %-22s bitwise=%d us=%.1f\n", R, name, (int)same, bench::Median(t));
    std::fflush(stdout);
  };
  run("engine", eng, true);
#define KV(V, NAME, CHECK) run(NAME, [&] { MixKnockKernel<V><<<grid, 256>>>(uf.p, w1.p, thi.p, tlo.p, ss.p, H.p, nullptr, R, xh.p, ldh); }, CHECK)
  KV(1, "copy", true);
  KV(2, "H 8-row window (diag)", false);
  KV(3, "no H (diag)", false);
  run("lds up", [&] { MixLdsKernel<<<grid, 256>>>(uf.p, w1.p, thi.p, tlo.p, ss.p, H.p, R, xh.p, ldh); }, true);
  run("nt2", [&] { MixNt2Kernel<<<dim3(kHidden / 32, R / 128), 256>>>(uf.p, w1.p, thi.p, tlo.p, ss.p, H.p, R, xh.p, ldh); }, true);
  run("engine again", eng, true);
  std::printf("%s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
