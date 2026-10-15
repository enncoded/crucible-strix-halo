// Round 4 Q3: the prefill HC mix is bound by L0 fragment loads (the prefill work
// v33 diagnostic: 4.9 GB per site at R4096; removing H and WMMA changes
// little). Two load changes that keep every WMMA operand and sequence, so
// the output must stay bitwise equal:
//   V0  PrefillHcMixWideKernel<true> (the engine kernel)
//   V1  the block's up slice (16 columns x 4 streams x 320 k) is loaded once
//       into LDS; the 8 waves read B fragments from LDS instead of each
//       loading the same slice from L0 (8x redundant today)
//   V2  V1, and each half-wave loads half of every A fragment (thi, tlo);
//       v_permlanex16 fills the other half (gfx11 WMMA needs both half-waves
//       to hold the full row)
//   V3  V2 with B streamed through LDS in 4 k-chunks (10 KB), for occupancy
//   V4  V0 loads (global A and B) with every fragment of step k+16 issued
//       before the 8 WMMAs of step k (one-step software pipeline)
// Build on the test machine against a snapshot of the prefill work (engine, kernels):
//   hipcc ... -I build/pl-3035f9f -o build/bench/hc_mix_frag bench/explore/hc_mix_frag.hip
// Usage: hc_mix_frag [R=4096] [reps=20]
#include <algorithm>
#include <cstring>
#include <random>

#include "engine/layers/prefill_hc.h"

using namespace crucible;

namespace {
__device__ __forceinline__ prefill::H16 LdsLoad16(const __half* p) {
  using U4 = uint32_t __attribute__((ext_vector_type(4)));
  prefill::H16 f;
  U4* fp = reinterpret_cast<U4*>(&f);
  fp[0] = reinterpret_cast<const U4*>(p)[0];
  fp[1] = reinterpret_cast<const U4*>(p)[1];
  return f;
}

// Lanes 0-15 load k 0-7 of their row, lanes 16-31 load k 8-15; one
// permlanex16 per dword hands each half the other half's words.
__device__ __forceinline__ prefill::H16 HalfLoad16(const __half* row, int k, int lane) {
  using U4 = uint32_t __attribute__((ext_vector_type(4)));
  const bool upper = lane >= 16;
  const U4 mine = *reinterpret_cast<const U4*>(row + k + (upper ? 8 : 0));
  U4 other;
#pragma unroll
  for (int i = 0; i < 4; ++i)
    other[i] = __builtin_amdgcn_permlanex16(mine[i], mine[i], 0x76543210u, 0xfedcba98u, false, false);
  prefill::H16 f;
  U4* fp = reinterpret_cast<U4*>(&f);
  fp[0] = upper ? other : mine;
  fp[1] = upper ? mine : other;
  return f;
}

template <int V>
__global__ void __launch_bounds__(256) MixVariant(const __half* up, const float* w1, const __half* thi,
                                                  const __half* tlo, const float* ss, const float* H,
                                                  float* mixed, int R, __half* xh) {
  // Row stride kChunkK + 8 halves: 16-byte aligned, and the 16 rows of a
  // fragment start on distinct 4-bank groups (no LDS bank conflicts).
  constexpr int kChunks = V == 3 ? 4 : 1, kChunkK = kRank / kChunks, kRowK = kChunkK + 8;
  __shared__ __attribute__((aligned(16))) __half bs[4 * 16 * kRowK];
  const int lane = threadIdx.x & 31, wave = threadIdx.x >> 5;
  const int r0 = blockIdx.y * 128 + wave * 16, d0 = blockIdx.x * 16, d = d0 + (lane & 15);
  const int ra = r0 + (lane & 15);
  prefill::F8 hi[4] = {}, lo[4] = {};
  for (int c = 0; c < kChunks; ++c) {
    // Stage up[s][d0..d0+15][c-chunk of k] (all 256 threads, 16-byte pieces).
    if (c) __syncthreads();
    constexpr int kPieces = 4 * 16 * kChunkK / 8;
    for (int p = threadIdx.x; p < kPieces; p += 256) {
      const int row = p / (kChunkK / 8), k8 = p % (kChunkK / 8) * 8, s = row / 16, dl = row % 16;
      *reinterpret_cast<uint4*>(&bs[row * kRowK + k8]) =
          *reinterpret_cast<const uint4*>(up + ((size_t)s * kHidden + d0 + dl) * kRank + c * kChunkK + k8);
    }
    __syncthreads();
    if (r0 >= R) continue;
#pragma unroll 2
    for (int kk = 0; kk < kChunkK; kk += 16) {
      const int k = c * kChunkK + kk;
      prefill::H16 ah, al;
      if constexpr (V >= 2) {
        const size_t ro = (size_t)min(ra, R - 1) * kRank;
        ah = HalfLoad16(thi + ro, k, lane);
        al = HalfLoad16(tlo + ro, k, lane);
        if (ra >= R) { ah = prefill::H16{}; al = prefill::H16{}; }
      } else {
        ah = ra < R ? PrefillLoad16(thi + (size_t)ra * kRank + k) : prefill::H16{};
        al = ra < R ? PrefillLoad16(tlo + (size_t)ra * kRank + k) : prefill::H16{};
      }
#pragma unroll
      for (int s = 0; s < 4; ++s) {
        const prefill::H16 b = LdsLoad16(&bs[(s * 16 + (lane & 15)) * kRowK + kk]);
        hi[s] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(ah, b, hi[s]);
        lo[s] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(al, b, lo[s]);
      }
    }
  }
  if (r0 >= R) return;
  // Epilogue copied from PrefillHcMixWideKernel.
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
      v[s] = Sigmoid(g) * H[((size_t)r * 4 + s) * kHidden + d] * inv * w[s];
    }
    const float x = RoundH(((v[0] + v[1]) + (v[2] + v[3])) * .25f);
    if (mixed) mixed[(size_t)r * kHidden + d] = x;
    if (xh) xh[(size_t)r * kHidden + d] = __float2half(x);
  }
}

// V4: PrefillHcMixWideKernel<true> with a one-step fragment pipeline.
__global__ void __launch_bounds__(256) MixPipelined(const __half* up, const float* w1, const __half* thi,
                                                   const __half* tlo, const float* ss, const float* H,
                                                   float* mixed, int R, __half* xh) {
  const int lane = threadIdx.x & 31, wave = threadIdx.x >> 5;
  const int r0 = blockIdx.y * 128 + wave * 16, d = blockIdx.x * 16 + (lane & 15);
  if (r0 >= R) return;
  const int ra = r0 + (lane & 15);
  const __half* bp[4];
#pragma unroll
  for (int s = 0; s < 4; ++s) bp[s] = up + ((size_t)s * kHidden + d) * kRank;
  auto load = [&](int k, prefill::H16& ah, prefill::H16& al, prefill::H16* b) {
    ah = ra < R ? PrefillLoad16(thi + (size_t)ra * kRank + k) : prefill::H16{};
    al = ra < R ? PrefillLoad16(tlo + (size_t)ra * kRank + k) : prefill::H16{};
#pragma unroll
    for (int s = 0; s < 4; ++s) b[s] = PrefillLoad16(bp[s] + k);
  };
  prefill::F8 hi[4] = {}, lo[4] = {};
  prefill::H16 ah, al, b[4], nah, nal, nb[4];
  load(0, ah, al, b);
#pragma unroll 1
  for (int k = 0; k < kRank; k += 16) {
    if (k + 16 < kRank) load(k + 16, nah, nal, nb);
#pragma unroll
    for (int s = 0; s < 4; ++s) {
      hi[s] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(ah, b[s], hi[s]);
      lo[s] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(al, b[s], lo[s]);
    }
    ah = nah; al = nal;
#pragma unroll
    for (int s = 0; s < 4; ++s) b[s] = nb[s];
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
      v[s] = Sigmoid(g) * H[((size_t)r * 4 + s) * kHidden + d] * inv * w[s];
    }
    const float x = RoundH(((v[0] + v[1]) + (v[2] + v[3])) * .25f);
    if (mixed) mixed[(size_t)r * kHidden + d] = x;
    if (xh) xh[(size_t)r * kHidden + d] = __float2half(x);
  }
}

template <class T> T* Upload(const std::vector<T>& v) {
  T* p;
  HIP_OK(hipMalloc(&p, v.size() * sizeof(T)));
  HIP_OK(hipMemcpy(p, v.data(), v.size() * sizeof(T), hipMemcpyHostToDevice));
  return p;
}
}  // namespace

int main(int argc, char** argv) {
  const int R = argc > 1 ? std::atoi(argv[1]) : 4096;
  const int reps = argc > 2 ? std::atoi(argv[2]) : 20;
  std::mt19937 rng(3); std::normal_distribution<float> nd; std::uniform_real_distribution<float> ud(0.5f, 2.f);
  std::vector<__half> up((size_t)4 * kHidden * kRank), th((size_t)R * kRank), tl((size_t)R * kRank);
  std::vector<float> w1(4 * kHidden), ss((size_t)R * 4), H((size_t)R * 4 * kHidden);
  for (auto& x : up) x = __float2half(nd(rng) * 0.05f);
  for (auto& x : th) x = __float2half(ud(rng));
  for (auto& x : tl) x = __float2half(nd(rng));
  for (auto& x : w1) x = nd(rng);
  for (auto& x : ss) x = ud(rng) * kHidden;
  for (auto& x : H) x = nd(rng);
  const __half *dup = Upload(up), *dth = Upload(th), *dtl = Upload(tl);
  const float *dw1 = Upload(w1), *dss = Upload(ss), *dH = Upload(H);
  constexpr int kVariants = 5;
  float* out[kVariants];
  __half* xh;
  for (auto& o : out) HIP_OK(hipMalloc(&o, (size_t)R * kHidden * 4));
  HIP_OK(hipMalloc(&xh, (size_t)R * kHidden * 2));
  const dim3 grid(kHidden / 16, (R + 127) / 128);
  auto launch = [&](int v, float* o) {
    if (v == 0) PrefillHcMixWideKernel<true><<<grid, 256>>>(dup, dw1, dth, dtl, dss, dH, o, R, xh);
    else if (v == 1) MixVariant<1><<<grid, 256>>>(dup, dw1, dth, dtl, dss, dH, o, R, xh);
    else if (v == 2) MixVariant<2><<<grid, 256>>>(dup, dw1, dth, dtl, dss, dH, o, R, xh);
    else if (v == 3) MixVariant<3><<<grid, 256>>>(dup, dw1, dth, dtl, dss, dH, o, R, xh);
    else MixPipelined<<<grid, 256>>>(dup, dw1, dth, dtl, dss, dH, o, R, xh);
    HIP_OK(hipGetLastError());
  };
  for (int v = 0; v < kVariants; ++v) launch(v, out[v]);
  HIP_OK(hipDeviceSynchronize());
  std::vector<float> ref((size_t)R * kHidden), got((size_t)R * kHidden);
  HIP_OK(hipMemcpy(ref.data(), out[0], ref.size() * 4, hipMemcpyDeviceToHost));
  bool ok = true;
  for (int v = 1; v < kVariants; ++v) {
    HIP_OK(hipMemcpy(got.data(), out[v], got.size() * 4, hipMemcpyDeviceToHost));
    const bool eq = std::memcmp(ref.data(), got.data(), ref.size() * 4) == 0;
    std::printf("V%d bitwise %s\n", v, eq ? "equal" : "DIFFERENT");
    ok &= eq;
  }
  const double frag_gb = (double)(R / 16) * (kHidden / 16) * (kRank / 16) * 6 * 1024 / 1e9;
  for (int pass = 0; pass < 2; ++pass)
    for (int v = 0; v < kVariants; ++v) {
      std::vector<float> t;
      for (int r = 0; r < reps + 1; ++r) {
        hipEvent_t a, b;
        HIP_OK(hipEventCreate(&a)); HIP_OK(hipEventCreate(&b));
        HIP_OK(hipEventRecord(a));
        launch(v, out[v]);
        HIP_OK(hipEventRecord(b)); HIP_OK(hipEventSynchronize(b));
        float ms; HIP_OK(hipEventElapsedTime(&ms, a, b));
        if (r) t.push_back(ms);
        HIP_OK(hipEventDestroy(a)); HIP_OK(hipEventDestroy(b));
      }
      std::sort(t.begin(), t.end());
      if (pass) std::printf("V%d R=%d median %.1f us (min %.1f)  [V0 fragment loads %.2f GB]\n", v, R,
                            t[t.size() / 2] * 1e3, t[0] * 1e3, frag_gb);
    }
  return ok ? 0 : 1;
}
