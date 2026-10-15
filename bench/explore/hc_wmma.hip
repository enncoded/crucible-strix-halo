// Probe: HC read (dots, mid, mix) on WMMA at R = 1, 4, 8, 16 decode rows.
// The decode HC is a weight stream: fn (324 x 10240 fp16) for the dots and
// up (10240 x 320 fp16) for the mix, 13.4 MB per site, 97 sites per forward.
// VALU kernels pay 2 FMA per weight per row, so their cost grows with R.
// WMMA pads R to 16 rows, so its ALU cost is the same for R = 1..16.
// The fp32 activations (H, t) take the prefill high/low split; fp16 weights
// are exact WMMA operands. This is not bitwise equal to the VALU order.
//
// Layout: fn and up are repacked offline so that one WMMA B fragment
// (16 columns x 16 k) is one contiguous 512-byte run.
// Usage: hc_wmma [sets]
#include <hip/hip_fp16.h>

#include <cmath>
#include <random>

#include "../common.h"

namespace {

using H16 = _Float16 __attribute__((ext_vector_type(16)));
using F8 = float __attribute__((ext_vector_type(8)));

constexpr int kHidden = 2560, kStreams = 4, kWide = 10240, kRank = 320, kM = 324;
constexpr int kNT = 21;               // 324 outputs padded to 336
constexpr int kKS = 8;                // dots K split: 320 k per wave
constexpr int kKC = kHidden / kKS;    // 320
constexpr int kDT = kHidden / 16;     // mix column tiles
constexpr float kRes = 4096.f, kEps = 1e-6f;
constexpr size_t kFnHalves = (size_t)kStreams * kNT * 16 * kHidden;  // packed, padded
constexpr size_t kUpHalves = (size_t)kStreams * kHidden * kRank;

__device__ __forceinline__ float Sigmoid(float x) { return 1.f / (1.f + __expf(-x)); }
__device__ __forceinline__ float Silu(float x) { return x * Sigmoid(x); }
__device__ __forceinline__ float RoundH(float x) { return __half2float(__float2half(x)); }

__device__ __forceinline__ H16 NtLoad(const H16* p) {
  return __builtin_nontemporal_load(p);
}

// Sum of squares per (row, stream).
template <int R>
__global__ void __launch_bounds__(128) Squares(const float* H, float* ss) {
  const int w = blockIdx.x * 4 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
  if (w >= R * kStreams) return;
  const float* h = H + (size_t)w * kHidden;
  float a = 0;
  for (int d = lane; d < kHidden; d += 32) a += h[d] * h[d];
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) a += __shfl_xor(a, o, 32);
  if (lane == 0) ss[w] = a;
}

// One wave: one 16-column tile, one stream, one 320-k chunk.
template <int R>
__global__ void __launch_bounds__(128) DotsW(const H16* __restrict__ fnp, const float* __restrict__ H,
                                             float* __restrict__ part) {
  const int id = blockIdx.x * 4 + (threadIdx.x >> 5), lane = threadIdx.x & 31, row = lane & 15;
  const int kc = id % kKS, s = (id / kKS) % kStreams, nt = id / (kKS * kStreams);
  if (nt >= kNT) return;
  const float* h = H + (size_t)row * kWide + s * kHidden + kc * kKC;
  const H16* b = fnp + ((size_t)(s * kNT + nt) * (kHidden / 16) + kc * (kKC / 16)) * 16 + row;
  F8 hi = {}, lo = {};
#pragma unroll 5
  for (int st = 0; st < kKC / 16; ++st) {
    const H16 bf = NtLoad(b + st * 16);
    H16 ah = {}, al = {};
    if (row < R) {
      const float4* hp = reinterpret_cast<const float4*>(h + st * 16);
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        const float4 v = hp[q];
        const float f[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          const _Float16 x = (_Float16)f[i];
          ah[q * 4 + i] = x;
          al[q * 4 + i] = (_Float16)((f[i] - (float)x) * kRes);
        }
      }
    }
    hi = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(ah, bf, hi);
    lo = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(al, bf, lo);
  }
#pragma unroll
  for (int z = 0; z < 8; ++z) {
    const int r = 2 * z + (lane >> 4);
    if (r < R) part[((size_t)(kc * R + r) * kStreams + s) * (kNT * 16) + nt * 16 + (lane & 15)] =
        hi[z] + lo[z] * (1.f / kRes);
  }
}

// Reduce the K chunks in a fixed order, then the HC mid (t, post).
template <int R>
__global__ void __launch_bounds__(352) MidW(const float* part, const float* ss, __half* thi, __half* tlo,
                                            float* post) {
  const int r = blockIdx.x, m = threadIdx.x;
  if (m >= kM) return;
  float a = 0;
#pragma unroll
  for (int s = 0; s < kStreams; ++s) {
    float d = 0;
    for (int kc = 0; kc < kKS; ++kc) d += part[((size_t)(kc * R + r) * kStreams + s) * (kNT * 16) + m];
    a += rsqrtf(ss[r * 4 + s] / kHidden + kEps) * d;
  }
  if (m < kRank) {
    const float v = Silu(a * .25f);
    const __half h = __float2half(v);
    thi[r * kRank + m] = h;
    tlo[r * kRank + m] = __float2half((v - __half2float(h)) * kRes);
  } else {
    post[r * 4 + m - kRank] = 2.f * Sigmoid(a * .25f);
  }
}

// One block: 16 hidden columns; wave s owns stream s.
template <int R>
__global__ void __launch_bounds__(128) MixW(const H16* __restrict__ upp, const __half* thi, const __half* tlo,
                                            const float* ss, const float* H, const float* w1, float* x) {
  __shared__ float g[kStreams][16][17];
  const int dt = blockIdx.x, s = threadIdx.x >> 5, lane = threadIdx.x & 31, row = lane & 15;
  const H16* b = upp + ((size_t)(s * kDT + dt) * (kRank / 16)) * 16 + row;
  F8 hi = {}, lo = {};
#pragma unroll 5
  for (int st = 0; st < kRank / 16; ++st) {
    const H16 bf = NtLoad(b + st * 16);
    H16 ah = {}, al = {};
    if (row < R) {
      ah = *reinterpret_cast<const H16*>(thi + row * kRank + st * 16);
      al = *reinterpret_cast<const H16*>(tlo + row * kRank + st * 16);
    }
    hi = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(ah, bf, hi);
    lo = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(al, bf, lo);
  }
#pragma unroll
  for (int z = 0; z < 8; ++z) g[s][2 * z + (lane >> 4)][lane & 15] = hi[z] + lo[z] * (1.f / kRes);
  __syncthreads();
  for (int i = threadIdx.x; i < R * 16; i += 128) {
    const int r = i >> 4, c = i & 15, d = dt * 16 + c;
    float v[4];
#pragma unroll
    for (int st = 0; st < 4; ++st) {
      const float inv = rsqrtf(ss[r * 4 + st] / kHidden + kEps);
      v[st] = Sigmoid(g[st][r][c]) * H[((size_t)r * 4 + st) * kHidden + d] * inv * w1[st * kHidden + d];
    }
    x[(size_t)r * kHidden + d] = RoundH(((v[0] + v[1]) + (v[2] + v[3])) * .25f);
  }
}

// v2: K split 16 (1344 waves, 10 fully unrolled loads in flight per lane),
// sum of squares fused (partial per K chunk), mix split over two K halves.
constexpr int kKS2 = 16, kKC2 = kHidden / kKS2;  // 160

template <int R>
__global__ void __launch_bounds__(128) DotsW2(const H16* __restrict__ fnp, const float* __restrict__ H,
                                              float* __restrict__ part, float* __restrict__ pss) {
  const int id = blockIdx.x * 4 + (threadIdx.x >> 5), lane = threadIdx.x & 31, row = lane & 15;
  const int kc = id % kKS2, s = (id / kKS2) % kStreams, nt = id / (kKS2 * kStreams);
  if (nt >= kNT) return;
  const float* h = H + (size_t)row * kWide + s * kHidden + kc * kKC2;
  const H16* b = fnp + ((size_t)(s * kNT + nt) * (kHidden / 16) + kc * (kKC2 / 16)) * 16 + row;
  H16 bf[kKC2 / 16];
#pragma unroll
  for (int st = 0; st < kKC2 / 16; ++st) bf[st] = NtLoad(b + st * 16);
  F8 hi = {}, lo = {};
  float sq = 0;
#pragma unroll
  for (int st = 0; st < kKC2 / 16; ++st) {
    H16 ah = {}, al = {};
    if (row < R) {
      const float4* hp = reinterpret_cast<const float4*>(h + st * 16);
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        const float4 v = hp[q];
        const float f[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          const _Float16 x = (_Float16)f[i];
          ah[q * 4 + i] = x;
          al[q * 4 + i] = (_Float16)((f[i] - (float)x) * kRes);
          sq += f[i] * f[i];
        }
      }
    }
    hi = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(ah, bf[st], hi);
    lo = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(al, bf[st], lo);
  }
  if (nt == 0 && lane < 16 && row < R) pss[(kc * R + row) * kStreams + s] = sq;
#pragma unroll
  for (int z = 0; z < 8; ++z) {
    const int r = 2 * z + (lane >> 4);
    if (r < R) part[((size_t)(kc * R + r) * kStreams + s) * (kNT * 16) + nt * 16 + (lane & 15)] =
        hi[z] + lo[z] * (1.f / kRes);
  }
}

template <int R>
__global__ void __launch_bounds__(352) MidW2(const float* part, const float* pss, float* ss, __half* thi,
                                             __half* tlo, float* post) {
  __shared__ float inv[kStreams];
  const int r = blockIdx.x, m = threadIdx.x;
  if (m < kStreams) {
    float a = 0;
    for (int kc = 0; kc < kKS2; ++kc) a += pss[(kc * R + r) * kStreams + m];
    ss[r * 4 + m] = a;
    inv[m] = rsqrtf(a / kHidden + kEps);
  }
  __syncthreads();
  if (m >= kM) return;
  float a = 0;
#pragma unroll
  for (int s = 0; s < kStreams; ++s) {
    float d = 0;
    for (int kc = 0; kc < kKS2; ++kc) d += part[((size_t)(kc * R + r) * kStreams + s) * (kNT * 16) + m];
    a += inv[s] * d;
  }
  if (m < kRank) {
    const float v = Silu(a * .25f);
    const __half h = __float2half(v);
    thi[r * kRank + m] = h;
    tlo[r * kRank + m] = __float2half((v - __half2float(h)) * kRes);
  } else {
    post[r * 4 + m - kRank] = 2.f * Sigmoid(a * .25f);
  }
}

// Eight waves: wave w takes stream w & 3 and K half w >> 2.
template <int R>
__global__ void __launch_bounds__(256) MixW2(const H16* __restrict__ upp, const __half* thi, const __half* tlo,
                                             const float* ss, const float* H, const float* w1, float* x) {
  __shared__ float g[2][kStreams][16][17];
  const int dt = blockIdx.x, w = threadIdx.x >> 5, s = w & 3, kh = w >> 2, lane = threadIdx.x & 31,
            row = lane & 15;
  constexpr int kSteps = kRank / 32;  // 10 per half
  const H16* b = upp + ((size_t)(s * kDT + dt) * (kRank / 16) + kh * kSteps) * 16 + row;
  H16 bf[kSteps];
#pragma unroll
  for (int st = 0; st < kSteps; ++st) bf[st] = NtLoad(b + st * 16);
  F8 hi = {}, lo = {};
#pragma unroll
  for (int st = 0; st < kSteps; ++st) {
    H16 ah = {}, al = {};
    const int k = (kh * kSteps + st) * 16;
    if (row < R) {
      ah = *reinterpret_cast<const H16*>(thi + row * kRank + k);
      al = *reinterpret_cast<const H16*>(tlo + row * kRank + k);
    }
    hi = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(ah, bf[st], hi);
    lo = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(al, bf[st], lo);
  }
#pragma unroll
  for (int z = 0; z < 8; ++z) g[kh][s][2 * z + (lane >> 4)][lane & 15] = hi[z] + lo[z] * (1.f / kRes);
  __syncthreads();
  for (int i = threadIdx.x; i < R * 16; i += 256) {
    const int r = i >> 4, c = i & 15, d = dt * 16 + c;
    float v[4];
#pragma unroll
    for (int st = 0; st < 4; ++st) {
      const float inv = rsqrtf(ss[r * 4 + st] / kHidden + kEps);
      v[st] = Sigmoid(g[0][st][r][c] + g[1][st][r][c]) * H[((size_t)r * 4 + st) * kHidden + d] * inv *
              w1[st * kHidden + d];
    }
    x[(size_t)r * kHidden + d] = RoundH(((v[0] + v[1]) + (v[2] + v[3])) * .25f);
  }
}

// Reference stream: read the same bytes with plain 16-byte loads.
__global__ void __launch_bounds__(256) Stream(const uint4* p, size_t n, uint32_t* sink) {
  uint32_t acc = 0;
  for (size_t i = (size_t)blockIdx.x * 256 + threadIdx.x; i < n; i += (size_t)gridDim.x * 256) {
    const uint4 v = p[i];
    acc ^= v.x ^ v.y ^ v.z ^ v.w;
  }
  if (acc == 0x9e3779b9u) sink[0] = acc;
}

struct Site {
  H16* fnp;
  H16* upp;
};

struct Bufs {
  float *H, *ss, *part, *post, *w1, *x, *pss;
  __half *thi, *tlo;
};

template <int R>
void RunSite(const Site& w, const Bufs& b, int which, bool v2 = false) {
  if (v2) {
    if (which & 1) {
      hipLaunchKernelGGL(DotsW2<R>, kNT * kStreams * kKS2 / 4, 128, 0, nullptr, w.fnp, b.H, b.part, b.pss);
      hipLaunchKernelGGL(MidW2<R>, R, 352, 0, nullptr, b.part, b.pss, b.ss, b.thi, b.tlo, b.post);
    }
    if (which & 2)
      hipLaunchKernelGGL(MixW2<R>, kDT, 256, 0, nullptr, w.upp, b.thi, b.tlo, b.ss, b.H, b.w1, b.x);
    return;
  }
  if (which & 1) {
    hipLaunchKernelGGL(Squares<R>, (R * 4 + 3) / 4, 128, 0, nullptr, b.H, b.ss);
    hipLaunchKernelGGL(DotsW<R>, kNT * kStreams * kKS / 4, 128, 0, nullptr, w.fnp, b.H, b.part);
    hipLaunchKernelGGL(MidW<R>, R, 352, 0, nullptr, b.part, b.ss, b.thi, b.tlo, b.post);
  }
  if (which & 2)
    hipLaunchKernelGGL(MixW<R>, kDT, 128, 0, nullptr, w.upp, b.thi, b.tlo, b.ss, b.H, b.w1, b.x);
}

template <int R>
void Time(const std::vector<Site>& sites, const Bufs& b, bool v2 = false) {
  const int reps = 4;
  double us[3];
  for (int which = 1; which <= 3; ++which) {
    for (const auto& s : sites) RunSite<R>(s, b, which, v2);  // warm
    bench::EventTimer t;
    t.Begin();
    for (int k = 0; k < reps; ++k)
      for (const auto& s : sites) RunSite<R>(s, b, which, v2);
    us[which - 1] = t.EndMs() * 1e3 / (reps * sites.size());
  }
  // Percent of DRAM: bytes / 245.5 GB/s over the measured time.
  const auto pct = [](double bytes, double t_us) { return bytes / 245.5e9 / (t_us * 1e-6) * 100; };
  const double fn = kFnHalves * 2.0, upb = kUpHalves * 2.0;
  std::printf("%s R=%2d  dots+mid %6.1f us (%4.1f%% of DRAM)  mix %6.1f us (%4.1f%%)  site %6.1f us (%4.1f%%)"
              "  x97 = %5.2f ms\n",
              v2 ? "v2" : "v1", R, us[0], pct(fn, us[0]), us[1], pct(upb, us[1]), us[2], pct(fn + upb, us[2]), us[2] * 97 / 1e3);
}

// CPU reference for one site at R rows (double accumulation).
void Reference(int R, const std::vector<__half>& fn, const std::vector<__half>& up, const std::vector<float>& H,
               const std::vector<float>& w1, std::vector<double>& dots, std::vector<float>& x) {
  dots.assign((size_t)R * kM * 4, 0);
  std::vector<double> ss(R * 4, 0), t((size_t)R * kRank);
  for (int r = 0; r < R; ++r)
    for (int s = 0; s < 4; ++s) {
      const float* h = &H[(size_t)r * kWide + s * kHidden];
      for (int d = 0; d < kHidden; ++d) ss[r * 4 + s] += (double)h[d] * h[d];
      for (int m = 0; m < kM; ++m) {
        double a = 0;
        for (int d = 0; d < kHidden; ++d) a += (double)__half2float(fn[(size_t)m * kWide + s * kHidden + d]) * h[d];
        dots[((size_t)r * kM + m) * 4 + s] = a;
      }
    }
  x.assign((size_t)R * kHidden, 0);
  for (int r = 0; r < R; ++r) {
    double inv[4];
    for (int s = 0; s < 4; ++s) inv[s] = 1.0 / std::sqrt(ss[r * 4 + s] / kHidden + kEps);
    for (int m = 0; m < kRank; ++m) {
      double a = 0;
      for (int s = 0; s < 4; ++s) a += inv[s] * dots[((size_t)r * kM + m) * 4 + s];
      const double z = a * .25;
      t[(size_t)r * kRank + m] = z / (1 + std::exp(-z));
    }
    for (int d = 0; d < kHidden; ++d) {
      double v = 0;
      for (int s = 0; s < 4; ++s) {
        double gsum = 0;
        for (int k = 0; k < kRank; ++k)
          gsum += t[(size_t)r * kRank + k] * __half2float(up[((size_t)s * kHidden + d) * kRank + k]);
        v += 1 / (1 + std::exp(-gsum)) * H[((size_t)r * 4 + s) * kHidden + d] * inv[s] * w1[s * kHidden + d];
      }
      x[(size_t)r * kHidden + d] = (float)(v * .25);
    }
  }
}

}  // namespace

int main(int argc, char** argv) {
  const int nsets = argc > 1 ? std::atoi(argv[1]) : 97;
  std::mt19937 rng(7);
  std::normal_distribution<float> nd(0.f, 1.f);
  std::vector<__half> fn((size_t)kM * kWide), up(kUpHalves);
  for (auto& v : fn) v = __float2half(nd(rng) * 0.02f);
  for (auto& v : up) v = __float2half(nd(rng) * 0.05f);
  std::vector<float> H((size_t)16 * kWide), w1(kWide);
  for (auto& v : H) v = nd(rng);
  for (auto& v : w1) v = 1.f + 0.1f * nd(rng);
  // Pack fn: [s][nt][kstep][16 cols][16 k]; pack up: [s][dt][kstep][16 cols][16 k].
  std::vector<__half> fnp(kFnHalves, __float2half(0.f)), upp(kUpHalves);
  for (int s = 0; s < 4; ++s)
    for (int n = 0; n < kNT * 16; ++n)
      for (int k = 0; k < kHidden; ++k) {
        const size_t at = ((((size_t)s * kNT + n / 16) * (kHidden / 16) + k / 16) * 16 + n % 16) * 16 + k % 16;
        if (n < kM) fnp[at] = fn[(size_t)n * kWide + s * kHidden + k];
      }
  for (int s = 0; s < 4; ++s)
    for (int d = 0; d < kHidden; ++d)
      for (int k = 0; k < kRank; ++k)
        upp[((((size_t)s * kDT + d / 16) * (kRank / 16) + k / 16) * 16 + d % 16) * 16 + k % 16] =
            up[((size_t)s * kHidden + d) * kRank + k];
  std::vector<Site> sites(nsets);
  for (auto& s : sites) {
    HIP_CHECK(hipMalloc(&s.fnp, kFnHalves * 2));
    HIP_CHECK(hipMalloc(&s.upp, kUpHalves * 2));
    HIP_CHECK(hipMemcpy(s.fnp, fnp.data(), kFnHalves * 2, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(s.upp, upp.data(), kUpHalves * 2, hipMemcpyHostToDevice));
  }
  Bufs b;
  HIP_CHECK(hipMalloc(&b.H, H.size() * 4));
  HIP_CHECK(hipMalloc(&b.ss, 64 * 4));
  HIP_CHECK(hipMalloc(&b.part, (size_t)kKS2 * 16 * 4 * kNT * 16 * 4));
  HIP_CHECK(hipMalloc(&b.pss, (size_t)kKS2 * 16 * 4 * 4));
  HIP_CHECK(hipMalloc(&b.post, 64 * 4));
  HIP_CHECK(hipMalloc(&b.w1, kWide * 4));
  HIP_CHECK(hipMalloc(&b.x, 16 * kHidden * 4));
  HIP_CHECK(hipMalloc(&b.thi, 16 * kRank * 2));
  HIP_CHECK(hipMalloc(&b.tlo, 16 * kRank * 2));
  HIP_CHECK(hipMemcpy(b.H, H.data(), H.size() * 4, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(b.w1, w1.data(), kWide * 4, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemset(b.thi, 0, 16 * kRank * 2));
  HIP_CHECK(hipMemset(b.tlo, 0, 16 * kRank * 2));

  // Accuracy at R=8 against a double reference.
  {
    constexpr int R = 8;
    RunSite<R>(sites[0], b, 3);
    HIP_CHECK(hipDeviceSynchronize());
    std::vector<float> part((size_t)kKS * R * 4 * kNT * 16), xg((size_t)R * kHidden);
    HIP_CHECK(hipMemcpy(part.data(), b.part, part.size() * 4, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(xg.data(), b.x, xg.size() * 4, hipMemcpyDeviceToHost));
    std::vector<double> dref;
    std::vector<float> xref;
    Reference(R, fn, up, H, w1, dref, xref);
    double worst = 0, scale = 0;
    for (int r = 0; r < R; ++r)
      for (int s = 0; s < 4; ++s)
        for (int m = 0; m < kM; ++m) {
          double d = 0;
          for (int kc = 0; kc < kKS; ++kc) d += part[((size_t)(kc * R + r) * 4 + s) * (kNT * 16) + m];
          worst = std::max(worst, std::fabs(d - dref[((size_t)r * kM + m) * 4 + s]));
          scale = std::max(scale, std::fabs(dref[((size_t)r * kM + m) * 4 + s]));
        }
    int ulp = 0, big = 0;
    for (size_t i = 0; i < xg.size(); ++i) {
      const float want = __half2float(__float2half(xref[i]));
      if (xg[i] != want) {
        ++ulp;
        const float step = std::fabs(want) * 1.f / 1024;
        if (std::fabs(xg[i] - want) > 1.01f * step) ++big;
      }
    }
    std::printf("accuracy R=8: dots max abs err %.3g (max |dot| %.3g); x fp16 differs from the rounded "
                "double reference at %d of %zu (%d by more than 1 ulp)\n",
                worst, scale, ulp, xg.size(), big);
  }

  // Stream reference over the same per-site bytes.
  {
    uint32_t* sink;
    HIP_CHECK(hipMalloc(&sink, 64));
    bench::EventTimer t;
    for (const auto& s : sites) hipLaunchKernelGGL(Stream, 160, 256, 0, nullptr, (const uint4*)s.fnp, kFnHalves / 8, sink);
    t.Begin();
    for (int k = 0; k < 4; ++k)
      for (const auto& s : sites) {
        hipLaunchKernelGGL(Stream, 160, 256, 0, nullptr, (const uint4*)s.fnp, kFnHalves / 8, sink);
        hipLaunchKernelGGL(Stream, 160, 256, 0, nullptr, (const uint4*)s.upp, kUpHalves / 8, sink);
      }
    const double us = t.EndMs() * 1e3 / (4 * sites.size());
    std::printf("stream ref: fn+up per site %6.1f us (%4.1f%% of DRAM), %zu sets\n", us,
                (kFnHalves + kUpHalves) * 2 / 245.5e9 / us * 1e8, sites.size());
  }
  {
    constexpr int R = 8;
    std::vector<float> x1((size_t)R * kHidden), x2(x1.size());
    RunSite<R>(sites[0], b, 3);
    HIP_CHECK(hipMemcpy(x1.data(), b.x, x1.size() * 4, hipMemcpyDeviceToHost));
    RunSite<R>(sites[0], b, 3, true);
    HIP_CHECK(hipMemcpy(x2.data(), b.x, x2.size() * 4, hipMemcpyDeviceToHost));
    int diff = 0;
    for (size_t i = 0; i < x1.size(); ++i) diff += x1[i] != x2[i];
    std::printf("v2 vs v1 x at R=8: %d of %zu differ (K split order changes)\n", diff, x1.size());
  }
  for (bool v2 : {false, true}) {
    Time<1>(sites, b, v2);
    Time<4>(sites, b, v2);
    Time<8>(sites, b, v2);
    Time<16>(sites, b, v2);
  }
  return 0;
}
