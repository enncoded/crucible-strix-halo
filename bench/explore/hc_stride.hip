// Probe: DRAM channel camping in the decode HC dots kernel. The fn rows are
// 20480 bytes apart and every wave starts at the same offset of its row, so
// waves may hit the same channels together (the prefill work found this for H
// rows 40960 bytes apart). Runs a copy of the engine's HcDotsKernel<1>
// (engine/layers/ops.h) with the row stride padded by 0, 64, 128 or 256 halves,
// over 96 distinct weight sets so that no set is in the MALL.
//
// Usage: hc_stride [sets]
#include <hip/hip_fp16.h>

#include "../common.h"

namespace {

constexpr int kHidden = 2560, kStreams = 4, kWide = 10240, kM = 324;

__device__ __forceinline__ float WaveSum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor(v, o, 32);
  return v;
}

__device__ __forceinline__ uint4 LoadNT(const void* p) {
  typedef unsigned int U32x4 __attribute__((ext_vector_type(4)));
  const U32x4 v = __builtin_nontemporal_load(reinterpret_cast<const U32x4*>(p));
  return make_uint4(v.x, v.y, v.z, v.w);
}

// Same body as HcDotsKernel<1> except the fn row stride (ld halves).
__global__ void __launch_bounds__(256) HcDots(const __half* __restrict__ fn, int ld,
                                              const float* __restrict__ H,
                                              float* __restrict__ dots) {
  const int w = blockIdx.x * 8 + (threadIdx.x >> 5);
  const int lane = threadIdx.x & 31;
  const int m = w >> 2, s = w & 3;
  if (m >= kM) return;
  const __half* wp = fn + (size_t)m * ld + s * kHidden + lane * 8;
  uint4 raw[10];
#pragma unroll
  for (int k = 0; k < 10; ++k) raw[k] = LoadNT(wp + k * 256);
  float acc = 0.f;
#pragma unroll
  for (int k = 0; k < 10; ++k) {
    const __half* hw = reinterpret_cast<const __half*>(&raw[k]);
    const float4* hr = reinterpret_cast<const float4*>(H + s * kHidden + k * 256 + lane * 8);
    const float4 a = hr[0], b = hr[1];
    acc += __half2float(hw[0]) * a.x + __half2float(hw[1]) * a.y + __half2float(hw[2]) * a.z +
           __half2float(hw[3]) * a.w + __half2float(hw[4]) * b.x + __half2float(hw[5]) * b.y +
           __half2float(hw[6]) * b.z + __half2float(hw[7]) * b.w;
  }
  const float v = WaveSum(acc);
  if (lane == 0) dots[m * kStreams + s] = v;
}

}  // namespace

int main(int argc, char** argv) {
  const int sets = argc > 1 ? atoi(argv[1]) : 96;
  float *H, *dots;
  HIP_CHECK(hipMalloc(&H, kWide * 4));
  HIP_CHECK(hipMalloc(&dots, kM * kStreams * 4));
  HIP_CHECK(hipMemset(H, 0, kWide * 4));
  const int blocks = (kM * kStreams + 7) / 8;
  for (int pad : {0, 64, 128, 256, 0}) {
    const int ld = kWide + pad;
    const size_t set_bytes = (size_t)kM * ld * 2;
    __half* fn;
    HIP_CHECK(hipMalloc(&fn, set_bytes * sets));
    HIP_CHECK(hipMemset(fn, 0, set_bytes * sets));
    std::vector<double> us;
    for (int rep = 0; rep < 4; ++rep) {
      bench::EventTimer t;
      t.Begin();
      for (int i = 0; i < sets; ++i)
        hipLaunchKernelGGL(HcDots, blocks, 256, 0, nullptr, fn + (size_t)i * kM * ld, ld, H, dots);
      const float ms = t.EndMs();
      if (rep) us.push_back(ms * 1000.0 / sets);
    }
    const double med = bench::Median(us);
    std::printf("pad %3d halves (row stride %6d B): %.2f us per call, %.1f GB/s logical\n", pad, ld * 2,
                med, (double)kM * kWide * 2 / med / 1e3);
    HIP_CHECK(hipFree(fn));
  }
  return 0;
}
