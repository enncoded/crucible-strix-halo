// Multi-row HC kernels that only benches launch.
#pragma once
#include "../../engine/layers/kernels/hc_multirow.h"

namespace crucible {

template <int R>
__global__ void __launch_bounds__(256) HcRowsDotsRegister(
    const __half* __restrict__ fn, int M, const float* __restrict__ H,
    float* __restrict__ dots, float* __restrict__ ss) {
  const int lane = threadIdx.x & 31;
  const int task = blockIdx.x * 8 + (threadIdx.x >> 5);
  const int m = __builtin_amdgcn_readfirstlane(task >> 2);
  const int s = __builtin_amdgcn_readfirstlane(task & 3);
  if (m > M) return;
  if (m == M) {
#pragma unroll 1
    for (int r = 0; r < R; ++r) {
      const float* h = H + ((size_t)r * kStreams + s) * kHidden;
      float a = 0.f;
      for (int d = lane; d < kHidden; d += 32) a += h[d] * h[d];
      a = WaveSum(a);
      if (lane == 0) ss[r * kStreams + s] = a;
    }
    return;
  }
  const __half* wp = fn + (size_t)m * kWide + s * kHidden + lane * 8;
  uint4 raw[10];
#pragma unroll
  for (int k = 0; k < 10; ++k) raw[k] = LoadNT(wp + k * 256);
#pragma unroll 1
  for (int r = 0; r < R; ++r) {
    float acc = 0.f;
#pragma unroll
    for (int k = 0; k < 10; ++k) {
      const __half* hw = reinterpret_cast<const __half*>(&raw[k]);
      asm volatile("" ::: "memory");
      const float4* hr = reinterpret_cast<const float4*>(
          H + (size_t)r * kWide + s * kHidden + k * 256 + lane * 8);
      const float4 a = hr[0], b = hr[1];
      acc += __half2float(hw[0]) * a.x + __half2float(hw[1]) * a.y +
             __half2float(hw[2]) * a.z + __half2float(hw[3]) * a.w +
             __half2float(hw[4]) * b.x + __half2float(hw[5]) * b.y +
             __half2float(hw[6]) * b.z + __half2float(hw[7]) * b.w;
    }
    acc = WaveSum(acc);
    if (lane == 0) dots[((size_t)r * M + m) * kStreams + s] = acc;
  }
}

}  // namespace crucible
