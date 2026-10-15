// Multi-row fp16 matvec (router, PLE dense) for verify rows.
//
// DenseGemvKernel gives each wave one output and reads every activation row
// from L2 again per wave: 513 router waves x R rows x 10 KB. Here a workgroup
// stages one 256-column chunk of all R rows in LDS and eight waves (times
// Outputs) share it. Each lane keeps DenseGemvKernel's per-lane expression and
// chunk order, so every row is bitwise equal to the R=1 kernel.
#pragma once

#include "ops.h"

namespace crucible {

template <int R, int Outputs>
__global__ void __launch_bounds__(256) DenseRowsKernel(const __half* __restrict__ W, int O,
                                                       const float* __restrict__ x, int ldx,
                                                       float* __restrict__ y, int ldy) {
  PublishWeights(W);
  __shared__ float act[R][256];
  const int lane = threadIdx.x & 31, wave = threadIdx.x >> 5;
  const int base = __builtin_amdgcn_readfirstlane(blockIdx.x * 8 * Outputs + wave);
  uint4 raw[Outputs][10];
#pragma unroll
  for (int e = 0; e < Outputs; ++e) {
    const int o = base + e * 8;
    if (o < O) {
      const __half* w = W + (size_t)o * kHidden + lane * 8;
#pragma unroll
      for (int k = 0; k < 10; ++k) raw[e][k] = LoadNT(w + k * 256);
    }
  }
  float acc[Outputs][R] = {};
#pragma unroll
  for (int k = 0; k < 10; ++k) {
#pragma unroll
    for (int r = 0; r < R; ++r) act[r][threadIdx.x] = x[(size_t)r * ldx + k * 256 + threadIdx.x];
    __syncthreads();
#pragma unroll
    for (int e = 0; e < Outputs; ++e) {
      if (base + e * 8 < O) {
        const __half* h = reinterpret_cast<const __half*>(&raw[e][k]);
#pragma unroll
        for (int r = 0; r < R; ++r) {
          // Keeps the LDS reads inside this row's step so the compiler cannot
          // hoist all rows' loads at once (register pressure, not values).
          asm volatile("" ::: "memory");
          const float4* xr = reinterpret_cast<const float4*>(&act[r][lane * 8]);
          const float4 a = xr[0], b = xr[1];
          acc[e][r] += __half2float(h[0]) * a.x + __half2float(h[1]) * a.y + __half2float(h[2]) * a.z +
                       __half2float(h[3]) * a.w + __half2float(h[4]) * b.x + __half2float(h[5]) * b.y +
                       __half2float(h[6]) * b.z + __half2float(h[7]) * b.w;
        }
      }
    }
    __syncthreads();
  }
#pragma unroll
  for (int e = 0; e < Outputs; ++e) {
    const int o = base + e * 8;
    if (o < O) {
#pragma unroll
      for (int r = 0; r < R; ++r) {
        const float v = WaveSum(acc[e][r]);
        if (lane == 0) y[(size_t)r * ldy + o] = v;
      }
    }
  }
}

template <int R, int Outputs>
inline void LaunchDenseRows(const __half* W, int O, const float* x, int ldx, float* y, int ldy,
                            hipStream_t stream = 0) {
  DenseRowsKernel<R, Outputs><<<(O + 8 * Outputs - 1) / (8 * Outputs), 256, 0, stream>>>(W, O, x, ldx, y, ldy);
}

}  // namespace crucible
