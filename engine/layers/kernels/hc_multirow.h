// Exact multi-row HC paths. Each row keeps the R=1 contraction order.
#pragma once

#include "../ops.h"

namespace crucible {

// Eight output rows share one stream's activation tile. The tile is small
// enough for LDS at R=8, unlike all 2560 activation columns at once.
template <int R, bool Prefetch = true, int Outputs = 1>
__global__ void __launch_bounds__(256) HcRowsDotsShared(
    const __half* __restrict__ fn, int M, const float* __restrict__ H,
    float* __restrict__ dots, float* __restrict__ ss) {
  PublishWeights(fn);
  __shared__ float activation[R][256];
  const int lane = threadIdx.x & 31, wave = threadIdx.x >> 5, s = blockIdx.y;
  const int base = __builtin_amdgcn_readfirstlane(blockIdx.x * 8 * Outputs + wave);
  if ((int)blockIdx.x == (M + 8 * Outputs - 1) / (8 * Outputs)) {
    if (wave < R) {
      const float* h = H + ((size_t)wave * kStreams + s) * kHidden;
      float a = 0.f;
      for (int d = lane; d < kHidden; d += 32) a += h[d] * h[d];
      a = WaveSum(a);
      if (lane == 0) ss[wave * kStreams + s] = a;
    }
    return;
  }
  uint4 raw[Outputs][10];
  if constexpr (Prefetch) {
#pragma unroll
    for (int e = 0; e < Outputs; ++e) {
      const int m = base + e * 8;
      const __half* wp = fn + (size_t)m * kWide + s * kHidden + lane * 8;
      if (m < M) {
#pragma unroll
        for (int k = 0; k < 10; ++k) raw[e][k] = LoadNT(wp + k * 256);
      }
    }
  }
  float acc[Outputs][R] = {};
#pragma unroll
  for (int k = 0; k < 10; ++k) {
#pragma unroll
    for (int r = 0; r < R; ++r)
      activation[r][threadIdx.x] = H[((size_t)r * kStreams + s) * kHidden + k * 256 + threadIdx.x];
    __syncthreads();
#pragma unroll
    for (int e = 0; e < Outputs; ++e) {
      const int m = base + e * 8;
      if (m < M) {
        if constexpr (!Prefetch)
          raw[e][k] = LoadNT(fn + (size_t)m * kWide + s * kHidden + k * 256 + lane * 8);
        const __half* hw = reinterpret_cast<const __half*>(&raw[e][k]);
#pragma unroll
        for (int r = 0; r < R; ++r) {
          asm volatile("" ::: "memory");
          const float4* h = reinterpret_cast<const float4*>(&activation[r][lane * 8]);
          const float4 a = h[0], b = h[1];
          acc[e][r] += __half2float(hw[0]) * a.x + __half2float(hw[1]) * a.y +
                       __half2float(hw[2]) * a.z + __half2float(hw[3]) * a.w +
                       __half2float(hw[4]) * b.x + __half2float(hw[5]) * b.y +
                       __half2float(hw[6]) * b.z + __half2float(hw[7]) * b.w;
        }
      }
    }
    __syncthreads();
  }
#pragma unroll
  for (int e = 0; e < Outputs; ++e) {
    const int m = base + e * 8;
    if (m < M) {
#pragma unroll
      for (int r = 0; r < R; ++r) {
        const float a = WaveSum(acc[e][r]);
        if (lane == 0) dots[((size_t)r * M + m) * kStreams + s] = a;
      }
    }
  }
}

enum class HcMixMode { Serial, Parallel, Pair, Quad, Ordered };

template <int R, HcMixMode Mode = HcMixMode::Serial>
__global__ void __launch_bounds__(256) HcRowsMix(
    const __half* __restrict__ up, const float* __restrict__ w1,
    const float* __restrict__ dots, int M, const float* __restrict__ ss,
    const float* __restrict__ H, float* __restrict__ mixed, float* __restrict__ post) {
  PublishWeights(up);
  __shared__ float t[R][kRank], inv[R][kStreams];
  if (threadIdx.x < R * kStreams) {
    const int r = threadIdx.x / kStreams, s = threadIdx.x % kStreams;
    inv[r][s] = rsqrtf(ss[r * kStreams + s] / kHidden + kEps);
  }
  __syncthreads();
  for (int i = threadIdx.x; i < R * kRank; i += blockDim.x) {
    const int r = i / kRank, m = i % kRank;
    const float* dm = dots + ((size_t)r * M + m) * kStreams;
    float a = 0.f;
#pragma unroll
    for (int s = 0; s < kStreams; ++s) a += inv[r][s] * dm[s];
    t[r][m] = Silu(a / kStreams);
  }
  if (blockIdx.x == 0 && post != nullptr && threadIdx.x < R * kStreams) {
    const int r = threadIdx.x / kStreams, s = threadIdx.x % kStreams;
    const float* dm = dots + ((size_t)r * M + kRank + s) * kStreams;
    float a = 0.f;
#pragma unroll
    for (int i = 0; i < kStreams; ++i) a += inv[r][i] * dm[i];
    post[r * kStreams + s] = 2.f * Sigmoid(a / kStreams);
  }
  __syncthreads();
  const int lane = threadIdx.x & 31, s = lane >> 3, li = lane & 7;
  const int d0 = (blockIdx.x * 8 + (threadIdx.x >> 5)) * 2;
  uint4 raw[2][5];
#pragma unroll
  for (int e = 0; e < 2; ++e) {
    const __half* w = up + ((size_t)s * kHidden + d0 + e) * kRank + li * 8;
#pragma unroll
    for (int k = 0; k < 5; ++k) raw[e][k] = LoadNT(w + k * 64);
  }
#pragma unroll
  for (int e = 0; e < 2; ++e) {
    const int d = d0 + e;
    auto finish = [&](int r, float g) {
      g += __shfl_xor(g, 1);
      g += __shfl_xor(g, 2);
      g += __shfl_xor(g, 4);
      const size_t hi = ((size_t)r * kStreams + s) * kHidden + d;
      float v = Sigmoid(g) * H[hi] * inv[r][s] * w1[s * kHidden + d];
      v += __shfl_xor(v, 8);
      v += __shfl_xor(v, 16);
      if (lane == 0) mixed[(size_t)r * kHidden + d] = RoundH(v / kStreams);
    };
    if constexpr (Mode == HcMixMode::Parallel || Mode == HcMixMode::Ordered) {
      float acc[R] = {};
      unsigned ready = li * 8;
#pragma unroll
      for (int k = 0; k < 5; ++k) {
        const __half* hw = reinterpret_cast<const __half*>(&raw[e][k]);
#pragma unroll
        for (int r = 0; r < R; ++r) {
          const int c = k * 64 + (Mode == HcMixMode::Ordered ? ready : li * 8);
          asm volatile("" ::: "memory");
#pragma unroll
          for (int i = 0; i < 8; ++i) acc[r] += __half2float(hw[i]) * t[r][c + i];
          // The tied index stays unchanged. Its compiler dependency prevents
          // future LDS reads from keeping all R activation rows in registers.
          if constexpr (Mode == HcMixMode::Ordered)
            asm volatile("" : "+v"(ready) : "v"(acc[r]) : "memory");
        }
      }
#pragma unroll
      for (int r = 0; r < R; ++r) finish(r, acc[r]);
    } else if constexpr (Mode == HcMixMode::Pair || Mode == HcMixMode::Quad) {
      constexpr int Tile = Mode == HcMixMode::Pair ? 2 : 4;
#pragma unroll 1
      for (int base = 0; base < R; base += Tile) {
        float acc[Tile] = {};
#pragma unroll
        for (int k = 0; k < 5; ++k) {
          const __half* hw = reinterpret_cast<const __half*>(&raw[e][k]);
          const int c = k * 64 + li * 8;
#pragma unroll
          for (int j = 0; j < Tile; ++j) {
            if (base + j < R) {
#pragma unroll
              for (int i = 0; i < 8; ++i) acc[j] += __half2float(hw[i]) * t[base + j][c + i];
            }
          }
        }
#pragma unroll
        for (int j = 0; j < Tile; ++j)
          if (base + j < R) finish(base + j, acc[j]);
      }
    } else {
#pragma unroll 1
      for (int r = 0; r < R; ++r) {
        float acc = 0.f;
#pragma unroll
        for (int k = 0; k < 5; ++k) {
          const __half* hw = reinterpret_cast<const __half*>(&raw[e][k]);
          const int c = k * 64 + li * 8;
          asm volatile("" ::: "memory");
#pragma unroll
          for (int i = 0; i < 8; ++i) acc += __half2float(hw[i]) * t[r][c + i];
        }
        finish(r, acc);
      }
    }
  }
}

}  // namespace crucible
