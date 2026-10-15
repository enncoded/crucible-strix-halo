// Stable expert-major dispatch for prompt chunks. No atomics on output order.
#pragma once
#include "../../gfx1151/common.h"
#include "../../model/flashnext/prefill_launch.h"

namespace crucible::prefill {
constexpr int kDispatchTile = 256;

struct MoeDispatch {
  const int* selected;  // [rows][10], expert IDs already validated by router
  int* tile_counts;     // [512][ceil(rows*10/256)]
  int* counts;          // [512]
  int* offsets;         // [513], exclusive offsets in compact lists
  int* pairs;           // [rows*10], row*10 + original routing slot
  int* row_ids;         // [rows*10], gathered GEMM input rows
  int rows;
};

__device__ __forceinline__ unsigned MatchMask(bool p) {
  return (unsigned)__ballot(p);
}

// Shared integer atomics count routes, but never decide output order. Work
// scales with routes, not experts*routes; all writes have exclusive owners.
__global__ void __launch_bounds__(256) MoeCountKernel(MoeDispatch a) {
  __shared__ int histogram[kExperts];
  const int tile = blockIdx.x, t = threadIdx.x, i = tile * 256 + t;
  histogram[t] = histogram[t + 256] = 0;
  __syncthreads();
  if (i < a.rows * kTopK) atomicAdd(histogram + a.selected[i], 1);
  __syncthreads();
  a.tile_counts[t * gridDim.x + tile] = histogram[t];
  a.tile_counts[(t + 256) * gridDim.x + tile] = histogram[t + 256];
}

// Converts tile counts to per-expert tile prefixes in place; one block per
// expert, then one tiny global exclusive prefix. Integer results are exact.
// A launch of more than 6,553 rows has more than 256 tiles: the block scans
// them 256 at a time and carries the running total.
__global__ void __launch_bounds__(256) MoePrefixTilesKernel(MoeDispatch a, int tiles) {
  __shared__ int waves[8];
  const int e = blockIdx.x, t = threadIdx.x, lane = t & 31, wave = t >> 5;
  int carry = 0;
  for (int base = 0; base < tiles; base += 256) {
    const int idx = base + t;
    const int value = idx < tiles ? a.tile_counts[e * tiles + idx] : 0;
    int sum = value;
#pragma unroll
    for (int step = 1; step < 32; step <<= 1) {
      const int previous = __shfl_up(sum, step);
      if (lane >= step) sum += previous;
    }
    if (lane == 31) waves[wave] = sum;
    __syncthreads();
    for (int w = 0; w < wave; ++w) sum += waves[w];
    int total = 0;
    for (int w = 0; w < 8; ++w) total += waves[w];
    if (idx < tiles) a.tile_counts[e * tiles + idx] = carry + sum - value;
    carry += total;
    __syncthreads();  // waves[] is rewritten by the next 256 tiles
  }
  if (t == 0) a.counts[e] = carry;
}

__global__ void MoePrefixExpertsKernel(MoeDispatch a) {
  if (threadIdx.x == 0) {
    int n = 0;
    for (int e = 0; e < kExperts; ++e) {
      a.offsets[e] = n;
      n += a.counts[e];
    }
    a.offsets[kExperts] = n;
  }
}

__global__ void __launch_bounds__(256) MoeScatterKernel(MoeDispatch a) {
  __shared__ int expert[256];
  const int tile = blockIdx.x, t = threadIdx.x, i = tile * 256 + t;
  const int e = i < a.rows * kTopK ? a.selected[i] : -1;
  expert[t] = e;
  __syncthreads();
  if (e < 0) return;
  int rank = 0;
  for (int prev = 0; prev < t; ++prev) rank += expert[prev] == e;
  const int out = a.offsets[e] + a.tile_counts[e * gridDim.x + tile] + rank;
  a.pairs[out] = i;
  a.row_ids[out] = i / kTopK;
}

inline void LaunchMoeDispatch(MoeDispatch a, hipStream_t stream = nullptr) {
  if (a.rows < 1 || a.rows > kMaxPrefillLaunchRows) {
    std::fprintf(stderr, "MoE prefill requires 1..%d rows\n", kMaxPrefillLaunchRows);
    std::abort();
  }
  const int tiles = (a.rows * kTopK + 255) / 256;
  MoeCountKernel<<<tiles, 256, 0, stream>>>(a);
  MoePrefixTilesKernel<<<kExperts, 256, 0, stream>>>(a, tiles);
  MoePrefixExpertsKernel<<<1, 32, 0, stream>>>(a);
  MoeScatterKernel<<<tiles, 256, 0, stream>>>(a);
}
}  // namespace crucible::prefill
