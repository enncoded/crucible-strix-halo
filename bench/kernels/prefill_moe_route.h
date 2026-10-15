// The pruned prefill MoE router that only benches launch.
#pragma once
#include "../../engine/layers/prefill_moe.h"

namespace crucible {

// MoeRouteKernel with candidate pruning; same selection, order and weights.
// T = 10th largest of the 16 wave maxima. At least 10 logits are >= T, so a
// logit below T has rank >= 10. Logits >= T form the candidate list, and each
// candidate's rank among the candidates equals its rank among all 512.
__global__ void __launch_bounds__(512) PrefillMoeRouteKernel(const float* __restrict__ logits,
    int* __restrict__ sel, float* __restrict__ wts) {
  __shared__ float wmax[16], cv[kExperts], sl[kTopK];
  __shared__ int ce[kExperts];
  __shared__ float threshold;
  __shared__ int count;
  const int r = blockIdx.x, e = threadIdx.x, lane = e & 31, wave = e >> 5;
  const float* lg = logits + (size_t)r * (kExperts + 1);
  const float v = lg[e];
  float m = v;
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor(m, o));
  if (lane == 0) wmax[wave] = m;
  if (e == 0) count = 0;
  __syncthreads();
  if (e < 16) {
    int rank = 0;
    for (int i = 0; i < 16; ++i) rank += (wmax[i] > wmax[e]) || (wmax[i] == wmax[e] && i < e);
    if (rank == kTopK - 1) threshold = wmax[e];
  }
  __syncthreads();
  const bool candidate = v >= threshold;
  if (candidate) {
    const int slot = atomicAdd(&count, 1);
    cv[slot] = v; ce[slot] = e;
  }
  __syncthreads();
  if (candidate) {
    int rank = 0;
    for (int i = 0; i < count; ++i) {
      const float o = cv[i];
      rank += (o > v) || (o == v && ce[i] < e);
    }
    if (rank < kTopK) {
      sel[r * kTopK + rank] = e;
      sl[rank] = v;
    }
  }
  __syncthreads();
  if (e < kTopK) {
    float sum = 0.f;
    for (int k = 0; k < kTopK; ++k) sum += __expf(sl[k] - sl[0]);
    wts[r * kSlots + e] = __expf(sl[e] - sl[0]) / sum;
  }
  if (e == 0) wts[r * kSlots + kTopK] = Sigmoid(lg[kExperts]);
}

}  // namespace crucible
