// Dense EXL3 ring kernels that only benches launch.
#pragma once
#include "../../engine/exl3/exl3_ring.h"

namespace crucible {

namespace lin_ring {
template <bool Front, size_t N = 0>
__device__ __forceinline__ void Dispatch(int K, const LinJob& job, const GemvItem& item,
                                        uint32_t* payload, Control& control, uint32_t* xs) {
  if (K == kSupportedDecodeK[N]) {
    if (threadIdx.x < 32 * kLoaders) Load<kSupportedDecodeK[N], Front>(job, item, payload, control);
    else Compute<kSupportedDecodeK[N]>(job, item, payload, control, xs);
  } else if constexpr (N + 1 < kSupportedDecodeK.size()) {
    Dispatch<Front, N + 1>(K, job, item, payload, control, xs);
  }
}
}  // namespace lin_ring

// offsets (optional): block b runs items [offsets[b], offsets[b + 1]), a
// balanced chain range (balanced_items.h); otherwise items are strided.
template <bool Front = false>
__global__ void __launch_bounds__(320) LinRingKernel(const LinJob* __restrict__ jobs,
                                                    const GemvItem* __restrict__ items, int n,
                                                    const int* __restrict__ offsets = nullptr) {
  __shared__ lin_ring::Control control;
  __shared__ uint32_t payload[lin_ring::kSlots * lin_ring::kSlotWords];
  extern __shared__ uint32_t xs[];
  if (blockIdx.x == 0 && threadIdx.x == 0) PublishWeights(jobs[items[0].job].tiles);
  const int first = offsets ? offsets[blockIdx.x] : (int)blockIdx.x;
  const int last = offsets ? offsets[blockIdx.x + 1] : n;
  const int step = offsets ? 1 : (int)gridDim.x;
  for (int it = first; it < last; it += step) {
    const GemvItem item = items[it];
    const LinJob job = jobs[item.job];
    const int K = __builtin_amdgcn_readfirstlane(job.K);
    const int span = job.I / 16 / job.S, a0 = item.s * span;
    const uint32_t* x = reinterpret_cast<const uint32_t*>(job.xh);
    control.Initialize(threadIdx.x);
    for (int i = threadIdx.x; i < span * 8; i += blockDim.x) xs[i] = x[a0 * 8 + i];
    __syncthreads();
    lin_ring::Dispatch<Front>(K, job, item, payload, control, xs);
    __syncthreads();
  }
}

}  // namespace crucible
