// Compute-only barriers let loader waves keep weight requests in flight.
#pragma once
#include <hip/hip_runtime.h>
#include <cstdint>

namespace crucible {

struct PersistentSync {
  uint32_t compute_arrivals, phase_ok;
};
struct ComputeContext {
  int t;
  uint32_t barrier_seq = 0, phase_seq = 0;
  PersistentSync* sync;
  uint32_t* grid;
  uint32_t* error;
};

__device__ __forceinline__ void PersistentWait(const uint32_t* flag, uint32_t target,
                                              int scope, uint32_t* error) {
  // Debug builds trap rather than leaving a broken persistent grid resident.
  uint32_t polls = 0;
  while (__hip_atomic_load(flag, __ATOMIC_RELAXED, scope) < target) {
    __builtin_amdgcn_s_sleep(1);
    if (++polls == (1u << 24)) {
      __hip_atomic_store(error, target, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
      __builtin_trap();
    }
  }
  asm volatile("" ::: "memory");
}

__device__ __forceinline__ void ComputeBarrier(ComputeContext& c) {
  const uint32_t target = ++c.barrier_seq * 8;
  __threadfence_block();
  if ((c.t & 31) == 0)
    __hip_atomic_fetch_add(&c.sync->compute_arrivals, 1u, __ATOMIC_RELAXED,
                            __HIP_MEMORY_SCOPE_WORKGROUP);
  PersistentWait(&c.sync->compute_arrivals, target, __HIP_MEMORY_SCOPE_WORKGROUP, c.error);
  __builtin_amdgcn_wave_barrier();
}

__device__ __forceinline__ void PersistentGridBarrier(ComputeContext& c) {
  // Every lane publishes its stores before the representative wave arrives.
  __threadfence();
  ComputeBarrier(c);
  const uint32_t phase = ++c.phase_seq;
  if (c.t == 0) {
    __hip_atomic_fetch_add(c.grid, 1u, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
    PersistentWait(c.grid, phase * gridDim.x, __HIP_MEMORY_SCOPE_AGENT, c.error);
    __atomic_thread_fence(__ATOMIC_ACQUIRE);
    __hip_atomic_store(&c.sync->phase_ok, phase, __ATOMIC_RELAXED,
                        __HIP_MEMORY_SCOPE_WORKGROUP);
  }
  PersistentWait(&c.sync->phase_ok, phase, __HIP_MEMORY_SCOPE_WORKGROUP, c.error);
  __atomic_thread_fence(__ATOMIC_ACQUIRE);
  __builtin_amdgcn_wave_barrier();
}

__device__ __forceinline__ float PersistentBlockSum(float v, float* red, ComputeContext& c) {
  v = WaveSum(v);
  ComputeBarrier(c);
  if ((c.t & 31) == 0) red[c.t >> 5] = v;
  ComputeBarrier(c);
  float result = 0.f;
#pragma unroll
  for (int i = 0; i < 8; ++i) result += red[i];
  return result;
}

}  // namespace crucible
