// Keep a coefficient load behind its fixed-order accumulator dependency.
#pragma once
#include <hip/hip_runtime.h>
#include <cstdint>

namespace crucible {
__device__ __forceinline__ float PersistentCoefficient(const float* p, float dependency) {
#if CRUCIBLE_PERSISTENT_GDN_DEPENDENT_LDS
  const uint32_t address = (uint32_t)(uintptr_t)p;
  float value;
  // The coefficient stays unchanged between compute barriers. A memory clobber
  // here makes the compiler retain extra copies of the private state array.
  asm volatile("ds_load_b32 %0, %1\n\ts_waitcnt lgkmcnt(0)"
               : "=v"(value) : "v"(address), "v"(dependency));
  return value;
#else
  return *p;
#endif
}
}  // namespace crucible
