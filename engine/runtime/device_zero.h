// Device check that spans hold only zero bytes (gen --jobs state checks).
#pragma once
#include <stdexcept>
#include <vector>
#include "../gfx1151/warmup.h"

namespace crucible {
__global__ void NonzeroWordsKernel(const uint32_t* words, size_t count, unsigned* found) {
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < count; i += (size_t)gridDim.x * blockDim.x)
    if (words[i]) {
      atomicOr(found, 1u);
      return;
    }
}

inline bool DeviceSpansZero(const std::vector<DeviceSpan>& spans) {
  unsigned* found = nullptr;
  HIP_OK(hipMalloc(&found, sizeof(unsigned)));
  HIP_OK(hipMemset(found, 0, sizeof(unsigned)));
  for (const DeviceSpan& span : spans) {
    if (span.bytes % sizeof(uint32_t) || reinterpret_cast<uintptr_t>(span.pointer) % sizeof(uint32_t))
      throw std::invalid_argument("device zero check needs 4-byte aligned spans");
    if (span.bytes) NonzeroWordsKernel<<<256, 256>>>(static_cast<const uint32_t*>(span.pointer), span.bytes / 4, found);
  }
  HIP_OK(hipGetLastError());
  unsigned result = 0;
  HIP_OK(hipMemcpy(&result, found, sizeof(unsigned), hipMemcpyDeviceToHost));
  HIP_OK(hipFree(found));
  return result == 0;
}
}  // namespace crucible
