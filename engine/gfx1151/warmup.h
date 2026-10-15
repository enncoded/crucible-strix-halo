#pragma once
#include <chrono>
#include <cstdio>
#include <fstream>
#include <string>
#include <thread>
#include <vector>
#include "common.h"
#include "migration_wait.h"

namespace crucible {

struct DeviceSpan { const void* pointer; size_t bytes; };

__global__ void WarmPagesKernel(const unsigned char* input, size_t bytes, unsigned* sink) {
  const int t = blockIdx.x*blockDim.x+threadIdx.x;
  unsigned value = 0;
  for (size_t i = (size_t)t*4096; i < bytes; i += (size_t)gridDim.x*blockDim.x*4096)
    value ^= input[i];
  sink[t] = value;
}

inline bool ReadMigrationCounter(uint64_t& result) {
  std::ifstream file("/proc/vmstat");
  std::string name; uint64_t value;
  while (file >> name >> value) {
    if (name == "pgmigrate_success") { result = value; return true; }
  }
  return false;
}

// Repeated page migration can evict all queues of a GPU process. Wait only at
// startup, and keep the wait bounded if other processes keep memory active.
inline void WarmDeviceBuffers(const std::vector<DeviceSpan>& spans) {
  using Clock = std::chrono::steady_clock;
  const auto begin = Clock::now();
  unsigned* sink;
  HIP_OK(hipMalloc(&sink,80*256*sizeof(unsigned)));
  size_t bytes = 0;
  for (const auto& span : spans) {
    WarmPagesKernel<<<80,256>>>(static_cast<const unsigned char*>(span.pointer),span.bytes,sink);
    bytes += span.bytes;
  }
  HIP_OK(hipDeviceSynchronize()); HIP_OK(hipFree(sink));
  const auto result = WaitForMigration(ReadMigrationCounter,[] {
    std::this_thread::sleep_for(std::chrono::seconds(1));
  });
  const double elapsed = std::chrono::duration<double>(Clock::now()-begin).count();
  const char* status = !result.available ? "counter_unavailable" : result.stable == 3 ? "ready" : "timeout";
  std::fprintf(stderr,"warmup touched=%zu bytes buffers=%zu elapsed=%.3f s migration_wait=%d s status=%s\n",
               bytes,spans.size(),elapsed,result.waited,status);
  if (result.stable != 3)
    std::fprintf(stderr,"warmup warning: migration did not settle; decode latency may vary\n");
}

}  // namespace crucible
