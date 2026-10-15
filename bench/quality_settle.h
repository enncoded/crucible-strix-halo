// Test-only copy of longctx/bench/qsa_settle.h, with bounded counter snapshots.
// This does not change the engine's startup policy or any system setting.
#pragma once
#include <chrono>
#include <filesystem>
#include <fstream>
#include <thread>
#include <vector>
#include <unistd.h>
#include "../engine/gfx1151/common.h"

namespace crucible {
struct TraceSpan { const void* pointer; size_t bytes; };
inline std::vector<TraceSpan> trace_spans;
inline void TraceBuffer(const void* pointer, size_t bytes) { trace_spans.push_back({pointer,bytes}); }

__global__ void TouchPagesKernel(const unsigned char* input, size_t bytes, unsigned* sink) {
  const int t = blockIdx.x*blockDim.x+threadIdx.x;
  unsigned value = 0;
  for (size_t i = (size_t)t*4096; i < bytes; i += (size_t)gridDim.x*blockDim.x*4096)
    value ^= input[i];
  sink[t] = value;
}

inline uint64_t MigrationCounter() {
  std::ifstream file("/proc/vmstat");
  std::string name; uint64_t value;
  while (file >> name >> value) if (name == "pgmigrate_success") return value;
  std::fprintf(stderr,"missing migration counter\n"); std::exit(1);
}

inline void CounterSnapshot(const char* phase) {
  std::ifstream file("/proc/vmstat");
  std::string name; uint64_t value;
  std::fprintf(stderr,"counters %s pid=%d",phase,getpid());
  while (file >> name >> value)
    if (name == "pgmigrate_success" || name.starts_with("compact_"))
      std::fprintf(stderr," %s=%llu",name.c_str(),(unsigned long long)value);
  std::error_code error;
  const std::string dir = "/sys/class/kfd/kfd/proc/" + std::to_string(getpid());
  const std::filesystem::directory_iterator entries(dir,error);
  if (error) std::fprintf(stderr," kfd_unavailable=1");
  for (const auto& entry : entries) {
    if (!entry.path().filename().string().starts_with("stats_")) continue;
    std::ifstream evicted(entry.path()/"evicted_ms");
    if (evicted >> value) std::fprintf(stderr," evicted_ms=%llu",(unsigned long long)value);
  }
  std::fprintf(stderr,"\n");
}

inline void TouchAndSettle() {
  unsigned* sink;
  HIP_OK(hipMalloc(&sink,80*256*sizeof(unsigned)));
  size_t total = 0;
  for (const auto& span : trace_spans) {
    TouchPagesKernel<<<80,256>>>(static_cast<const unsigned char*>(span.pointer),span.bytes,sink);
    total += span.bytes;
  }
  HIP_OK(hipDeviceSynchronize()); HIP_OK(hipFree(sink));
  uint64_t previous = MigrationCounter();
  int stable = 0, seconds = 0;
  while (seconds < 30 && stable < 3) {
    std::this_thread::sleep_for(std::chrono::seconds(1)); ++seconds;
    const uint64_t current = MigrationCounter();
    stable = current == previous ? stable+1 : 0; previous = current;
  }
  std::fprintf(stderr,"settle touched_bytes=%zu buffers=%zu wait_seconds=%d stable_seconds=%d\n",
               total,trace_spans.size(),seconds,stable);
  if (stable < 3) { std::fprintf(stderr,"migration did not settle\n"); std::exit(1); }
}
}  // namespace crucible
