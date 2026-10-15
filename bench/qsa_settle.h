// Benchmark-only page touching and migration settling. No system settings change.
#pragma once
#include <thread>
#include <vector>

namespace crucible {
struct TraceSpan { const void* pointer; size_t bytes; };
inline std::vector<TraceSpan> trace_spans;
void TraceBuffer(const void* pointer, size_t bytes) { trace_spans.push_back({pointer,bytes}); }

__global__ void TouchPagesKernel(const unsigned char* input, size_t bytes, unsigned* sink) {
  const int t = blockIdx.x*blockDim.x+threadIdx.x;
  unsigned value = 0;
  for (size_t i = (size_t)t*4096; i < bytes; i += (size_t)gridDim.x*blockDim.x*4096)
    value ^= input[i];
  sink[t] = value;
}

uint64_t MigrationCounter() {
  std::ifstream file("/proc/vmstat");
  if (!file) { std::fprintf(stderr,"cannot read vmstat\n"); std::exit(1); }
  std::string name; uint64_t value;
  while (file >> name >> value) if (name == "pgmigrate_success") return value;
  std::fprintf(stderr,"missing migration counter\n"); std::exit(1);
}

void TouchAndSettle() {
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
