// a session replay tail save pinned its snapshot chunks on the batch worker (0.65 s for 3.5 GB at 128K, every
// decoder idle). This probe measures (1) a save with a cold pool (pin, then copy: the old path) against a save from
// the reserve (snapshot_bytes.h Reserve: the helper pinned the chunks first), and (2) whether the helper's pins slow a
// chain of decode-like kernel passes on the null stream while they run (ABAB rounds).
// Usage: snapshot_pin_bench [MiB=3310] [passes=40] [rounds=3]
#include "common.h"
#include <numeric>
#define HIP_OK(x) HIP_CHECK(x)
#include "../engine/runtime/snapshot_bytes.h"

namespace {
using crucible::SnapshotBytes;
using crucible::SnapshotChunkPool;
struct Span { const void* pointer; size_t bytes; };

double Now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

// One read pass over `count` uint4 values (a memory-bound decode kernel); the sum keeps the reads.
__global__ void ReadKernel(const uint4* src, size_t count, unsigned* out) {
  unsigned sum = 0;
  for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count; i += size_t(gridDim.x) * blockDim.x)
    sum += src[i].x ^ src[i].w;
  if (sum == 0x9e3779b9u) *out = sum;
}

void WaitReserve(SnapshotChunkPool& pool) {
  while (pool.Filling()) std::this_thread::sleep_for(std::chrono::milliseconds(1));
}

// A save of `spans` as ConversationStateIO::Save does it: take the chunks, then the copy kernel and its wait.
void Save(const std::vector<Span>& spans, size_t total, const char* name) {
  const double t0 = Now();
  SnapshotBytes bytes(total);
  const double t1 = Now();
  crucible::CopySnapshotSpans(spans, bytes, true);
  const double t2 = Now();
  std::printf("%s: chunks %.1f ms (%.2f GB/s), copy %.1f ms (%.1f GB/s), save %.1f ms\n", name, (t1 - t0) * 1e3,
              total / (t1 - t0) / 1e9, (t2 - t1) * 1e3, total / (t2 - t1) / 1e9, (t2 - t0) * 1e3);
}

// Decode-like passes: `launches` read kernels, then the host waits (a pass reads its tokens).
std::vector<double> Passes(const uint4* buffer, size_t count, unsigned* out, int passes, int launches) {
  std::vector<double> times;
  for (int pass = 0; pass < passes; ++pass) {
    const double t0 = Now();
    for (int i = 0; i < launches; ++i) hipLaunchKernelGGL(ReadKernel, dim3(1024), dim3(256), 0, 0, buffer, count, out);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipStreamSynchronize(0));
    times.push_back((Now() - t0) * 1e3);
  }
  return times;
}

void Report(const char* name, const std::vector<double>& times) {
  const double mean = std::accumulate(times.begin(), times.end(), 0.0) / double(times.size());
  std::printf("%s: passes %zu, mean %.2f ms, median %.2f ms, max %.2f ms\n", name, times.size(), mean,
              bench::Median(times), *std::max_element(times.begin(), times.end()));
}
}  // namespace

int main(int argc, char** argv) {
  const size_t mib = argc > 1 ? std::stoul(argv[1]) : 3310;  // 3.47 GB: one 128K tail of the floor-gap run
  const int passes = argc > 2 ? std::stoi(argv[2]) : 40;
  const int rounds = argc > 3 ? std::stoi(argv[3]) : 3;
  const size_t total = mib << 20;
  constexpr int kSpans = 400;
  const size_t each = total / kSpans / 16 * 16;
  uint8_t* device = nullptr;
  HIP_CHECK(hipMalloc(&device, each * kSpans));
  HIP_CHECK(hipMemset(device, 0x5a, each * kSpans));
  std::vector<Span> spans;
  for (int i = 0; i < kSpans; ++i) spans.push_back({device + size_t(i) * each, each});
  const size_t bytes = each * kSpans;
  // The pass buffer: 256 MiB read per launch (about 1.1 ms at 240 GB/s), 75 launches (about a decode pass).
  const size_t pass_count = (size_t(256) << 20) / sizeof(uint4);
  uint4* pass_buffer = nullptr;
  unsigned* out = nullptr;
  HIP_CHECK(hipMalloc(&pass_buffer, pass_count * sizeof(uint4)));
  HIP_CHECK(hipMemset(pass_buffer, 1, pass_count * sizeof(uint4)));
  HIP_CHECK(hipMalloc(&out, sizeof(unsigned)));
  constexpr int kLaunches = 75;
  std::printf("# %.1f MiB in %d spans; passes of %d launches over 256 MiB\n", bytes / 1048576.0, kSpans, kLaunches);
  auto& pool = SnapshotChunkPool::Instance();
  bench::Sampler sampler;
  sampler.Start();
  for (int round = 0; round < rounds; ++round) {
    // (1) The old path: every chunk pinned on this thread, then the copy.
    pool.SetPinnedLimit(2 * bytes + SnapshotChunkPool::kChunkBytes);
    Save(spans, bytes, "cold save");
    pool.SetPinnedLimit(0);  // frees every idle chunk (outside the timed parts)
    // The new path: the helper pinned the reserve first.
    pool.SetPinnedLimit(2 * bytes + SnapshotChunkPool::kChunkBytes);
    pool.Reserve(bytes);
    WaitReserve(pool);
    Save(spans, bytes, "reserved save");
    pool.Reserve(0);
    WaitReserve(pool);
    pool.SetPinnedLimit(0);
    // (2) A: passes alone. B: the same passes while the helper pins `bytes`.
    Report("passes alone", Passes(pass_buffer, pass_count, out, passes, kLaunches));
    pool.SetPinnedLimit(2 * bytes + SnapshotChunkPool::kChunkBytes);
    const double start = Now();
    pool.Reserve(bytes);
    std::vector<double> during, after;
    double pinned_at = 0;
    while (int(during.size() + after.size()) < passes) {
      const bool filling = pool.Filling();
      if (!filling && !pinned_at) pinned_at = Now();
      const auto one = Passes(pass_buffer, pass_count, out, 1, kLaunches);
      (filling ? during : after).push_back(one.front());
    }
    WaitReserve(pool);
    if (!pinned_at) pinned_at = Now();
    std::printf("helper pinned %.1f MiB in %.1f ms (%.2f GB/s) during the passes\n", bytes / 1048576.0,
                (pinned_at - start) * 1e3, bytes / (pinned_at - start) / 1e9);
    if (!during.empty()) Report("passes while pinning", during);
    if (!after.empty()) Report("passes after pinning", after);
    pool.Reserve(0);
    WaitReserve(pool);
    pool.SetPinnedLimit(0);
  }
  sampler.Stop();
  std::printf("sclk mean %.0f MHz\n", sampler.MeanMhz());
  pool.StopReserve();
  HIP_CHECK(hipFree(device));
  HIP_CHECK(hipFree(pass_buffer));
  HIP_CHECK(hipFree(out));
  return 0;
}
