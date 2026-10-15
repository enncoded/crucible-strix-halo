// Snapshot capture paths (oracle-cache item 8): where the time of a device-to-host state save goes.
// Usage: snapshot_copy_bench [MiB=730] [spans=200] [reps=5]
#include "common.h"
#include <cstring>
#include <numeric>
#define HIP_OK(x) HIP_CHECK(x)
#include "../engine/runtime/snapshot_bytes.h"

namespace {
struct Span { const void* src; size_t bytes; };

__global__ void GatherKernel(const uint4* const* srcs, const size_t* starts, const size_t* counts, int n, uint4* dst) {
  // One block range per span: blockIdx.y picks the span, x strides across it.
  const int s = blockIdx.y;
  if (s >= n) return;
  const uint4* src = srcs[s];
  uint4* out = dst + starts[s];
  const size_t count = counts[s];
  for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count; i += size_t(gridDim.x) * blockDim.x)
    out[i] = src[i];
}

double Now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

void Check(const std::vector<Span>& spans, const uint8_t* host, const std::vector<std::vector<uint8_t>>& want,
           const char* name) {
  size_t offset = 0;
  for (size_t i = 0; i < spans.size(); ++i) {
    if (std::memcmp(host + offset, want[i].data(), spans[i].bytes) != 0) {
      std::printf("MISMATCH %s span %zu\n", name, i);
      std::exit(1);
    }
    offset += spans[i].bytes;
  }
}
}  // namespace

int main(int argc, char** argv) {
  const size_t mib = argc > 1 ? std::stoul(argv[1]) : 730;
  const int count = argc > 2 ? std::stoi(argv[2]) : 200;
  const int reps = argc > 3 ? std::stoi(argv[3]) : 5;
  // Mixed span sizes, all multiples of 16 bytes, as in the snapshot layout (GDN state, rings, K, V, pools).
  std::vector<size_t> sizes(static_cast<size_t>(count));
  const size_t total_target = mib << 20;
  for (int i = 0; i < count; ++i) sizes[size_t(i)] = (i % 4 == 0 ? 3 : 1);
  const size_t weight = std::accumulate(sizes.begin(), sizes.end(), size_t(0));
  size_t total = 0;
  for (auto& size : sizes) { size = (total_target / weight * size) / 16 * 16; total += size; }
  std::vector<Span> spans;
  std::vector<std::vector<uint8_t>> want;
  for (size_t i = 0; i < sizes.size(); ++i) {
    void* p;
    HIP_CHECK(hipMalloc(&p, sizes[i]));
    std::vector<uint8_t> host(sizes[i]);
    for (size_t b = 0; b < host.size(); ++b) host[b] = uint8_t((b * 131 + i * 7) >> 3);
    HIP_CHECK(hipMemcpy(p, host.data(), host.size(), hipMemcpyHostToDevice));
    spans.push_back({p, sizes[i]});
    want.push_back(std::move(host));
  }
  std::printf("# %zu spans, %.1f MiB\n", spans.size(), total / 1048576.0);
  const auto rate = [&](double s) { return total / s / 1e9; };

  for (int rep = 0; rep < reps; ++rep) {
    // A: today's Save: allocate pinned, zero fill (vector resize), one hipMemcpy per span.
    double t0 = Now();
    uint8_t* fresh;
    HIP_CHECK(hipHostMalloc(&fresh, total));
    double t1 = Now();
    std::memset(fresh, 0, total);
    double t2 = Now();
    size_t offset = 0;
    for (const auto& span : spans) {
      HIP_CHECK(hipMemcpy(fresh + offset, span.src, span.bytes, hipMemcpyDeviceToHost));
      offset += span.bytes;
    }
    HIP_CHECK(hipStreamSynchronize(0));
    double t3 = Now();
    Check(spans, fresh, want, "today");
    std::printf("today: alloc %.1f ms, fill %.1f ms, copy %.1f ms (%.1f GB/s); total %.1f ms (%.1f GB/s)\n",
                (t1 - t0) * 1e3, (t2 - t1) * 1e3, (t3 - t2) * 1e3, rate(t3 - t2), (t3 - t0) * 1e3, rate(t3 - t0));
    HIP_CHECK(hipHostFree(fresh));

    // B: allocate pinned without fill, then copy.
    t0 = Now();
    HIP_CHECK(hipHostMalloc(&fresh, total));
    t1 = Now();
    offset = 0;
    for (const auto& span : spans) {
      HIP_CHECK(hipMemcpy(fresh + offset, span.src, span.bytes, hipMemcpyDeviceToHost));
      offset += span.bytes;
    }
    t2 = Now();
    Check(spans, fresh, want, "nofill");
    std::printf("nofill: alloc %.1f ms, copy (first touch) %.1f ms (%.1f GB/s)\n", (t1 - t0) * 1e3, (t2 - t1) * 1e3,
                rate(t2 - t1));

    // C: the same buffer again (pre-allocated arena, pages touched): sync copies, async copies, gather kernel.
    std::memset(fresh, 0xEE, total);
    t0 = Now();
    offset = 0;
    for (const auto& span : spans) {
      HIP_CHECK(hipMemcpy(fresh + offset, span.src, span.bytes, hipMemcpyDeviceToHost));
      offset += span.bytes;
    }
    t1 = Now();
    Check(spans, fresh, want, "arena-sync");
    std::memset(fresh, 0xEE, total);
    t2 = Now();
    offset = 0;
    for (const auto& span : spans) {
      HIP_CHECK(hipMemcpyAsync(fresh + offset, span.src, span.bytes, hipMemcpyDeviceToHost, 0));
      offset += span.bytes;
    }
    HIP_CHECK(hipStreamSynchronize(0));
    t3 = Now();
    Check(spans, fresh, want, "arena-async");
    std::printf("arena: sync copies %.1f ms (%.1f GB/s), async copies %.1f ms (%.1f GB/s)\n", (t1 - t0) * 1e3,
                rate(t1 - t0), (t3 - t2) * 1e3, rate(t3 - t2));

    // D: one gather kernel into the pinned arena (the GPU writes host memory directly).
    std::vector<const uint4*> srcs;
    std::vector<size_t> starts, counts;
    offset = 0;
    for (const auto& span : spans) {
      srcs.push_back(static_cast<const uint4*>(span.src));
      starts.push_back(offset / 16);
      counts.push_back(span.bytes / 16);
      offset += span.bytes;
    }
    const uint4** d_srcs; size_t *d_starts, *d_counts;
    HIP_CHECK(hipMalloc(&d_srcs, srcs.size() * sizeof(void*)));
    HIP_CHECK(hipMalloc(&d_starts, starts.size() * sizeof(size_t)));
    HIP_CHECK(hipMalloc(&d_counts, counts.size() * sizeof(size_t)));
    for (int pass = 0; pass < 2; ++pass) {
      std::memset(fresh, 0xEE, total);
      t0 = Now();
      HIP_CHECK(hipMemcpy(d_srcs, srcs.data(), srcs.size() * sizeof(void*), hipMemcpyHostToDevice));
      HIP_CHECK(hipMemcpy(d_starts, starts.data(), starts.size() * sizeof(size_t), hipMemcpyHostToDevice));
      HIP_CHECK(hipMemcpy(d_counts, counts.data(), counts.size() * sizeof(size_t), hipMemcpyHostToDevice));
      void* dst;
      HIP_CHECK(hipHostGetDevicePointer(&dst, fresh, 0));
      hipLaunchKernelGGL(GatherKernel, dim3(16, unsigned(spans.size())), dim3(256), 0, nullptr, d_srcs, d_starts,
                         d_counts, int(spans.size()), static_cast<uint4*>(dst));
      HIP_CHECK(hipGetLastError());
      HIP_CHECK(hipStreamSynchronize(0));
      t1 = Now();
      Check(spans, fresh, want, "kernel");
      std::printf("arena: gather kernel pass %d %.1f ms (%.1f GB/s)\n", pass, (t1 - t0) * 1e3, rate(t1 - t0));
    }
    HIP_CHECK(hipFree(d_srcs)); HIP_CHECK(hipFree(d_starts)); HIP_CHECK(hipFree(d_counts));
    HIP_CHECK(hipHostFree(fresh));

    // E: allocation flags for an arena block: non-coherent pinned memory.
    t0 = Now();
    HIP_CHECK(hipHostMalloc(&fresh, total, hipHostMallocNonCoherent));
    t1 = Now();
    std::memset(fresh, 0xEE, total);
    t2 = Now();
    offset = 0;
    for (const auto& span : spans) {
      HIP_CHECK(hipMemcpyAsync(fresh + offset, span.src, span.bytes, hipMemcpyDeviceToHost, 0));
      offset += span.bytes;
    }
    HIP_CHECK(hipStreamSynchronize(0));
    t3 = Now();
    Check(spans, fresh, want, "noncoherent");
    std::printf("noncoherent: alloc %.1f ms, async copies %.1f ms (%.1f GB/s)\n", (t1 - t0) * 1e3, (t3 - t2) * 1e3,
                rate(t3 - t2));
    HIP_CHECK(hipHostFree(fresh));
  }
  // F: the engine path (engine/runtime/snapshot_bytes.h): pooled 8 MiB chunks and one gather kernel, then a restore.
  struct DeviceSpan { const void* pointer; size_t bytes; };
  std::vector<DeviceSpan> device_spans;
  for (const auto& span : spans) device_spans.push_back({span.src, span.bytes});
  auto& pool = crucible::SnapshotChunkPool::Instance();
  pool.SetPinnedLimit(total + crucible::SnapshotChunkPool::kChunkBytes);
  for (int rep = 0; rep < reps + 1; ++rep) {
    const double t0 = Now();
    crucible::SnapshotBytes bytes(total);
    const double t1 = Now();
    crucible::CopySnapshotSpans(device_spans, bytes, true);
    const double t2 = Now();
    const auto flat = bytes.Copy();
    Check(spans, flat.data(), want, "engine-save");
    for (const auto& span : spans) HIP_CHECK(hipMemset(const_cast<void*>(span.src), 0, span.bytes));
    HIP_CHECK(hipDeviceSynchronize());
    const double t3 = Now();
    crucible::CopySnapshotSpans(device_spans, bytes, false);
    const double t4 = Now();
    for (size_t i = 0; i < spans.size(); ++i) {
      std::vector<uint8_t> back(spans[i].bytes);
      HIP_CHECK(hipMemcpy(back.data(), spans[i].src, back.size(), hipMemcpyDeviceToHost));
      if (back != want[i]) { std::printf("MISMATCH engine-restore span %zu\n", i); return 1; }
    }
    std::printf("engine %s: take %.1f ms, save copy %.1f ms (%.1f GB/s), save total %.1f ms (%.1f GB/s); "
                "restore %.1f ms (%.1f GB/s)\n", rep ? "warm" : "cold", (t1 - t0) * 1e3, (t2 - t1) * 1e3,
                rate(t2 - t1), (t2 - t0) * 1e3, rate(t2 - t0), (t4 - t3) * 1e3, rate(t4 - t3));
  }
  // Release cost of today's path for comparison: hipHostFree of one snapshot-sized block.
  {
    uint8_t* block;
    HIP_CHECK(hipHostMalloc(&block, total));
    std::memset(block, 1, total);
    const double t0 = Now();
    HIP_CHECK(hipHostFree(block));
    std::printf("hipHostFree of one %.0f MiB block: %.1f ms\n", total / 1048576.0, (Now() - t0) * 1e3);
  }
  for (const auto& span : spans) HIP_CHECK(hipFree(const_cast<void*>(span.src)));
  return 0;
}
