// MALL lookahead: can a small helper kernel on a second stream keep DRAM busy
// through the launch gaps, ramps and tails of a dependent kernel sequence?
//
// The main stream runs a chain of dependent "matvec-like" kernels, each
// streaming its own chunk of a weight pool once (decode reads every weight
// once per token). Each kernel publishes its start offset. A persistent
// prefetcher with a few workgroups reads the pool ahead of the chain, never
// more than W bytes past the running kernel's start, so the data is still in
// the 32 MiB MALL when its kernel arrives.
//
//   lookahead [total_mib]
#include <cstdio>
#include <vector>

#include "common.h"

// External linkage on purpose: an internal global that is never read lets the
// compiler delete the store and every load feeding it.
__device__ uint32_t g_sink;
__device__ unsigned g_blocks;
__device__ unsigned g_slabs;

namespace {

constexpr int kThreads = 256;
constexpr size_t kSlab = 256 << 10;  // prefetch unit claimed by one workgroup

__global__ void __launch_bounds__(kThreads) ChunkKernel(const uint4* __restrict__ p, size_t n,
                                                        unsigned long long* progress,
                                                        unsigned long long start) {
  if (blockIdx.x == 0 && threadIdx.x == 0)
    __hip_atomic_store(progress, start, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  uint32_t acc = 0;
  const size_t stride = (size_t)gridDim.x * kThreads;
#pragma unroll 8
  for (size_t i = (size_t)blockIdx.x * kThreads + threadIdx.x; i < n; i += stride) {
    const uint4 v = p[i];
    acc ^= v.x ^ v.y ^ v.z ^ v.w;
  }
  if (acc == 0x9e3779b9u) g_sink = acc;  // keeps the loads alive
  if (threadIdx.x == 0) atomicAdd(&g_blocks, 1u);
}

// Stand-in for the forward's small dependent kernels (routing, recurrences).
// wall_clock64 ticks at a constant 100 MHz; the gfx11 shader cycle counter
// is only 20 bits wide.
__global__ void SpinKernel(long long ticks) {
  const long long t0 = wall_clock64();
  while (wall_clock64() - t0 < ticks) __builtin_amdgcn_s_sleep(1);
}

__global__ void __launch_bounds__(kThreads) PrefetchKernel(const uint4* __restrict__ base,
                                                           unsigned long long total,
                                                           unsigned long long window,
                                                           const unsigned long long* progress,
                                                           unsigned* next_slab) {
  __shared__ unsigned slab;
  uint32_t acc = 0;
  for (;;) {
    if (threadIdx.x == 0) {
      // Never read behind the running kernel: once the chain has passed a
      // slab, fetching it only steals bandwidth from the chain.
      const unsigned long long p =
          __hip_atomic_load(progress, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
      atomicMax(next_slab, (unsigned)(p / kSlab));
      slab = atomicAdd(next_slab, 1u);
    }
    __syncthreads();
    const unsigned long long off = (unsigned long long)slab * kSlab;
    __syncthreads();
    if (off >= total) break;
    // Throttle: stay within `window` bytes of the running kernel's start.
    if (threadIdx.x == 0) {
      long long spins = 0;
      while (off >= __hip_atomic_load(progress, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) + window) {
        __builtin_amdgcn_s_sleep(2);
        if (++spins > (1ll << 26)) break;  // never hang the GPU
      }
    }
    __syncthreads();
    if (threadIdx.x == 0) atomicAdd(&g_slabs, 1u);
    const uint4* p = base + off / 16;
    const size_t n = (size_t)std::min<unsigned long long>(kSlab, total - off) / 16;
#pragma unroll 16
    for (size_t i = threadIdx.x; i < n; i += kThreads) {
      const uint4 v = p[i];
      acc ^= v.x ^ v.y ^ v.z ^ v.w;
    }
  }
  if (acc == 0x9e3779b9u) g_sink = acc;
}

struct Case {
  std::vector<size_t> chunks;  // bytes per dependent kernel
  int spin_every;              // a SpinKernel after every Nth chunk (0 = none)
  long long spin_cycles;
};

double RunOnce(const Case& c, const uint8_t* region, int pf_blocks, size_t window, int grid,
               hipStream_t main, hipStream_t side, unsigned long long* progress,
               unsigned* next_slab) {
  size_t total = 0;
  for (size_t b : c.chunks) total += b;
  HIP_CHECK(hipMemset(progress, 0, sizeof(*progress)));
  HIP_CHECK(hipMemset(next_slab, 0, sizeof(*next_slab)));
  HIP_CHECK(hipDeviceSynchronize());
  bench::EventTimer t;
  if (pf_blocks > 0)
    hipLaunchKernelGGL(PrefetchKernel, dim3(pf_blocks), dim3(kThreads), 0, side,
                       reinterpret_cast<const uint4*>(region), (unsigned long long)total,
                       (unsigned long long)window, progress, next_slab);
  t.Begin(main);
  size_t off = 0;
  for (size_t i = 0; i < c.chunks.size(); ++i) {
    hipLaunchKernelGGL(ChunkKernel, dim3(grid), dim3(kThreads), 0, main,
                       reinterpret_cast<const uint4*>(region + off), c.chunks[i] / 16, progress,
                       (unsigned long long)off);
    off += c.chunks[i];
    if (c.spin_every && (i + 1) % c.spin_every == 0)
      hipLaunchKernelGGL(SpinKernel, dim3(1), dim3(32), 0, main, c.spin_cycles);
  }
  HIP_CHECK(hipGetLastError());
  const double ms = t.EndMs(main);
  HIP_CHECK(hipDeviceSynchronize());
  return ms;
}

}  // namespace

int main(int argc, char** argv) {
  setvbuf(stdout, nullptr, _IONBF, 0);
  const size_t total = (argc > 1 ? std::stoul(argv[1]) : 1024) << 20;
  constexpr int kRegions = 4;  // rotate so no rep starts with a warm MALL
  uint8_t* pool;
  HIP_CHECK(hipMalloc(&pool, total * kRegions));
  HIP_CHECK(hipMemset(pool, 1, total * kRegions));
  unsigned long long* progress;
  unsigned* next_slab;
  HIP_CHECK(hipMalloc(&progress, sizeof(*progress)));
  HIP_CHECK(hipMalloc(&next_slab, sizeof(*next_slab)));
  hipStream_t main, side;
  HIP_CHECK(hipStreamCreateWithFlags(&main, hipStreamNonBlocking));
  HIP_CHECK(hipStreamCreateWithFlags(&side, hipStreamNonBlocking));

  // A Flash-Next-like layer: HC dots 6.6, HC mix 6.6, projections 26.2, out 9.8,
  // HC 6.6 + 6.6, router 2.6, experts 14.3 and 7.1 MB, with small kernels between.
  std::vector<size_t> layer_mb10 = {66, 66, 262, 98, 66, 66, 26, 143, 71};
  std::vector<Case> cases;
  for (size_t mb : {1, 2, 4, 8}) {
    Case c{{}, 0, 0};
    for (size_t b = 0; b + (mb << 20) <= total; b += mb << 20) c.chunks.push_back(mb << 20);
    cases.push_back(c);
  }
  {
    Case c{{}, 1, 1500};  // 15 us of dependent non-weight work per kernel
    for (size_t b = 0;;) {
      bool full = false;
      for (size_t m : layer_mb10) {
        const size_t bytes = ((m << 20) / 10) & ~size_t(255);
        if (b + bytes > total) {
          full = true;
          break;
        }
        c.chunks.push_back(bytes);
        b += bytes;
      }
      if (full) break;
    }
    cases.push_back(c);
  }
  {
    unsigned zero = 0, got = 0;
    HIP_CHECK(hipMemcpyToSymbol(HIP_SYMBOL(g_blocks), &zero, 4));
    RunOnce(cases[0], pool, 0, 0, 80, main, side, progress, next_slab);
    HIP_CHECK(hipMemcpyFromSymbol(&got, HIP_SYMBOL(g_blocks), 4));
    std::printf("sanity: %zu chunk kernels x 80 blocks expected %zu, ran %u\n", cases[0].chunks.size(),
                cases[0].chunks.size() * 80, got);
  }
  const char* names[] = {"1 MiB x N", "2 MiB x N", "4 MiB x N", "8 MiB x N", "layer mix + 15us gaps"};
  std::printf("%-22s %4s %6s %8s %8s %8s %7s\n", "case", "pf", "W MiB", "ms", "GB/s", "vs base",
              "fetched");
  for (size_t ci = 0; ci < cases.size(); ++ci) {
    size_t bytes = 0;
    for (size_t b : cases[ci].chunks) bytes += b;
    double base = 0;
    struct Cfg { int pf; size_t w; };
    for (Cfg cfg : {Cfg{0, 0}, Cfg{2, 16}, Cfg{4, 16}, Cfg{8, 16}, Cfg{4, 8}, Cfg{4, 24}}) {
      std::vector<double> ms;
      unsigned zero = 0, slabs = 0;
      HIP_CHECK(hipMemcpyToSymbol(HIP_SYMBOL(g_slabs), &zero, 4));
      for (int rep = 0; rep < 8; ++rep)
        ms.push_back(RunOnce(cases[ci], pool + (rep % kRegions) * total, cfg.pf, cfg.w << 20, 80,
                             main, side, progress, next_slab));
      const double m = bench::Median(ms);
      if (cfg.pf == 0) base = m;
      HIP_CHECK(hipMemcpyFromSymbol(&slabs, HIP_SYMBOL(g_slabs), 4));
      std::printf("%-22s %4d %6zu %8.3f %8.1f %7.2fx %6.1f%%\n", names[ci], cfg.pf, cfg.w, m,
                  bytes / m / 1e6, base / m, 100.0 * slabs * kSlab / (8.0 * bytes));
    }
  }
  // Floor: the chain's own cost when every chunk already sits in the MALL.
  for (size_t mb : {1, 2, 4}) {
    Case c{{}, 0, 0};
    for (int i = 0; i < 256; ++i) c.chunks.push_back(mb << 20);
    std::vector<double> ms;
    for (int rep = 0; rep < 8; ++rep) {
      // Re-use one 16 MiB window: chunk i reads window offset (i * mb) % 16 MiB.
      bench::EventTimer t;
      t.Begin(main);
      for (int i = 0; i < 256; ++i)
        hipLaunchKernelGGL(ChunkKernel, dim3(80), dim3(kThreads), 0, main,
                           reinterpret_cast<const uint4*>(pool + ((i * (mb << 20)) % (16 << 20))),
                           (mb << 20) / 16, progress, 0ull);
      ms.push_back(t.EndMs(main));
    }
    const double m = bench::Median(ms);
    std::printf("MALL-resident %zu MiB x 256: %.3f ms, %.2f us per kernel, %.0f GB/s\n", mb, m,
                1000 * m / 256, 256.0 * (mb << 20) / m / 1e6);
  }
  return 0;
}
