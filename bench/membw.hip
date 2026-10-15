// DRAM / cache bandwidth probes for gfx1151.
//
//   membw occ     launch-shape sweep on a cold 2 GiB stream (find the peak)
//   membw hot     same buffer re-read: L2 / MALL / DRAM plateaus by size
//   membw cold    one kernel per fresh chunk of an 8 GiB pool: what a kernel
//                 streaming S bytes it has never touched actually gets
//   membw alloc   hipMalloc vs host/managed/fine-grained allocations
//   membw rw      write-only and copy at the best read shape
#include <cstring>
#include <functional>
#include <map>

#include "common.h"

using bench::EventTimer;
using bench::Sampler;

constexpr uint32_t kSentinel = 0x9e3779b9u;

template <int UNROLL, bool NT, bool CHUNKED>
__global__ void ReadKernel(const uint4* __restrict__ p, size_t n,
                           uint32_t* out) {
  uint4 acc = {0, 0, 0, 0};
  auto eat = [&](uint4 v) {
    acc.x ^= v.x;
    acc.y ^= v.y;
    acc.z ^= v.z;
    acc.w ^= v.w;
  };
  auto load = [&](size_t i) -> uint4 {
    if constexpr (NT) {
      typedef unsigned int u32x4 __attribute__((ext_vector_type(4)));
      u32x4 v = __builtin_nontemporal_load(reinterpret_cast<const u32x4*>(p) + i);
      return make_uint4(v.x, v.y, v.z, v.w);
    }
    return p[i];
  };
  size_t begin, end, stride;
  if constexpr (CHUNKED) {
    // Each block owns one contiguous range; threads stride inside it.
    size_t per = (n + gridDim.x - 1) / gridDim.x;
    begin = blockIdx.x * per + threadIdx.x;
    end = std::min(n, (blockIdx.x + 1) * per);
    stride = blockDim.x;
  } else {
    begin = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    end = n;
    stride = (size_t)gridDim.x * blockDim.x;
  }
  size_t i = begin;
  for (; i + (UNROLL - 1) * stride < end; i += UNROLL * stride) {
    uint4 v[UNROLL];
#pragma unroll
    for (int u = 0; u < UNROLL; ++u) v[u] = load(i + u * stride);
#pragma unroll
    for (int u = 0; u < UNROLL; ++u) eat(v[u]);
  }
  for (; i < end; i += stride) eat(load(i));
  if ((acc.x ^ acc.y ^ acc.z ^ acc.w) == kSentinel) out[0] = acc.x;
}

__global__ void WriteKernel(uint4* __restrict__ p, size_t n, uint32_t seed) {
  size_t stride = (size_t)gridDim.x * blockDim.x;
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n;
       i += stride)
    p[i] = make_uint4(seed, (uint32_t)i, seed ^ (uint32_t)i, 7);
}

__global__ void CopyKernel(const uint4* __restrict__ src, uint4* __restrict__ dst,
                           size_t n) {
  size_t stride = (size_t)gridDim.x * blockDim.x;
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n;
       i += stride)
    dst[i] = src[i];
}

using ReadFn = void (*)(const uint4*, size_t, uint32_t*);

struct Shape {
  int unroll = 4;
  bool nt = false;
  bool chunked = false;
  int threads = 256;
  int blocks_per_cu = 8;
};

static ReadFn Pick(const Shape& s) {
#define K(U, N, C) \
  if (s.unroll == U && s.nt == N && s.chunked == C) return ReadKernel<U, N, C>;
#define K2(U) K(U, false, false) K(U, true, false) K(U, false, true) K(U, true, true)
  K2(1) K2(2) K2(4) K2(8) K2(16)
#undef K2
#undef K
  std::fprintf(stderr, "bad shape\n");
  std::exit(1);
}

static uint32_t* g_out;

static void LaunchRead(const Shape& s, const void* p, size_t bytes, int cus,
                       hipStream_t st = nullptr) {
  size_t n = bytes / sizeof(uint4);
  int blocks = s.blocks_per_cu * cus;
  // Small buffers: never launch more threads than 16-byte elements.
  size_t max_blocks = (n + s.threads - 1) / s.threads;
  if ((size_t)blocks > max_blocks) blocks = (int)std::max<size_t>(1, max_blocks);
  hipLaunchKernelGGL(Pick(s), dim3(blocks), dim3(s.threads), 0, st,
                     static_cast<const uint4*>(p), n, g_out);
}

static void Fill(void* p, size_t bytes, int cus) {
  hipLaunchKernelGGL(WriteKernel, dim3(cus * 16), dim3(256), 0, nullptr,
                     static_cast<uint4*>(p), bytes / 16, 12345u);
  HIP_CHECK(hipDeviceSynchronize());
}

static double TimeHot(const Shape& s, const void* p, size_t bytes, int cus,
                      int reps) {
  EventTimer t;
  for (int i = 0; i < 3; ++i) LaunchRead(s, p, bytes, cus);
  std::vector<double> ms;
  for (int r = 0; r < reps; ++r) {
    t.Begin();
    LaunchRead(s, p, bytes, cus);
    ms.push_back(t.EndMs());
  }
  return bench::Median(ms);
}

// Streams a pool once per pass so no read can hit the 32 MB MALL.
static double TimeCold(const Shape& s, const char* pool, size_t pool_bytes,
                       size_t bytes, int cus, int passes) {
  EventTimer t;
  std::vector<double> gbs;
  size_t off = 0;
  for (int pass = 0; pass < passes; ++pass) {
    if (off + bytes > pool_bytes) off = 0;
    t.Begin();
    LaunchRead(s, pool + off, bytes, cus);
    double ms = t.EndMs();
    gbs.push_back(bytes / ms / 1e6);
    off += bytes;
  }
  return bench::Median(gbs);
}

static void Occ(int cus) {
  const size_t kBytes = 2ull << 30;
  char* pool;
  const size_t kPool = 6ull << 30;
  HIP_CHECK(hipMalloc(&pool, kPool));
  Fill(pool, kPool, cus);
  std::printf("# occ: cold 2 GiB stream from a 6 GiB pool, median of 5\n");
  std::printf("unroll nt chunked threads blocks_per_cu GBps\n");
  double best = 0;
  Shape bs;
  Sampler smp;
  smp.Start();
  for (int nt = 0; nt < 2; ++nt)
    for (int ch = 0; ch < 2; ++ch)
      for (int u : {1, 2, 4, 8, 16})
        for (int th : {64, 128, 256, 512, 1024})
          for (int bpc : {1, 2, 4, 8, 16, 32}) {
            if (th * bpc > 1024 * 32) continue;  // 32 waves of 32 per CU max
            Shape s{u, nt != 0, ch != 0, th, bpc};
            double g = TimeCold(s, pool, kPool, kBytes, cus, 5);
            std::printf("%d %d %d %d %d %.1f\n", u, nt, ch, th, bpc, g);
            if (g > best) {
              best = g;
              bs = s;
            }
          }
  smp.Stop();
  std::printf("# best %.1f GB/s: unroll=%d nt=%d chunked=%d threads=%d bpc=%d  [%s]\n",
              best, bs.unroll, bs.nt, bs.chunked, bs.threads, bs.blocks_per_cu,
              smp.Summary().c_str());
  HIP_CHECK(hipFree(pool));
}

static Shape ParseShape(int argc, char** argv, int from) {
  Shape s;
  for (int i = from; i + 1 < argc; i += 2) {
    std::string k = argv[i];
    int v = std::atoi(argv[i + 1]);
    if (k == "--unroll") s.unroll = v;
    if (k == "--nt") s.nt = v != 0;
    if (k == "--chunked") s.chunked = v != 0;
    if (k == "--threads") s.threads = v;
    if (k == "--bpc") s.blocks_per_cu = v;
  }
  return s;
}

static void Hot(const Shape& s, int cus) {
  std::printf("# hot: same buffer re-read, median of 20\nbytes GBps us\n");
  char* p;
  const size_t kMax = 4ull << 30;
  HIP_CHECK(hipMalloc(&p, kMax));
  Fill(p, kMax, cus);
  for (size_t b = 64 << 10; b <= kMax; b *= 2) {
    for (size_t bb : {b, b + b / 2}) {
      if (bb > kMax) continue;
      double ms = TimeHot(s, p, bb, cus, 20);
      std::printf("%zu %.1f %.2f\n", bb, bb / ms / 1e6, ms * 1e3);
    }
  }
  HIP_CHECK(hipFree(p));
}

static void Cold(const Shape& s, int cus) {
  std::printf("# cold: one kernel per fresh chunk of an 8 GiB pool\n");
  std::printf("bytes GBps_median us_per_kernel\n");
  char* pool;
  const size_t kPool = 8ull << 30;
  HIP_CHECK(hipMalloc(&pool, kPool));
  Fill(pool, kPool, cus);
  for (size_t b = 64 << 10; b <= (2ull << 30); b *= 2) {
    for (size_t bb : {b, b + b / 2}) {
      int passes = (int)std::clamp<size_t>((1ull << 30) / bb, 5, 400);
      double g = TimeCold(s, pool, kPool, bb, cus, passes);
      std::printf("%zu %.1f %.2f\n", bb, g, bb / g / 1e3);
    }
  }
  HIP_CHECK(hipFree(pool));
}

static void Alloc(const Shape& s, int cus) {
  struct A {
    const char* name;
    std::function<void*(size_t)> make;
    std::function<void(void*)> free;
  };
  std::vector<A> kinds = {
      {"hipMalloc", [](size_t n) { void* p; HIP_CHECK(hipMalloc(&p, n)); return p; },
       [](void* p) { HIP_CHECK(hipFree(p)); }},
      {"hipExtMalloc_finegrained",
       [](size_t n) { void* p; HIP_CHECK(hipExtMallocWithFlags(&p, n, hipDeviceMallocFinegrained)); return p; },
       [](void* p) { HIP_CHECK(hipFree(p)); }},
      {"hipExtMalloc_uncached",
       [](size_t n) { void* p; HIP_CHECK(hipExtMallocWithFlags(&p, n, hipDeviceMallocUncached)); return p; },
       [](void* p) { HIP_CHECK(hipFree(p)); }},
      {"hipHostMalloc_coherent",
       [](size_t n) { void* p; HIP_CHECK(hipHostMalloc(&p, n, hipHostMallocCoherent)); return p; },
       [](void* p) { HIP_CHECK(hipHostFree(p)); }},
      {"hipHostMalloc_noncoherent",
       [](size_t n) { void* p; HIP_CHECK(hipHostMalloc(&p, n, hipHostMallocNonCoherent)); return p; },
       [](void* p) { HIP_CHECK(hipHostFree(p)); }},
      {"hipMallocManaged",
       [](size_t n) { void* p; HIP_CHECK(hipMallocManaged(&p, n)); return p; },
       [](void* p) { HIP_CHECK(hipFree(p)); }},
  };
  std::printf("# alloc: hot 1 MiB (L2), hot 16 MiB (MALL), hot 256 MiB, cold 1 GiB of a 4 GiB pool\n");
  std::printf("kind hot1M hot16M hot256M cold1G\n");
  for (auto& k : kinds) {
    const size_t kPool = 4ull << 30;
    char* p = static_cast<char*>(k.make(kPool));
    Fill(p, kPool, cus);
    double h1 = (1 << 20) / TimeHot(s, p, 1 << 20, cus, 50) / 1e6;
    double h16 = (16 << 20) / TimeHot(s, p, 16 << 20, cus, 50) / 1e6;
    double h256 = (256 << 20) / TimeHot(s, p, 256 << 20, cus, 20) / 1e6;
    double c1 = TimeCold(s, p, kPool, 1ull << 30, cus, 8);
    std::printf("%s %.1f %.1f %.1f %.1f\n", k.name, h1, h16, h256, c1);
    k.free(p);
  }
}

static void ReadWrite(const Shape& s, int cus) {
  const size_t kBytes = 2ull << 30;
  char *a, *b;
  HIP_CHECK(hipMalloc(&a, kBytes));
  HIP_CHECK(hipMalloc(&b, kBytes));
  Fill(a, kBytes, cus);
  Fill(b, kBytes, cus);
  EventTimer t;
  std::vector<double> w, c, r;
  Sampler smp;
  smp.Start();
  for (int i = 0; i < 10; ++i) {
    t.Begin();
    hipLaunchKernelGGL(WriteKernel, dim3(cus * s.blocks_per_cu), dim3(s.threads), 0,
                       nullptr, reinterpret_cast<uint4*>(a), kBytes / 16, (uint32_t)i);
    w.push_back(kBytes / t.EndMs() / 1e6);
    t.Begin();
    hipLaunchKernelGGL(CopyKernel, dim3(cus * s.blocks_per_cu), dim3(s.threads), 0,
                       nullptr, reinterpret_cast<const uint4*>(a),
                       reinterpret_cast<uint4*>(b), kBytes / 16);
    c.push_back(2 * kBytes / t.EndMs() / 1e6);
    t.Begin();
    LaunchRead(s, i % 2 ? a : b, kBytes, cus);
    r.push_back(kBytes / t.EndMs() / 1e6);
  }
  smp.Stop();
  std::printf("# 2 GiB, median of 10: read %.1f GB/s, write %.1f GB/s, copy %.1f GB/s (read+write)  [%s]\n",
              bench::Median(r), bench::Median(w), bench::Median(c),
              smp.Summary().c_str());
  HIP_CHECK(hipFree(a));
  HIP_CHECK(hipFree(b));
}

// MoE-like traffic: one kernel streams K chunks at random chunk-aligned
// offsets. A pool far larger than the GPU TLB reach shows page-walk cost.
__global__ void ScatterKernel(const uint4* __restrict__ pool,
                              const uint64_t* __restrict__ offs, int k,
                              size_t chunk_vec, uint32_t* out) {
  uint4 acc = {0, 0, 0, 0};
  for (int c = blockIdx.x; c < k; c += gridDim.x) {
    const uint4* p = pool + offs[c];
    size_t i = threadIdx.x;
    for (; i + 7 * blockDim.x < chunk_vec; i += 8 * blockDim.x) {
      uint4 v[8];
#pragma unroll
      for (int u = 0; u < 8; ++u) v[u] = p[i + u * blockDim.x];
#pragma unroll
      for (int u = 0; u < 8; ++u) {
        acc.x ^= v[u].x;
        acc.y ^= v[u].y;
        acc.z ^= v[u].z;
        acc.w ^= v[u].w;
      }
    }
    for (; i < chunk_vec; i += blockDim.x) {
      uint4 v = p[i];
      acc.x ^= v.x;
      acc.w ^= v.w;
    }
  }
  if ((acc.x ^ acc.y ^ acc.z ^ acc.w) == kSentinel) out[0] = acc.x;
}

static void Scatter(int cus, size_t pool_gib) {
  size_t pool_bytes = pool_gib << 30;
  char* pool;
  HIP_CHECK(hipMalloc(&pool, pool_bytes));
  Fill(pool, pool_bytes, cus);
  uint64_t* d_offs;
  HIP_CHECK(hipMalloc(&d_offs, 1 << 20));
  std::printf("# scatter: one kernel reads K random chunks (~512 MiB total), %zu GiB pool\n", pool_gib);
  std::printf("chunk_bytes span_gib threads blocks_per_wgp GBps\n");
  uint64_t rng = 88172645463325252ull;
  for (size_t chunk : {(size_t)64 << 10, (size_t)256 << 10, (size_t)640 << 10, (size_t)1280 << 10, (size_t)4 << 20}) {
    for (size_t span_gib : {(size_t)1, (size_t)4, pool_gib}) {
      size_t span = std::min(span_gib << 30, pool_bytes);
      int k = (int)std::min<size_t>((512ull << 20) / chunk, span / chunk);
      std::vector<uint64_t> offs(k);
      for (auto& o : offs) {
        rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
        o = (rng % (span / chunk)) * (chunk / 16);
      }
      HIP_CHECK(hipMemcpy(d_offs, offs.data(), k * 8, hipMemcpyHostToDevice));
      for (int th : {256, 512}) {
        for (int bpw : {2, 4, 8}) {
          bench::EventTimer t;
          std::vector<double> g;
          for (int r = 0; r < 6; ++r) {
            t.Begin();
            hipLaunchKernelGGL(ScatterKernel, dim3(cus * bpw), dim3(th), 0, nullptr,
                               reinterpret_cast<const uint4*>(pool), d_offs, k,
                               chunk / 16, g_out);
            double ms = t.EndMs();
            if (r > 0) g.push_back((double)k * chunk / ms / 1e6);
          }
          std::printf("%zu %zu %d %d %.1f\n", chunk, span >> 30, th, bpw, bench::Median(g));
        }
      }
    }
  }
  HIP_CHECK(hipFree(pool));
  HIP_CHECK(hipFree(d_offs));
}

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: membw occ|hot|cold|alloc|rw [--unroll N --nt 0|1 --chunked 0|1 --threads N --bpc N]\n");
    return 2;
  }
  int cus = bench::NumCUs();
  HIP_CHECK(hipMalloc(&g_out, 64));
  std::string mode = argv[1];
  Shape s = ParseShape(argc, argv, 2);
  if (mode == "occ") Occ(cus);
  else if (mode == "hot") Hot(s, cus);
  else if (mode == "cold") Cold(s, cus);
  else if (mode == "alloc") Alloc(s, cus);
  else if (mode == "rw") ReadWrite(s, cus);
  else if (mode == "scatter") Scatter(cus, argc > 2 ? std::atoi(argv[2]) : 48);
  else return 2;
  return 0;
}
