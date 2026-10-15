// Split the real dense ring's bytes-only gap to the synthetic probe:
// work balance, tile layout in memory, and chunk depth. Two loaders, eight
// compute waves. Repacked layouts hold the same tiles, so every column keeps
// its accumulation order and the full mode must stay bitwise equal.
//   layout 0  engine order: column-major tiles, eight 128K-byte runs per chunk
//   layout 1  item order: each chunk of an item is one contiguous run
//   layout 2  front order: chunk q of every item of a job is adjacent
// Usage: mega_ablate2 CKPT [grid=80]
#define main DenseHarnessMain
#include "mega_dense.hip"
#undef main

namespace ab2 {
constexpr int L = 2, C = 8, MaxK = 6;
template <int Chunk> constexpr int SlotWords = 8 * Chunk * 8 * MaxK;
// Chunk 8 keeps the LDS per block near Chunk 4 so four blocks fit one WGP.
template <int Chunk> constexpr int SlotsFor = Chunk == 4 ? 3 : 2;
struct Phase { const LinJob* jobs; const GemvItem* items; int n, start; };

// Byte offset of chunk vector vi relative to job.tiles.
template <int Layout, int Chunk, int T>
__device__ __forceinline__ int Offset(const LinJob& job, const GemvItem& item, int seq, int vi) {
  constexpr int ColVec = Chunk * T / 4, ChunkBytes = 8 * Chunk * T * 4;
  const int A = job.I / 16, span = A / job.S, count = span / Chunk;
  const int ncg = job.O / 128, idx = item.s * ncg + item.b0 / 8;
  if constexpr (Layout == 0) {
    const int col = vi / ColVec, off = vi % ColVec;
    return ((item.b0 + col) * A + item.s * span) * T * 4 + off * 16 + seq * Chunk * T * 4;
  } else if constexpr (Layout == 1) {
    return (idx * count + seq) * ChunkBytes + vi * 16;
  } else {
    return (seq * job.S * ncg + idx) * ChunkBytes + vi * 16;
  }
}

template <int K, int Layout, int Chunk>
__device__ __forceinline__ void Loader(const LinJob& job, const GemvItem& item, uint32_t* data,
                                      LdsStreamRing<SlotsFor<Chunk>, L, C>& ring, uint32_t base) {
  constexpr int Slots = SlotsFor<Chunk>;
  constexpr int T = 8 * K, NV = (8 * Chunk * T / 4 + 32 * L - 1) / (32 * L);
  constexpr int Total = 8 * Chunk * T / 4;
  const int lane = threadIdx.x & 31, lw = threadIdx.x >> 5;
  const int count = job.I / 16 / job.S / Chunk;
  const auto rs = StreamRsrc(job.tiles, (uint32_t)((size_t)job.I / 16 * (job.O / 16) * T * 4));
  auto ld = [&](int seq, int j) -> uint4 {
    const int vi = j * 32 * L + lw * 32 + lane;
    const auto raw = __builtin_amdgcn_raw_buffer_load_b128(
        rs, Offset<Layout, Chunk, T>(job, item, seq, vi < Total ? vi : Total - 1), 0, 0);
    return *reinterpret_cast<const uint4*>(&raw);
  };
#define A2_ONE(R, S, J) if constexpr (NV > J) R##J = ld(S, J)
#define A2_LD(R, S) A2_ONE(R,S,0);A2_ONE(R,S,1);A2_ONE(R,S,2);A2_ONE(R,S,3);A2_ONE(R,S,4);A2_ONE(R,S,5);A2_ONE(R,S,6);A2_ONE(R,S,7);A2_ONE(R,S,8);A2_ONE(R,S,9);A2_ONE(R,S,10);A2_ONE(R,S,11)
#define A2_STORE(R, J) if constexpr (NV > J) { if (J * 32 * L + lw * 32 + lane < Total) d[J * 32 * L] = R##J; }
#define A2_PUT(R, S) if ((S) < count) { const uint32_t g = base + (S); ring.WaitWritable(g); \
    uint4* d = reinterpret_cast<uint4*>(data + (g % Slots) * SlotWords<Chunk>) + lw * 32 + lane; \
    A2_STORE(R,0);A2_STORE(R,1);A2_STORE(R,2);A2_STORE(R,3);A2_STORE(R,4);A2_STORE(R,5); \
    A2_STORE(R,6);A2_STORE(R,7);A2_STORE(R,8);A2_STORE(R,9);A2_STORE(R,10);A2_STORE(R,11); ring.Publish(g, lane); }
  uint4 A0,A1,A2,A3,A4,A5,A6,A7,A8,A9,A10,A11,B0,B1,B2,B3,B4,B5,B6,B7,B8,B9,B10,B11;
  A2_LD(A, 0);
  for (int seq = 0; seq < count; seq += 2) {
    if (seq + 1 < count) A2_LD(B, seq + 1);
    A2_PUT(A, seq);
    if (seq + 2 < count) A2_LD(A, seq + 2);
    A2_PUT(B, seq + 1);
  }
#undef A2_PUT
#undef A2_STORE
#undef A2_LD
#undef A2_ONE
}

template <int K, int Mode, int Chunk>
__device__ __forceinline__ uint32_t Consumer(const LinJob& job, const GemvItem& item,
                                            const uint32_t* data, LdsStreamRing<SlotsFor<Chunk>, L, C>& ring,
                                            const uint32_t* xs, uint32_t base, uint32_t sink) {
  constexpr int Slots = SlotsFor<Chunk>;
  constexpr int T = 8 * K;
  const int lane = threadIdx.x & 31, cw = __builtin_amdgcn_readfirstlane((threadIdx.x >> 5) - L);
  const int span = job.I / 16 / job.S, q = lane & 3;
  const LaneWindow<K> win(lane);
  float acc0 = 0.f, acc1 = 0.f;
  for (int seq = 0, a = 0; a < span; ++seq, a += Chunk) {
    const uint32_t g = base + seq;
    ring.WaitReady(g);
    if constexpr (Mode == 0) {
      const uint4* raw = reinterpret_cast<const uint4*>(data + (g % Slots) * SlotWords<Chunk>);
#pragma unroll
      for (int j = 0; j < (8 * Chunk * T / 4 + 32 * C - 1) / (32 * C); ++j) {
        const int i = j * 32 * C + cw * 32 + lane;
        if (i < 8 * Chunk * T / 4) { const uint4 v = raw[i]; sink ^= v.x ^ v.y ^ v.z ^ v.w; }
      }
    } else if (cw < item.nb) {
      const uint32_t* column = data + (g % Slots) * SlotWords<Chunk> + cw * Chunk * T;
#pragma unroll
      for (int f = 0; f < Chunk; ++f) {
        const uint32_t* tile = column + f * T;
        __half2 P[4];
        DecodeLane<K, kMulMode>(tile[win.d0], tile[(win.d0 + 1) % T], tile[(win.d0 + 2) % T], win.shift, P);
        const uint32_t xa = xs[(a + f) * 8 + q], xb = xs[(a + f) * 8 + 4 + q];
        const __half2 ha = *reinterpret_cast<const __half2*>(&xa), hb = *reinterpret_cast<const __half2*>(&xb);
        acc0 = Dot2(P[0], ha, acc0); acc0 = Dot2(P[1], hb, acc0);
        acc1 = Dot2(P[2], ha, acc1); acc1 = Dot2(P[3], hb, acc1);
      }
    }
    ring.Release(g, lane);
  }
  if constexpr (Mode != 0) {
    if (cw < item.nb) {
      acc0 = acc0 + __shfl_xor(acc0, 1); acc0 = acc0 + __shfl_xor(acc0, 2);
      acc1 = acc1 + __shfl_xor(acc1, 1); acc1 = acc1 + __shfl_xor(acc1, 2);
      if (q == 0) {
        const int col = 16 * (item.b0 + cw) + lane / 4;
        float* out = job.partial + (size_t)item.s * job.O;
        out[col] = acc0; out[col + 8] = acc1;
      }
    }
  }
  return sink;
}

template <int Mode, int Layout, int Chunk, bool Load, size_t N = 0>
__device__ __forceinline__ uint32_t Dispatch(int K, const LinJob& job, const GemvItem& item, uint32_t* data,
                                            LdsStreamRing<SlotsFor<Chunk>, L, C>& ring, uint32_t* xs, uint32_t base,
                                            uint32_t sink) {
  if (K == kSupportedDecodeK[N]) {
    if constexpr (Load) Loader<kSupportedDecodeK[N], Layout, Chunk>(job, item, data, ring, base);
    else sink = Consumer<kSupportedDecodeK[N], Mode, Chunk>(job, item, data, ring, xs, base, sink);
  } else if constexpr (N + 1 < kSupportedDecodeK.size()) {
    sink = Dispatch<Mode, Layout, Chunk, Load, N + 1>(K, job, item, data, ring, xs, base, sink);
  }
  return sink;
}

__device__ __forceinline__ void ComputeMeet(uint32_t* arrived, uint32_t& seq, int t) {
  const uint32_t target = ++seq * C;
  asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
  if ((t & 31) == 0) __hip_atomic_fetch_add(arrived, 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_WORKGROUP);
  while (__hip_atomic_load(arrived, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_WORKGROUP) < target)
    __builtin_amdgcn_s_sleep(1);
  asm volatile("" ::: "memory");
  __builtin_amdgcn_wave_barrier();
}

// Phase.start rotates the first block of each phase. A running start over
// the whole stream balances the per-block byte totals; zero repeats today's
// assignment, which gives the low blocks every remainder item.
template <int Mode, int Layout, int Chunk>
__global__ void __launch_bounds__(32 * (L + C)) Kernel(const Phase* phases, int nph, uint32_t* sinks,
                                                       uint64_t* clocks) {
  const uint64_t t0 = wall_clock64();
  __shared__ LdsStreamRing<SlotsFor<Chunk>, L, C> ring;
  __shared__ uint32_t data[SlotsFor<Chunk> * SlotWords<Chunk>], arrived;
  extern __shared__ uint32_t xs[];
  ring.Initialize(threadIdx.x);
  if (threadIdx.x == 0) arrived = 0;
  __syncthreads();
  const bool load = threadIdx.x < 32 * L;
  const int t = threadIdx.x - 32 * L;
  uint32_t seq = 0, meet = 0, sink = 0;
  for (int p = 0; p < nph; ++p) {
    const Phase ph = StreamConstant(phases + p);
    const int first = ((int)blockIdx.x - ph.start % (int)gridDim.x + (int)gridDim.x) % (int)gridDim.x;
    for (int it = first; it < ph.n; it += gridDim.x) {
      const GemvItem item = StreamConstant(ph.items + it);
      const LinJob job = StreamConstant(ph.jobs + item.job);
      const int span = job.I / 16 / job.S;
      if (load) {
        Dispatch<Mode, Layout, Chunk, true>(job.K, job, item, data, ring, xs, seq, sink);
      } else {
        const uint32_t* x = reinterpret_cast<const uint32_t*>(job.xh);
        for (int i = t; i < span * 8; i += 32 * C) xs[i] = x[item.s * span * 8 + i];
        ComputeMeet(&arrived, meet, t);
        sink = Dispatch<Mode, Layout, Chunk, false>(job.K, job, item, data, ring, xs, seq, sink);
        ComputeMeet(&arrived, meet, t);
      }
      seq += span / Chunk;
    }
  }
  if (!load) sinks[(size_t)blockIdx.x * 32 * C + t] = sink;
  __syncthreads();
  if (threadIdx.x == 0) { clocks[2 * blockIdx.x] = t0; clocks[2 * blockIdx.x + 1] = wall_clock64(); }
}

using Fn = void (*)(const Phase*, int, uint32_t*, uint64_t*);
template <int Mode, int Chunk> Fn PickLayout(int layout) {
  return layout == 0 ? Kernel<Mode, 0, Chunk> : layout == 1 ? Kernel<Mode, 1, Chunk> : Kernel<Mode, 2, Chunk>;
}
Fn Pick(int mode, int layout, int chunk) {
  if (chunk == 4) return mode == 0 ? PickLayout<0, 4>(layout) : PickLayout<2, 4>(layout);
  return mode == 0 ? PickLayout<0, 8>(layout) : PickLayout<2, 8>(layout);
}

// The HIP occupancy query assumes 64 KB of LDS per WGP on gfx1151. Use the
// 1,536-entry VGPR file (granule 8), 16 waves per SIMD and 128 KB per WGP.
int ResidentBlocks(Fn fn, size_t dynamic_lds) {
  hipFuncAttributes a;
  HIP_CHECK(hipFuncGetAttributes(&a, reinterpret_cast<const void*>(fn)));
  const int vgpr = (a.numRegs + 7) / 8 * 8, waves = std::min(16, 1536 / std::max(8, vgpr));
  const int by_vgpr = 4 * waves / (L + C), by_lds = (128 << 10) / (int)(a.sharedSizeBytes + dynamic_lds);
  return 20 * std::min(by_vgpr, by_lds);
}

// Host repack of one engine-order job into item or front order.
uint32_t* Repack(const LinJob& j, int layout, int chunk) {
  const int A = j.I / 16, B = j.O / 16, T = 8 * j.K, span = A / j.S, count = span / chunk, ncg = B / 8;
  const size_t words = (size_t)A * B * T;
  std::vector<uint32_t> src(words), dst(words);
  HIP_CHECK(hipMemcpy(src.data(), j.tiles, words * 4, hipMemcpyDeviceToHost));
  for (int s = 0; s < j.S; ++s)
    for (int cg = 0; cg < ncg; ++cg)
      for (int q = 0; q < count; ++q)
        for (int c = 0; c < 8; ++c)
          for (int f = 0; f < chunk; ++f) {
            const int idx = s * ncg + cg;
            const size_t chunk_index = layout == 1 ? (size_t)idx * count + q : (size_t)q * j.S * ncg + idx;
            const size_t d = (chunk_index * 8 * chunk + c * chunk + f) * T;
            const size_t from = ((size_t)(cg * 8 + c) * A + s * span + q * chunk + f) * T;
            std::memcpy(dst.data() + d, src.data() + from, T * 4);
          }
  uint32_t* out;
  HIP_CHECK(hipMalloc(&out, words * 4));
  HIP_CHECK(hipMemcpy(out, dst.data(), words * 4, hipMemcpyHostToDevice));
  return out;
}
}  // namespace ab2

int main(int argc, char** argv) {
  if (argc < 2) { std::fprintf(stderr, "usage: mega_ablate2 CKPT [grid=80]\n"); return 2; }
  const int grid = argc > 2 ? std::atoi(argv[2]) : 80;
  if (grid < 1 || grid > 1024) return 2;
  st::Checkpoint ck(argv[1]);
  auto groups = LoadGroups(ck, true);
  std::mt19937 rng(17); std::normal_distribution<float> norm;
  std::vector<__half> hx(10240);
  for (auto& h : hx) h = __float2half(norm(rng));
  __half* x; float* partial; uint32_t* sinks;
  HIP_CHECK(hipMalloc(&x, hx.size() * 2));
  HIP_CHECK(hipMemcpy(x, hx.data(), hx.size() * 2, hipMemcpyHostToDevice));
  HIP_CHECK(hipMalloc(&partial, 64 << 20));
  HIP_CHECK(hipMalloc(&sinks, (size_t)grid * 256 * 4));
  uint64_t* clocks;
  HIP_CHECK(hipMalloc(&clocks, (size_t)grid * 16));
  std::vector<Plan> plans;
  size_t bytes = 0, lds = 0, items = 0;
  for (const auto& g : groups) {
    plans.push_back(MakePlan(g, x, partial));
    bytes += plans.back().bytes; lds = std::max(lds, plans.back().lds);
  }
  for (const auto& p : plans) for (const auto& j : p.jobs) {
    const int span = j.I / 16 / j.S;
    if (span % 8 || (j.O / 16) % 8) { std::fprintf(stderr, "span %d or O %d not chunk aligned\n", span, j.O); return 1; }
  }
  // phases[layout][chunk index][balanced]
  std::vector<ab2::Phase> host[3][2][2];
  std::vector<LinJob*> device_jobs[3][2];
  for (int layout = 0; layout < 3; ++layout)
    for (int ci = 0; ci < 2; ++ci) {
      const int chunk = ci ? 8 : 4;
      if (layout == 0 && ci == 1) { device_jobs[0][1] = device_jobs[0][0]; continue; }
      for (const auto& p : plans) {
        std::vector<LinJob> jobs = p.jobs;
        if (layout) for (auto& j : jobs) j.tiles = ab2::Repack(j, layout, chunk);
        LinJob* dj;
        HIP_CHECK(hipMalloc(&dj, jobs.size() * sizeof(LinJob)));
        HIP_CHECK(hipMemcpy(dj, jobs.data(), jobs.size() * sizeof(LinJob), hipMemcpyHostToDevice));
        device_jobs[layout][ci].push_back(dj);
      }
    }
  for (int layout = 0; layout < 3; ++layout)
    for (int ci = 0; ci < 2; ++ci)
      for (int bal = 0; bal < 2; ++bal) {
        int start = 0;
        for (size_t i = 0; i < plans.size(); ++i) {
          host[layout][ci][bal].push_back({device_jobs[layout][ci][i], plans[i].di, (int)plans[i].items.size(), bal ? start : 0});
          start += (int)plans[i].items.size();
        }
      }
  ab2::Phase* dph[3][2][2];
  for (int a = 0; a < 3; ++a) for (int b = 0; b < 2; ++b) for (int c = 0; c < 2; ++c) {
    HIP_CHECK(hipMalloc(&dph[a][b][c], plans.size() * sizeof(ab2::Phase)));
    HIP_CHECK(hipMemcpy(dph[a][b][c], host[a][b][c].data(), plans.size() * sizeof(ab2::Phase), hipMemcpyHostToDevice));
  }
  for (const auto& p : plans) items += p.items.size();
  std::printf("# phases=%zu items=%zu bytes=%.9fGB grid=%d L2 C8 dynLDS=%zu roof=245.5GB/s\n",
              plans.size(), items, bytes / 1e9, grid, lds);
  // Exactness: every layout and chunk against LinGemvKernel<1> on representative phases.
  for (int layout = 0; layout < 3; ++layout)
    for (int chunk : {4, 8})
      for (int i : {0, 1, 6, 7, 96}) {
        const Plan& p = plans[i];
        std::vector<float> reference(p.partials), actual(p.partials);
        Launch(p, false, grid);
        HIP_CHECK(hipMemcpy(reference.data(), partial, p.partials * 4, hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemset(partial, 0xff, p.partials * 4));
        hipLaunchKernelGGL(ab2::Pick(2, layout, chunk), grid, 320, lds, nullptr, dph[layout][chunk == 8][0] + i, 1, sinks, clocks);
        HIP_CHECK(hipMemcpy(actual.data(), partial, p.partials * 4, hipMemcpyDeviceToHost));
        if (std::memcmp(reference.data(), actual.data(), p.partials * 4)) {
          std::fprintf(stderr, "full mismatch layout%d chunk%d phase%d\n", layout, chunk, i);
          return 1;
        }
      }
  std::printf("# full partials exact for 3 layouts x 2 chunks on five representative phases\n");
  struct Case { int mode, layout, chunk, bal; bool persistent; };
  std::vector<Case> cases;
  for (int mode : {0, 2})
    for (int layout : {0, 1, 2})
      for (int chunk : {4, 8})
        for (int bal : {0, 1}) {
          cases.push_back({mode, layout, chunk, bal, true});
          if (bal == 0) cases.push_back({mode, layout, chunk, 0, false});
        }
  for (const auto& c : cases) {
    const auto fn = ab2::Pick(c.mode, c.layout, c.chunk);
    const ab2::Phase* ph = dph[c.layout][c.chunk == 8][c.bal];
    const int resident = ab2::ResidentBlocks(fn, lds), g = std::min(grid, resident);
    std::vector<double> times;
    for (int rep = 0; rep < 7; ++rep) {
      bench::EventTimer timer; timer.Begin();
      if (c.persistent) hipLaunchKernelGGL(fn, g, 320, lds, nullptr, ph, (int)plans.size(), sinks, clocks);
      else for (size_t i = 0; i < plans.size(); ++i) hipLaunchKernelGGL(fn, g, 320, lds, nullptr, ph + i, 1, sinks, clocks);
      const float ms = timer.EndMs(); HIP_CHECK(hipGetLastError());
      if (rep) times.push_back(ms);
    }
    const double ms = bench::Median(times), bw = bytes / ms / 1e6;
    // Last repetition, persistent only: block end spread as a share of the kernel span.
    std::vector<uint64_t> clk(2 * g);
    HIP_CHECK(hipMemcpy(clk.data(), clocks, clk.size() * 8, hipMemcpyDeviceToHost));
    uint64_t s0 = ~0ull, e0 = ~0ull, e1 = 0;
    for (int b = 0; b < g; ++b) { s0 = std::min(s0, clk[2 * b]); e0 = std::min(e0, clk[2 * b + 1]); e1 = std::max(e1, clk[2 * b + 1]); }
    const double spread = c.persistent ? 100.0 * (e1 - e0) / std::max<uint64_t>(1, e1 - s0) : 0.0;
    std::printf("%-5s layout%d chunk%d %-8s %-10s %8.3fms %7.2fGB/s %6.2f%%DRAM min=%.3f max=%.3f endspread=%.1f%% grid=%d\n",
                c.mode ? "full" : "bytes", c.layout, c.chunk, c.bal ? "balanced" : "zero",
                c.persistent ? "persistent" : "per-phase", ms, bw, bw / 245.5 * 100,
                *std::min_element(times.begin(), times.end()), *std::max_element(times.begin(), times.end()), spread, g);
    std::fflush(stdout);
  }
  return 0;
}
