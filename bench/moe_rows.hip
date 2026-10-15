// Routed-expert MoE cost at R verify rows, without the 23 s checkpoint load.
//
//   moe_rows [layers] [iters]
//
// Random EXL3 tiles (K=3 routed, K=5 shared) for several layers of 512
// experts, so consecutive calls read cold weights as in a real forward. Each
// call runs the engine's MoE sequence after the router: unique, gate/up
// (LinIn + LinGemv), mid, down, combine. Distinct experts per call follow the
// measured verify counts.
#include <algorithm>
#include <chrono>
#include <cstring>
#include <map>
#include <random>
#include <string>

#include "../engine/model/flashnext/model.h"
#include "../engine/layers/ops.h"
#include "../engine/exl3/plans.h"
#include "kernels/moe_rows.h"

using namespace crucible;

namespace {

constexpr int kUnique[kMaxRows + 1] = {0, 10, 16, 22, 27, 32, 37, 41, 45};

__global__ void FillWords(uint32_t* p, size_t n, uint32_t seed) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    uint64_t z = (i + 1) * 0x9E3779B97F4A7C15ull + seed;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    p[i] = (uint32_t)(z ^ (z >> 31));
  }
}

// Sign times a value in [0.5, 1.5) times scale, like EXL3 sign/scale vectors.
__global__ void FillHalves(__half* p, size_t n, uint32_t seed, float scale) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    uint32_t z = (uint32_t)(i * 2654435761u) ^ seed;
    z ^= z >> 13;
    z *= 0x5bd1e995u;
    z ^= z >> 15;
    const float v = (0.5f + (z & 0xffff) / 65536.f) * scale;
    p[i] = __float2half((z >> 16) & 1 ? v : -v);
  }
}

__global__ void FillFloats(float* p, size_t n, uint32_t seed, float scale) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    uint32_t z = (uint32_t)(i * 2246822519u) ^ seed;
    z ^= z >> 15;
    z *= 0x2c1b3c6du;
    z ^= z >> 12;
    p[i] = ((z & 0xffffff) / 16777216.f - 0.5f) * scale;
  }
}

template <typename T>
T* Alloc(size_t n) {
  T* p;
  HIP_OK(hipMalloc(&p, n * sizeof(T)));
  return p;
}

uint32_t* Words(size_t n, uint32_t seed) {
  uint32_t* p = Alloc<uint32_t>(n);
  FillWords<<<1024, 256>>>(p, n, seed);
  return p;
}

__half* Halves(size_t n, uint32_t seed, float scale) {
  __half* p = Alloc<__half>(n);
  FillHalves<<<256, 256>>>(p, n, seed, scale);
  return p;
}

size_t TileWords(int I, int O, int K) { return (size_t)I * O / 256 * 8 * K; }

Lin MakeLin(int I, int O, int K, uint32_t seed) {
  return Lin{I, O, K, Words(TileWords(I, O, K), seed), Halves(I, seed + 1, 1.f), Halves(O, seed + 2, 0.02f)};
}

struct LayerW {
  ExpertPtrs* d_experts = nullptr;
  Lin sg, su, sd;
};

struct RowsScratch {
  float *x, *partial, *H, *post, *wts;
  __half* xh;
  int *sel, *map;
};

LinJob Job(const Lin& L, const float* src, int lds, int R) {
  LinJob j{};
  j.tiles = L.tiles;
  j.suh = L.suh;
  j.svh = L.svh;
  j.src = src;
  j.I = L.I;
  j.O = L.O;
  j.K = L.K;
  j.S = 1;
  j.R = R;
  j.lds = lds;
  j.epi = kEpiStore;
  return j;
}

bool g_shared_first = false;
int g_order = 0;

// Engine::Build with fixedS (the routed-expert plan).
void Build(Group& g, std::vector<LinJob> jobs, int R, float* partial, __half* xh, int ib = 8) {
  size_t poff = 0, xoff = 0;
  int maxspan = 0;
  std::vector<GemvItem> items;
  std::vector<XfItem> in, out;
  for (size_t ji = 0; ji < jobs.size(); ++ji) {
    LinJob& j = jobs[ji];
    j.partial = partial + poff;
    poff += (size_t)j.S * j.R * j.O;
    j.xh = xh + xoff;
    xoff += (size_t)j.R * j.I;
    maxspan = std::max(maxspan, j.I / 16 / j.S);
    for (int s = 0; s < j.S; ++s)
      for (int b0 = 0; b0 < j.O / 16; b0 += ib) items.push_back({(int)ji, s, b0, std::min(ib, j.O / 16 - b0)});
    for (int r = 0; r < j.R; ++r) {
      for (int b = 0; b < j.I / 128; ++b) in.push_back({(int)ji, r, b});
      for (int b = 0; b < j.O / 128; ++b) out.push_back({(int)ji, r, b});
    }
  }
  if (g_shared_first) {
    // The shared expert runs every row: its workgroups are the longest, so
    // dispatch them first instead of in the launch tail. Job indices and
    // buffers stay the same.
    const int last = (int)jobs.size() - 1, lastgu = (int)jobs.size() - 2;
    const bool gu = jobs.size() >= 2 && jobs[lastgu].I == jobs[last].I && jobs[last].I == kHidden;
    std::stable_partition(items.begin(), items.end(), [&](const GemvItem& it) {
      return it.job == last || (gu && it.job == lastgu);
    });
  }
  g.jobs = jobs;
  g.d_jobs = Upload(jobs);
  g.d_items = Upload(items);
  g.d_in = Upload(in);
  g.d_out = Upload(out);
  g.nitems = (int)items.size();
  g.nin = (int)in.size();
  g.nout = (int)out.size();
  g.R = R;
  g.lds = (size_t)R * maxspan * 32;
}

struct Plan {
  Group gu, dn;
};

Plan MakePlan(const LayerW& L, const ExpertPtrs& e0, int R, RowsScratch& s, float* partial_dn, __half* xh_dn,
              int ib = 8) {
  Lin rg{kHidden, kMoeInter, 3, (uint32_t*)e0.tiles[0], (__half*)e0.suh[0], (__half*)e0.svh[0]};
  Lin ru{kHidden, kMoeInter, 3, (uint32_t*)e0.tiles[1], (__half*)e0.suh[1], (__half*)e0.svh[1]};
  Lin rd{kMoeInter, kHidden, 3, (uint32_t*)e0.tiles[2], (__half*)e0.suh[2], (__half*)e0.svh[2]};
  std::vector<LinJob> gu, dn;
  for (int u = 0; u <= R * kTopK; ++u) {
    const bool sh = u == R * kTopK;
    gu.push_back(Job(sh ? L.sg : rg, s.x, kHidden, R));
    gu.push_back(Job(sh ? L.su : ru, s.x, kHidden, R));
    dn.push_back(Job(sh ? L.sd : rd, nullptr, 0, R));
  }
  Plan p;
  Build(p.gu, gu, R, s.partial, s.xh, ib);
  Build(p.dn, dn, R, partial_dn, xh_dn, ib);
  return p;
}

template <int R>
void Gemv(const Group& g) {
  LinGemvKernel<R><<<g.nitems, 256, g.lds>>>(g.d_jobs, g.d_items, g.nitems);
}

void RunGemv(const Group& g) {
  switch (g.R) {
    case 1: Gemv<1>(g); break;
    case 2: Gemv<2>(g); break;
    case 3: Gemv<3>(g); break;
    case 4: Gemv<4>(g); break;
    case 5: Gemv<5>(g); break;
    case 6: Gemv<6>(g); break;
    case 7: Gemv<7>(g); break;
    default: Gemv<8>(g); break;
  }
}

int g_pf = 8;
int g_mode = 0;  // 0 matvec, 1 weight stream only, 2 weight stream plus LDS prologue

// The same items, waves and buffer loads as MoeGemvBody without the decode
// and dot products: the bandwidth this access pattern can reach.
template <int R, bool Prologue>
__global__ void __launch_bounds__(256) StreamOnlyKernel(const LinJob* __restrict__ jobs,
                                                        const GemvItem* __restrict__ items, int n,
                                                        uint32_t* __restrict__ sink) {
  extern __shared__ uint32_t xs[];
  const GemvItem item = items[blockIdx.x];
  const LinJob& job = jobs[item.job];
  const int K = __builtin_amdgcn_readfirstlane(job.K);
  if (K == 0) return;
  const int rows = __builtin_amdgcn_readfirstlane(job.R);
  const int lane = threadIdx.x & 31, wave = threadIdx.x >> 5;
  const int T = 8 * K, A = job.I >> 4, span = A;
  if constexpr (Prologue) {
    const uint32_t* xg = reinterpret_cast<const uint32_t*>(job.xh);
    for (int i = threadIdx.x; i < rows * span * 8; i += blockDim.x) xs[i] = xg[i];
    __syncthreads();
  }
  if (wave >= item.nb) return;
  const int b = item.b0 + wave;
  const __amdgpu_buffer_rsrc_t rs = StreamRsrc(job.tiles + (size_t)b * A * T, (uint32_t)span * T * 4);
  const int d0 = (lane * 3) % T;
  const int o0 = 4 * d0, o1 = 4 * ((d0 + 1) % T), o2 = 4 * ((d0 + 2) % T);
  uint32_t acc = Prologue ? xs[lane] : 0u;
  constexpr int PF = 8;
  uint32_t ring[PF][3];
#pragma unroll
  for (int f = 0; f < PF; ++f) {
    ring[f][0] = __builtin_amdgcn_raw_buffer_load_b32(rs, o0, f * T * 4, 0);
    ring[f][1] = __builtin_amdgcn_raw_buffer_load_b32(rs, o1, f * T * 4, 0);
    ring[f][2] = __builtin_amdgcn_raw_buffer_load_b32(rs, o2, f * T * 4, 0);
  }
  for (int a = 0; a < span; a += PF) {
#pragma unroll
    for (int f = 0; f < PF; ++f) {
      acc ^= ring[f][0] + ring[f][1] * 3u + ring[f][2] * 5u;
      const int so = (a + f + PF) * T * 4;
      ring[f][0] = __builtin_amdgcn_raw_buffer_load_b32(rs, o0, so, 0);
      ring[f][1] = __builtin_amdgcn_raw_buffer_load_b32(rs, o1, so, 0);
      ring[f][2] = __builtin_amdgcn_raw_buffer_load_b32(rs, o2, so, 0);
    }
  }
  if (acc == 0x12345678u) sink[0] = acc;
}

uint32_t* g_sink = nullptr;

__constant__ int g_chunked_dev = 0;

struct WgTime {
  uint64_t t0, t1;
  int job, rows, K, cu;
};

// MoeRowsGemvKernel with a REALTIME (100 MHz) stamp at entry and exit of
// each workgroup, for the launch timeline.
template <int R>
__global__ void __launch_bounds__(256) MoeRowsGemvTimed(const LinJob* __restrict__ jobs,
                                                        const GemvItem* __restrict__ items, int n,
                                                        WgTime* __restrict__ out) {
  extern __shared__ uint32_t xs[];
  const uint64_t t0 = wall_clock64();
  const GemvItem item = items[blockIdx.x];
  const LinJob& job = jobs[item.job];
  const int K = __builtin_amdgcn_readfirstlane(job.K);
  const int rows = __builtin_amdgcn_readfirstlane(job.R);
  if (g_chunked_dev) {
    if (K == 3) MoeRowsChunkedBody<3, R, 8>(rows, job, item, xs);
    else if (K == 5 && rows == R) MoeGemvChunked<5, R, 8>(job, item, xs);
  } else {
    if (K == 3) MoeRowsBody<3, R, 8>(rows, job, item, xs);
    else if (K == 5 && rows == R) MoeGemvBody<5, R, 8>(job, item, xs);
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    unsigned hw;
    asm volatile("s_getreg_b32 %0, hwreg(HW_REG_HW_ID1)" : "=s"(hw));
    out[blockIdx.x] = {t0, (uint64_t)wall_clock64(), item.job, rows, K, (int)hw};
  }
}

template <int R>
void TimedLaunch(const Group& g, WgTime* out) {
  MoeRowsGemvTimed<R><<<g.nitems, 256, std::max(g.lds, MoeChunkLds(R))>>>(g.d_jobs, g.d_items, g.nitems, out);
}

void Timeline(const Group& g, int R, const char* name) {
  WgTime* d = nullptr;
  HIP_OK(hipMalloc(&d, sizeof(WgTime) * g.nitems));
  HIP_OK(hipMemset(d, 0, sizeof(WgTime) * g.nitems));
  switch (R) {
    case 1: TimedLaunch<1>(g, d); break;
    case 3: TimedLaunch<3>(g, d); break;
    case 5: TimedLaunch<5>(g, d); break;
    default: TimedLaunch<8>(g, d); break;
  }
  std::vector<WgTime> h(g.nitems);
  HIP_OK(hipMemcpy(h.data(), d, sizeof(WgTime) * g.nitems, hipMemcpyDeviceToHost));
  HIP_OK(hipFree(d));
  uint64_t lo = ~0ull, hi = 0;
  int active = 0;
  for (auto& w : h)
    if (w.t1) {
      lo = std::min(lo, w.t0);
      hi = std::max(hi, w.t1);
      ++active;
    }
  std::vector<double> ends, starts;
  double shared_dur = 0, routed_dur = 0;
  int ns = 0, nr = 0;
  for (auto& w : h) {
    if (!w.t1) continue;
    starts.push_back((w.t0 - lo) / 100.0);
    ends.push_back((w.t1 - lo) / 100.0);
    const double dur = (w.t1 - w.t0) / 100.0;
    if (w.K == 5) shared_dur += dur, ++ns;
    else routed_dur += dur, ++nr;
  }
  std::sort(ends.begin(), ends.end());
  std::sort(starts.begin(), starts.end());
  auto pct = [](const std::vector<double>& v, double p) { return v[std::min(v.size() - 1, (size_t)(p * v.size()))]; };
  std::printf("  timeline %s R=%d: %d WGs (%d with work), span %.1f us; starts p50 %.1f p90 %.1f max %.1f; "
              "ends p10 %.1f p50 %.1f p90 %.1f max %.1f; mean dur shared %.1f routed %.1f us\n",
              name, R, g.nitems, active, (hi - lo) / 100.0, pct(starts, .5), pct(starts, .9), starts.back(),
              pct(ends, .1), pct(ends, .5), pct(ends, .9), ends.back(), ns ? shared_dur / ns : 0, nr ? routed_dur / nr : 0);
  // Concurrency profile in 10 us bins.
  std::printf("  running WGs per 10 us:");
  for (double t = 0; t < (hi - lo) / 100.0; t += 10) {
    int c = 0;
    for (auto& w : h)
      if (w.t1 && (w.t0 - lo) / 100.0 <= t + 5 && (w.t1 - lo) / 100.0 > t + 5) ++c;
    std::printf(" %d", c);
  }
  std::printf("\n");
}

// The workgroup's whole item region (nb column blocks, contiguous), read with
// all 256 threads coalesced: the best case for DRAM locality.
template <int R>
__global__ void __launch_bounds__(256) StreamBlockKernel(const LinJob* __restrict__ jobs,
                                                         const GemvItem* __restrict__ items, int n,
                                                         uint32_t* __restrict__ sink) {
  const GemvItem item = items[blockIdx.x];
  const LinJob& job = jobs[item.job];
  const int K = __builtin_amdgcn_readfirstlane(job.K);
  if (K == 0) return;
  const int T = 8 * K, A = job.I >> 4;
  const uint32_t bytes = (uint32_t)item.nb * A * T * 4;
  const __amdgpu_buffer_rsrc_t rs = StreamRsrc(job.tiles + (size_t)item.b0 * A * T, bytes);
  uint32_t acc = 0;
  constexpr int PF = 8;
  uint4 ring[PF];
#pragma unroll
  for (int f = 0; f < PF; ++f)
    ring[f] = __builtin_bit_cast(uint4, __builtin_amdgcn_raw_buffer_load_b128(rs, (f * 256 + threadIdx.x) * 16, 0, 0));
  for (uint32_t o = 0; o < bytes; o += PF * 4096) {
#pragma unroll
    for (int f = 0; f < PF; ++f) {
      acc ^= ring[f].x + ring[f].w * 3u;
      ring[f] = __builtin_bit_cast(uint4,
          __builtin_amdgcn_raw_buffer_load_b128(rs, o + (PF + f) * 4096 + threadIdx.x * 16, 0, 0));
    }
  }
  if (acc == 0x12345678u) sink[0] = acc;
}

// Same per-wave byte ranges as StreamOnlyKernel, read with 16-byte loads.
template <int R>
__global__ void __launch_bounds__(256) StreamWideKernel(const LinJob* __restrict__ jobs,
                                                        const GemvItem* __restrict__ items, int n,
                                                        uint32_t* __restrict__ sink) {
  const GemvItem item = items[blockIdx.x];
  const LinJob& job = jobs[item.job];
  const int K = __builtin_amdgcn_readfirstlane(job.K);
  if (K == 0) return;
  const int lane = threadIdx.x & 31, wave = threadIdx.x >> 5;
  if (wave >= item.nb) return;
  const int T = 8 * K, A = job.I >> 4;
  const int b = item.b0 + wave;
  const uint32_t bytes = (uint32_t)A * T * 4;
  const __amdgpu_buffer_rsrc_t rs = StreamRsrc(job.tiles + (size_t)b * A * T, bytes);
  uint32_t acc = 0;
  constexpr int PF = 8;
  uint4 ring[PF];
#pragma unroll
  for (int f = 0; f < PF; ++f)
    ring[f] = __builtin_bit_cast(uint4, __builtin_amdgcn_raw_buffer_load_b128(rs, (f * 32 + lane) * 16, 0, 0));
  for (uint32_t o = 0; o < bytes; o += PF * 512) {
#pragma unroll
    for (int f = 0; f < PF; ++f) {
      acc ^= ring[f].x + ring[f].w * 3u;
      ring[f] = __builtin_bit_cast(uint4, __builtin_amdgcn_raw_buffer_load_b128(rs, o + (PF + f) * 512 + lane * 16, 0, 0));
    }
  }
  if (acc == 0x12345678u) sink[0] = acc;
}

template <int R>
void RowsGemv(const Group& g) {
  if (g_mode == 6) {
    MoeRowsChunkedKernel<R><<<g.nitems, 256, MoeChunkLds(R)>>>(g.d_jobs, g.d_items, g.nitems);
    return;
  }
  if (g_mode == 1) {
    StreamOnlyKernel<R, false><<<g.nitems, 256, g.lds>>>(g.d_jobs, g.d_items, g.nitems, g_sink);
    return;
  }
  if (g_mode == 5) {
    StreamBlockKernel<R><<<g.nitems, 256>>>(g.d_jobs, g.d_items, g.nitems, g_sink);
    return;
  }
  if (g_mode == 4) {
    StreamWideKernel<R><<<g.nitems, 256>>>(g.d_jobs, g.d_items, g.nitems, g_sink);
    return;
  }
  if (g_mode == 2) {
    StreamOnlyKernel<R, true><<<g.nitems, 256, g.lds>>>(g.d_jobs, g.d_items, g.nitems, g_sink);
    return;
  }
  if (g_pf == 16) MoeRowsGemvKernel<R, 16><<<g.nitems, 256, g.lds>>>(g.d_jobs, g.d_items, g.nitems);
  else if (g_pf == 12) MoeRowsGemvKernel<R, 12><<<g.nitems, 256, g.lds>>>(g.d_jobs, g.d_items, g.nitems);
  else MoeRowsGemvKernel<R, 8><<<g.nitems, 256, g.lds>>>(g.d_jobs, g.d_items, g.nitems);
}

void RunRowsGemv(const Group& g) {
  switch (g.R) {
    case 1: RowsGemv<1>(g); break;
    case 2: RowsGemv<2>(g); break;
    case 3: RowsGemv<3>(g); break;
    case 4: RowsGemv<4>(g); break;
    case 5: RowsGemv<5>(g); break;
    case 6: RowsGemv<6>(g); break;
    case 7: RowsGemv<7>(g); break;
    default: RowsGemv<8>(g); break;
  }
}

template <int R>
void RowsGemvPrefetch(const Group& g, const Group& dn) {
  // Shared down (K=5) is the largest job; size the chunk count for it.
  const int per_job = MoePrefetchChunks(kMoeInter * kHidden / 256 * 32 * 5);
  const int n = g.nitems + (int)dn.jobs.size() * per_job;
  MoeRowsGemvPrefetchKernel<R><<<n, 256, g.lds>>>(g.d_jobs, g.d_items, g.nitems, dn.d_jobs, per_job, g_sink);
}

void RunRowsGemvPrefetch(const Group& g, const Group& dn) {
  switch (g.R) {
    case 1: RowsGemvPrefetch<1>(g, dn); break;
    case 2: RowsGemvPrefetch<2>(g, dn); break;
    case 3: RowsGemvPrefetch<3>(g, dn); break;
    case 4: RowsGemvPrefetch<4>(g, dn); break;
    case 5: RowsGemvPrefetch<5>(g, dn); break;
    case 6: RowsGemvPrefetch<6>(g, dn); break;
    case 7: RowsGemvPrefetch<7>(g, dn); break;
    default: RowsGemvPrefetch<8>(g, dn); break;
  }
}

unsigned* g_done = nullptr;

template <int R, int PF, int C>
void FusedV(const Group& gu, const Group& dn) {
  MoeFusedKernel<R, 3, PF, 0, C><<<gu.nitems + dn.nitems, 256, std::max(gu.lds, dn.lds)>>>(
      gu.d_jobs, gu.d_items, gu.nitems, dn.d_jobs, dn.d_items, dn.nitems, g_done);
}

// Mode 7: the engine kernel. Modes 12 and 13: two column blocks per wave
// (16-block items) with prefetch depth 8 or 4.
template <int R>
void Fused(const Group& gu, const Group& dn) {
  if (g_mode == 12) FusedV<R, 8, 2>(gu, dn);
  else if (g_mode == 13) FusedV<R, 4, 2>(gu, dn);
  else FusedV<R, 8, 1>(gu, dn);
}

void RunFused(const Group& gu, const Group& dn) {
  switch (gu.R) {
    case 1: Fused<1>(gu, dn); break;
    case 2: Fused<2>(gu, dn); break;
    case 3: Fused<3>(gu, dn); break;
    case 4: Fused<4>(gu, dn); break;
    case 5: Fused<5>(gu, dn); break;
    case 6: Fused<6>(gu, dn); break;
    case 7: Fused<7>(gu, dn); break;
    default: Fused<8>(gu, dn); break;
  }
}

// Each row draws 10 distinct experts from a pool of U; every pool expert is used.
std::vector<int> Selection(int R, std::mt19937& rng) {
  const int U = kUnique[R];
  std::vector<int> all(kExperts);
  for (int i = 0; i < kExperts; ++i) all[i] = i;
  std::shuffle(all.begin(), all.end(), rng);
  std::vector<int> pool(all.begin(), all.begin() + U);
  std::vector<std::vector<int>> rows(R);
  for (int i = 0; i < U; ++i) rows[i % R].push_back(pool[i]);
  for (auto& row : rows) {
    while ((int)row.size() < kTopK) {
      const int e = pool[rng() % U];
      if (std::find(row.begin(), row.end(), e) == row.end()) row.push_back(e);
    }
    std::shuffle(row.begin(), row.end(), rng);
  }
  std::vector<int> sel;
  for (auto& row : rows) sel.insert(sel.end(), row.begin(), row.end());
  return sel;
}

double Now() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

}  // namespace

int main(int argc, char** argv) {
  const int nl = argc > 1 ? std::atoi(argv[1]) : 6;
  const int iters = argc > 2 ? std::atoi(argv[2]) : 96;
  std::vector<LayerW> layers(nl);
  std::vector<std::vector<ExpertPtrs>> hosts(nl);
  const size_t gw = TileWords(kHidden, kMoeInter, 3), dw = TileWords(kMoeInter, kHidden, 3);
  for (int l = 0; l < nl; ++l) {
    // One allocation per layer and projection keeps experts contiguous like the loader.
    uint32_t* g = Words(gw * kExperts, 100 * l + 1);
    uint32_t* u = Words(gw * kExperts, 100 * l + 2);
    uint32_t* d = Words(dw * kExperts, 100 * l + 3);
    __half* gs = Halves((size_t)kExperts * (kHidden + kMoeInter), 100 * l + 4, 1.f);
    __half* us = Halves((size_t)kExperts * (kHidden + kMoeInter), 100 * l + 5, 1.f);
    __half* ds = Halves((size_t)kExperts * (kHidden + kMoeInter), 100 * l + 6, 1.f);
    auto& h = hosts[l];
    h.resize(kExperts);
    for (int e = 0; e < kExperts; ++e) {
      const size_t so = (size_t)e * (kHidden + kMoeInter);
      h[e].tiles[0] = g + gw * e;
      h[e].tiles[1] = u + gw * e;
      h[e].tiles[2] = d + dw * e;
      h[e].suh[0] = gs + so;
      h[e].svh[0] = gs + so + kHidden;
      h[e].suh[1] = us + so;
      h[e].svh[1] = us + so + kHidden;
      h[e].suh[2] = ds + so;
      h[e].K[0] = h[e].K[1] = h[e].K[2] = 3;
      h[e].svh[2] = ds + so + kMoeInter;
    }
    layers[l].d_experts = Upload(h);
    layers[l].sg = MakeLin(kHidden, kMoeInter, 5, 100 * l + 7);
    layers[l].su = MakeLin(kHidden, kMoeInter, 5, 100 * l + 10);
    layers[l].sd = MakeLin(kMoeInter, kHidden, 5, 100 * l + 13);
  }
  RowsScratch s{};
  s.x = Alloc<float>(kMaxRows * kHidden);
  FillFloats<<<64, 256>>>(s.x, kMaxRows * kHidden, 7, 2.f);
  s.partial = Alloc<float>(size_t(16) << 20);
  s.xh = Alloc<__half>(size_t(4) << 20);
  float* partial_dn = Alloc<float>(size_t(16) << 20);
  __half* xh_dn = Alloc<__half>(size_t(4) << 20);
  s.H = Alloc<float>(kMaxRows * kWide);
  s.post = Alloc<float>(kMaxRows * kStreams);
  FillFloats<<<4, 64>>>(s.post, kMaxRows * kStreams, 9, 1.f);
  s.wts = Alloc<float>(kMaxRows * kSlots);
  FillFloats<<<4, 64>>>(s.wts, kMaxRows * kSlots, 11, 0.2f);
  s.map = Alloc<int>(kMaxRows * kTopK);
  int* mapj = Alloc<int>(kMaxRows * kTopK);
  int* rowl = Alloc<int>((kMaxRows * kTopK + 1) * kMaxRows);
  float* H0 = Alloc<float>(kMaxRows * kWide);
  FillFloats<<<64, 256>>>(H0, kMaxRows * kWide, 13, 1.f);
  HIP_OK(hipDeviceSynchronize());

  const double expert_mb = (2.0 * gw + dw) * 4 / 1e6, shared_mb = 3.0 * TileWords(kHidden, kMoeInter, 5) * 4 / 1e6;
  std::printf("expert %.3f MB, shared %.3f MB, %d layers, %d calls per R\n", expert_mb, shared_mb, nl, iters);
  std::mt19937 rng(1234);
  hipEvent_t ev[8];
  for (auto& e : ev) HIP_OK(hipEventCreate(&e));
  const char* stage[] = {"unique", "gateup", "mid", "down", "combine"};
  const int rmin = argc > 3 ? std::atoi(argv[3]) : 1;
  g_pf = argc > 4 ? std::atoi(argv[4]) : 8;
  g_mode = argc > 5 ? std::atoi(argv[5]) : 0;
  g_order = argc > 6 ? std::atoi(argv[6]) : 0;
  g_sink = Alloc<uint32_t>(1);
  g_done = Alloc<unsigned>(kMaxRows * kTopK + 1);
  std::printf("variant prefetch depth %d, mode %d\n", g_pf, g_mode);
  for (int R = rmin; R <= kMaxRows; ++R) {
    std::vector<Plan> plans;
    for (int l = 0; l < nl; ++l) plans.push_back(MakePlan(layers[l], hosts[l][0], R, s, partial_dn, xh_dn));
    std::vector<int*> sels;
    for (int i = 0; i < iters; ++i) sels.push_back(Upload(Selection(R, rng)));
    // Row-sparse variant: its own job tables, since it rewrites job.R.
    std::vector<Plan> vplans;
    g_shared_first = g_order == 1;
    const int ib = g_mode == 12 || g_mode == 13 ? 16 : 8;
    for (int l = 0; l < nl; ++l) vplans.push_back(MakePlan(layers[l], hosts[l][0], R, s, partial_dn, xh_dn, ib));
    g_shared_first = false;
    auto vcall = [&](int i, bool timed, double* acc) {
      const int l = i % nl;
      const Plan& p = vplans[l];
      if (timed) HIP_OK(hipEventRecord(ev[0], 0));
      MoeRowsUniqueKernel<<<1, 512>>>(sels[i], R, layers[l].d_experts, p.gu.d_jobs, p.dn.d_jobs, s.map, mapj, rowl,
                                      g_done);
      if (timed) HIP_OK(hipEventRecord(ev[1], 0));
      if (g_mode == 7 || g_mode == 12 || g_mode == 13) {
        MoeRowsInKernel<<<(p.gu.nin + 7) / 8, 256>>>(p.gu.d_jobs, p.gu.d_in, p.gu.nin, rowl);
        RunFused(p.gu, p.dn);
        if (timed) {
          HIP_OK(hipEventRecord(ev[2], 0));
          HIP_OK(hipEventRecord(ev[3], 0));
          HIP_OK(hipEventRecord(ev[4], 0));
        }
        MoeRowsCombineKernel<<<(R * 20 + 7) / 8, 256>>>(p.dn.d_jobs, s.map, mapj, s.wts, s.post, s.H, R);
        if (timed) {
          HIP_OK(hipEventRecord(ev[5], 0));
          HIP_OK(hipEventSynchronize(ev[5]));
          for (int k = 0; k < 5; ++k) {
            float ms;
            HIP_OK(hipEventElapsedTime(&ms, ev[k], ev[k + 1]));
            acc[k] += ms;
          }
        }
        return;
      }
      MoeRowsInKernel<<<(p.gu.nin + 7) / 8, 256>>>(p.gu.d_jobs, p.gu.d_in, p.gu.nin, rowl);
      if (g_mode == 3) RunRowsGemvPrefetch(p.gu, p.dn);
      else RunRowsGemv(p.gu);
      if (timed) HIP_OK(hipEventRecord(ev[2], 0));
      MoeRowsMidKernel<<<((R * kTopK + 1) * R * 5 + 7) / 8, 256>>>(p.gu.d_jobs, p.dn.d_jobs, R * kTopK + 1, R);
      if (timed) HIP_OK(hipEventRecord(ev[3], 0));
      RunRowsGemv(p.dn);
      if (timed) HIP_OK(hipEventRecord(ev[4], 0));
      MoeRowsCombineKernel<<<(R * 20 + 7) / 8, 256>>>(p.dn.d_jobs, s.map, mapj, s.wts, s.post, s.H, R);
      if (timed) {
        HIP_OK(hipEventRecord(ev[5], 0));
        HIP_OK(hipEventSynchronize(ev[5]));
        for (int k = 0; k < 5; ++k) {
          float ms;
          HIP_OK(hipEventElapsedTime(&ms, ev[k], ev[k + 1]));
          acc[k] += ms;
        }
      }
    };
    const int nj = R * kTopK + 1;
    auto call = [&](int i, bool timed, double* acc) {
      const int l = i % nl;
      const Plan& p = plans[l];
      if (timed) HIP_OK(hipEventRecord(ev[0], 0));
      MoeUniqueKernel<<<1, 512>>>(sels[i], R, layers[l].d_experts, p.gu.d_jobs, p.dn.d_jobs, s.map);
      if (timed) HIP_OK(hipEventRecord(ev[1], 0));
      LinInKernel<<<(p.gu.nin + 7) / 8, 256>>>(p.gu.d_jobs, p.gu.d_in, p.gu.nin);
      RunGemv(p.gu);
      if (timed) HIP_OK(hipEventRecord(ev[2], 0));
      MoeMidKernel<<<(nj * R * 5 + 7) / 8, 256>>>(p.gu.d_jobs, p.dn.d_jobs, nj, R);
      if (timed) HIP_OK(hipEventRecord(ev[3], 0));
      RunGemv(p.dn);
      if (timed) HIP_OK(hipEventRecord(ev[4], 0));
      MoeCombineKernel<<<(R * 20 + 7) / 8, 256>>>(p.dn.d_jobs, s.map, s.wts, s.post, s.H, R);
      if (timed) {
        HIP_OK(hipEventRecord(ev[5], 0));
        HIP_OK(hipEventSynchronize(ev[5]));
        for (int k = 0; k < 5; ++k) {
          float ms;
          HIP_OK(hipEventElapsedTime(&ms, ev[k], ev[k + 1]));
          acc[k] += ms;
        }
      }
    };
    double acc[5] = {0, 0, 0, 0, 0};
    for (int i = 0; i < nl; ++i) call(i, false, acc);
    HIP_OK(hipDeviceSynchronize());
    for (int i = 0; i < iters; ++i) call(i, true, acc);
    // Untimed back-to-back calls: the stream cost per layer call.
    HIP_OK(hipDeviceSynchronize());
    const double t0 = Now();
    for (int i = 0; i < iters; ++i) call(i, false, acc);
    HIP_OK(hipDeviceSynchronize());
    const double stream_us = (Now() - t0) / iters * 1e6;
    const double mb = kUnique[R] * expert_mb + shared_mb;
    std::printf("R=%d U=%d %.1f MB: stream %.1f us/call (%.0f GB/s, floor %.1f us);", R, kUnique[R], mb, stream_us,
                mb / stream_us * 1e3, mb / 245e3 * 1e6);
    for (int k = 0; k < 5; ++k) std::printf(" %s %.1f", stage[k], acc[k] / iters * 1e3);
    std::printf("\n");
    if ((g_mode == 9 || g_mode == 10) && (R == 1 || R == 3 || R == 5 || R == 8)) {
      const int chunked = g_mode == 10;
      HIP_OK(hipMemcpyToSymbol(HIP_SYMBOL(g_chunked_dev), &chunked, sizeof(int)));
      for (int i = 0; i < 3; ++i) {
        const int l = i % nl;
        const Plan& p = vplans[l];
        MoeRowsUniqueKernel<<<1, 512>>>(sels[i], R, layers[l].d_experts, p.gu.d_jobs, p.dn.d_jobs, s.map, mapj, rowl);
        MoeRowsInKernel<<<(p.gu.nin + 7) / 8, 256>>>(p.gu.d_jobs, p.gu.d_in, p.gu.nin, rowl);
        Timeline(p.gu, R, "gateup");
        MoeRowsMidKernel<<<((R * kTopK + 1) * R * 5 + 7) / 8, 256>>>(p.gu.d_jobs, p.dn.d_jobs, R * kTopK + 1, R);
        Timeline(p.dn, R, "down");
      }
    }
    // Variant: same inputs, compare the residual stack bit for bit.
    {
      double vacc[5] = {0, 0, 0, 0, 0};
      bool same = true;
      for (int i = 0; i < std::min(iters, 2 * nl); ++i) {
        HIP_OK(hipMemcpy(s.H, H0, sizeof(float) * kMaxRows * kWide, hipMemcpyDeviceToDevice));
        call(i, false, acc);
        std::vector<float> a(kMaxRows * kWide), b(kMaxRows * kWide);
        HIP_OK(hipMemcpy(a.data(), s.H, a.size() * 4, hipMemcpyDeviceToHost));
        HIP_OK(hipMemcpy(s.H, H0, sizeof(float) * kMaxRows * kWide, hipMemcpyDeviceToDevice));
        vcall(i, false, vacc);
        HIP_OK(hipMemcpy(b.data(), s.H, b.size() * 4, hipMemcpyDeviceToHost));
        same = same && std::memcmp(a.data(), b.data(), (size_t)R * kWide * 4) == 0;
      }
      std::fill(vacc, vacc + 5, 0.0);
      for (int i = 0; i < iters; ++i) vcall(i, true, vacc);
      HIP_OK(hipDeviceSynchronize());
      const double v0 = Now();
      for (int i = 0; i < iters; ++i) vcall(i, false, vacc);
      HIP_OK(hipDeviceSynchronize());
      const double vus = (Now() - v0) / iters * 1e6;
      std::printf("  variant: stream %.1f us/call (%.0f GB/s), %s;", vus, mb / vus * 1e3,
                  same ? "bitwise equal" : "DIFFERENT");
      for (int k = 0; k < 5; ++k) std::printf(" %s %.1f", stage[k], vacc[k] / iters * 1e3);
      std::printf("\n");
    }
    std::fflush(stdout);
  }
  return 0;
}
