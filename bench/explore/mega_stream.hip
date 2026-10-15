// Probe: the DRAM rate of one serial decode forward's weight stream under
// three program structures. The phases, sizes and order follow the 3.05
// checkpoint at R=1 (docs/FLASHNEXT_FORWARD.md section 5): 48 layers, each
// with HC dots/mix, attention in/core/out, HC dots/mix, router, gate/up of
// 10 random experts plus the shared expert, down; PLE at layer 1; the final
// mixer and the LM head. Routed experts come from a large random pool.
//
//   A  one kernel per phase (today's structure)
//   B  one persistent kernel, a grid barrier after each phase
//   D  one persistent kernel; loader waves stream every granule through an
//      LDS ring without waiting for barriers, compute waves consume it and
//      meet at a grid barrier after each phase (a megakernel's dataflow)
//   F  one persistent kernel, all granules, no barriers (upper bound)
//
// v5 (mega-advisor): optional ALU-only phases after each GDN state phase
// and each QSA input phase (the recurrence and the attention core, which
// add time but no weight bytes), and an optional prefetch wave per block in
// modes D/E that loads one dword per 128-byte line PF granules ahead of the
// loader waves, so DRAM keeps filling the MALL while compute waves spin.
//
// Each consumed 16-byte vector costs `alu` dependent-free mad24 ops per
// lane, set per phase to the measured decode cost (EXL3 K3 ~288, K5 ~173,
// fp16 dot ~6). Results do not depend on data; a sink store keeps the loads.
//
// Usage: mega_stream POOL_GB BLOCKS SLOTS ALU_SCALE REPS U R DEP NC MODES
//                    GDN_US QSA_US PF
#include <random>

#include "../common.h"

namespace {

constexpr int kGran = 8192;  // bytes per granule
constexpr int kGBits = 21;   // mode D sequence entry: granule | phase << kGBits
constexpr uint32_t kGMask = (1u << kGBits) - 1;
constexpr int kLoaders = 2;  // mode D: loader waves per block

struct Phase {
  uint32_t g0, g1;  // granule range
  int alu;          // mad24 ops per 16-byte vector per lane
  int store;        // 1: also write the bytes (GDN state)
  int dep;          // 1: addresses known only after the previous phase (experts after the router)
  int spin;         // wall-clock ticks of ALU-only work per compute wave after the granules
};

struct Ctx {
  const uint8_t* const* gran;  // granule pointers in forward order
  const Phase* ph;
  int nph;
  uint8_t* stout;     // store target for store phases
  uint32_t* bar;      // grid barrier counter
  uint32_t* sink;
  const uint32_t* bseq;  // mode D: per-block granule sequence
  const uint32_t* boff;
  const int* blen;
  int barriers;  // mode D: grid barrier after each phase (E: off)
  int pf;        // modes D/E: prefetch wave runs this many granules ahead (0: no wave)
};

// Table reads through the constant address space become scalar loads, so
// they never make a wave wait (vmcnt) for its own in-flight weight loads.
template <class T>
__device__ __forceinline__ T LdC(const T* p) {
  using CT = const __attribute__((address_space(4))) T;
  return *(CT*)(p);
}

__device__ __forceinline__ Phase LdPhase(const Phase* p) {
  const uint32_t* q = reinterpret_cast<const uint32_t*>(p);
  return Phase{LdC(q), LdC(q + 1), (int)LdC(q + 2), (int)LdC(q + 3), (int)LdC(q + 4), (int)LdC(q + 5)};
}

__device__ __forceinline__ uint32_t Alu(uint4 v, int n) {
  uint32_t a = v.x, b = v.y, c = v.z, d = v.w;
  for (int i = 0; i < n; i += 4) {
    a = __umul24(a, 0x193u) + b;
    b = __umul24(b, 0x1a7u) + c;
    c = __umul24(c, 0x1b1u) + d;
    d = __umul24(d, 0x1c3u) + a;
  }
  return a ^ b ^ c ^ d;
}

// ALU-only work: a fixed count of dependent mad24 blocks, calibrated on the
// host to microseconds. (v5 first polled wall_clock64; with 640 waves polling
// at once each read took about 100 us, so every phase overran.)
__device__ __forceinline__ uint32_t Spin(uint32_t acc, int iters) {
  for (int i = 0; i < iters; ++i) acc = Alu(make_uint4(acc, acc ^ 1u, acc ^ 2u, acc ^ 3u), 64);
  return acc;
}

// Calibration: the compute waves of every block spin, loaders idle.
__global__ void SpinKernel(uint32_t* sink, int iters, int skip_waves) {
  if ((int)(threadIdx.x >> 5) < skip_waves) return;
  const uint32_t acc = Spin(threadIdx.x, iters);
  if (acc == 0x12345678u) sink[blockIdx.x] = acc;
}

__device__ __forceinline__ void GridBarrier(uint32_t* bar, uint32_t target) {
  __hip_atomic_fetch_add(bar, 1u, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
  while (__hip_atomic_load(bar, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) < target)
    __builtin_amdgcn_s_sleep(1);
  __atomic_thread_fence(__ATOMIC_ACQUIRE);
}

// 256 threads read 8 granules per pass: 16 x 16-byte loads in flight per lane.
// Loads past the phase end repeat the last granule and are not counted.
__device__ __forceinline__ uint32_t StreamPhase(const Ctx& c, const Phase& p, int blk, int nblk,
                                                uint32_t acc) {
  const int t = threadIdx.x;
  for (uint32_t b = p.g0 + blk * 8; b < p.g1; b += nblk * 8) {
    uint4 v[16];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const uint32_t g = b + j < p.g1 ? b + j : p.g1 - 1;
      const uint4* q = reinterpret_cast<const uint4*>(LdC(c.gran + g));
      v[2 * j] = q[t];
      v[2 * j + 1] = q[t + 256];
    }
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      if (b + j >= p.g1) break;
      acc += Alu(v[2 * j], p.alu) + Alu(v[2 * j + 1], p.alu);
      if (p.store) {
        uint4* o = reinterpret_cast<uint4*>(c.stout + (size_t)(b + j - p.g0) * kGran);
        o[t] = v[2 * j];
        o[t + 256] = v[2 * j + 1];
      }
    }
  }
  if (p.spin) acc = Spin(acc, p.spin);
  return acc;
}

__global__ void __launch_bounds__(256) PhaseKernel(Ctx c, int pi) {
  uint32_t acc = StreamPhase(c, LdPhase(c.ph + pi), blockIdx.x, gridDim.x, 0);
  if (acc == 0x12345678u) c.sink[blockIdx.x] = acc;
}

__global__ void __launch_bounds__(256) PersistentKernel(Ctx c, int barriers) {
  uint32_t acc = 0;
  for (int pi = 0; pi < c.nph; ++pi) {
    acc = StreamPhase(c, LdPhase(c.ph + pi), blockIdx.x, gridDim.x, acc);
    if (barriers) {
      __syncthreads();
      if (threadIdx.x == 0) GridBarrier(c.bar, (pi + 1) * gridDim.x);
      __syncthreads();
    }
  }
  if (acc == 0x12345678u) c.sink[blockIdx.x] = acc;
}

// Mode D. Granule k of block w in phase p is g0 + w + k * nblk. Loader wave
// l writes half of each granule into ring slot (seq % NS); compute waves
// read the slot after both halves arrived. LDS flags use relaxed atomics and
// lgkmcnt waits only, so the loaders keep global loads in flight.
template <int NS, int kCompute>
__global__ void __launch_bounds__(32 * (kLoaders + kCompute + 1)) RingKernel(Ctx c) {
  constexpr int kPer = 512 / kCompute;  // 16-byte vectors per compute wave per granule
  __shared__ uint4 ring[NS][kGran / 16];
  __shared__ uint32_t filled[NS], consumed[NS], wdone, phase_ok, lpos;
  const int wave = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int nblk = gridDim.x, w = blockIdx.x;
  if (threadIdx.x < NS) filled[threadIdx.x] = consumed[threadIdx.x] = 0;
  if (threadIdx.x == 0) wdone = phase_ok = lpos = 0;
  __syncthreads();
  // Dependent phases (experts): no load may be issued before every block
  // finished the previous phase (the router).
  auto dep_at = [&](const uint32_t* sq, int k, int n) -> uint32_t {  // barrier target, 0 if none
    if (!c.barriers || k >= n) return 0;
    const uint32_t e = LdC(sq + k), pi = e >> kGBits;
    if (k > 0 && (LdC(sq + k - 1) >> kGBits) == pi) return 0;
    return LdC(reinterpret_cast<const uint32_t*>(c.ph + pi) + 4) ? pi * nblk : 0;
  };
  auto wait_bar = [&](uint32_t target) {
    while (__hip_atomic_load(c.bar, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) < target)
      __builtin_amdgcn_s_sleep(1);
  };
  if (wave == kLoaders + kCompute) {
    // Prefetch wave: one dword per 128-byte line, two granules per step,
    // at most c.pf granules ahead of the loader waves.
    const uint32_t* sq = c.bseq + LdC(c.boff + w);
    const int n = LdC(c.blen + w);
    uint32_t acc = 0;
    for (int k = 0; k < n; k += 2) {
      while ((int)__hip_atomic_load(&lpos, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_WORKGROUP) + c.pf < k)
        __builtin_amdgcn_s_sleep(1);
      if (const uint32_t t = dep_at(sq, k, n)) wait_bar(t);
      if (const uint32_t t = dep_at(sq, k + 1, n)) wait_bar(t);
      const uint32_t* q0 = reinterpret_cast<const uint32_t*>(LdC(c.gran + (LdC(sq + k) & kGMask)));
      const int k1 = k + 1 < n ? k + 1 : k;
      const uint32_t* q1 = reinterpret_cast<const uint32_t*>(LdC(c.gran + (LdC(sq + k1) & kGMask)));
      acc += q0[lane * 32] + q0[1024 + lane * 32] + q1[lane * 32] + q1[1024 + lane * 32];
    }
    if (acc == 0x12345678u) c.sink[w] = acc;
    return;
  }
  if (wave < kLoaders) {
    // 4 KB per loader wave per granule (8 vectors per lane). Two register
    // sets alternate, so granule k+1 is in flight while k goes to LDS.
    const uint32_t* sq = c.bseq + LdC(c.boff + w);
    const int n = LdC(c.blen + w);
    if (n == 0) return;
#define MS_LD(r, k)                                                                            \
  {                                                                                            \
    const int kk = (k) < n ? (k) : n - 1;                                                      \
    const uint4* q = reinterpret_cast<const uint4*>(LdC(c.gran + (LdC(sq + kk) & kGMask))) + wave * 256; \
    r##0 = q[0 * 32 + lane]; r##1 = q[1 * 32 + lane]; r##2 = q[2 * 32 + lane];                \
    r##3 = q[3 * 32 + lane]; r##4 = q[4 * 32 + lane]; r##5 = q[5 * 32 + lane];                \
    r##6 = q[6 * 32 + lane]; r##7 = q[7 * 32 + lane];                                          \
  }
#define MS_PUT(r, k)                                                                           \
  if ((k) < n) {                                                                               \
    const int s = (k) % NS;                                                                    \
    const uint32_t need = kCompute * ((k) / NS);                                               \
    while (__hip_atomic_load(&consumed[s], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_WORKGROUP) <   \
           need)                                                                               \
      __builtin_amdgcn_s_sleep(1);                                                             \
    uint4* d = &ring[s][wave * 256 + lane];                                                    \
    d[0] = r##0; d[32] = r##1; d[64] = r##2; d[96] = r##3;                                     \
    d[128] = r##4; d[160] = r##5; d[192] = r##6; d[224] = r##7;                                \
    asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");                                         \
    if (lane == 0)                                                                             \
      __hip_atomic_fetch_add(&filled[s], 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_WORKGROUP);  \
    if (lane == 0 && wave == 0)                                                                \
      __hip_atomic_store(&lpos, (uint32_t)(k), __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_WORKGROUP); \
  }
    // A dependent phase: the loader first hands over its pending granule,
    // then waits for the barrier.
    uint4 A0, A1, A2, A3, A4, A5, A6, A7, B0, B1, B2, B3, B4, B5, B6, B7;
    MS_LD(A, 0);
    for (int k = 0; k < n; k += 2) {
      bool done = false;
      if (const uint32_t t = dep_at(sq, k + 1, n)) {
        MS_PUT(A, k);
        done = true;
        wait_bar(t);
      }
      MS_LD(B, k + 1);
      if (!done) MS_PUT(A, k);
      done = false;
      if (const uint32_t t = dep_at(sq, k + 2, n)) {
        MS_PUT(B, k + 1);
        done = true;
        wait_bar(t);
      }
      MS_LD(A, k + 2);
      if (!done) MS_PUT(B, k + 1);
    }
#undef MS_LD
#undef MS_PUT
    return;
  }
  const int cw = wave - kLoaders;
  uint32_t acc = 0, seq = 0;
  for (int pi = 0; pi < c.nph; ++pi) {
    const Phase p = LdPhase(c.ph + pi);
    for (uint32_t g = p.g0 + w; g < p.g1; g += nblk, ++seq) {
      const int s = seq % NS;
      const uint32_t need = kLoaders * (seq / NS + 1);
      while (__hip_atomic_load(&filled[s], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_WORKGROUP) < need)
        __builtin_amdgcn_s_sleep(1);
      asm volatile("" ::: "memory");
      uint4 v[kPer / 32];
#pragma unroll
      for (int j = 0; j < kPer / 32; ++j) v[j] = ring[s][cw * kPer + j * 32 + lane];
      asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
      if (lane == 0)
        __hip_atomic_fetch_add(&consumed[s], 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_WORKGROUP);
#pragma unroll
      for (int j = 0; j < kPer / 32; ++j) {
        acc += Alu(v[j], p.alu);
        if (p.store) {
          uint4* o = reinterpret_cast<uint4*>(c.stout + (size_t)(g - p.g0) * kGran);
          o[cw * kPer + j * 32 + lane] = v[j];
        }
      }
    }
    if (p.spin) acc = Spin(acc, p.spin);
    // Phase end: the compute waves of this block meet, one lane arrives at
    // the grid barrier, the others wait on an LDS phase word.
    if (lane == 0) __hip_atomic_fetch_add(&wdone, 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_WORKGROUP);
    if (cw == 0) {
      if (lane == 0) {
        while (__hip_atomic_load(&wdone, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_WORKGROUP) <
               (uint32_t)kCompute * (pi + 1))
          __builtin_amdgcn_s_sleep(1);
        if (c.barriers) GridBarrier(c.bar, (pi + 1) * nblk);
        __hip_atomic_store(&phase_ok, (uint32_t)(pi + 1), __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_WORKGROUP);
      }
    } else if (lane == 0) {
      while (__hip_atomic_load(&phase_ok, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_WORKGROUP) < (uint32_t)(pi + 1))
        __builtin_amdgcn_s_sleep(1);
    }
    __builtin_amdgcn_wave_barrier();
  }
  if (acc == 0x12345678u) c.sink[w] = acc;
}

// Byte sizes from FLASHNEXT_FORWARD.md section 5 (3.05 checkpoint).
constexpr size_t kHcDots = 324ull * 10240 * 2, kHcMix = 10240ull * 320 * 2;
constexpr size_t kGdnIn = 16409604 + 9847812, kQsaIn = 19690500 + 2 * 825348 + 620804;
constexpr size_t kOut = 9847812, kRouter = 2621440, kState = 48ull * 128 * 128 * 4;
constexpr size_t kExpMat = 620804, kShMat = 1030404, kPle = 65536000, kHead = 397813764;

size_t Gr(size_t bytes) { return (bytes + kGran - 1) / kGran; }

struct Plan {
  std::vector<Phase> ph;
  std::vector<const uint8_t*> gran;
  size_t bytes = 0;
};

// Static weights are laid out in forward order in `stat`; experts are
// picked at random from `pool` (npool experts of 3 matrices each).
Plan BuildPlan(uint8_t* stat, uint8_t* state, uint8_t* pool, size_t npool, std::mt19937_64& rng,
               int alu_scale, int U, int R, int dep_experts, int gdn_ticks, int qsa_ticks) {
  // Per-weight lane ops: EXL3 decode ~6.25 plus one v_dot2 per 2 weights per
  // row; fp16 HC/router: one fp32 FMA per weight per row plus conversion.
  // Routed experts run only their routed rows (moe_rows.h): 10 R / U rows
  // per distinct expert on average, not R.
  const double rr = 10.0 * R / U;
  const int kAluK3 = (int)((6.25 + 0.5 * rr) * 128 / 3), kAluK5 = (int)((6.25 + 0.5 * R) * 128 / 5);
  const int kAluF16 = (int)((0.5 + R) * 8);
  Plan P;
  size_t soff = 0;
  auto add_static = [&](size_t bytes, int alu, int store = 0) {
    Phase p{(uint32_t)P.gran.size(), 0, alu * alu_scale, store, 0};
    for (size_t g = 0; g < Gr(bytes); ++g) P.gran.push_back(stat + soff + g * kGran);
    soff += Gr(bytes) * kGran;
    p.g1 = (uint32_t)P.gran.size();
    P.ph.push_back(p);
    P.bytes += bytes;
  };
  const size_t exp_stride = Gr(kExpMat) * kGran * 3;
  auto add_spin = [&](int ticks) {
    if (ticks > 0) P.ph.push_back(Phase{(uint32_t)P.gran.size(), (uint32_t)P.gran.size(), 0, 0, 0, ticks});
  };
  for (int l = 0; l < 48; ++l) {
    const bool attn = l % 4 == 3;
    if (l == 1) add_static(kPle, kAluF16);
    add_static(kHcDots, kAluF16);
    add_static(kHcMix, kAluF16);
    add_static(attn ? kQsaIn : kGdnIn, kAluK5);
    if (attn) add_spin(qsa_ticks);
    if (!attn) {  // GDN state: read the previous state, write the new one
      Phase p{(uint32_t)P.gran.size(), 0, kAluF16 * alu_scale, 1, 0};
      for (size_t g = 0; g < Gr(kState); ++g) P.gran.push_back(state + (size_t)l * Gr(kState) * kGran + g * kGran);
      p.g1 = (uint32_t)P.gran.size();
      P.ph.push_back(p);
      P.bytes += 2 * kState;
      add_spin(gdn_ticks);
    }
    add_static(kOut, kAluK5);
    add_static(kHcDots, kAluF16);
    add_static(kHcMix, kAluF16);
    add_static(kRouter, kAluF16);
    std::vector<size_t> ex;
    while ((int)ex.size() < U) {
      size_t e = rng() % npool;
      if (std::find(ex.begin(), ex.end(), e) == ex.end()) ex.push_back(e);
    }
    for (int part = 0; part < 2; ++part) {  // 0: gate/up, 1: down
      Phase p{(uint32_t)P.gran.size(), 0, kAluK3 * alu_scale, 0, part == 0 ? dep_experts : 0};
      for (size_t e : ex)
        for (int m = part == 0 ? 0 : 2; m < (part == 0 ? 2 : 3); ++m)
          for (size_t g = 0; g < Gr(kExpMat); ++g)
            P.gran.push_back(pool + e * exp_stride + m * Gr(kExpMat) * kGran + g * kGran);
      P.bytes += (part == 0 ? 2 : 1) * U * kExpMat;
      const size_t sh = part == 0 ? 2 * kShMat : kShMat;  // shared expert, K5
      for (size_t g = 0; g < Gr(sh); ++g) P.gran.push_back(stat + soff + g * kGran);
      soff += Gr(sh) * kGran;
      P.bytes += sh;
      p.g1 = (uint32_t)P.gran.size();
      P.ph.push_back(p);
    }
  }
  add_static(kHcDots, kAluF16);
  add_static(kHcMix, kAluF16);
  add_static(kHead, kAluK5);
  return P;
}

size_t StaticBytes() { return size_t(4) << 30; }

}  // namespace

int main(int argc, char** argv) {
  const double pool_gb = argc > 1 ? atof(argv[1]) : 8.0;
  const int nblk = argc > 2 ? atoi(argv[2]) : 40;
  const int slots = argc > 3 ? atoi(argv[3]) : 6;
  const int alu_scale = argc > 4 ? atoi(argv[4]) : 1;
  const int reps = argc > 5 ? atoi(argv[5]) : 5;
  const int U = argc > 6 ? atoi(argv[6]) : 10;  // distinct routed experts per layer
  const int R = argc > 7 ? atoi(argv[7]) : 1;   // rows (scales the ALU per byte)
  const int dep = argc > 8 ? atoi(argv[8]) : 1;  // mode D: expert loads wait for the router
  const int nc = argc > 9 ? atoi(argv[9]) : 4;   // mode D/E: compute waves per block (4 or 8)
  const std::string modes = argc > 10 ? argv[10] : "ABDEF";
  const double gdn_us = argc > 11 ? atof(argv[11]) : 0;  // ALU-only phase after each GDN state phase
  const double qsa_us = argc > 12 ? atof(argv[12]) : 0;  // ALU-only phase after each QSA input phase
  const int pf = argc > 13 ? atoi(argv[13]) : 0;          // modes D/E: prefetch granules ahead per block
  // Spin iterations per microsecond with the ring's wave count per block.
  double ipu = 0;
  {
    uint32_t* csink;
    HIP_CHECK(hipMalloc(&csink, 4096 * 4));
    std::vector<double> t;
    for (int r = 0; r < 6; ++r) {
      bench::EventTimer et;
      et.Begin();
      hipLaunchKernelGGL(SpinKernel, nblk, 32 * (kLoaders + nc), 0, nullptr, csink, 2000, kLoaders);
      const float ms = et.EndMs();
      if (r) t.push_back(ms);
    }
    ipu = 2000.0 / (bench::Median(t) * 1000.0);
    HIP_CHECK(hipFree(csink));
  }
  const int clk_khz = (int)(ipu * 1000);  // printed as iterations per ms
  const int gdn_ticks = (int)(gdn_us * ipu + 0.5), qsa_ticks = (int)(qsa_us * ipu + 0.5);
  const size_t exp_bytes = Gr(kExpMat) * kGran * 3;
  const size_t npool = (size_t)(pool_gb * 1e9) / exp_bytes;
  uint8_t *stat, *pool, *state, *stout;
  uint32_t *bar, *sink;
  HIP_CHECK(hipMalloc(&stat, StaticBytes()));
  HIP_CHECK(hipMalloc(&pool, npool * exp_bytes));
  HIP_CHECK(hipMalloc(&state, 48 * Gr(kState) * kGran));
  HIP_CHECK(hipMalloc(&stout, Gr(kState) * kGran));
  HIP_CHECK(hipMalloc(&bar, 4));
  HIP_CHECK(hipMalloc(&sink, 4096 * 4));
  HIP_CHECK(hipMemset(stat, 1, StaticBytes()));
  HIP_CHECK(hipMemset(pool, 2, npool * exp_bytes));
  HIP_CHECK(hipMemset(state, 3, 48 * Gr(kState) * kGran));
  std::mt19937_64 rng(1234);
  Plan P0 = BuildPlan(stat, state, pool, npool, rng, alu_scale, U, R, dep, gdn_ticks, qsa_ticks);
  const uint8_t** dgran;
  Phase* dph;
  HIP_CHECK(hipMalloc(&dgran, P0.gran.size() * sizeof(void*)));
  HIP_CHECK(hipMalloc(&dph, P0.ph.size() * sizeof(Phase)));
  uint32_t *dseq, *doff;
  int* dlen;
  HIP_CHECK(hipMalloc(&dseq, (P0.gran.size() + 16) * 4));
  HIP_CHECK(hipMalloc(&doff, nblk * (P0.ph.size() + 1) * 4));
  HIP_CHECK(hipMalloc(&dlen, nblk * (P0.ph.size() + 1) * 4));
  if (P0.gran.size() > kGMask) { std::fprintf(stderr, "too many granules\n"); return 1; }
  std::printf("dep %d, pool %.1f GB (%zu experts), U %d, R %d, %zu phases, %zu granules, %.3f GB logical, blocks %d, slots %d, alu x%d\n",
              dep, npool * exp_bytes / 1e9, npool, U, R, P0.ph.size(), P0.gran.size(), P0.bytes / 1e9, nblk, slots,
              alu_scale);
  std::printf("nc %d, ALU-only phases: GDN %.1f us, QSA %.1f us (%d spin iterations per ms); prefetch %d granules ahead (%.1f MB total)\n",
              nc, gdn_us, qsa_us, clk_khz, pf, (double)pf * kGran * nblk / 1e6);
  const char* names[] = {"A kernels", "B barrier", "D ring", "F flat", "E ring/phase"};
  for (int mode : {0, 1, 2, 4, 3}) {
    if (modes.find("ABDFE"[mode]) == std::string::npos) continue;
    if (mode == 2) {
      // A persistent grid that is not fully co-resident hangs at the first barrier.
      auto k = nc == 8 ? (slots >= 3 ? RingKernel<3, 8> : RingKernel<2, 8>)
                       : (slots >= 6 ? RingKernel<6, 4> : slots >= 3 ? RingKernel<3, 4> : RingKernel<2, 4>);
      const int threads = 32 * (kLoaders + nc + (pf > 0 ? 1 : 0));
      // hipOccupancyMaxActiveBlocksPerMultiprocessor reports 2 per WGP here,
      // but v4 ran 80 blocks of 10 waves. Count per WGP (4 SIMDs, 1536 VGPRs
      // and 16 wave slots each, 128 KB LDS) instead: 80 fits, 120 hung in v4.
      hipFuncAttributes fa{};
      HIP_CHECK(hipFuncGetAttributes(&fa, reinterpret_cast<const void*>(k)));
      int nwgp = 0;
      HIP_CHECK(hipDeviceGetAttribute(&nwgp, hipDeviceAttributeMultiprocessorCount, 0));
      const int waves = threads / 32, vgpr = std::max(8, (fa.numRegs + 7) / 8 * 8);
      const int wps = std::min(16, 1536 / vgpr);
      const int per_wgp = std::min(4 * wps / waves, (int)(131072 / std::max<size_t>(fa.sharedSizeBytes, 1)));
      if (per_wgp * nwgp < nblk) {
        std::printf("D ring     skipped: %d blocks of %d waves; %d VGPRs, %zu B LDS: %d per WGP x %d\n", nblk, waves,
                    fa.numRegs, fa.sharedSizeBytes, per_wgp, nwgp);
        continue;
      }
    }
    std::vector<double> ms;
    for (int r = 0; r < reps + 1; ++r) {
      Plan P = BuildPlan(stat, state, pool, npool, rng, alu_scale, U, R, dep, gdn_ticks, qsa_ticks);
      HIP_CHECK(hipMemcpy(dgran, P.gran.data(), P.gran.size() * sizeof(void*), hipMemcpyHostToDevice));
      HIP_CHECK(hipMemcpy(dph, P.ph.data(), P.ph.size() * sizeof(Phase), hipMemcpyHostToDevice));
      // Per-block granule sequences; entries [0, nblk) cover the forward,
      // entries [nblk * (1 + pi), ...) cover phase pi alone (mode E).
      const size_t nph = P.ph.size();
      std::vector<uint32_t> bseq, boff(nblk * (nph + 1));
      std::vector<int> blen(nblk * (nph + 1));
      for (int w = 0; w < nblk; ++w) {
        boff[w] = (uint32_t)bseq.size();
        for (size_t pi = 0; pi < nph; ++pi) {
          const Phase& p = P.ph[pi];
          boff[nblk * (1 + pi) + w] = (uint32_t)bseq.size();
          for (uint32_t g = p.g0 + w; g < p.g1; g += nblk) bseq.push_back(g | (uint32_t)pi << kGBits);
          blen[nblk * (1 + pi) + w] = (int)(bseq.size() - boff[nblk * (1 + pi) + w]);
        }
        blen[w] = (int)(bseq.size() - boff[w]);
      }
      HIP_CHECK(hipMemcpy(dseq, bseq.data(), bseq.size() * 4, hipMemcpyHostToDevice));
      HIP_CHECK(hipMemcpy(doff, boff.data(), boff.size() * 4, hipMemcpyHostToDevice));
      HIP_CHECK(hipMemcpy(dlen, blen.data(), blen.size() * 4, hipMemcpyHostToDevice));
      HIP_CHECK(hipMemset(bar, 0, 4));
      Ctx c{dgran, dph, (int)P.ph.size(), stout, bar, sink, dseq, doff, dlen, 1, pf};
      HIP_CHECK(hipDeviceSynchronize());
      bench::EventTimer t;
      t.Begin();
      if (mode == 0) {
        for (int pi = 0; pi < c.nph; ++pi) {
          const uint32_t n = P.ph[pi].g1 - P.ph[pi].g0;
          // ALU-only phases use the full grid, like the ring modes.
          const int g = n == 0 ? nblk : std::max(1, std::min<int>((n + 7) / 8, 1024));
          hipLaunchKernelGGL(PhaseKernel, g, 256, 0, nullptr, c, pi);
        }
      } else if (mode == 1 || mode == 3) {
        hipLaunchKernelGGL(PersistentKernel, nblk, 256, 0, nullptr, c, mode == 1 ? 1 : 0);
      } else {
        auto k = nc == 8 ? (slots >= 3 ? RingKernel<3, 8> : RingKernel<2, 8>)
                         : (slots >= 6 ? RingKernel<6, 4> : slots >= 3 ? RingKernel<3, 4> : RingKernel<2, 4>);
        const int threads = 32 * (kLoaders + nc + (pf > 0 ? 1 : 0));
        if (mode == 2) {
          hipLaunchKernelGGL(k, nblk, threads, 0, nullptr, c);
        } else {
          for (int pi = 0; pi < c.nph; ++pi) {
            Ctx e = c;
            e.ph = dph + pi;
            e.nph = 1;
            e.boff = doff + nblk * (1 + pi);
            e.blen = dlen + nblk * (1 + pi);
            e.barriers = 0;
            hipLaunchKernelGGL(k, nblk, threads, 0, nullptr, e);
          }
        }
      }
      const float m = t.EndMs();
      HIP_CHECK(hipGetLastError());
      if (r > 0) ms.push_back(m);
    }
    const double med = bench::Median(ms);
    std::printf("%-10s median %7.3f ms  %6.1f GB/s  (min %.3f max %.3f)\n", names[mode], med,
                P0.bytes / med / 1e6, *std::min_element(ms.begin(), ms.end()),
                *std::max_element(ms.begin(), ms.end()));
  }
  return 0;
}
