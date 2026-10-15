// Decode hyper-connection read (dots, mid, mix) on WMMA, one launch per site.
//
// The fp32 activations (H, t) take the prefill high/low fp16 split; the fp16
// weights fn and up are exact WMMA operands. Every R = 1..16 runs the same
// code: rows pad to one 16-row tile, a row's result depends only on its own
// A row, and every reduction (K chunks, streams, square sums) has a fixed
// order. So each row is bitwise equal for any R (Gates 2 and 3). It is not
// bitwise equal to the VALU HcDotsKernel/HcMixKernel order (Gate 1(b)).
//
// Grid: dots blocks first, then kHidden / 16 mix blocks. A dots block owns 7
// column tiles of fn for one stream and K chunk. The last dots block to
// finish reduces the K chunks, computes t (high/low) and the inject gates,
// and publishes `epoch`. Mix blocks load their weights and epilogue operands
// first and then wait for that flag; workgroups dispatch in index order, so
// every dots block is resident or done before any mix block waits, and the
// wait cannot deadlock.
#pragma once

#include "ops.h"
#include "../gfx1151/prefill_wmma.h"
#include "../exl3/plans.h"

namespace crucible {

constexpr int kHcWRows = 16;                             // one WMMA row tile
constexpr int kHcWTiles = (kRank + kStreams + 15) / 16;  // 21 column tiles of fn
constexpr int kHcWPad = kHcWTiles * 16;                  // 336
constexpr int kHcWMixBlocks = kHidden / 16;              // 160
constexpr int kHcWMixSteps = kRank / 32;                 // 16-k steps per mix K half
constexpr int kHcWGroups = 3, kHcWGroupTiles = 7;        // dots column groups
static_assert(kHcWGroups * kHcWGroupTiles == kHcWTiles);

struct HcWScratch {
  float* part;      // [KS][kStreams][16][kHcWPad] dots partials
  float* pss;       // [KS][kStreams][16] square-sum partials
  float* inv;       // [16][kStreams] 1 / rms
  __half* thi;      // [16][kRank] t, high half
  __half* tlo;      // [16][kRank] t, low half (scaled)
  unsigned* count;  // finished dots blocks (reset by the last one)
  unsigned* flag;   // last published epoch
};

template <int KS>
constexpr int HcWDotsBlocks() { return KS * kStreams * kHcWGroups; }

template <int KS>
constexpr size_t HcWPartFloats() { return (size_t)KS * kStreams * kHcWRows * kHcWPad; }

// LDS row stride of the staged activations: 8 halves of padding spread the
// rows of a fragment read over the banks.
template <int KS>
constexpr int HcWStride() { return kHidden / KS + 8; }

// Dynamic LDS bytes for R rows (the larger of the two block kinds).
template <int KS>
inline size_t HcWLds(int R) {
  const size_t dots = (size_t)2 * R * HcWStride<KS>() * sizeof(_Float16);
  const size_t mix = (size_t)2 * R * kRank * sizeof(_Float16) + (size_t)2 * kStreams * R * 17 * sizeof(float);
  return dots > mix ? dots : mix;
}

// One block: the dots of 7 fn column tiles with the R rows of stream s over
// K chunk kc. The block stages that slice of H once, split into high/low
// fp16, so every wave reads its A fragments from LDS. Lanes 16..31 repeat
// lanes 0..15, as wave32 WMMA requires.
template <int KS, int PF>
__device__ __forceinline__ void HcWDots(int db, const __half* __restrict__ fn, int M,
                                        const float* __restrict__ H, int R, const HcWScratch& sc,
                                        unsigned char* smem) {
  constexpr int KC = kHidden / KS, STEPS = KC / 16, LD = HcWStride<KS>(), V = KC / 4;
  constexpr int NV = kHcWRows * V / 256;  // float4 staging items per thread at R = 16
  static_assert(STEPS % PF == 0 && kHcWRows * V % 256 == 0 && V % 16 == 0);
  _Float16* ahi = reinterpret_cast<_Float16*>(smem);
  _Float16* alo = ahi + R * LD;
  const int t = threadIdx.x, lane = t & 31, wave = t >> 5, row = lane & 15;
  const int ntg = db % kHcWGroups, s = (db / kHcWGroups) % kStreams, kc = db / (kHcWGroups * kStreams);
  const int nt = ntg * kHcWGroupTiles + wave;
  const bool work = wave < kHcWGroupTiles && nt * 16 < M;
  // Columns past M read column M - 1 and are never stored.
  const int m = min(nt * 16 + row, M - 1);
  const auto* bp = reinterpret_cast<const prefill::H16*>(fn + (size_t)m * kWide + s * kHidden + kc * KC);
  prefill::H16 ring[PF];
  if (work) {
#pragma unroll
    for (int f = 0; f < PF; ++f) ring[f] = bp[f];
  }
  const float* hs = H + s * kHidden + kc * KC;
  {
    float4 v[NV];
#pragma unroll
    for (int j = 0; j < NV; ++j) {
      const int i = t + 256 * j, r = i / V, k4 = i - r * V;
      if (r < R) v[j] = *reinterpret_cast<const float4*>(hs + (size_t)r * kWide + k4 * 4);
    }
#pragma unroll
    for (int j = 0; j < NV; ++j) {
      const int i = t + 256 * j, r = i / V, k4 = i - r * V;
      if (r >= R) continue;
      const float f[4] = {v[j].x, v[j].y, v[j].z, v[j].w};
      typedef _Float16 H4 __attribute__((ext_vector_type(4)));
      H4 h4, l4;
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        h4[e] = (_Float16)f[e];
        l4[e] = (_Float16)((f[e] - (float)h4[e]) * prefill::kResidualScale);
      }
      *reinterpret_cast<H4*>(ahi + r * LD + k4 * 4) = h4;
      *reinterpret_cast<H4*>(alo + r * LD + k4 * 4) = l4;
    }
  }
  // Square sums of this K chunk: 16 threads per row, each over float4 items
  // l, l + 16, ... in order, then a fixed shuffle tree. Group 0 writes them.
  if (ntg == 0) {
    const int r = t >> 4, l = t & 15;
    float4 v[V / 16];
    if (r < R) {
#pragma unroll
      for (int j = 0; j < V / 16; ++j)
        v[j] = *reinterpret_cast<const float4*>(hs + (size_t)r * kWide + (l + 16 * j) * 4);
    }
    float sq = 0.f;
    if (r < R) {
#pragma unroll
      for (int j = 0; j < V / 16; ++j) {
        sq += v[j].x * v[j].x;
        sq += v[j].y * v[j].y;
        sq += v[j].z * v[j].z;
        sq += v[j].w * v[j].w;
      }
    }
#pragma unroll
    for (int o = 1; o < 16; o <<= 1) sq += __shfl_xor(sq, o);
    if (l == 0 && r < R) sc.pss[(kc * kStreams + s) * kHcWRows + r] = sq;
  }
  __syncthreads();
  if (!work) return;
  // A tile t holds the high halves of rows 8t..8t+7 in its rows 0..7 and
  // their low halves in rows 8..15, so R <= 8 needs one WMMA per step. Each
  // output element still accumulates only its own A row: values do not
  // depend on the packing or on R.
  const int NT = (R + 7) >> 3, i7 = row & 7;
  const _Float16* ar = (row < 8 ? ahi : alo) + i7 * LD;
  const bool live0 = i7 < R, live1 = 8 + i7 < R;
  prefill::F8 acc[2] = {};
  for (int st0 = 0; st0 < STEPS; st0 += PF) {
#pragma unroll
    for (int f = 0; f < PF; ++f) {
      const int st = st0 + f;
      prefill::H16 a0 = {};
      if (live0) a0 = *reinterpret_cast<const prefill::H16*>(ar + st * 16);
      acc[0] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a0, ring[f], acc[0]);
      if (NT > 1) {
        prefill::H16 a1 = {};
        if (live1) a1 = *reinterpret_cast<const prefill::H16*>(ar + 8 * LD + st * 16);
        acc[1] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a1, ring[f], acc[1]);
      }
      // The last round re-reads the final step; those values are unused.
      ring[f] = bp[min(st + PF, STEPS - 1)];
      // Without it the scheduler hoists the LDS fragments of every step of
      // the round (8 VGPRs each).
      __builtin_amdgcn_sched_barrier(0);
    }
  }
  // Fragment rows z and z + 4 of a lane are the high and low parts of one row.
  float* out = sc.part + ((size_t)(kc * kStreams + s) * kHcWRows) * kHcWPad + nt * 16 + row;
#pragma unroll
  for (int t2 = 0; t2 < 2; ++t2) {
#pragma unroll
    for (int z = 0; z < 4; ++z) {
      const int r = 8 * t2 + prefill::FragmentRow(z);
      if (r < R) out[(size_t)r * kHcWPad] = acc[t2][z] + acc[t2][z + 4] * (1.f / prefill::kResidualScale);
    }
  }
}

// The last dots block: K-chunk sums in chunk order, then the HC mid. Items
// go four per thread per round so their partial loads are all in flight.
template <int KS>
__device__ __forceinline__ void HcWMid(int M, int R, float* __restrict__ post, const HcWScratch& sc) {
  __shared__ float inv[kHcWRows][kStreams];
  const int t = threadIdx.x;
  if (t < R * kStreams) {
    const int r = t / kStreams, s = t % kStreams;
    float a = 0.f;
#pragma unroll
    for (int kc = 0; kc < KS; ++kc) a += sc.pss[(kc * kStreams + s) * kHcWRows + r];
    inv[r][s] = rsqrtf(a / kHidden + kEps);
    sc.inv[r * kStreams + s] = inv[r][s];
  }
  __syncthreads();
  constexpr int B = 4;
  for (int i0 = t; i0 < R * M; i0 += B * blockDim.x) {
    float d[B][kStreams][KS];
#pragma unroll
    for (int b = 0; b < B; ++b) {
      const int i = i0 + b * blockDim.x;
      if (i >= R * M) continue;
      const int r = i / M, m = i - r * M;
#pragma unroll
      for (int s = 0; s < kStreams; ++s)
#pragma unroll
        for (int kc = 0; kc < KS; ++kc)
          d[b][s][kc] = sc.part[((size_t)(kc * kStreams + s) * kHcWRows + r) * kHcWPad + m];
    }
#pragma unroll
    for (int b = 0; b < B; ++b) {
      const int i = i0 + b * blockDim.x;
      if (i >= R * M) continue;
      const int r = i / M, m = i - r * M;
      float a = 0.f;
#pragma unroll
      for (int s = 0; s < kStreams; ++s) {
        float sum = 0.f;
#pragma unroll
        for (int kc = 0; kc < KS; ++kc) sum += d[b][s][kc];
        a += inv[r][s] * sum;
      }
      if (m < kRank) {
        const float v = Silu(a * .25f);
        const __half h = __float2half(v);
        sc.thi[r * kRank + m] = h;
        sc.tlo[r * kRank + m] = __float2half((v - __half2float(h)) * prefill::kResidualScale);
      } else if (post) {
        post[r * kStreams + m - kRank] = 2.f * Sigmoid(a * .25f);
      }
    }
  }
}

// One block: 16 hidden columns. Wave w owns stream w % 4 and K half w / 4.
__device__ __forceinline__ void HcWMix(int b, const __half* __restrict__ up, const float* __restrict__ w1,
                                       const float* __restrict__ H, int R, float* __restrict__ x,
                                       const HcWScratch& sc, unsigned epoch, unsigned char* smem) {
  constexpr int STEPS = kHcWMixSteps, KT = kRank / 16;
  auto* ts = reinterpret_cast<prefill::H16*>(smem);  // [2][R][KT]: t high, then low
  float* g = reinterpret_cast<float*>(ts + 2 * R * KT);  // [2][kStreams][R][17]
  const int t = threadIdx.x, lane = t & 31, wave = t >> 5, row = lane & 15;
  const int s = wave & 3, kh = wave >> 2, d0 = b * 16;
  const auto* bp = reinterpret_cast<const prefill::H16*>(up + ((size_t)s * kHidden + d0 + row) * kRank) + kh * STEPS;
  prefill::H16 w[STEPS];
#pragma unroll
  for (int st = 0; st < STEPS; ++st) w[st] = bp[st];
  // Epilogue operands of thread (row er, column ec), loaded before the wait.
  const int er = t >> 4, ec = t & 15, d = d0 + ec;
  float hv[kStreams], wv[kStreams];
#pragma unroll
  for (int st = 0; st < kStreams; ++st) {
    wv[st] = w1[st * kHidden + d];
    hv[st] = er < R ? H[((size_t)er * kStreams + st) * kHidden + d] : 0.f;
  }
  if (t == 0)
    while (__hip_atomic_load(sc.flag, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) != epoch)
      __builtin_amdgcn_s_sleep(1);
  __syncthreads();
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
  float iv[kStreams];
#pragma unroll
  for (int st = 0; st < kStreams; ++st) iv[st] = er < R ? sc.inv[er * kStreams + st] : 0.f;
  // Stage t rows < R: at most 2 x 16 x 20 fragments, three per thread.
#pragma unroll
  for (int j = 0; j < 3; ++j) {
    const int i = t + 256 * j;
    if (i >= 2 * R * KT) break;
    const int hl = i / (R * KT), rem = i - hl * R * KT;
    ts[i] = *reinterpret_cast<const prefill::H16*>((hl ? sc.tlo : sc.thi) + rem * 16);
  }
  __syncthreads();
  // The same packing as the dots: t rows 8u..8u+7 high, then low.
  const int NT = (R + 7) >> 3, i7 = row & 7;
  const prefill::H16* ar = ts + ((row < 8 ? 0 : R) + i7) * KT + kh * STEPS;
  const bool live0 = i7 < R, live1 = 8 + i7 < R;
  prefill::F8 acc[2] = {};
#pragma unroll
  for (int st = 0; st < STEPS; ++st) {
    prefill::H16 a0 = {};
    if (live0) a0 = ar[st];
    acc[0] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a0, w[st], acc[0]);
    if (NT > 1) {
      prefill::H16 a1 = {};
      if (live1) a1 = ar[8 * KT + st];
      acc[1] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a1, w[st], acc[1]);
    }
    __builtin_amdgcn_sched_barrier(0);
  }
#pragma unroll
  for (int t2 = 0; t2 < 2; ++t2) {
#pragma unroll
    for (int z = 0; z < 4; ++z) {
      const int r = 8 * t2 + prefill::FragmentRow(z);
      if (r < R)
        g[((kh * kStreams + s) * R + r) * 17 + row] = acc[t2][z] + acc[t2][z + 4] * (1.f / prefill::kResidualScale);
    }
  }
  __syncthreads();
  if (er < R) {
    float v[kStreams];
#pragma unroll
    for (int st = 0; st < kStreams; ++st) {
      const float gs = g[(st * R + er) * 17 + ec] + g[((kStreams + st) * R + er) * 17 + ec];
      v[st] = Sigmoid(gs) * hv[st] * iv[st] * wv[st];
    }
    x[(size_t)er * kHidden + d] = RoundH(((v[0] + v[1]) + (v[2] + v[3])) * .25f);
  }
}

// Launch with HcWDotsBlocks<KS>() + kHcWMixBlocks blocks of 256 threads and
// HcWLds<KS>(R) bytes of dynamic LDS.
template <int KS, int PF>
__global__ void __launch_bounds__(256) HcWmmaKernel(const __half* __restrict__ fn, int M,
                                                    const __half* __restrict__ up,
                                                    const float* __restrict__ w1,
                                                    const float* __restrict__ H, int R,
                                                    float* __restrict__ x, float* __restrict__ post,
                                                    HcWScratch sc, unsigned epoch, int first = 0) {
  constexpr int ND = HcWDotsBlocks<KS>();
  extern __shared__ __attribute__((aligned(16))) unsigned char smem[];
  // `first` > 0 skips blocks (bench: the mix alone after a published flag).
  const int blk = blockIdx.x + first;
  if (blk >= ND) {
    HcWMix(blk - ND, up, w1, H, R, x, sc, epoch, smem);
    return;
  }
  __shared__ bool last;
  // The lookahead prefetches past the last published weights (block 0 only),
  // like HcDotsKernel (fn) and HcMixKernel (up) do.
  PublishWeights(fn);
  HcWDots<KS, PF>(blk, fn, M, H, R, sc, smem);
  __syncthreads();
  PublishWeights(up);
  if (threadIdx.x == 0) {
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
    last = __hip_atomic_fetch_add(sc.count, 1u, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT) == ND - 1;
  }
  __syncthreads();
  if (!last) return;
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
  HcWMid<KS>(M, R, post, sc);
  __syncthreads();
  if (threadIdx.x == 0) {
    *sc.count = 0;
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
    __hip_atomic_store(sc.flag, epoch, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  }
}

// Engine shape: 4 K chunks, 20 fn fragments in flight per wave.
constexpr int kHcWKs = 4, kHcWPf = 20;

// Scratch for one site at a time (sites on one stream run in order).
inline HcWScratch HcWAlloc() {
  HcWScratch sc;
  sc.part = Scratch<float>(HcWPartFloats<kHcWKs>());
  sc.pss = Scratch<float>((size_t)kHcWKs * kStreams * kHcWRows);
  sc.inv = Scratch<float>(kHcWRows * kStreams);
  sc.thi = Scratch<__half>((size_t)kHcWRows * kRank);
  sc.tlo = Scratch<__half>((size_t)kHcWRows * kRank);
  sc.count = Zeros<unsigned>(1);
  sc.flag = Zeros<unsigned>(1);
  return sc;
}

// x[r] (and post[r] if non-null) for rows r < R <= 16 of H. `epoch` must
// differ from the previous launch on the same scratch.
inline void HcWLaunch(const __half* fn, int M, const __half* up, const float* w1, const float* H, int R,
                      float* x, float* post, const HcWScratch& sc, unsigned epoch) {
  HcWmmaKernel<kHcWKs, kHcWPf><<<HcWDotsBlocks<kHcWKs>() + kHcWMixBlocks, 256, HcWLds<kHcWKs>(R)>>>(
      fn, M, up, w1, H, R, x, post, sc, epoch);
}

}  // namespace crucible
