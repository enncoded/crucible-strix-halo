// MoE kernels that only benches launch: the unique-expert path of ops.h and
// the compact-row kernels of moe_rows.h that MoeFusedKernel replaced.
#pragma once
#include "../../engine/layers/moe_rows.h"

namespace crucible {

template <int K, int R, int PF, bool Fill = true>
__device__ __forceinline__ void MoeGemvBody(const LinJob& job, const GemvItem& item, uint32_t* xs) {
  MoeGemvCore<K, R, PF, 1, 0>(job, item, xs, [&] {
    if constexpr (Fill) MoeFillX(job, item, R, xs);
  });
}

template <int K, int R, int PF, int C = kMoeChunk>
__device__ __forceinline__ void MoeGemvChunked(const LinJob& job, const GemvItem& item, uint32_t* xs) {
  static_assert(C % PF == 0, "chunks hold whole prefetch rounds");
  const int lane = threadIdx.x & 31;
  const int wave = __builtin_amdgcn_readfirstlane(threadIdx.x >> 5);
  const LaneWindow<K> win(lane);
  const int q = lane & 3;
  constexpr int T = 8 * K;
  const int A = job.I >> 4;
  const int span = A / job.S;
  const int a0 = item.s * span;
  const uint32_t* xg = reinterpret_cast<const uint32_t*>(job.xh);
  auto fill = [&](int c0, uint32_t* dst) {
    const int n = min(C, span - c0) * 8;
    for (int i = threadIdx.x; i < R * C * 8; i += blockDim.x) {
      const int r = i / (C * 8), o = i - r * (C * 8);
      if (o < n) dst[i] = xg[(size_t)r * A * 8 + (a0 + c0) * 8 + o];
    }
  };
  const bool active = wave < item.nb;
  const int b = item.b0 + (active ? wave : 0);
  const __amdgpu_buffer_rsrc_t rs = StreamRsrc(job.tiles + ((size_t)b * A + a0) * T, (uint32_t)span * T * 4);
  const int o0 = 4 * win.d0, o1 = 4 * ((win.d0 + 1) % T), o2 = 4 * ((win.d0 + 2) % T);
  auto load_tile = [&](int tile, uint32_t* dst) {
    const int so = tile * T * 4;
    dst[0] = __builtin_amdgcn_raw_buffer_load_b32(rs, o0, so, 0);
    dst[1] = __builtin_amdgcn_raw_buffer_load_b32(rs, o1, so, 0);
    dst[2] = __builtin_amdgcn_raw_buffer_load_b32(rs, o2, so, 0);
  };
  float acc[R][2];
#pragma unroll
  for (int r = 0; r < R; ++r) acc[r][0] = acc[r][1] = 0.f;
  uint32_t ring[PF][3];
  if (active) {
#pragma unroll
    for (int f = 0; f < PF; ++f) load_tile(f, ring[f]);
  }
  fill(0, xs);
  __syncthreads();
  for (int c0 = 0, buf = 0; c0 < span; c0 += C, buf ^= 1) {
    if (c0 + C < span) fill(c0 + C, xs + (buf ^ 1) * R * C * 8);
    const uint32_t* xc = xs + buf * R * C * 8;
    if (active) {
      const int end = min(C, span - c0);
      for (int a = 0; a < end; a += PF) {
#pragma unroll
        for (int f = 0; f < PF; ++f) {
          if (a + f >= end) break;
          const uint32_t w0 = ring[f][0], w1 = ring[f][1], w2 = ring[f][2];
          load_tile(c0 + a + f + PF, ring[f]);
          __half2 P[4];
          DecodeLane<K, kMulMode>(w0, w1, w2, win.shift, P);
#pragma unroll
          for (int r = 0; r < R; ++r) {
            const uint32_t* xr = xc + (r * C + a + f) * 8;
            uint32_t xa = xr[q], xb = xr[4 + q];
            __half2 ha = *reinterpret_cast<__half2*>(&xa);
            __half2 hb = *reinterpret_cast<__half2*>(&xb);
            acc[r][0] = Dot2(P[0], ha, acc[r][0]);
            acc[r][0] = Dot2(P[1], hb, acc[r][0]);
            acc[r][1] = Dot2(P[2], ha, acc[r][1]);
            acc[r][1] = Dot2(P[3], hb, acc[r][1]);
          }
        }
      }
    }
    __syncthreads();
  }
  if (active) {
#pragma unroll
    for (int r = 0; r < R; ++r) {
#pragma unroll
      for (int c = 0; c < 2; ++c) {
        float v = acc[r][c];
        v = v + __shfl_xor(v, 1);
        v = v + __shfl_xor(v, 2);
        acc[r][c] = v;
      }
    }
    if (q == 0) {
      const int col = 16 * b + lane / 4;
      float* out = job.partial + (size_t)item.s * R * job.O;
#pragma unroll
      for (int r = 0; r < R; ++r) {
        out[r * job.O + col] = acc[r][0];
        out[r * job.O + col + 8] = acc[r][1];
      }
    }
  }
}

// Dynamic LDS bytes for MoeGemvChunked at R launch rows.
constexpr size_t MoeChunkLds(int R) { return (size_t)2 * R * kMoeChunk * 32; }

template <int K, int N, int PF>
__device__ __forceinline__ void MoeRowsChunkedBody(int rows, const LinJob& job, const GemvItem& item, uint32_t* xs) {
  if constexpr (N >= 1) {
    if (rows == N) MoeGemvChunked<K, N, PF>(job, item, xs);
    else MoeRowsChunkedBody<K, N - 1, PF>(rows, job, item, xs);
  }
}

template <int K, int N, int PF>
__device__ __forceinline__ void MoeRowsBody(int rows, const LinJob& job, const GemvItem& item, uint32_t* xs) {
  if constexpr (N >= 1) {
    if (rows == N) MoeGemvBody<K, N, PF>(job, item, xs);
    else MoeRowsBody<K, N - 1, PF>(rows, job, item, xs);
  }
}

// Down-projection weights of every routed job are known once the unique
// kernel has run. Workgroups appended to the gate/up launch read them early
// so the down launch finds them in MALL. Those workgroups dispatch after
// all gate/up items, which fills the launch tail where DRAM has idle slots.
constexpr int kMoePrefetchChunk = 64 << 10;  // bytes per prefetch workgroup
constexpr int MoePrefetchChunks(int bytes) { return (bytes + kMoePrefetchChunk - 1) / kMoePrefetchChunk; }

__device__ __forceinline__ void MoePrefetchBlock(const LinJob* __restrict__ dn, int p, int per_job,
                                                 uint32_t* __restrict__ sink) {
  const LinJob& d = dn[p / per_job];
  const int K = __builtin_amdgcn_readfirstlane(d.K);
  if (K == 0) return;
  const uint32_t bytes = (uint32_t)d.I * d.O / 256 * 32 * K;
  const uint32_t off = (uint32_t)(p % per_job) * kMoePrefetchChunk;
  if (off >= bytes) return;
  const __amdgpu_buffer_rsrc_t rs = StreamRsrc(d.tiles, bytes);
  uint32_t acc = 0;
#pragma unroll
  for (int i = 0; i < kMoePrefetchChunk / (256 * 16); ++i) {
    const uint32_t o = off + (uint32_t)(i * 256 + threadIdx.x) * 16;
    const uint4 v = __builtin_bit_cast(uint4, __builtin_amdgcn_raw_buffer_load_b128(rs, o, 0, 0));
    acc ^= v.x ^ v.w;
  }
  if (acc == 0x9e3779b9u) sink[0] = acc;  // keeps the loads; practically never stores
}

// Deduplicates the routed experts of all rows: unique expert u gets one
// gate/up/down job applied to every row (weights stream once per forward).
// map[r][k] = u keeps each row's own selection order for the combine, so
// results are bitwise those of a one-row forward. Unused jobs get K = 0.
// gu: [U + 1][2], dn: [U + 1] with U = R * kTopK; index U is the shared expert.
__global__ void __launch_bounds__(512) MoeUniqueKernel(const int* __restrict__ sel, int R,
                                                       const ExpertPtrs* __restrict__ experts,
                                                       LinJob* gu, LinJob* dn,
                                                       int* __restrict__ map) {
  __shared__ int seen[kExperts];
  __shared__ int list[kMaxRows * kTopK];
  __shared__ int n;
  const int t = threadIdx.x;
  seen[t] = -1;
  __syncthreads();
  if (t == 0) {
    int c = 0;
    for (int i = 0; i < R * kTopK; ++i) {
      const int e = sel[i];
      if (seen[e] < 0) {
        seen[e] = c;
        list[c++] = e;
      }
      map[i] = seen[e];
    }
    n = c;
  }
  __syncthreads();
  const int U = R * kTopK;
  for (int u = t; u < U; u += blockDim.x) {
    LinJob* g = gu + (size_t)u * 2;
    LinJob* d = dn + u;
    if (u < n) {
      const ExpertPtrs& ep = experts[list[u]];
      for (int k = 0; k < 2; ++k) {
        g[k].tiles = ep.tiles[k];
        g[k].suh = ep.suh[k];
        g[k].svh = ep.svh[k];
        g[k].K = ep.K[k];
      }
      d->tiles = ep.tiles[2];
      d->suh = ep.suh[2];
      d->svh = ep.svh[2];
      d->K = ep.K[2];
    } else {
      g[0].K = g[1].K = d->K = 0;
    }
  }
}

// act = SiLU(gate) * up per (expert job, row), then the down projection's
// input transform. One wave per (job, row, 128-block); 5 blocks of 640.
__global__ void __launch_bounds__(256) MoeMidKernel(const LinJob* __restrict__ gu,
                                                    const LinJob* __restrict__ dn, int njobs,
                                                    int R) {
  const int wv = blockIdx.x * 8 + (threadIdx.x >> 5);
  if (wv >= njobs * R * 5) return;
  const int jb = wv / (R * 5), r = (wv / 5) % R, blk = wv % 5, lane = threadIdx.x & 31;
  const LinJob& d = dn[jb];
  if (d.K == 0) return;
  float gv[4], uv[4];
  LinOutBlock(gu[jb * 2], r, blk, gv);
  LinOutBlock(gu[jb * 2 + 1], r, blk, uv);
  const int c0 = blk * 128 + lane * 4;
  float v[4];
#pragma unroll
  for (int i = 0; i < 4; ++i)
    v[i] = RoundH(RoundH(Silu(gv[i]) * uv[i]) * __half2float(d.suh[c0 + i]));
  Wht128(v);
#pragma unroll
  for (int i = 0; i < 4; ++i) d.xh[(size_t)r * kMoeInter + c0 + i] = __float2half(v[i] * kHadNorm);
}

// y = sum_k wts[k] * down_k (the row's selection order, then shared), then the
// hyper-connection write-back H[s] += post[s] * y. One wave per (row, block).
// StoreY (ablation.h): store y to H as [R][kHidden] instead; post is unused.
template <bool StoreY = false>
__global__ void __launch_bounds__(256) MoeCombineKernel(const LinJob* __restrict__ dn,
                                                        const int* __restrict__ map,
                                                        const float* __restrict__ wts,
                                                        const float* __restrict__ post,
                                                        float* __restrict__ H, int R) {
  const int wv = blockIdx.x * 8 + (threadIdx.x >> 5);
  if (wv >= R * 20) return;
  const int r = wv / 20, blk = wv % 20, lane = threadIdx.x & 31;
  float y[4] = {0.f, 0.f, 0.f, 0.f};
  for (int k = 0; k < kSlots; ++k) {
    const int u = k < kTopK ? map[r * kTopK + k] : R * kTopK;
    float v[4];
    LinOutBlock(dn[u], r, blk, v);
    const float wk = wts[r * kSlots + k];
#pragma unroll
    for (int i = 0; i < 4; ++i) y[i] += wk * v[i];
  }
  const int c0 = blk * 128 + lane * 4;
  if constexpr (StoreY) {
#pragma unroll
    for (int i = 0; i < 4; ++i) H[(size_t)r * kHidden + c0 + i] = y[i];
    return;
  }
#pragma unroll
  for (int s = 0; s < kStreams; ++s) {
    const float gs = post[r * kStreams + s];
    float* h = H + ((size_t)r * kStreams + s) * kHidden + c0;
#pragma unroll
    for (int i = 0; i < 4; ++i) h[i] += gs * y[i];
  }
}

// Input transform of the gate/up jobs for compact rows only.
__global__ void __launch_bounds__(256) MoeRowsInKernel(const LinJob* __restrict__ jobs,
                                                       const XfItem* __restrict__ items, int n,
                                                       const int* __restrict__ rows) {
  const int w = blockIdx.x * 8 + (threadIdx.x >> 5);
  if (w >= n) return;
  const int lane = threadIdx.x & 31;
  const XfItem it = items[w];
  const LinJob& j = jobs[it.job];
  if (j.K == 0 || it.r >= j.R) return;
  const int c0 = it.blk * 128 + lane * 4;
  const float* s = j.src + (size_t)rows[(it.job >> 1) * kMaxRows + it.r] * j.lds + c0;
  float v[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) v[i] = RoundH(RoundH(s[i]) * __half2float(j.suh[c0 + i]));
  Wht128(v);
  __half* d = j.xh + (size_t)it.r * j.I + c0;
#pragma unroll
  for (int i = 0; i < 4; ++i) d[i] = __float2half(v[i] * kHadNorm);
}

// MoeRowsGemvKernel with chunked activation staging (launch with MoeChunkLds(R)).
template <int R, int PF = 8>
__global__ void __launch_bounds__(256) MoeRowsChunkedKernel(const LinJob* __restrict__ jobs,
                                                            const GemvItem* __restrict__ items, int n) {
  extern __shared__ uint32_t xs[];
  for (int it = blockIdx.x; it < n; it += gridDim.x) {
    const GemvItem item = items[it];
    const LinJob& job = jobs[item.job];
    const int K = __builtin_amdgcn_readfirstlane(job.K);
    const int rows = __builtin_amdgcn_readfirstlane(job.R);
    if (K == 3) MoeRowsChunkedBody<3, R, PF>(rows, job, item, xs);
    else if (K == 5 && rows == R) MoeGemvChunked<5, R, PF>(job, item, xs);
  }
}

// One launch over all expert jobs; each routed job runs only its own row
// count. Other widths (a quality-tier shared expert) take the generic body.
template <int R, int PF = 8>
__global__ void __launch_bounds__(256) MoeRowsGemvKernel(const LinJob* __restrict__ jobs,
                                                         const GemvItem* __restrict__ items, int n) {
  extern __shared__ uint32_t xs[];
  for (int it = blockIdx.x; it < n; it += gridDim.x) {
    const GemvItem item = items[it];
    const LinJob& job = jobs[item.job];
    const int K = __builtin_amdgcn_readfirstlane(job.K);
    const int rows = __builtin_amdgcn_readfirstlane(job.R);
    if (K == 3) MoeRowsBody<3, R, PF>(rows, job, item, xs);
    else if (K == 5 && rows == R) MoeGemvBody<5, R, PF>(job, item, xs);
    else DispatchLinGemv<R>(K, job, item, xs);
  }
}

template <int R, int PF = 8>
__global__ void __launch_bounds__(256) MoeRowsGemvPrefetchKernel(const LinJob* __restrict__ jobs,
                                                                 const GemvItem* __restrict__ items, int n,
                                                                 const LinJob* __restrict__ dn, int per_job,
                                                                 uint32_t* __restrict__ sink) {
  extern __shared__ uint32_t xs[];
  if ((int)blockIdx.x >= n) {
    MoePrefetchBlock(dn, blockIdx.x - n, per_job, sink);
    return;
  }
  const GemvItem item = items[blockIdx.x];
  const LinJob& job = jobs[item.job];
  const int K = __builtin_amdgcn_readfirstlane(job.K);
  const int rows = __builtin_amdgcn_readfirstlane(job.R);
  if (K == 3) MoeRowsBody<3, R, PF>(rows, job, item, xs);
  else if (K == 5 && rows == R) MoeGemvBody<5, R, PF>(job, item, xs);
  else DispatchLinGemv<R>(K, job, item, xs);
}

// SiLU(gate) * up and the down input transform for compact rows.
__global__ void __launch_bounds__(256) MoeRowsMidKernel(const LinJob* __restrict__ gu,
                                                        const LinJob* __restrict__ dn, int njobs, int R) {
  const int wv = blockIdx.x * 8 + (threadIdx.x >> 5);
  if (wv >= njobs * R * 5) return;
  const int jb = wv / (R * 5), r = (wv / 5) % R, blk = wv % 5, lane = threadIdx.x & 31;
  const LinJob& d = dn[jb];
  if (d.K == 0 || r >= d.R) return;
  float gv[4], uv[4];
  LinOutBlock(gu[jb * 2], r, blk, gv);
  LinOutBlock(gu[jb * 2 + 1], r, blk, uv);
  const int c0 = blk * 128 + lane * 4;
  float v[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) v[i] = RoundH(RoundH(Silu(gv[i]) * uv[i]) * __half2float(d.suh[c0 + i]));
  Wht128(v);
#pragma unroll
  for (int i = 0; i < 4; ++i) d.xh[(size_t)r * kMoeInter + c0 + i] = __float2half(v[i] * kHadNorm);
}

}  // namespace crucible
