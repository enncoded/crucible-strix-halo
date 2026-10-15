// Bench-only exact wider projections. Keep each row's contraction order unchanged.
#pragma once
#include "../engine/exl3/exl3.h"
#include "exact_projection_geometry.h"

namespace crucible::wide_probe {
static_assert(kPrefetch==8,"rolling geometry requires PF8");
template <int N, int K, int CACHE, bool Front, bool Complete>
__device__ __forceinline__ void RollingBody(const LinJob& job, const GemvItem& item, uint32_t* xs) {
  constexpr int PF = kPrefetch, T = 8 * K;
  const int lane = threadIdx.x & 31, wave = threadIdx.x >> 5, q = lane & 3;
  const int A = job.I >> 4, span = A / job.S, a0 = item.s * span;
  const LaneWindow<K> win(lane);
  const int b = item.b0 + wave;
  const int items=job.S*(job.O>>7),idx=item.s*(job.O>>7)+(b>>3);
  const auto rs = Front ? StreamRsrc(job.tiles+((size_t)idx*32+(b&7)*4)*T,
                                    (uint32_t)((size_t)(span/4-1)*items*32+4)*T*4)
                        : StreamRsrc(job.tiles+((size_t)b*A+a0)*T,(uint32_t)span*T*4);
  const int offsets[3] = {4 * win.d0, 4 * ((win.d0 + 1) % T), 4 * ((win.d0 + 2) % T)};
  auto load = [&](int tile, uint32_t* dst) {
#pragma unroll
    for (int i = 0; i < 3; ++i) {
      const int so=Front?(tile<span?((tile>>2)*items*32+(tile&3))*T*4:0x40000000):tile*T*4;
      dst[i]=__builtin_amdgcn_raw_buffer_load_b32(rs,offsets[i],so,CACHE);
    }
  };
  float acc[N][2];
#pragma unroll
  for (int r = 0; r < N; ++r) acc[r][0] = acc[r][1] = 0.f;
  uint32_t ring[PF][3];
  if (Complete || wave < item.nb) {
#pragma unroll
    for (int f = 0; f < PF; ++f) load(f, ring[f]);
  }
  const auto* xg = reinterpret_cast<const uint32_t*>(job.xh);
  for (int a = 0; a < span; a += PF) {
    for (int i = threadIdx.x; i < N * PF * 8; i += 256) {
      const int r = i / (PF * 8), f = (i / 8) % PF, word = i % 8;
      xs[i] = Complete || a + f < span ? xg[((size_t)r * A + a0 + a + f) * 8 + word] : 0;
    }
    __syncthreads();
    if (Complete || wave < item.nb) {
#pragma unroll
      for (int f = 0; f < PF; ++f) {
        if (!Complete && a + f >= span) break;
        const uint32_t w0 = ring[f][0], w1 = ring[f][1], w2 = ring[f][2];
        load(a + f + PF, ring[f]);
        __half2 P[4];
        DecodeLane<K, kMulMode>(w0, w1, w2, win.shift, P);
#pragma unroll
        for (int r = 0; r < N; ++r) {
          // Match DenseRows: keep each row's LDS reads inside its contraction step.
          asm volatile("" ::: "memory");
          const uint32_t* xr = xs + RollingSlot(r,f,0);
          uint32_t xa = xr[q], xb = xr[4 + q];
          const __half2 ha = *reinterpret_cast<const __half2*>(&xa);
          const __half2 hb = *reinterpret_cast<const __half2*>(&xb);
          acc[r][0] = Dot2(P[0], ha, acc[r][0]);
          acc[r][0] = Dot2(P[1], hb, acc[r][0]);
          acc[r][1] = Dot2(P[2], ha, acc[r][1]);
          acc[r][1] = Dot2(P[3], hb, acc[r][1]);
        }
      }
    }
    // The next activation chunk must not overwrite data still used by another wave.
    __syncthreads();
  }
  if (Complete || wave < item.nb) {
#pragma unroll
    for (int r = 0; r < N; ++r) {
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
      float* out = job.partial + (size_t)item.s * N * job.O;
#pragma unroll
      for (int r = 0; r < N; ++r) {
        out[r * job.O + col] = acc[r][0];
        out[r * job.O + col + 8] = acc[r][1];
      }
    }
  }
}

template <int N, bool Front, bool Complete, size_t Index = 0>
__device__ __forceinline__ void DispatchRolling(const LinJob& job, const GemvItem& item, uint32_t* xs) {
  if (job.K == kSupportedDecodeK[Index]) {
    if (job.cache_policy == 4) RollingBody<N, kSupportedDecodeK[Index], 4, Front, Complete>(job, item, xs);
    else RollingBody<N, kSupportedDecodeK[Index], 0, Front, Complete>(job, item, xs);
  } else if constexpr (Index + 1 < kSupportedDecodeK.size()) DispatchRolling<N, Front, Complete, Index + 1>(job, item, xs);
}

template <int N, bool Front, bool Complete=false>
__global__ void __launch_bounds__(256) RollingKernel(const LinJob* jobs, const GemvItem* items) {
  extern __shared__ uint32_t xs[];
  const GemvItem item = items[blockIdx.x];
  DispatchRolling<N,Front,Complete>(jobs[item.job], item, xs);
}

template <int N, bool Front>
__global__ void __launch_bounds__(256) FullKernel(const LinJob* jobs, const GemvItem* items) {
  extern __shared__ uint32_t xs[];
  const GemvItem item = items[blockIdx.x];
  const LinJob& job = jobs[item.job];
  DispatchLinGemv<N,0,6,Front>(job.K, job, item, xs);
}

template <int N, bool Front, bool Interleave=false>
__global__ void __launch_bounds__(256) StripeKernel(const LinJob* jobs, const GemvItem* items) {
  extern __shared__ uint32_t xs[];
  const GemvItem item = items[Interleave?InterleavedItem(blockIdx.x,N):blockIdx.x];
  const LinJob& base = jobs[item.job];
  const int row = Interleave?InterleavedRow(blockIdx.x,N):blockIdx.y*8;
  LinJob view = base;
  view.R = 8;
  view.xh += (size_t)row * base.I;
  // LinGemvBody uses template R for the split stride. Keep the global N-row layout.
  view.partial += StripePartialBase(item.s,N,row,base.O);
  DispatchLinGemv<8,0,6,Front>(view.K, view, item, xs);
}
}  // namespace crucible::wide_probe
