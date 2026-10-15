// Stage K slices once, then share each B tile across four row waves.
#pragma once
#include "../../engine/exl3/kernels/exl3_gemm.h"

namespace crucible {
__device__ __forceinline__ void GemmTileWrite(const __half2 p[4], uint32_t* dst, int lane) {
  const int q = lane & 3, c = lane / 4;
  const auto* v = reinterpret_cast<const uint32_t*>(p);
  dst[c * kGemmBStride + q] = v[0];
  dst[c * kGemmBStride + q + 4] = v[1];
  dst[(c + 8) * kGemmBStride + q] = v[2];
  dst[(c + 8) * kGemmBStride + q + 4] = v[3];
}
__device__ __forceinline__ GemmH16 GemmTileRead(const uint32_t* src, int lane) {
  GemmH16 b;
  auto* v = reinterpret_cast<uint32_t*>(&b);
#pragma unroll
  for (int z = 0; z < 8; ++z) v[z] = src[(lane & 15) * kGemmBStride + z];
  return b;
}

template <int K, int RM, int CHUNK, bool FULL_M>
__device__ __forceinline__ void GemmTiledBody(const Exl3GemmJob& j,
    const Exl3GemmItem& it, uint32_t* weights) {
  const int lane = threadIdx.x & 31;
  const int wave = __builtin_amdgcn_readfirstlane(threadIdx.x >> 5);
  const int b = it.b0;
  if (b * 16 >= j.O) return;
  const int row0 = it.m0 + wave * RM * 16;
  const int A = j.I / 16;
  const int begin = A * it.s / j.S, end = A * (it.s + 1) / j.S;
  constexpr int T = 8 * K;
  const LaneWindow<K> win(lane);
  const auto wr = StreamRsrc(j.tiles + ((size_t)b * A + begin) * T, (end - begin) * T * 4);
  const auto xr = StreamRsrc(reinterpret_cast<const uint32_t*>(j.x), (uint32_t)j.Xrows * j.I * 2);
  uint32_t xo[RM];
#pragma unroll
  for (int r = 0; r < RM; ++r) {
    const int row = row0 + r * 16 + (lane & 15);
    const int source = FULL_M || row < j.M ? (j.row_ids ? j.row_ids[row] : row) : j.Xrows;
    xo[r] = (uint32_t)source * j.I * 2;
  }
  GemmF8 acc[RM] = {};
  for (int a0 = begin; a0 < end; a0 += CHUNK) {
    // Each wave owns different K tiles in the producer phase. The next phase
    // shares these tiles across all row waves, without four separate decodes.
#pragma unroll
    for (int t = 0; t < CHUNK / 4; ++t) {
      const int f = wave + 4 * t;
      if (a0 + f >= end) break;
      const int so = (a0 + f - begin) * T * 4;
      const uint32_t w0 = __builtin_amdgcn_raw_buffer_load_b32(wr, win.d0 * 4 + so, 0, 0);
      const uint32_t w1 = __builtin_amdgcn_raw_buffer_load_b32(wr, ((win.d0 + 1) % T) * 4 + so, 0, 0);
      uint32_t w2 = 0;
      if constexpr (K >= 5)
        w2 = __builtin_amdgcn_raw_buffer_load_b32(wr, ((win.d0 + 2) % T) * 4 + so, 0, 0);
      __half2 p[4];
      DecodeLane<K, kMulTrue16>(w0, w1, w2, win.shift, p);
      GemmTileWrite(p, weights + f * kGemmWaveWords, lane);
    }
    GemmGroupSync();
    if (FULL_M || row0 < j.M) {
#pragma unroll
      for (int f = 0; f < CHUNK; ++f) {
        if (a0 + f >= end) break;
        const GemmH16 bfrag = GemmTileRead(weights + f * kGemmWaveWords, lane);
#pragma unroll
        for (int r = 0; r < RM; ++r) {
          if constexpr (!FULL_M) { if (row0 + r * 16 >= j.M) break; }
          const GemmH16 afrag = GemmAFragment<false>(xr, xo[r], a0 + f, nullptr, 0);
          acc[r] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(afrag, bfrag, acc[r]);
        }
      }
    }
    // Every row wave must finish before producers reuse the same LDS slice.
    GemmGroupSync();
  }
#pragma unroll
  for (int r = 0; r < RM; ++r)
#pragma unroll
    for (int z = 0; z < 8; ++z) {
      const int row = row0 + r * 16 + 2 * z + lane / 16;
      if (FULL_M || row < j.M)
        j.partial[((size_t)it.s * j.M + row) * j.O + b * 16 + (lane & 15)] = acc[r][z];
    }
}

// Items advance m0 by 64*RM and b0 by one 16-column tile. The job ABI and
// partial layout stay unchanged. CHUNK amortizes the two group barriers.
template <int RM = 2, int CHUNK = 8, bool FULL_M = false>
__global__ void __launch_bounds__(128) Exl3GemmTiledKernel(const Exl3GemmJob* jobs,
    const Exl3GemmItem* items, int n) {
  static_assert(RM == 1 || RM == 2 || RM == 4 || RM == 8 || RM == 16);
  static_assert(CHUNK == 4 || CHUNK == 8 || CHUNK == 16);
  __shared__ uint32_t weights[CHUNK * kGemmWaveWords];
  if (blockIdx.x >= n) return;
  const Exl3GemmItem it = items[blockIdx.x];
  const Exl3GemmJob j = jobs[it.job];
  if (j.K == 2) GemmTiledBody<2, RM, CHUNK, FULL_M>(j, it, weights);
  else if (j.K == 3) GemmTiledBody<3, RM, CHUNK, FULL_M>(j, it, weights);
  else if (j.K == 4) GemmTiledBody<4, RM, CHUNK, FULL_M>(j, it, weights);
  else if (j.K == 5) GemmTiledBody<5, RM, CHUNK, FULL_M>(j, it, weights);
  else if (j.K == 6) GemmTiledBody<6, RM, CHUNK, FULL_M>(j, it, weights);
}
}  // namespace crucible
