// Register blocks share exact fp32 indexer operands across rows and keys.
#pragma once
#include "../qsa_long.h"

namespace crucible::prefill {

// Each dot keeps dimensions 0..127 in order. Reuse changes only which
// independent dots execute between two FMAs of the same dot.
// RowsFast puts row tiles on grid x, so resident groups share one key tile
// through L2 instead of each streaming the whole pool from the MALL.
template<int MR, int NK, int PF, bool BufferLoads = false, bool RowsFast = false>
__global__ void __launch_bounds__(128) QsaPrefillExactScoreKernel(QsaArgs a) {
  constexpr int Threads = 128;
  static_assert(MR >= 1 && MR <= 8 && NK >= 1 && NK <= 8 && 128%PF == 0);
  const int row_tile = RowsFast ? blockIdx.x : blockIdx.y, key_tile = RowsFast ? blockIdx.y : blockIdx.x;
  const int r0 = row_tile*MR, b0 = key_tile*(Threads*NK)+threadIdx.x;
  const int total = (a.pos0+a.R)/4, row_total = (a.pos0+min(r0+MR,a.R))/4;
  if (row_total <= kSelBlocks || key_tile*(Threads*NK) >= row_total) return;
  __shared__ float iq[MR][512];
  for (int i = threadIdx.x; i < MR*512; i += Threads) {
    const int r = r0+i/512;
    iq[i/512][i%512] = r < a.R ? a.idx[(size_t)r*kQsaIdxWidth+i%512] : 0.f;
  }
  __syncthreads();
  const int lane = threadIdx.x&31;
  const int wave = __builtin_amdgcn_readfirstlane(threadIdx.x/32);
  const auto pool = StreamRsrc(reinterpret_cast<const uint32_t*>(a.c.ik_pool),
      (uint32_t)QsaPoolFloats(a.pos0+a.R)*sizeof(float));
  // The OOB check may skip soffset, so dead tiles must stay inside the pool.
  const int last_tile = (int)(QsaPoolFloats(a.pos0+a.R)/4096)-1;
  float dots[MR][4][NK] = {};
#pragma clang loop unroll(disable)
  for (int dim = 0; dim < 128; dim += PF) {
    float keys[PF][NK];
#pragma unroll
    for (int j = 0; j < PF; ++j)
#pragma unroll
      for (int n = 0; n < NK; ++n) {
        const int b = b0+n*Threads;
        if constexpr (BufferLoads) {
          const int tile = min(last_tile,key_tile*(Threads*NK/32)+wave+n*(Threads/32));
          const int scalar = tile*4096*sizeof(float)+(dim+j)*32*sizeof(float);
          const uint32_t bits = __builtin_amdgcn_raw_buffer_load_b32(pool,lane*sizeof(float),scalar,0);
          keys[j][n] = __uint_as_float(bits);
        } else keys[j][n] = b < total ? a.c.ik_pool[QsaPoolOffset(b,dim+j)] : 0.f;
      }
#pragma unroll
    for (int j = 0; j < PF; ++j)
#pragma unroll
      for (int r = 0; r < MR; ++r)
#pragma unroll
        for (int h = 0; h < 4; ++h) {
          const float q = iq[r][h*128+dim+j];
#pragma unroll
          for (int n = 0; n < NK; ++n) dots[r][h][n] = fmaf(q,keys[j][n],dots[r][h][n]);
        }
  }
#pragma unroll
  for (int r = 0; r < MR; ++r) {
    const int row = r0+r, nb = (a.pos0+row+1)/4;
#pragma unroll
    for (int n = 0; n < NK; ++n) {
      const int b = b0+n*Threads;
      if (row < a.R && nb > kSelBlocks && b < nb) {
#if CRUCIBLE_DEBUG_NONFINITE & 1
        for (int h = 0; h < 4; ++h) if (Nonfinite(dots[r][h][n])) FlagNonfinite(kNfQsaScore);
#endif
        float s = fmaxf(dots[r][0][n],0.f);
        s += fmaxf(dots[r][1][n],0.f); s += fmaxf(dots[r][2][n],0.f);
        s += fmaxf(dots[r][3][n],0.f);
        a.scores[(size_t)row*a.sstride+b] = s*0.08838834764831845f;
      }
    }
  }
}

template<int MR = 4, int NK = 4, int PF = 8, bool BufferLoads = false, bool RowsFast = false>
inline void LaunchQsaPrefillScores(QsaArgs a, hipStream_t stream = nullptr) {
  const int nb = (a.pos0+a.R)/4;
  const int keys = (nb+128*NK-1)/(128*NK), rows = (a.R+MR-1)/MR;
  if (nb > kSelBlocks)
    QsaPrefillExactScoreKernel<MR,NK,PF,BufferLoads,RowsFast>
        <<<RowsFast ? dim3(rows,keys) : dim3(keys,rows),128,0,stream>>>(a);
}

}  // namespace crucible::prefill
