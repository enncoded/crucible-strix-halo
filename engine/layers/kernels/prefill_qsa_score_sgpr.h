// Exact prefill indexer scores with wave-uniform query values in SGPRs, no LDS.
#pragma once
#include "prefill_qsa_score.h"

namespace crucible::prefill {

// Query tiles [R/MR][128 dims][MR rows][4 heads], so one dim of a row tile is
// 4*MR contiguous floats: MR/4 scalar 512-bit loads.
template<int MR>
__global__ void QsaQueryTileKernel(QsaArgs a, float* __restrict__ qt) {
  const int tile = blockIdx.x, t = threadIdx.x;
  for (int i = t; i < 512*MR; i += 256) {
    const int dim = i/(4*MR), r = i/4%MR, h = i%4, row = tile*MR+r;
    qt[(size_t)tile*512*MR+i] = row < a.R ? a.idx[(size_t)row*kQsaIdxWidth+h*128+dim] : 0.f;
  }
}

// Each dot keeps dimensions 0..127 in order, as in QsaPrefillExactScoreKernel.
template<int MR, int NK>
__global__ void __launch_bounds__(128) QsaPrefillSgprScoreKernel(QsaArgs a, const float* __restrict__ qt) {
  constexpr int Threads = 128;
  static_assert(MR == 4 || MR == 8 || MR == 16);
  const int row_tile = blockIdx.x, key_tile = blockIdx.y;
  const int r0 = row_tile*MR, b0 = key_tile*(Threads*NK)+threadIdx.x;
  const int row_total = (a.pos0+min(r0+MR,a.R))/4;
  if (row_total <= kSelBlocks || key_tile*(Threads*NK) >= row_total) return;
  const int lane = threadIdx.x&31;
  const int wave = __builtin_amdgcn_readfirstlane(threadIdx.x/32);
  const auto pool = StreamRsrc(reinterpret_cast<const uint32_t*>(a.c.ik_pool),
      (uint32_t)QsaPoolFloats(a.pos0+a.R)*sizeof(float));
  const int last_tile = (int)(QsaPoolFloats(a.pos0+a.R)/4096)-1;
  const float* q = qt+(size_t)row_tile*512*MR;
  float dots[MR][4][NK] = {};
#pragma clang loop unroll(disable)
  for (int dim = 0; dim < 128; ++dim) {
    float keys[NK], qv[4*MR];
#pragma unroll
    for (int n = 0; n < NK; ++n) {
      const int tile = min(last_tile,key_tile*(Threads*NK/32)+wave+n*(Threads/32));
      keys[n] = __uint_as_float(__builtin_amdgcn_raw_buffer_load_b32(pool,lane*sizeof(float),
          tile*4096*(int)sizeof(float)+dim*32*(int)sizeof(float),0));
    }
#pragma unroll
    for (int i = 0; i < 4*MR; ++i) qv[i] = q[dim*4*MR+i];
#pragma unroll
    for (int r = 0; r < MR; ++r)
#pragma unroll
      for (int h = 0; h < 4; ++h)
#pragma unroll
        for (int n = 0; n < NK; ++n) dots[r][h][n] = fmaf(qv[r*4+h],keys[n],dots[r][h][n]);
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

constexpr size_t QsaQueryTileFloats(int R, int MR = 8) { return (size_t)(R+MR-1)/MR*512*MR; }

// qt needs QsaQueryTileFloats(R, MR) floats.
template<int MR = 8, int NK = 4>
inline void LaunchQsaPrefillSgprScores(QsaArgs a, float* qt, hipStream_t stream = nullptr) {
  const int nb = (a.pos0+a.R)/4;
  if (nb <= kSelBlocks) return;
  const int rows = (a.R+MR-1)/MR, keys = (nb+128*NK-1)/(128*NK);
  QsaQueryTileKernel<MR><<<rows,256,0,stream>>>(a,qt);
  QsaPrefillSgprScoreKernel<MR,NK><<<dim3(rows,keys),128,0,stream>>>(a,qt);
}

}  // namespace crucible::prefill
