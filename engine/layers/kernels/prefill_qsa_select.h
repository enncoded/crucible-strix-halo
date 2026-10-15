// Fused exact scores and tile-local top-512. Only candidates reach DRAM.
#pragma once
#include "prefill_qsa_score.h"
#include "prefill_qsa_score_sgpr.h"
#include <cstdlib>
#include <string>

namespace crucible::prefill {

// The two-pass contract needs at most 16*512 candidates per row. Keeping
// 4096 keys per tile preserves the existing scratch allocation bound.
template<int MR = 2, int PF = 8, bool CaptureScores = false, bool BufferLoads = false>
__global__ void __launch_bounds__(1024) QsaPrefillScoreTopKernel(QsaArgs a, float* scores, int* ids) {
  static_assert(MR == 1 || MR == 2);
  constexpr int NK = 4, Threads = 1024;
  __shared__ float iq[MR][512], cached[MR][kQsaTopTile];
  __shared__ unsigned hist[16][256], prefix_gt[32], prefix_eq[32], threshold, remaining;
  const int r0 = blockIdx.y*MR, base = blockIdx.x*kQsaTopTile, t = threadIdx.x;
  const int total = (a.pos0+a.R)/4;
  if (total <= kSelBlocks) return;
  for (int i = t; i < MR*512; i += Threads) {
    const int r = r0+i/512;
    iq[i/512][i%512] = r < a.R ? a.idx[(size_t)r*kQsaIdxWidth+i%512] : 0.f;
  }
  __syncthreads();
  const int lane = t&31, wave = __builtin_amdgcn_readfirstlane(t/32);
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
        const int b = base+t+n*Threads;
        if constexpr (BufferLoads) {
          const int tile = min(last_tile,(int)blockIdx.x*(kQsaTopTile/32)+wave+n*(Threads/32));
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
  for (int r = 0; r < MR; ++r)
#pragma unroll
    for (int n = 0; n < NK; ++n) {
      const int local = t+n*Threads;
#if CRUCIBLE_DEBUG_NONFINITE & 1
      if (r0+r < a.R && (a.pos0+r0+r+1)/4 > kSelBlocks && base+local < (a.pos0+r0+r+1)/4)
        for (int h = 0; h < 4; ++h) if (Nonfinite(dots[r][h][n])) FlagNonfinite(kNfQsaScore);
#endif
      float s = fmaxf(dots[r][0][n],0.f);
      s += fmaxf(dots[r][1][n],0.f); s += fmaxf(dots[r][2][n],0.f);
      s += fmaxf(dots[r][3][n],0.f);
      s *= 0.08838834764831845f;
      cached[r][local] = s;
      if constexpr (CaptureScores) {
        const int row = r0+r, nb = (a.pos0+row+1)/4;
        if (row < a.R && nb > kSelBlocks && base+local < nb)
          a.scores[(size_t)row*a.sstride+base+local] = s;
      }
    }
  __syncthreads();
#pragma unroll
  for (int r = 0; r < MR; ++r) {
    const int row = r0+r, nb = (a.pos0+row+1)/4;
    if (row >= a.R || nb <= kSelBlocks) continue;
    const int count = min(kQsaTopTile,max(0,nb-base));
    const size_t offset = (size_t)row*kQsaCandidates+blockIdx.x*kSelBlocks;
    int* selected = ids+offset;
    float* selected_scores = scores+offset;
    if (count <= kSelBlocks) {
      if (t < count) { selected[t] = base+t; selected_scores[t] = cached[r][t]; }
      continue;
    }
    if (t == 0) { threshold = 0; remaining = kSelBlocks; }
    __syncthreads();
    unsigned mask = 0;
    for (int shift = 24; shift >= 0; shift -= 8) {
      QsaRadixPass<16>(cached[r],count,shift,mask,hist,prefix_gt,threshold,remaining);
      mask |= 255u << shift;
    }
    QsaEmitSelected(cached[r],count,threshold,selected,prefix_gt,prefix_eq,nullptr,base,selected_scores);
    __syncthreads();
  }
}

template<int MR = 2, int PF = 8, bool CaptureScores = false, bool BufferLoads = false>
inline void LaunchQsaPrefillScoreTop(QsaArgs a, float* scratch, hipStream_t stream = nullptr) {
  const int nb = (a.pos0+a.R)/4;
  if (nb <= kQsaCandidates || nb > kQsaTopMaxBlocks || !scratch) {
    LaunchQsaPrefillScores(a,stream); LaunchQsaTop(a,scratch,stream); return;
  }
  int* ids = reinterpret_cast<int*>(scratch+(size_t)a.R*kQsaCandidates);
  QsaPrefillScoreTopKernel<MR,PF,CaptureScores,BufferLoads><<<dim3((nb+kQsaTopTile-1)/kQsaTopTile,(a.R+MR-1)/MR),1024,0,stream>>>(a,scratch,ids);
  QsaTopPassKernel<2><<<a.R,1024,0,stream>>>(a,scratch,ids);
}

#ifndef CRUCIBLE_PF_QSA_SCORE_MODE
#define CRUCIBLE_PF_QSA_SCORE_MODE 1
#endif
static_assert(CRUCIBLE_PF_QSA_SCORE_MODE >= 0 && CRUCIBLE_PF_QSA_SCORE_MODE <= 2);

// The SGPR MR8/NK4 kernel reads tiled queries with scalar loads, so its FMA
// loop has no LDS traffic. qtiles needs QsaQueryTileFloats(R) floats; without
// it, MR4/NK8 reuses each LDS query value over eight keys, and the row-fast
// grid keeps the current key tile in L2 for all resident row tiles.
inline void LaunchQsaBlockedPrefillSelect(QsaArgs a, float* scratch, float* qtiles, hipStream_t stream) {
  if (a.R <= 8) { LaunchQsaSelect(a,scratch,stream); return; }
  if (qtiles) LaunchQsaPrefillSgprScores<8,4>(a,qtiles,stream);
  else LaunchQsaPrefillScores<4,8,2,true,true>(a,stream);
  LaunchQsaTop(a,scratch,stream);
}

// Mode 0 keeps the original selector, mode 1 uses exact blocked scores, and
// mode 2 adds the slower tile-local fused selection prototype.
inline void LaunchQsaPrefillSelect(QsaArgs a, float* scratch = nullptr, float* qtiles = nullptr,
    hipStream_t stream = nullptr) {
  if constexpr (CRUCIBLE_PF_QSA_SCORE_MODE == 0) LaunchQsaSelect(a,scratch,stream);
  else if constexpr (CRUCIBLE_PF_QSA_SCORE_MODE == 1) LaunchQsaBlockedPrefillSelect(a,scratch,qtiles,stream);
  else if ((a.pos0+a.R)/4 <= kQsaCandidates) LaunchQsaBlockedPrefillSelect(a,scratch,qtiles,stream);
  else LaunchQsaPrefillScoreTop<2,4,false,true>(a,scratch,stream);
}

}  // namespace crucible::prefill
