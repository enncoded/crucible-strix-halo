// QSA decode kernels that only benches launch: the old select and attention
// baselines. The engine runs QSA decode through engine/layers/qsa_long.h.
#pragma once
#include "../../engine/layers/ops.h"

namespace crucible {

// Benchmark baseline only: this kernel expects the OLD row-major pooled-key
// layout, not QsaPoolOffset. The engine uses LaunchQsaSelect in qsa_long.h.
// Top-512 by score desc / block id asc, or dense for <=512 complete blocks.
__global__ void __launch_bounds__(1024) QsaSelectKernel(QsaArgs a) {
  __shared__ float iq[kIdxHeads * kIdxDim];
  __shared__ unsigned cnt;
  __shared__ int nsel;
  const int r = blockIdx.x, p = a.pos0 + r, nb = (p + 1) / 4, j = threadIdx.x;
  if (nb <= kSelBlocks) {
    if (j == 0) a.nsel[r] = -1;
    return;
  }
  for (int i = j; i < kIdxHeads * kIdxDim; i += 1024) iq[i] = a.idx[(size_t)r * kQsaIdxWidth + i];
  __syncthreads();
  float* sc = a.scores + (size_t)r * a.sstride;
  for (int b = j; b < nb; b += 1024) {
    const float* pk = a.c.ik_pool + (size_t)b * kIdxDim;
    float s = 0.f;
#pragma unroll
    for (int hh = 0; hh < kIdxHeads; ++hh) {
      float d = 0.f;
      for (int i = 0; i < kIdxDim; ++i) d += iq[hh * kIdxDim + i] * pk[i];
      s += fmaxf(d, 0.f);
    }
    sc[b] = s * 0.08838834764831845f;
  }
  __syncthreads();
  // Scores are >= 0, so their bit patterns sort like the values. Find the
  // largest threshold t with count(score >= t) >= 512, MSB first.
  unsigned thr = 0;
  for (int bit = 31; bit >= 0; --bit) {
    const unsigned cand = thr | (1u << bit);
    if (j == 0) cnt = 0;
    __syncthreads();
    unsigned c = 0;
    for (int b = j; b < nb; b += 1024) c += __float_as_uint(sc[b]) >= cand;
    atomicAdd(&cnt, c);
    __syncthreads();
    if (cnt >= kSelBlocks) thr = cand;
    __syncthreads();
  }
  // Strictly-greater blocks all win; ties at thr fill the rest by lowest id.
  // Serial emission keeps ascending order and the tie rule exact.
  if (j == 0) {
    int n = 0, gt = 0;
    for (int b = 0; b < nb; ++b) gt += __float_as_uint(sc[b]) > thr;
    int ties = kSelBlocks - gt;
    for (int b = 0; b < nb && n < kSelBlocks; ++b) {
      const unsigned u = __float_as_uint(sc[b]);
      if (u > thr || (u == thr && ties-- > 0)) a.sel[r * kSelBlocks + n++] = b;
    }
    a.nsel[r] = n;
  }
}

// One block per (head, row): online softmax over the attended positions, eight
// waves striding positions, fixed-order merge. Writes the gated context times
// suh_o, Hadamard-transformed, as the o_proj input.
__global__ void __launch_bounds__(256) QsaAttnKernel(QsaArgs a) {
  __shared__ float qv[256], wm[8], wl[8], wacc[8][256];
  const int h = blockIdx.x, r = blockIdx.y, p = a.pos0 + r, g = h / 12;
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
  const float* q = a.q + ((size_t)r * kQHeads + h) * 512;
  qv[threadIdx.x] = q[threadIdx.x];
  __syncthreads();
  float qr[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) qr[i] = qv[lane * 8 + i];
  const int ns = a.nsel[r];
  const int nb = (p + 1) / 4;
  const int n = ns < 0 ? p + 1 : ns * 4 + (p + 1 - nb * 4);
  float m = -INFINITY, l = 0.f, acc[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) acc[i] = 0.f;
  for (int t = w; t < n; t += 8) {
    int pos = t;
    if (ns >= 0) pos = t < ns * 4 ? a.sel[r * kSelBlocks + t / 4] * 4 + (t & 3) : nb * 4 + (t - ns * 4);
    const __half* kr = a.c.k + ((size_t)pos * kKvHeads + g) * kHeadDim + lane * 8;
    const uint4 kraw = *reinterpret_cast<const uint4*>(kr);
    const __half* kh = reinterpret_cast<const __half*>(&kraw);
    float d = 0.f;
#pragma unroll
    for (int i = 0; i < 8; ++i) d += qr[i] * __half2float(kh[i]);
    d = WaveSum(d) * 0.0625f;
    const float mn = fmaxf(m, d);
    const float sc = __expf(m - mn), e = __expf(d - mn);
    const __half* vr = a.c.v + ((size_t)pos * kKvHeads + g) * kHeadDim + lane * 8;
    const uint4 vraw = *reinterpret_cast<const uint4*>(vr);
    const __half* vh = reinterpret_cast<const __half*>(&vraw);
#pragma unroll
    for (int i = 0; i < 8; ++i) acc[i] = acc[i] * sc + e * __half2float(vh[i]);
    l = l * sc + e;
    m = mn;
  }
  if (lane == 0) {
    wm[w] = m;
    wl[w] = l;
  }
#pragma unroll
  for (int i = 0; i < 8; ++i) wacc[w][lane * 8 + i] = acc[i];
  __syncthreads();
  const int d = threadIdx.x;
  float M = -INFINITY;
  for (int i = 0; i < 8; ++i) M = fmaxf(M, wm[i]);
  float L = 0.f, o = 0.f;
  for (int i = 0; i < 8; ++i) {
    if (wl[i] == 0.f) continue;
    const float f = __expf(wm[i] - M);
    L += wl[i] * f;
    o += wacc[i][d] * f;
  }
  if (a.context) a.context[((size_t)r * kQHeads + h) * 256 + d] = o / L;
  o = o / L * Sigmoid(q[256 + d]);
  __syncthreads();
  qv[d] = RoundH(RoundH(o) * __half2float(a.suh_o[h * 256 + d]));
  __syncthreads();
  if (w < 2) {
    float v[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) v[i] = qv[w * 128 + lane * 4 + i];
    Wht128(v);
    __half* dst = a.xh_o + (size_t)r * kQsaOutWidth + h * 256 + w * 128 + lane * 4;
#pragma unroll
    for (int i = 0; i < 4; ++i) dst[i] = __float2half(v[i] * kHadNorm);
  }
}

}  // namespace crucible
