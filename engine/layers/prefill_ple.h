// Prefill PLE mix in two passes. The decode kernel walks rows in order because
// of the dilated conv; its normalized values depend only on their own row, so
// prefill computes them for all rows first and then applies the conv. Both
// passes keep the decode kernel's block shape, so the sums are bitwise equal.
#pragma once
#include "ops.h"

namespace crucible {
// nv is fp16 because the decode kernel already rounds it to fp16.
__global__ void __launch_bounds__(512) PrefillPleNormKernel(PleArgs a, __half* nv, float* gates) {
  __shared__ float red[16];
  const int s = blockIdx.x, r = blockIdx.y, j = threadIdx.x;
  constexpr int C = 5;
  const float* key = a.kv + (size_t)r * 12800 + s * kHidden;
  const float* val = a.kv + (size_t)r * 12800 + kWide;
  const float* hs = a.H + (size_t)r * a.ldh + s * kHidden;
  float kk[C], qq[C];
  float sk = 0.f, sq = 0.f;
#pragma unroll
  for (int c = 0; c < C; ++c) {
    const int d = j + c * 512;
    kk[c] = key[d];
    qq[c] = hs[d];
    sk += kk[c] * kk[c];
    sq += qq[c] * qq[c];
  }
  sk = BlockSum<512>(sk, red);
  sq = BlockSum<512>(sq, red);
  const float ik = rsqrtf(sk / kHidden + kEps), iqn = rsqrtf(sq / kHidden + kEps);
  float dot = 0.f;
#pragma unroll
  for (int c = 0; c < C; ++c) {
    const int ch = s * kHidden + j + c * 512;
    dot += (kk[c] * ik * a.norm_key1[ch]) * (qq[c] * iqn * a.norm_query1[ch]);
  }
  dot = BlockSum<512>(dot, red) * 0.019764235376052372f;  // 1/sqrt(2560)
  const float sg = dot > 0.f ? 1.f : (dot < 0.f ? -1.f : 0.f);
  const float gate = Sigmoid(sg * sqrtf(fmaxf(fabsf(dot), 1e-6f)));
  float gv[C], sn = 0.f;
#pragma unroll
  for (int c = 0; c < C; ++c) {
    gv[c] = gate * val[j + c * 512];
    sn += gv[c] * gv[c];
  }
  sn = BlockSum<512>(sn, red);
  const float in = rsqrtf(sn / kHidden + kEps);
#pragma unroll
  for (int c = 0; c < C; ++c) {
    const int ch = s * kHidden + j + c * 512;
    nv[(size_t)r * kWide + ch] = __float2half(gv[c] * in * a.norm_conv1[ch]);
  }
  if (j == 0) gates[r * kStreams + s] = gate;
}

// One thread per (row, channel). Rows before this chunk come from the ring.
__global__ void __launch_bounds__(256) PrefillPleConvKernel(PleArgs a, const __half* nv,
                                                           const float* gates) {
  const int ch = blockIdx.x * 256 + threadIdx.x, r = blockIdx.y;
  if (ch >= kWide) return;
  const int s = ch / kHidden, d = ch % kHidden, p = a.pos0 + r;
  const float* w = a.conv + (size_t)ch * 4;
  float cv = 0.f;
#pragma unroll
  for (int t = 0; t < 4; ++t) {
    const int src = p - 9 + 3 * t;
    if (src < 0) continue;
    const float x = src >= a.pos0 ? __half2float(nv[(size_t)(src - a.pos0) * kWide + ch])
                                  : a.ring[(size_t)(src & 15) * kWide + ch];
    cv += w[t] * x;
  }
  const float gv = gates[r * kStreams + s] * a.kv[(size_t)r * 12800 + kWide + d];
  a.H[(size_t)r * a.ldh + s * kHidden + d] += gv + Silu(cv);
}

__global__ void __launch_bounds__(256) PrefillPleRingKernel(PleArgs a, const __half* nv) {
  const int ch = blockIdx.x * 256 + threadIdx.x, r = a.R - 1 - blockIdx.y;
  if (ch >= kWide || r < 0) return;
  a.ring[(size_t)((a.pos0 + r) & 15) * kWide + ch] = __half2float(nv[(size_t)r * kWide + ch]);
}

inline void LaunchPrefillPle(PleArgs a, __half* nv, float* gates) {
  PrefillPleNormKernel<<<dim3(kStreams, a.R), 512>>>(a, nv, gates);
  PrefillPleConvKernel<<<dim3(kWide / 256, a.R), 256>>>(a, nv, gates);
  PrefillPleRingKernel<<<dim3(kWide / 256, a.R < 16 ? a.R : 16), 256>>>(a, nv);
}
}  // namespace crucible
