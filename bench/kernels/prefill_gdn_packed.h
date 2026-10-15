// Vector-packed high/low operands for the same fp32 WY/UT recurrence.
#pragma once
#include "../../engine/layers/kernels/prefill_gdn.h"

namespace crucible::prefill {
struct Packed { __half *hi, *lo; };
struct PackedGdn {
  GdnPrefill a;
  Packed q, k, inverse, qk, wk, v, w, u, end, state;
};

__device__ __forceinline__ void StorePacked(Packed p, size_t i, float x) {
  p.hi[i] = __float2half(x);
  p.lo[i] = __float2half((x - __half2float(p.hi[i])) * kResidualScale);
}

__device__ __forceinline__ H16 PackedFragment(const __half* x) {
  return *reinterpret_cast<const H16*>(x);
}

__device__ __forceinline__ F8 PackedProduct(int k, Packed a, Packed b) {
  const int lane = threadIdx.x & 31;
  F8 high = {}, cross = {};
#pragma clang loop unroll(disable)
  for (int start = 0; start < k; start += 16) {
    const H16 ah = PackedFragment(a.hi + (lane & 15) * k + start);
    const H16 al = PackedFragment(a.lo + (lane & 15) * k + start);
    const H16 bh = PackedFragment(b.hi + (lane & 15) * k + start);
    const H16 bl = PackedFragment(b.lo + (lane & 15) * k + start);
    high = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(ah, bh, high);
    cross = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(ah, bl, cross);
    cross = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(al, bh, cross);
  }
  return high + cross * (1.f / kResidualScale);
}

__device__ __forceinline__ Packed Slice(Packed p, size_t i) { return {p.hi + i, p.lo + i}; }

// Matrix transpose uses a padded LDS tile. Global reads and writes are both
// coalesced. Scalar scattered half stores would waste most transactions.
__global__ void __launch_bounds__(256) GdnPackInputKernel(PackedGdn p) {
  __shared__ float tile[32][33];
  const auto a = p.a;
  const int kind = blockIdx.z % 4, hc = blockIdx.z / 4;
  const int x = threadIdx.x & 31, y = threadIdx.x >> 5;
  const int n0 = blockIdx.x * 32, m0 = blockIdx.y * 32;
  const size_t base = (size_t)hc * 64 * 128;
  const float* src = kind == 0 ? a.q : kind == 1 ? a.k : kind == 2 ? a.wk : a.v;
  const Packed dst = kind == 0 ? p.q : kind == 1 ? p.k : kind == 2 ? p.wk : p.v;
  if (kind < 2) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const size_t idx = base + (m0 + y + i * 8) * 128 + n0 + x;
      StorePacked(dst, idx, src[idx]);
    }
    return;
  }
#pragma unroll
  for (int i = 0; i < 4; ++i) tile[y + i * 8][x] = src[base + (m0 + y + i * 8) * 128 + n0 + x];
  __syncthreads();
#pragma unroll
  for (int i = 0; i < 4; ++i)
    StorePacked(dst, base + (n0 + y + i * 8) * 64 + m0 + x, tile[x][y + i * 8]);
}

__global__ void __launch_bounds__(128) GdnPackedGramKernel(PackedGdn p) {
  const auto a = p.a;
  const int m0 = blockIdx.x * 16, n0 = (threadIdx.x >> 5) * 16;
  const int c = blockIdx.y, h = blockIdx.z, lane = threadIdx.x & 31;
  const size_t b = GdnBase(a, h, c);
  const F8 kk = PackedProduct(128, Slice(p.k, (b + m0) * 128), Slice(p.k, (b + n0) * 128));
  const F8 qk = PackedProduct(128, Slice(p.q, (b + m0) * 128), Slice(p.k, (b + n0) * 128));
#pragma unroll
  for (int z = 0; z < 8; ++z) {
    const int t = m0 + FragmentRow(z), s = n0 + (lane & 15);
    const float decay = s <= t ? expf(a.g[b + t] - a.g[b + s]) : 0.f;
    const size_t idx = b * 64 + t * 64 + s;
    a.lower[idx] = s < t ? a.beta[b + t] * kk[z] * decay : 0.f;
    StorePacked(p.qk, idx, s <= t ? qk[z] * decay : 0.f);
  }
}

__global__ void GdnPackInverseKernel(PackedGdn p) {
  const size_t i = (size_t)blockIdx.x * 256 + threadIdx.x;
  if (i < (size_t)48 * p.a.chunks * 64 * 64) StorePacked(p.inverse, i, p.a.lower[i]);
}

__global__ void __launch_bounds__(128) GdnPackedWyKernel(PackedGdn p) {
  const auto a = p.a;
  const int hc = blockIdx.z / 2, kind = blockIdx.z & 1;
  const int h = hc / a.chunks, c = hc % a.chunks;
  const int m0 = blockIdx.y * 16, n0 = blockIdx.x * 64 + (threadIdx.x >> 5) * 16;
  const int lane = threadIdx.x & 31;
  const size_t b = GdnBase(a, h, c);
  const Packed src = kind == 0 ? p.wk : p.v;
  const F8 out = PackedProduct(64, Slice(p.inverse, (b + m0) * 64), Slice(src, b * 128 + n0 * 64));
#pragma unroll
  for (int z = 0; z < 8; ++z) {
    const size_t idx = (b + m0 + FragmentRow(z)) * 128 + n0 + (lane & 15);
    if (kind == 0) StorePacked(p.w, idx, out[z]);
    else a.ut[idx] = out[z];
  }
}

__global__ void __launch_bounds__(256) GdnPackEndKernel(PackedGdn p) {
  __shared__ float tile[32][33];
  const auto a = p.a;
  const int hc = blockIdx.z, c = hc % a.chunks;
  const size_t b = (size_t)hc * 64;
  const int end = min(64, a.input.R - c * 64) - 1;
  const int x = threadIdx.x & 31, y = threadIdx.x >> 5;
  const int n0 = blockIdx.x * 32, m0 = blockIdx.y * 32;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int t = m0 + y + i * 8;
    const float decay = t <= end ? expf(a.g[b + end] - a.g[b + t]) : 0.f;
    tile[y + i * 8][x] = decay * a.k[(b + t) * 128 + n0 + x];
  }
  __syncthreads();
#pragma unroll
  for (int i = 0; i < 4; ++i)
    StorePacked(p.end, b * 128 + (n0 + y + i * 8) * 64 + m0 + x, tile[x][y + i * 8]);
}

__global__ void __launch_bounds__(256) GdnPackStateKernel(PackedGdn p) {
  __shared__ float tile[32][33];
  const int x = threadIdx.x & 31, y = threadIdx.x >> 5;
  const int n0 = blockIdx.x * 32, m0 = blockIdx.y * 32;
  const size_t b = (size_t)blockIdx.z * 128 * 128;
#pragma unroll
  for (int i = 0; i < 4; ++i)
    tile[y + i * 8][x] = p.a.input.state[b + (m0 + y + i * 8) * 128 + n0 + x];
  __syncthreads();
#pragma unroll
  for (int i = 0; i < 4; ++i)
    StorePacked(p.state, b + (n0 + y + i * 8) * 128 + m0 + x, tile[x][y + i * 8]);
}

__global__ void __launch_bounds__(128) GdnPackedCarryKernel(PackedGdn p, int c) {
  __shared__ float transpose[4][16][16];
  const auto a = p.a;
  const int m0 = blockIdx.y * 16, n0 = blockIdx.x * 64 + (threadIdx.x >> 5) * 16;
  const int h = blockIdx.z, lane = threadIdx.x & 31, wave = threadIdx.x >> 5;
  const size_t b = GdnBase(a, h, c);
  const Packed s = Slice(p.state, (size_t)h * 128 * 128 + n0 * 128);
  const F8 ws = PackedProduct(128, Slice(p.w, (b + m0) * 128), s);
  const F8 qs = PackedProduct(128, Slice(p.q, (b + m0) * 128), s);
#pragma unroll
  for (int z = 0; z < 8; ++z) {
    const int m = m0 + FragmentRow(z), n = n0 + (lane & 15);
    const size_t idx = (b + m) * 128 + n;
    const float u = a.ut[idx] - ws[z];
    transpose[wave][FragmentRow(z)][lane & 15] = u;
    a.y[idx] = expf(a.g[b + m]) * qs[z];
  }
  __syncwarp();
#pragma unroll
  for (int z = 0; z < 8; ++z) {
    const int n = n0 + FragmentRow(z), m = m0 + (lane & 15);
    StorePacked(p.u, b * 128 + n * 64 + m, transpose[wave][lane & 15][FragmentRow(z)]);
  }
}

__global__ void __launch_bounds__(128) GdnPackedOutputKernel(PackedGdn p, int c) {
  const auto a = p.a;
  const int m0 = blockIdx.y * 16, n0 = blockIdx.x * 64 + (threadIdx.x >> 5) * 16;
  const int h = blockIdx.z, lane = threadIdx.x & 31;
  const size_t b = GdnBase(a, h, c);
  const F8 local = PackedProduct(64, Slice(p.qk, (b + m0) * 64), Slice(p.u, b * 128 + n0 * 64));
#pragma unroll
  for (int z = 0; z < 8; ++z)
    a.y[(b + m0 + FragmentRow(z)) * 128 + n0 + (lane & 15)] += local[z];
}

__global__ void __launch_bounds__(128) GdnPackedStateKernel(PackedGdn p, int c) {
  __shared__ float transpose[4][16][16];
  const auto a = p.a;
  const int m0 = blockIdx.y * 16, n0 = blockIdx.x * 64 + (threadIdx.x >> 5) * 16;
  const int h = blockIdx.z, lane = threadIdx.x & 31, wave = threadIdx.x >> 5;
  const size_t b = GdnBase(a, h, c);
  const int end = min(64, a.input.R - c * 64) - 1;
  const float decay = expf(a.g[b + end]);
  const F8 update = PackedProduct(64, Slice(p.end, b * 128 + m0 * 64), Slice(p.u, b * 128 + n0 * 64));
#pragma unroll
  for (int z = 0; z < 8; ++z) {
    const size_t idx = (size_t)h * 128 * 128 + (m0 + FragmentRow(z)) * 128 + n0 + (lane & 15);
    const float state = decay * a.input.state[idx] + update[z];
    a.input.state[idx] = state;
    transpose[wave][FragmentRow(z)][lane & 15] = state;
  }
  __syncwarp();
#pragma unroll
  for (int z = 0; z < 8; ++z) {
    const size_t idx = (size_t)h * 128 * 128 + (n0 + FragmentRow(z)) * 128 + m0 + (lane & 15);
    StorePacked(p.state, idx, transpose[wave][lane & 15][FragmentRow(z)]);
  }
}

inline void LaunchGdnPacked(PackedGdn p, hipStream_t stream = nullptr) {
  const auto a = p.a;
  ValidateGdnPrefill(a);
  GdnPrefillPrepKernel<<<dim3(48, a.chunks * 64), 256, 0, stream>>>(a);
  GdnPrefillPrefixKernel<<<dim3(48, a.chunks), 128, 0, stream>>>(a);
  GdnPackInputKernel<<<dim3(4, 2, 48 * a.chunks * 4), 256, 0, stream>>>(p);
  GdnPackedGramKernel<<<dim3(4, a.chunks, 48), 128, 0, stream>>>(p);
  GdnPrefillInverseKernel<<<dim3(48, a.chunks), 64, 0, stream>>>(a);
  GdnPackInverseKernel<<<(48 * a.chunks * 64 * 64 + 255) / 256, 256, 0, stream>>>(p);
  GdnPackedWyKernel<<<dim3(2, 4, 48 * a.chunks * 2), 128, 0, stream>>>(p);
  GdnPackEndKernel<<<dim3(4, 2, 48 * a.chunks), 256, 0, stream>>>(p);
  GdnPackStateKernel<<<dim3(4, 4, 48), 256, 0, stream>>>(p);
  for (int c = 0; c < a.chunks; ++c) {
    GdnPackedCarryKernel<<<dim3(2, 4, 48), 128, 0, stream>>>(p, c);
    GdnPackedOutputKernel<<<dim3(2, 4, 48), 128, 0, stream>>>(p, c);
    GdnPackedStateKernel<<<dim3(2, 8, 48), 128, 0, stream>>>(p, c);
  }
  GdnPrefillGateKernel<<<dim3(48, a.input.R), 256, 0, stream>>>(a);
  GdnPrefillHistoryKernel<<<40, 256, 0, stream>>>(a);
}
}  // namespace crucible::prefill
