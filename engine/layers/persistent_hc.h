// HC read for the persistent R=1 kernel. Every lane computes exactly what it
// computes in HcDotsKernel<1> and HcMixKernel<1>; only the assignment of rows
// to waves and the load schedule change. Each wave keeps the next unit's
// weights in flight while it computes the current one, so the phase streams
// instead of running latency-bound rounds of one unit per wave.
#pragma once
#include "ops.h"
#include "../exl3/exl3_ring.h"
#include "../gfx1151/persistent_sync.h"

namespace crucible {

// Block b serves stream b % 4 (gridDim.x % 4 == 0) and stages that raw stream
// in hs (kHidden floats of LDS), so the dot loads never wait behind the next
// unit's weights. Units are rows m = 0..M of the stream; m == M is the sum
// of squares.
__device__ __forceinline__ void PersistentHcDotsPipe(ComputeContext& ct, const __half* fn, int M,
                                                    const float* H, float* dots, float* ss,
                                                    float* hs) {
  const int s = blockIdx.x & 3, group = blockIdx.x >> 2, stride = (gridDim.x >> 2) * 8;
  const float4* src = reinterpret_cast<const float4*>(H + s * kHidden);
  for (int i = ct.t; i < kHidden / 4; i += 256) reinterpret_cast<float4*>(hs)[i] = src[i];
  ComputeBarrier(ct);
  const int lane = ct.t & 31;
  auto load = [&](uint4* raw, int m) {
    if (m >= M) return;
    const __half* wp = fn + (size_t)m * kWide + s * kHidden + lane * 8;
#pragma unroll
    for (int k = 0; k < 10; ++k) raw[k] = LoadNT(wp + k * 256);
  };
  auto compute = [&](const uint4* raw, int m) {
    if (m > M) return;
    if (m == M) {
      float a = 0.f;
      for (int d = lane; d < kHidden; d += 32) a += hs[d] * hs[d];
      a = WaveSum(a);
      if (lane == 0) ss[s] = a;
      return;
    }
    float acc = 0.f;
#pragma unroll
    for (int k = 0; k < 10; ++k) {
      const __half* hw = reinterpret_cast<const __half*>(&raw[k]);
      float wf[8];
#pragma unroll
      for (int i = 0; i < 8; ++i) wf[i] = __half2float(hw[i]);
      const float4* hr = reinterpret_cast<const float4*>(hs + k * 256 + lane * 8);
      const float4 a = hr[0], b = hr[1];
      acc += wf[0] * a.x + wf[1] * a.y + wf[2] * a.z + wf[3] * a.w + wf[4] * b.x + wf[5] * b.y +
             wf[6] * b.z + wf[7] * b.w;
    }
    const float v = WaveSum(acc);
    if (lane == 0) dots[(size_t)m * kStreams + s] = v;
  };
  uint4 A[10], B[10];
  int m = group * 8 + (ct.t >> 5);
  load(A, m);
  for (; m <= M; m += 2 * stride) {
    load(B, m + stride);
    compute(A, m);
    load(A, m + 2 * stride);
    compute(B, m + stride);
  }
  // hs is the ring's activation buffer in the next phases.
  ComputeBarrier(ct);
}

// Units are output pairs d0 = 2u, u < kHidden / 2, strided over all waves.
// The H and w1 values of a unit load before the next unit's weights.
__device__ __forceinline__ void PersistentHcMixPipe(ComputeContext& ct, const __half* up,
                                                   const float* w1, const float* dots, int M,
                                                   const float* ss, const float* H, float* mixed,
                                                   float* post) {
  __shared__ float t[kRank];
  __shared__ float inv[kStreams];
  if (ct.t < kStreams) inv[ct.t] = rsqrtf(ss[ct.t] / kHidden + kEps);
  ComputeBarrier(ct);
  for (int m = ct.t; m < kRank; m += 256) {
    const float* dm = dots + (size_t)m * kStreams;
    float a = 0.f;
#pragma unroll
    for (int s = 0; s < kStreams; ++s) a += inv[s] * dm[s];
    t[m] = Silu(a / kStreams);
  }
  if (blockIdx.x == 0 && post != nullptr && ct.t < kStreams) {
    const int s = ct.t;
    const float* dm = dots + (size_t)(kRank + s) * kStreams;
    float a = 0.f;
#pragma unroll
    for (int s2 = 0; s2 < kStreams; ++s2) a += inv[s2] * dm[s2];
    post[s] = 2.f * Sigmoid(a / kStreams);
  }
  ComputeBarrier(ct);
  const int lane = ct.t & 31, s = lane >> 3, li = lane & 7;
  const int stride = gridDim.x * 8, units = kHidden / 2;
  auto load = [&](uint4 (*raw)[5], int u) {
    if (u >= units) return;
#pragma unroll
    for (int e = 0; e < 2; ++e) {
      const __half* w = up + ((size_t)s * kHidden + 2 * u + e) * kRank + li * 8;
#pragma unroll
      for (int k = 0; k < 5; ++k) raw[e][k] = LoadNT(w + k * 64);
    }
  };
  auto side = [&](float* hv, float* wv, int u) {
    if (u >= units) return;
#pragma unroll
    for (int e = 0; e < 2; ++e) {
      hv[e] = H[(size_t)s * kHidden + 2 * u + e];
      wv[e] = w1[s * kHidden + 2 * u + e];
    }
  };
  auto compute = [&](const uint4 (*raw)[5], const float* hv, const float* wv, int u) {
    if (u >= units) return;
#pragma unroll
    for (int e = 0; e < 2; ++e) {
      const int d = 2 * u + e;
      float acc = 0.f;
#pragma unroll
      for (int k = 0; k < 5; ++k) {
        const __half* hw = reinterpret_cast<const __half*>(&raw[e][k]);
        const int c = k * 64 + li * 8;
#pragma unroll
        for (int i = 0; i < 8; ++i) acc += __half2float(hw[i]) * t[c + i];
      }
      float g = acc;
      g += __shfl_xor(g, 1);
      g += __shfl_xor(g, 2);
      g += __shfl_xor(g, 4);
      float v = Sigmoid(g) * hv[e] * inv[s] * wv[e];
      v += __shfl_xor(v, 8);
      v += __shfl_xor(v, 16);
      if (lane == 0) mixed[d] = RoundH(v / kStreams);
    }
  };
  uint4 A[2][5], B[2][5];
  float ha[2], wa[2], hb[2], wb[2];
  int u = blockIdx.x * 8 + (ct.t >> 5);
  load(A, u);
  for (; u < units; u += 2 * stride) {
    side(ha, wa, u);
    load(B, u + stride);
    compute(A, ha, wa, u);
    side(hb, wb, u + stride);
    load(A, u + 2 * stride);
    compute(B, hb, wb, u + stride);
  }
}

}  // namespace crucible

// ---- HC read through the loader ring ------------------------------------
// The loader waves stream every HC weight byte through the LDS ring, as they
// do for the projections, so DRAM stays busy across the grid barriers. One
// 5 KB unit per ring chunk; unit i of a block belongs to compute wave i % 8.
// Each block owns a contiguous unit range, so per-block bytes differ by at
// most one unit. The lane arithmetic is the one of HcDotsKernel<1> and
// HcMixKernel<1>.

namespace crucible {
namespace persistent_hc {

constexpr int kUnitVectors = 320;  // 16-byte vectors per 5 KB unit
static_assert(kUnitVectors * 4 <= lin_ring::kSlotWords);

struct DotsRange {
  int s, r0, r1;
};
// Block b serves stream b % 4 and a contiguous range of the M weight rows.
__device__ __forceinline__ DotsRange DotsUnits(int M) {
  const int groups = gridDim.x >> 2, group = blockIdx.x >> 2;
  return {(int)(blockIdx.x & 3), M * group / groups, M * (group + 1) / groups};
}
__device__ __forceinline__ int2 MixUnits() {
  constexpr int units = kHidden / 2;
  return {units * (int)blockIdx.x / (int)gridDim.x, units * ((int)blockIdx.x + 1) / (int)gridDim.x};
}

// Loader side: offset(unit, vector) gives the byte offset of one 16-byte vector.
template <class Offset>
__device__ __forceinline__ void LoadUnits(const void* base, uint32_t bytes, int u0, int u1,
                                         Offset offset, uint32_t* payload, lin_ring::Control& ring,
                                         uint32_t& seq) {
  const int L = threadIdx.x, lane = L & 31;  // both loader waves
  const auto rs = StreamRsrc(reinterpret_cast<const uint32_t*>(base), bytes);
  auto ld = [&](uint4* r, int u) {
    if (u >= u1) return;
#pragma unroll
    for (int j = 0; j < 5; ++j) {
      const auto raw = __builtin_amdgcn_raw_buffer_load_b128(rs, offset(u, j * 64 + L), 0, 0);
      r[j] = *reinterpret_cast<const uint4*>(&raw);
    }
  };
  auto put = [&](const uint4* r, int u) {
    if (u >= u1) return;
    const uint32_t g = seq + (u - u0);
    ring.WaitWritable(g);
    uint4* d = reinterpret_cast<uint4*>(payload + (g % lin_ring::kSlots) * lin_ring::kSlotWords) + L;
#pragma unroll
    for (int j = 0; j < 5; ++j) d[j * 64] = r[j];
    ring.Publish(g, lane);
  };
  uint4 A[5], B[5];
  ld(A, u0);
  for (int u = u0; u < u1; u += 2) {
    ld(B, u + 1);
    put(A, u);
    ld(A, u + 2);
    put(B, u + 1);
  }
  seq += u1 - u0;
}

__device__ __forceinline__ void LoadDots(const __half* fn, int M, uint32_t* payload,
                                        lin_ring::Control& ring, uint32_t& seq) {
  const DotsRange r = DotsUnits(M);
  LoadUnits(fn, (uint32_t)M * kWide * 2, r.r0, r.r1,
            [&](int m, int v) { return (m * kWide + r.s * kHidden) * 2 + v * 16; }, payload, ring, seq);
}

// A mix unit is four 1,280-byte runs: rows 2u and 2u + 1 of each stream.
__device__ __forceinline__ void LoadMix(const __half* up, uint32_t* payload, lin_ring::Control& ring,
                                       uint32_t& seq) {
  const int2 r = MixUnits();
  LoadUnits(up, (uint32_t)kWide * kRank * 2, r.x, r.y,
            [&](int u, int v) { return ((v / 80) * kHidden + 2 * u) * kRank * 2 + (v % 80) * 16; },
            payload, ring, seq);
}

}  // namespace persistent_hc

__device__ __forceinline__ void PersistentHcDotsRing(ComputeContext& ct, int M, const float* H,
                                                    float* dots, float* ss, float* hs,
                                                    const uint32_t* payload, lin_ring::Control& ring,
                                                    uint32_t& seq) {
  const persistent_hc::DotsRange r = persistent_hc::DotsUnits(M);
  const int s = r.s;
  const float4* src = reinterpret_cast<const float4*>(H + s * kHidden);
  for (int i = ct.t; i < kHidden / 4; i += 256) reinterpret_cast<float4*>(hs)[i] = src[i];
  ComputeBarrier(ct);
  const int lane = ct.t & 31, warp = ct.t >> 5;
  for (int m = r.r0; m < r.r1; ++m) {
    const uint32_t g = seq + (m - r.r0);
    const bool mine = ((m - r.r0) & 7) == warp;
    ring.WaitReady(g);
    uint4 raw[10];
    if (mine) {
      const uint4* slot = reinterpret_cast<const uint4*>(payload + (g % lin_ring::kSlots) * lin_ring::kSlotWords);
#pragma unroll
      for (int k = 0; k < 10; ++k) raw[k] = slot[k * 32 + lane];
    }
    ring.Release(g, lane);
    if (!mine) continue;
    float acc = 0.f;
#pragma unroll
    for (int k = 0; k < 10; ++k) {
      const __half* hw = reinterpret_cast<const __half*>(&raw[k]);
      float wf[8];
#pragma unroll
      for (int i = 0; i < 8; ++i) wf[i] = __half2float(hw[i]);
      const float4* hr = reinterpret_cast<const float4*>(hs + k * 256 + lane * 8);
      const float4 a = hr[0], b = hr[1];
      acc += wf[0] * a.x + wf[1] * a.y + wf[2] * a.z + wf[3] * a.w + wf[4] * b.x + wf[5] * b.y +
             wf[6] * b.z + wf[7] * b.w;
    }
    const float v = WaveSum(acc);
    if (lane == 0) dots[(size_t)m * kStreams + s] = v;
  }
  seq += r.r1 - r.r0;
  // The last block of each stream adds the sum of squares (row M).
  if (r.r1 == M && warp == 0) {
    float a = 0.f;
    for (int d = lane; d < kHidden; d += 32) a += hs[d] * hs[d];
    a = WaveSum(a);
    if (lane == 0) ss[s] = a;
  }
  ComputeBarrier(ct);  // hs is the ring's activation buffer in later phases
}

__device__ __forceinline__ void PersistentHcMixRing(ComputeContext& ct, const float* w1,
                                                   const float* dots, int M, const float* ss,
                                                   const float* H, float* mixed, float* post,
                                                   const uint32_t* payload, lin_ring::Control& ring,
                                                   uint32_t& seq) {
  __shared__ float t[kRank];
  __shared__ float inv[kStreams];
  if (ct.t < kStreams) inv[ct.t] = rsqrtf(ss[ct.t] / kHidden + kEps);
  ComputeBarrier(ct);
  for (int m = ct.t; m < kRank; m += 256) {
    const float* dm = dots + (size_t)m * kStreams;
    float a = 0.f;
#pragma unroll
    for (int s = 0; s < kStreams; ++s) a += inv[s] * dm[s];
    t[m] = Silu(a / kStreams);
  }
  if (blockIdx.x == 0 && post != nullptr && ct.t < kStreams) {
    const int s = ct.t;
    const float* dm = dots + (size_t)(kRank + s) * kStreams;
    float a = 0.f;
#pragma unroll
    for (int s2 = 0; s2 < kStreams; ++s2) a += inv[s2] * dm[s2];
    post[s] = 2.f * Sigmoid(a / kStreams);
  }
  ComputeBarrier(ct);
  const int lane = ct.t & 31, warp = ct.t >> 5, s = lane >> 3, li = lane & 7;
  const int2 r = persistent_hc::MixUnits();
  for (int u = r.x; u < r.y; ++u) {
    const uint32_t g = seq + (u - r.x);
    const bool mine = ((u - r.x) & 7) == warp;
    ring.WaitReady(g);
    uint4 raw[2][5];
    if (mine) {
      const uint4* slot = reinterpret_cast<const uint4*>(payload + (g % lin_ring::kSlots) * lin_ring::kSlotWords);
#pragma unroll
      for (int e = 0; e < 2; ++e)
#pragma unroll
        for (int k = 0; k < 5; ++k) raw[e][k] = slot[s * 80 + e * 40 + k * 8 + li];
    }
    ring.Release(g, lane);
    if (!mine) continue;
#pragma unroll
    for (int e = 0; e < 2; ++e) {
      const int d = 2 * u + e;
      float acc = 0.f;
#pragma unroll
      for (int k = 0; k < 5; ++k) {
        const __half* hw = reinterpret_cast<const __half*>(&raw[e][k]);
        const int c = k * 64 + li * 8;
#pragma unroll
        for (int i = 0; i < 8; ++i) acc += __half2float(hw[i]) * t[c + i];
      }
      float g2 = acc;
      g2 += __shfl_xor(g2, 1);
      g2 += __shfl_xor(g2, 2);
      g2 += __shfl_xor(g2, 4);
      float v = Sigmoid(g2) * H[(size_t)s * kHidden + d] * inv[s] * w1[s * kHidden + d];
      v += __shfl_xor(v, 8);
      v += __shfl_xor(v, 16);
      if (lane == 0) mixed[d] = RoundH(v / kStreams);
    }
  }
  seq += r.y - r.x;
}

}  // namespace crucible
