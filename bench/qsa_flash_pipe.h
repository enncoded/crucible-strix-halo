// Pipelined variants of QsaUnionHalfQKernel (bench only). Same operands, same
// WMMA chains, same softmax and P V order: only load timing and barriers move.
//   LA: the key metadata of step s+1 is built during step s from a list entry
//       loaded one step earlier (double-buffered keypos), so the step loses one
//       barrier and the list load latency.
//   VH: the V loader waves issue all their V loads before the first LDS store
//       (production waits for each 16-byte load before the next one).
//   IL: the two key tiles of a QK half run as two interleaved WMMA chains, with
//       the K fragments of c+D loaded before the WMMAs of c (production waits
//       for each fragment pair before its WMMA).
//   KT: waves 0..2 touch the second line of their keys' first half and both
//       lines of the second half at the step start, so these misses overlap.
// W sets amdgpu_waves_per_eu (production 8).
#pragma once
#include "../engine/layers/kernels/prefill_qsa_union.h"

namespace crucible::prefill {
constexpr int kPipeLA = 1, kPipeVH = 2, kPipeIL1 = 4, kPipeKT = 8, kPipeIL2 = 16, kPipeIL = kPipeIL1 | kPipeIL2;
// VM: each V load instruction covers 4 keys x one 128-byte line (8 lanes per
// key) instead of 32 keys x 16 bytes. V^T row d moves by (d >> 3) x 16
// bytes, so the 8 lanes of a key store to 8 different bank groups.
// Diagnostics (wrong output): NOV no V loads, NOK no K loads, NOPV no P V.
constexpr int kPipeVM = 32, kPipeNoV = 64, kPipeNoK = 128, kPipeNoPV = 256;
// GX: grid (2, tiles), so the two KV groups of a tile run side by side and
// read the two 512-byte halves of the same K and V rows at the same steps.
constexpr int kPipeGX = 512;
// P2: second-half chains one key tile at a time, fragment c + 1 loaded before the WMMA of c.
constexpr int kPipeP2 = 1024;
constexpr int kPipeVtBytes = kUnionVtBytes + 31 * 16;
__device__ __forceinline__ int PipeVtRow(int d) { return d * kUnionPad + (d >> 3) * 8; }

// Two interleaved chains: acc[nt] += q[c] k_nt[c] for c = 0..7, the same
// order per chain as the production loop. Fragments of c + D load before c.
template <int D>
__device__ __forceinline__ void QkChains(const H16 (&qf)[8], const __half* kp0, const __half* kp1, F8& h0, F8& h1) {
  H16 b0[D], b1[D];
#pragma unroll
  for (int c = 0; c < D; ++c) { b0[c] = HalfFragment(kp0 + c * 16); b1[c] = HalfFragment(kp1 + c * 16); }
#pragma unroll
  for (int c = 0; c < 8; ++c) {
    h0 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(qf[c], b0[c % D], h0);
    h1 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(qf[c], b1[c % D], h1);
    if (c + D < 8) { b0[c % D] = HalfFragment(kp0 + (c + D) * 16); b1[c % D] = HalfFragment(kp1 + (c + D) * 16); }
    // Keep the compiler from hoisting all 16 fragment loads (VGPR spills).
    __builtin_amdgcn_sched_barrier(0);
  }
}

template <bool Gate, int F, int D>
__device__ __forceinline__ void QsaUnionPipeBody(QsaPrefill a, QsaUnion u) {
  __shared__ __attribute__((aligned(16))) unsigned char smem[kUnionScoreBytes + kUnionPhiBytes + kPipeVtBytes];
  __shared__ float maxima[48], sums[48], scale[48];
  __shared__ int keypos_[2][kUnionKeys], keymask_[2][kUnionKeys], stepmask_[2];
  auto score = reinterpret_cast<float(*)[kUnionKeys]>(smem);
  auto phi = reinterpret_cast<_Float16(*)[kUnionPad]>(smem + kUnionScoreBytes);
  auto vt = reinterpret_cast<_Float16(*)[kUnionPad]>(smem + kUnionScoreBytes + kUnionPhiBytes);
  _Float16* vtm = reinterpret_cast<_Float16*>(smem + kUnionScoreBytes + kUnionPhiBytes);
  constexpr bool LA = F & kPipeLA, VH = F & kPipeVH, IL1 = F & kPipeIL1, IL2 = F & kPipeIL2, KT = F & kPipeKT;
  constexpr bool VM = F & kPipeVM, NOV = F & kPipeNoV, NOK = F & kPipeNoK, NOPV = F & kPipeNoPV;
  const QsaArgs q = a.input;
  const int tile = F & kPipeGX ? blockIdx.y : blockIdx.x, g = F & kPipeGX ? blockIdx.x : blockIdx.y;
  const int r0 = tile * kUnionRows;
  const int t = threadIdx.x, lane = t & 31, wave = __builtin_amdgcn_readfirstlane(t >> 5);
  const int rows = min(kUnionRows, q.R - r0);
  const int n = u.count[tile] * 4;
  const int* list = u.list + (size_t)tile * kUnionCap;
  if (t < 48) { maxima[t] = -INFINITY; sums[t] = 0.f; }
  F8 out[3][2] = {};
  const bool first = wave < 3, second = wave >= 3 && wave < 6;
  const int mt = first ? wave : second ? wave - 3 : 0, c0 = second ? 8 : 0;
  const int qm = mt * 16 + (lane & 15), qrow = qm / 12, qhead = qm % 12;
  const size_t qi = ((size_t)(r0 + min(qrow, rows - 1)) * 24 + g * 12 + qhead) * 256 + c0 * 16;
  unsigned rowpack = 0;
#pragma unroll
  for (int z = 0; z < 8; ++z) rowpack |= (unsigned)((mt * 16 + FragmentRow(z)) / 12) << (4 * z);
  H16 qf[8];
  if (wave < 6) {
#pragma unroll
    for (int c = 0; c < 8; ++c) qf[c] = HalfFragment(a.q_hi + qi + c * 16);
  }
  // Metadata of the step at `start` into slot `slot`, by one full wave.
  auto meta = [&](int start, int e, int slot) {
    const int idx = start + lane;
    const int pos = (e & 0xffffff) * 4 + (idx & 3);
    keypos_[slot][lane] = idx < n && pos < q.pos0 + q.R ? pos : -1;
    int km = idx < n ? (e >> 24) & 0xff : 0;
    keymask_[slot][lane] = km;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) km |= __shfl_xor(km, o);
    if (lane == 0) stepmask_[slot] = km;
  };
  auto entry = [&](int start) { const int idx = start + lane; return idx < n ? list[idx >> 2] : 0; };
  int enext = 0;
  unsigned sink = 0;
  if constexpr (LA) {
    if (wave == 0) meta(0, entry(0), 0);
    if (wave == 7) enext = entry(kUnionKeys);
  }
  for (int start = 0, s = 0; start < n; start += kUnionKeys, ++s) {
    const int slot = LA ? s & 1 : 0;
    __syncthreads();
    if constexpr (LA) {
      if (wave == 7) {
        meta(start + kUnionKeys, enext, slot ^ 1);
        enext = entry(start + 2 * kUnionKeys);
      }
    } else {
      if (t < kUnionKeys) meta(start, entry(start), 0);
      __syncthreads();
    }
    const int* keypos = keypos_[slot];
    const int* keymask = keymask_[slot];
    const int sm = stepmask_[slot];
    const bool live0 = sm & 3, live1 = sm & 6, live2 = sm & 12;
    const bool live = mt == 0 ? live0 : mt == 1 ? live1 : live2;
    const int pos0 = keypos[lane & 15], pos1 = keypos[16 + (lane & 15)];
    if (first) {
      if (live && IL1) {
        const __half* kp0 = q.c.k + ((size_t)max(pos0, 0) * 2 + g) * 256;
        const __half* kp1 = q.c.k + ((size_t)max(pos1, 0) * 2 + g) * 256;
        unsigned touch[3] = {};
        if constexpr (KT) {
          // Lane l touches key l: line 1 of the first half, lines 2 and 3.
          const __half* kl = q.c.k + ((size_t)max(keypos[lane], 0) * 2 + g) * 256;
#pragma unroll
          for (int line = 1; line < 4; ++line) touch[line - 1] = *reinterpret_cast<const unsigned*>(kl + line * 64);
        }
        F8 high[2] = {};
        QkChains<D>(qf, kp0, kp1, high[0], high[1]);
        if constexpr (KT) sink ^= touch[0] ^ touch[1] ^ touch[2];
#pragma unroll
        for (int nt = 0; nt < 2; ++nt)
#pragma unroll
          for (int z = 0; z < 8; ++z) score[mt * 16 + FragmentRow(z)][nt * 16 + (lane & 15)] = high[nt][z];
      } else if (live) {
        F8 high[2] = {};
#pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
          const __half* kp = NOK ? a.q_hi + qi : q.c.k + ((size_t)max(nt ? pos1 : pos0, 0) * 2 + g) * 256;
#pragma unroll
          for (int c = 0; c < 8; ++c) {
            high[nt] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(qf[c], HalfFragment(kp + c * 16), high[nt]);
            if (c % 4 == 3) __builtin_amdgcn_sched_barrier(0);
          }
#pragma unroll
          for (int z = 0; z < 8; ++z) score[mt * 16 + FragmentRow(z)][nt * 16 + (lane & 15)] = high[nt][z];
        }
      }
    } else if (NOV) {
    } else if (VM) {
      // Instruction ii = 0..31 of waves 3..7: keys 4 (ii & 7) .. +3, line ii >> 3.
      for (int ii = wave - 3; ii < 32; ii += 5) {
        const int key = 4 * (ii & 7) + (lane & 3), v0 = (8 * (ii >> 3) + (lane >> 2)) * 8;
        const int pos = keypos[key];
        uint4 raw = {};
        if (pos >= 0) raw = *reinterpret_cast<const uint4*>(q.c.v + ((size_t)pos * 2 + g) * 256 + v0);
        const _Float16* h = reinterpret_cast<const _Float16*>(&raw);
#pragma unroll
        for (int k = 0; k < 8; ++k) vtm[PipeVtRow(v0 + k) + key] = h[k];
      }
    } else if (VH) {
      // Parts t - 96 + 160 i, i = 0..6, in two batches (4 + 3): each batch
      // issues its loads before the first LDS store.
#pragma unroll
      for (int i0 = 0; i0 < 7; i0 += 4) {
        uint4 raw[4];
#pragma unroll
        for (int i = 0; i < 4 && i0 + i < 7; ++i) {
          const int part = t - 96 + 160 * (i0 + i), key = part & 31, v0 = (part >> 5 & 31) * 8;
          const int pos = keypos[key];
          raw[i] = uint4{};
          if (part < 1024) {
            const uint4 x = *reinterpret_cast<const uint4*>(q.c.v + ((size_t)max(pos, 0) * 2 + g) * 256 + v0);
            if (pos >= 0) raw[i] = x;
          }
        }
#pragma unroll
        for (int i = 0; i < 4 && i0 + i < 7; ++i) {
          const int part = t - 96 + 160 * (i0 + i), key = part & 31, v0 = (part >> 5) * 8;
          if (part < 1024) {
            const _Float16* h = reinterpret_cast<const _Float16*>(&raw[i]);
#pragma unroll
            for (int k = 0; k < 8; ++k) vt[v0 + k][key] = h[k];
          }
        }
        __builtin_amdgcn_sched_barrier(0);
      }
    } else {
      for (int part = t - 96; part < 1024; part += 160) {
        const int key = part & 31, v0 = (part >> 5) * 8;
        const int pos = keypos[key];
        uint4 raw = {};
        if (pos >= 0) raw = *reinterpret_cast<const uint4*>(q.c.v + ((size_t)pos * 2 + g) * 256 + v0);
        const _Float16* h = reinterpret_cast<const _Float16*>(&raw);
#pragma unroll
        for (int k = 0; k < 8; ++k) vt[v0 + k][key] = h[k];
      }
    }
    __syncthreads();
    if (second && IL2) {
      if (!live) {
#pragma unroll
        for (int nt = 0; nt < 2; ++nt)
#pragma unroll
          for (int z = 0; z < 8; ++z) score[mt * 16 + FragmentRow(z)][nt * 16 + (lane & 15)] = -INFINITY;
      } else {
        F8 high[2];
#pragma unroll
        for (int nt = 0; nt < 2; ++nt)
#pragma unroll
          for (int z = 0; z < 8; ++z) high[nt][z] = score[mt * 16 + FragmentRow(z)][nt * 16 + (lane & 15)];
        __builtin_amdgcn_sched_barrier(0);
        QkChains<D>(qf, q.c.k + ((size_t)max(pos0, 0) * 2 + g) * 256 + 128,
            q.c.k + ((size_t)max(pos1, 0) * 2 + g) * 256 + 128, high[0], high[1]);
        __builtin_amdgcn_sched_barrier(0);
        unsigned rz = rowpack;
        asm volatile("" : "+v"(rz));
#pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
          const int key = nt * 16 + (lane & 15), pos = nt ? pos1 : pos0;
#pragma unroll
          for (int z = 0; z < 8; ++z) {
            const int m = mt * 16 + FragmentRow(z), row = rz >> (4 * z) & 15;
            const bool visible = pos >= 0 && row < rows && (keymask[key] >> row & 1)
                && pos <= q.pos0 + r0 + row;
            score[m][key] = visible ? high[nt][z] * .0625f : -INFINITY;
          }
        }
      }
    } else if (second) {
#pragma unroll
      for (int nt = 0; nt < 2; ++nt) {
        const int key = nt * 16 + (lane & 15), pos = nt ? pos1 : pos0;
        if (!live) {
#pragma unroll
          for (int z = 0; z < 8; ++z) score[mt * 16 + FragmentRow(z)][key] = -INFINITY;
          continue;
        }
        F8 high;
#pragma unroll
        for (int z = 0; z < 8; ++z) high[z] = score[mt * 16 + FragmentRow(z)][key];
        unsigned rz = rowpack;
        asm volatile("" : "+v"(rz));
        const __half* kp = NOK ? a.q_hi + qi : q.c.k + ((size_t)max(pos, 0) * 2 + g) * 256 + 128;
        if constexpr ((F & kPipeP2) != 0) {
          H16 b = HalfFragment(kp);
#pragma unroll
          for (int c = 0; c < 8; ++c) {
            const H16 cur = b;
            if (c < 7) b = HalfFragment(kp + (c + 1) * 16);
            high = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(qf[c], cur, high);
            __builtin_amdgcn_sched_barrier(0);
          }
        } else {
#pragma unroll
          for (int c = 0; c < 8; ++c) {
            high = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(qf[c], HalfFragment(kp + c * 16), high);
            if (c % 4 == 3) __builtin_amdgcn_sched_barrier(0);
          }
        }
#pragma unroll
        for (int z = 0; z < 8; ++z) {
          const int m = mt * 16 + FragmentRow(z), row = rz >> (4 * z) & 15;
          const bool visible = pos >= 0 && row < rows && (keymask[key] >> row & 1)
              && pos <= q.pos0 + r0 + row;
          score[m][key] = visible ? high[z] * .0625f : -INFINITY;
        }
      }
    }
    __syncthreads();
    if (t < 192 && ((t >> 6) == 0 ? live0 : (t >> 6) == 1 ? live1 : live2)) {
      const int m = t >> 2, part = t & 3;
      float maximum = -INFINITY;
#pragma unroll
      for (int i = 0; i < 8; ++i) maximum = fmaxf(maximum, score[m][part * 8 + i]);
      maximum = fmaxf(maximum, __shfl_xor(maximum, 1));
      maximum = fmaxf(maximum, __shfl_xor(maximum, 2));
      const float old_max = maxima[m], new_max = fmaxf(old_max, maximum);
      const float base = new_max == -INFINITY ? 0.f : new_max;
      float sum = 0.f;
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        const int col = part * 8 + i;
        const float p = __expf(score[m][col] - base);
        phi[m][col] = (_Float16)p;
        sum += p;
      }
      sum += __shfl_xor(sum, 1);
      sum += __shfl_xor(sum, 2);
      if (part == 0) {
        const float s = old_max == -INFINITY ? 0.f : __expf(old_max - base);
        maxima[m] = new_max; scale[m] = s; sums[m] = sums[m] * s + sum;
      }
    }
    __syncthreads();
#pragma unroll
    for (int m3 = 0; m3 < 3; ++m3) {
      if (NOPV || !(m3 == 0 ? live0 : m3 == 1 ? live1 : live2)) continue;
      F8 high[2] = {};
#pragma unroll
      for (int k = 0; k < kUnionKeys; k += 16) {
        const H16 ah = *reinterpret_cast<const H16*>(&phi[m3 * 16 + (lane & 15)][k]);
#pragma unroll
        for (int d2 = 0; d2 < 2; ++d2) {
          const int d = wave * 32 + d2 * 16 + (lane & 15);
          const H16 b = *reinterpret_cast<const H16*>(VM ? vtm + PipeVtRow(d) + k : &vt[d][k]);
          high[d2] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(ah, b, high[d2]);
        }
      }
#pragma unroll
      for (int d2 = 0; d2 < 2; ++d2)
#pragma unroll
        for (int z = 0; z < 8; ++z)
          out[m3][d2][z] = out[m3][d2][z] * scale[m3 * 16 + FragmentRow(z)] + high[d2][z];
    }
  }
  if constexpr (KT) asm volatile("" :: "v"(sink));
  if constexpr (!Gate) {
#pragma unroll
    for (int m3 = 0; m3 < 3; ++m3)
#pragma unroll
      for (int d2 = 0; d2 < 2; ++d2)
#pragma unroll
        for (int z = 0; z < 8; ++z) {
          const int m = m3 * 16 + FragmentRow(z), row = m / 12, head = m % 12;
          const int d = wave * 32 + d2 * 16 + (lane & 15);
          if (row < rows)
            a.context[((size_t)(r0 + row) * 24 + g * 12 + head) * 256 + d] = out[m3][d2][z] / sums[m];
        }
    return;
  }
  auto gated = reinterpret_cast<_Float16(*)[256]>(smem);
  __syncthreads();
#pragma unroll
  for (int m3 = 0; m3 < 3; ++m3)
#pragma unroll
    for (int d2 = 0; d2 < 2; ++d2)
#pragma unroll
      for (int z = 0; z < 8; ++z) {
        const int m = m3 * 16 + FragmentRow(z), row = m / 12, h = g * 12 + m % 12;
        const int d = wave * 32 + d2 * 16 + (lane & 15);
        if (row < rows) {
          const float context = out[m3][d2][z] / sums[m];
          const float gate = Sigmoid(q.q[((size_t)(r0 + row) * 24 + h) * 512 + 256 + d]);
          const float x = context * gate;
          gated[m][d] = (_Float16)(RoundH(x) * __half2float(q.suh_o[h * 256 + d]));
        }
      }
  __syncthreads();
  for (int job = wave; job < 96; job += 8) {
    const int m = job >> 1, half = job & 1, row = m / 12, h = g * 12 + m % 12;
    if (row >= rows) continue;
    float v[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) v[i] = (float)gated[m][half * 128 + lane * 4 + i];
    Wht128(v);
#pragma unroll
    for (int i = 0; i < 4; ++i)
      q.xh_o[OutProjIndex(r0 + row, h * 256 + half * 128 + lane * 4 + i, q.pack_rows, q.pack_off)] =
          __float2half(v[i] * kHadNorm);
  }
}

template <bool Gate, int F, int W = 8, int D = 2>
__global__ void __launch_bounds__(256) __attribute__((amdgpu_waves_per_eu(W, W)))
QsaUnionPipeKernel(QsaPrefill a, QsaUnion u) { QsaUnionPipeBody<Gate, F, D>(a, u); }

// The same body in CU mode: all 8 waves of a workgroup share one CU and its
// L0, and workgroup barriers need no L0 invalidate. 2 workgroups per CU, as
// in WGP mode (LDS 64 KB per CU, 8 waves per SIMD).
template <bool Gate, int F, int D = 2>
__global__ void __launch_bounds__(256) __attribute__((amdgpu_waves_per_eu(8, 8), target("cumode")))
QsaUnionPipeCuKernel(QsaPrefill a, QsaUnion u) { QsaUnionPipeBody<Gate, F, D>(a, u); }
}  // namespace crucible::prefill
