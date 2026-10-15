// Bench-only copy of QsaUnionFlashKernel<false> (no-lo) with knockouts, to
// split its time between V transpose, QK, softmax and PV. KO = 0 must stay
// bitwise equal to the production kernel.
#pragma once
#include "../engine/layers/kernels/prefill_qsa_union.h"

namespace crucible::prefill {
enum : int {
  kKoVLoad = 1,      // no global V loads (zeros go through the transpose)
  kKoVAll = 2,       // no V loads and no V^T stores
  kKoKLoad = 4,      // QK B operand from registers, not global K
  kKoQLoad = 8,      // QK A operand from registers, not global q
  kKoQk = 16,        // no QK at all (scores from the key mask)
  kKoPv = 32,        // no PV WMMA
  kKoExp = 64,       // softmax without exp
};

template <int KO>
__global__ void __launch_bounds__(256) QsaUnionKoKernel(QsaPrefill a, QsaUnion u) {
  __shared__ float score[48][kUnionKeys], maxima[48], sums[48], scale[48];
  __shared__ __attribute__((aligned(16))) _Float16 phi[48][kUnionPad];
  __shared__ __attribute__((aligned(16))) _Float16 vt[256][kUnionPad];
  __shared__ int keypos[kUnionKeys], keymask[kUnionKeys], stepmask;
  const QsaArgs q = a.input;
  const int tile = blockIdx.x, g = blockIdx.y, r0 = tile * kUnionRows;
  const int t = threadIdx.x, lane = t & 31, wave = t >> 5;
  const int rows = min(kUnionRows, q.R - r0);
  const int n = u.count[tile] * 4;
  const int* list = u.list + (size_t)tile * kUnionCap;
  if (t < 48) { maxima[t] = -INFINITY; sums[t] = 0.f; }
  F8 out[3][2] = {};
  const int mt = wave >> 1, nt = wave & 1;
  const int qm = mt * 16 + (lane & 15), qrow = qm / 12, qhead = qm % 12;
  const bool qvalid = qrow < rows;
  const size_t qi = ((size_t)(r0 + min(qrow, rows - 1)) * 24 + g * 12 + qhead) * 256;
  H16 qreg = {}, kreg = {};
  if constexpr ((KO & kKoQLoad) != 0) qreg = HalfFragment(a.q_hi + qi);
  for (int start = 0; start < n; start += kUnionKeys) {
    __syncthreads();
    if (t < kUnionKeys) {
      const int idx = start + t;
      const int e = idx < n ? list[idx >> 2] : 0;
      const int pos = (e & 0xffffff) * 4 + (idx & 3);
      keypos[t] = idx < n && pos < q.pos0 + q.R ? pos : -1;
      int km = idx < n ? (e >> 24) & 0xff : 0;
      keymask[t] = km;
#pragma unroll
      for (int o = 1; o < 32; o <<= 1) km |= __shfl_xor(km, o);
      if (t == 0) stepmask = km;
    }
    __syncthreads();
    const int sm = stepmask;
    const bool live0 = sm & 3, live1 = sm & 6, live2 = sm & 12;
    if constexpr ((KO & kKoVAll) == 0) {
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int part = t + i * 256, key = part & 31, d0 = (part >> 5) * 8;
        const int pos = keypos[key];
        uint4 raw = {};
        if constexpr ((KO & kKoVLoad) != 0) raw = {(unsigned)start & 0x3ff, 0u, (unsigned)key, 0u};
        else if (pos >= 0) raw = *reinterpret_cast<const uint4*>(q.c.v + ((size_t)pos * 2 + g) * 256 + d0);
        const _Float16* h = reinterpret_cast<const _Float16*>(&raw);
#pragma unroll
        for (int k = 0; k < 8; ++k) vt[d0 + k][key] = h[k];
      }
    }
    if (wave < 6 && !(mt == 0 ? live0 : mt == 1 ? live1 : live2)) {
#pragma unroll
      for (int z = 0; z < 8; ++z) score[mt * 16 + FragmentRow(z)][nt * 16 + (lane & 15)] = -INFINITY;
    } else if (wave < 6) {
      const int key = nt * 16 + (lane & 15), pos = keypos[key];
      const __half* kp = q.c.k + ((size_t)max(pos, 0) * 2 + g) * 256;
      F8 high = {};
      if constexpr ((KO & kKoKLoad) != 0) kreg = HalfFragment(kp);
      if constexpr ((KO & kKoQk) == 0) {
#pragma unroll 4
        for (int k = 0; k < 256; k += 16) {
          H16 ah, b;
          if constexpr ((KO & kKoQLoad) != 0) ah = qreg;
          else ah = qvalid ? HalfFragment(a.q_hi + qi + k) : H16{};
          if constexpr ((KO & kKoKLoad) != 0) b = kreg;
          else b = pos >= 0 ? HalfFragment(kp + k) : H16{};
          high = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(ah, b, high);
        }
      } else {
#pragma unroll
        for (int z = 0; z < 8; ++z) high[z] = (float)(pos & 7) * .01f;
      }
#pragma unroll
      for (int z = 0; z < 8; ++z) {
        const int m = mt * 16 + FragmentRow(z), row = m / 12;
        const bool visible = pos >= 0 && row < rows && (keymask[key] >> row & 1)
            && pos <= q.pos0 + r0 + row;
        score[m][key] = visible ? high[z] * .0625f : -INFINITY;
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
        const float p = (KO & kKoExp) ? fmaxf(score[m][col] - base + 1.f, 0.f) : __expf(score[m][col] - base);
        phi[m][col] = (_Float16)p;
        sum += p;
      }
      sum += __shfl_xor(sum, 1);
      sum += __shfl_xor(sum, 2);
      if (part == 0) {
        const float s = old_max == -INFINITY ? 0.f
            : (KO & kKoExp) ? fmaxf(old_max - base + 1.f, 0.f) : __expf(old_max - base);
        maxima[m] = new_max; scale[m] = s; sums[m] = sums[m] * s + sum;
      }
    }
    __syncthreads();
#pragma unroll
    for (int m3 = 0; m3 < 3; ++m3) {
      if (!(m3 == 0 ? live0 : m3 == 1 ? live1 : live2)) continue;
      F8 high[2] = {};
      if constexpr ((KO & kKoPv) == 0) {
#pragma unroll
        for (int k = 0; k < kUnionKeys; k += 16) {
          const H16 ah = *reinterpret_cast<const H16*>(&phi[m3 * 16 + (lane & 15)][k]);
#pragma unroll
          for (int d2 = 0; d2 < 2; ++d2) {
            const H16 b = *reinterpret_cast<const H16*>(&vt[wave * 32 + d2 * 16 + (lane & 15)][k]);
            high[d2] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(ah, b, high[d2]);
          }
        }
      } else {
#pragma unroll
        for (int d2 = 0; d2 < 2; ++d2)
#pragma unroll
          for (int z = 0; z < 8; ++z)
            high[d2][z] = (float)vt[wave * 32 + d2 * 16 + (lane & 15)][z] + (float)phi[m3 * 16 + z][lane & 15];
      }
#pragma unroll
      for (int d2 = 0; d2 < 2; ++d2)
#pragma unroll
        for (int z = 0; z < 8; ++z)
          out[m3][d2][z] = out[m3][d2][z] * scale[m3 * 16 + FragmentRow(z)] + high[d2][z];
    }
  }
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
}

// Bitwise-preserving restructures of the no-lo union kernel (bench candidates).
// VAR bit 1: prefetch the next step's list entries one step ahead.
// VAR bit 2: waves 6..7 load and transpose V while waves 0..5 run QK.
// VAR bit 4: waves 0..2 each run both key tiles of one M tile (one q load
//            feeds two WMMA chains); waves 4..7 load and transpose V.
// VAR bit 8: V loads in batches of 4 instead of 8; bit 16: QK unroll 2.
enum : int { kVarList = 1, kVarVWaves = 2, kVarQk2 = 4, kVarVb4 = 8, kVarU2 = 16 };

template <int VAR>
__global__ void __launch_bounds__(256) QsaUnionVarKernel(QsaPrefill a, QsaUnion u) {
  static_assert(!((VAR & kVarVWaves) && (VAR & kVarQk2)));
  __shared__ float score[48][kUnionKeys], maxima[48], sums[48], scale[48];
  __shared__ __attribute__((aligned(16))) _Float16 phi[48][kUnionPad];
  __shared__ __attribute__((aligned(16))) _Float16 vt[256][kUnionPad];
  __shared__ int keypos[kUnionKeys], keymask[kUnionKeys], stepmask;
  const QsaArgs q = a.input;
  const int tile = blockIdx.x, g = blockIdx.y, r0 = tile * kUnionRows;
  const int t = threadIdx.x, lane = t & 31, wave = t >> 5;
  const int rows = min(kUnionRows, q.R - r0);
  const int n = u.count[tile] * 4;
  const int* list = u.list + (size_t)tile * kUnionCap;
  if (t < 48) { maxima[t] = -INFINITY; sums[t] = 0.f; }
  F8 out[3][2] = {};
  constexpr bool kQk2 = (VAR & kVarQk2) != 0;
  constexpr int kQkWaves = kQk2 ? 3 : 6;
  // V^T producers: all threads, waves 6..7, or waves 4..7.
  constexpr int kVFirst = (VAR & kVarVWaves) ? 192 : kQk2 ? 128 : 0;
  constexpr int kVThreads = 256 - kVFirst, kVParts = 1024 / kVThreads, kVB = (VAR & kVarVb4) ? 4 : 8;
  const int mt = kQk2 ? wave : wave >> 1, nt = kQk2 ? 0 : wave & 1;
  const int qm = mt * 16 + (lane & 15), qrow = qm / 12, qhead = qm % 12;
  const bool qvalid = qrow < rows;
  const size_t qi = ((size_t)(r0 + min(qrow, rows - 1)) * 24 + g * 12 + qhead) * 256;
  int enext = 0;
  if constexpr ((VAR & kVarList) != 0) if (t < kUnionKeys) enext = t < n ? list[t >> 2] : 0;
  for (int start = 0; start < n; start += kUnionKeys) {
    __syncthreads();
    if (t < kUnionKeys) {
      const int idx = start + t;
      int e;
      if constexpr ((VAR & kVarList) != 0) {
        e = enext;
        const int nidx = idx + kUnionKeys;
        enext = nidx < n ? list[nidx >> 2] : 0;
      } else {
        e = idx < n ? list[idx >> 2] : 0;
      }
      const int pos = (e & 0xffffff) * 4 + (idx & 3);
      keypos[t] = idx < n && pos < q.pos0 + q.R ? pos : -1;
      int km = idx < n ? (e >> 24) & 0xff : 0;
      keymask[t] = km;
#pragma unroll
      for (int o = 1; o < 32; o <<= 1) km |= __shfl_xor(km, o);
      if (t == 0) stepmask = km;
    }
    __syncthreads();
    const int sm = stepmask;
    const bool live0 = sm & 3, live1 = sm & 6, live2 = sm & 12;
    if (t >= kVFirst) {
      const int tv = t - kVFirst;
#pragma unroll
      for (int i0 = 0; i0 < kVParts; i0 += kVB) {
        uint4 raw[kVB];
#pragma unroll
        for (int j = 0; j < kVB && i0 + j < kVParts; ++j) {
          const int part = tv + (i0 + j) * kVThreads, key = part & 31, d0 = (part >> 5) * 8;
          const int pos = keypos[key];
          raw[j] = {};
          if (pos >= 0) raw[j] = *reinterpret_cast<const uint4*>(q.c.v + ((size_t)pos * 2 + g) * 256 + d0);
        }
#pragma unroll
        for (int j = 0; j < kVB && i0 + j < kVParts; ++j) {
          const int part = tv + (i0 + j) * kVThreads, key = part & 31, d0 = (part >> 5) * 8;
          const _Float16* h = reinterpret_cast<const _Float16*>(&raw[j]);
#pragma unroll
          for (int k = 0; k < 8; ++k) vt[d0 + k][key] = h[k];
        }
      }
    }
    if (wave < kQkWaves) {
      const bool live = mt == 0 ? live0 : mt == 1 ? live1 : live2;
      constexpr int NT = kQk2 ? 2 : 1;
      if (!live) {
#pragma unroll
        for (int j = 0; j < NT; ++j)
#pragma unroll
          for (int z = 0; z < 8; ++z) score[mt * 16 + FragmentRow(z)][(nt + j) * 16 + (lane & 15)] = -INFINITY;
      } else {
        int pos[NT];
        const __half* kp[NT];
#pragma unroll
        for (int j = 0; j < NT; ++j) {
          pos[j] = keypos[(nt + j) * 16 + (lane & 15)];
          kp[j] = q.c.k + ((size_t)max(pos[j], 0) * 2 + g) * 256;
        }
        F8 high[NT] = {};
        constexpr int kU = (VAR & kVarU2) ? 2 : 4;
#pragma unroll kU
        for (int k = 0; k < 256; k += 16) {
          const H16 ah = qvalid ? HalfFragment(a.q_hi + qi + k) : H16{};
#pragma unroll
          for (int j = 0; j < NT; ++j) {
            const H16 b = pos[j] >= 0 ? HalfFragment(kp[j] + k) : H16{};
            high[j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(ah, b, high[j]);
          }
        }
#pragma unroll
        for (int j = 0; j < NT; ++j) {
          const int key = (nt + j) * 16 + (lane & 15);
#pragma unroll
          for (int z = 0; z < 8; ++z) {
            const int m = mt * 16 + FragmentRow(z), row = m / 12;
            const bool visible = pos[j] >= 0 && row < rows && (keymask[key] >> row & 1)
                && pos[j] <= q.pos0 + r0 + row;
            score[m][key] = visible ? high[j][z] * .0625f : -INFINITY;
          }
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
      if (!(m3 == 0 ? live0 : m3 == 1 ? live1 : live2)) continue;
      F8 high[2] = {};
#pragma unroll
      for (int k = 0; k < kUnionKeys; k += 16) {
        const H16 ah = *reinterpret_cast<const H16*>(&phi[m3 * 16 + (lane & 15)][k]);
#pragma unroll
        for (int d2 = 0; d2 < 2; ++d2) {
          const H16 b = *reinterpret_cast<const H16*>(&vt[wave * 32 + d2 * 16 + (lane & 15)][k]);
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
}

// K staged once per step in LDS (aliasing V^T) instead of three per-lane
// fragment loads per key tile. VAR bit 1: list prefetch. VAR bit 2: issue the
// V loads before QK and hold them in registers (else after QK).
template <int VAR>
__global__ void __launch_bounds__(256) QsaUnionKldsKernel(QsaPrefill a, QsaUnion u) {
  constexpr int kKPad = 256 + 8;
  static_assert(kUnionKeys * kKPad <= 256 * kUnionPad);
  __shared__ float score[48][kUnionKeys], maxima[48], sums[48], scale[48];
  __shared__ __attribute__((aligned(16))) _Float16 phi[48][kUnionPad];
  __shared__ __attribute__((aligned(16))) _Float16 vt[256][kUnionPad];
  __shared__ int keypos[kUnionKeys], keymask[kUnionKeys], stepmask;
  _Float16 (*kb)[kKPad] = reinterpret_cast<_Float16 (*)[kKPad]>(&vt[0][0]);
  const QsaArgs q = a.input;
  const int tile = blockIdx.x, g = blockIdx.y, r0 = tile * kUnionRows;
  const int t = threadIdx.x, lane = t & 31, wave = t >> 5;
  const int rows = min(kUnionRows, q.R - r0);
  const int n = u.count[tile] * 4;
  const int* list = u.list + (size_t)tile * kUnionCap;
  if (t < 48) { maxima[t] = -INFINITY; sums[t] = 0.f; }
  F8 out[3][2] = {};
  const int mt = wave >> 1, nt = wave & 1;
  const int qm = mt * 16 + (lane & 15), qrow = qm / 12, qhead = qm % 12;
  const bool qvalid = qrow < rows;
  const size_t qi = ((size_t)(r0 + min(qrow, rows - 1)) * 24 + g * 12 + qhead) * 256;
  int enext = 0;
  if constexpr ((VAR & 1) != 0) if (t < kUnionKeys) enext = t < n ? list[t >> 2] : 0;
  for (int start = 0; start < n; start += kUnionKeys) {
    __syncthreads();
    if (t < kUnionKeys) {
      const int idx = start + t;
      int e;
      if constexpr ((VAR & 1) != 0) {
        e = enext;
        const int nidx = idx + kUnionKeys;
        enext = nidx < n ? list[nidx >> 2] : 0;
      } else {
        e = idx < n ? list[idx >> 2] : 0;
      }
      const int pos = (e & 0xffffff) * 4 + (idx & 3);
      keypos[t] = idx < n && pos < q.pos0 + q.R ? pos : -1;
      int km = idx < n ? (e >> 24) & 0xff : 0;
      keymask[t] = km;
#pragma unroll
      for (int o = 1; o < 32; o <<= 1) km |= __shfl_xor(km, o);
      if (t == 0) stepmask = km;
    }
    __syncthreads();
    const int sm = stepmask;
    const bool live0 = sm & 3, live1 = sm & 6, live2 = sm & 12;
    // K rows: 32 keys x 512 bytes, 16-byte pieces; absent keys are zero.
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int part = t + i * 256, key = part >> 5, d0 = (part & 31) * 8;
      const int pos = keypos[key];
      uint4 raw = {};
      if (pos >= 0) raw = *reinterpret_cast<const uint4*>(q.c.k + ((size_t)pos * 2 + g) * 256 + d0);
      *reinterpret_cast<uint4*>(&kb[key][d0]) = raw;
    }
    uint4 vraw[4];
    auto vload = [&] {
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int part = t + i * 256, key = part & 31, d0 = (part >> 5) * 8;
        const int pos = keypos[key];
        vraw[i] = {};
        if (pos >= 0) vraw[i] = *reinterpret_cast<const uint4*>(q.c.v + ((size_t)pos * 2 + g) * 256 + d0);
      }
    };
    if constexpr ((VAR & 2) != 0) vload();
    __syncthreads();
    if (wave < 6 && !(mt == 0 ? live0 : mt == 1 ? live1 : live2)) {
#pragma unroll
      for (int z = 0; z < 8; ++z) score[mt * 16 + FragmentRow(z)][nt * 16 + (lane & 15)] = -INFINITY;
    } else if (wave < 6) {
      const int key = nt * 16 + (lane & 15), pos = keypos[key];
      F8 high = {};
#pragma unroll 4
      for (int k = 0; k < 256; k += 16) {
        const H16 ah = qvalid ? HalfFragment(a.q_hi + qi + k) : H16{};
        const H16 b = *reinterpret_cast<const H16*>(&kb[key][k]);
        high = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(ah, b, high);
      }
#pragma unroll
      for (int z = 0; z < 8; ++z) {
        const int m = mt * 16 + FragmentRow(z), row = m / 12;
        const bool visible = pos >= 0 && row < rows && (keymask[key] >> row & 1)
            && pos <= q.pos0 + r0 + row;
        score[m][key] = visible ? high[z] * .0625f : -INFINITY;
      }
    }
    __syncthreads();
    if constexpr ((VAR & 2) == 0) vload();
    // V^T into the same LDS (QK no longer reads K).
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int part = t + i * 256, key = part & 31, d0 = (part >> 5) * 8;
      const _Float16* h = reinterpret_cast<const _Float16*>(&vraw[i]);
#pragma unroll
      for (int k = 0; k < 8; ++k) vt[d0 + k][key] = h[k];
    }
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
      if (!(m3 == 0 ? live0 : m3 == 1 ? live1 : live2)) continue;
      F8 high[2] = {};
#pragma unroll
      for (int k = 0; k < kUnionKeys; k += 16) {
        const H16 ah = *reinterpret_cast<const H16*>(&phi[m3 * 16 + (lane & 15)][k]);
#pragma unroll
        for (int d2 = 0; d2 < 2; ++d2) {
          const H16 b = *reinterpret_cast<const H16*>(&vt[wave * 32 + d2 * 16 + (lane & 15)][k]);
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
}
}  // namespace crucible::prefill
