// Wave32 WMMA primitives. High/low expansions preserve fp32 state contractions.
#pragma once
#include "common.h"

namespace crucible::prefill {
using H16 = _Float16 __attribute__((ext_vector_type(16)));
using F8 = float __attribute__((ext_vector_type(8)));
constexpr float kResidualScale = 4096.f;

struct Fragment { H16 hi, lo; };

template <class Load>
__device__ __forceinline__ Fragment LoadFragment(Load load) {
  Fragment f;
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    const float x = load(i);
    f.hi[i] = (_Float16)x;
    // Scaling keeps the low part normal even for normalized q/k values.
    f.lo[i] = (_Float16)((x - (float)f.hi[i]) * kResidualScale);
  }
  return f;
}

__device__ __forceinline__ F8 Mma(Fragment a, Fragment b, F8 hh, F8& cross) {
  hh = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a.hi, b.hi, hh);
  cross = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a.hi, b.lo, cross);
  cross = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a.lo, b.hi, cross);
  return hh;
}

// A is row-major MxK, B is a callable (k,n). Each wave owns one 16x16 tile.
// The omitted low*low term is at most about 6e-8 of |A| |B|; this is an
// approximation to fp32, not a bitwise fp32 emulation.
template <class LoadA, class LoadB>
__device__ __forceinline__ F8 Product(int k, LoadA la, LoadB lb) {
  const int lane = threadIdx.x & 31;
  F8 hi = {}, low = {};
  for (int begin = 0; begin < k; begin += 16) {
    const Fragment a = LoadFragment([&](int i) { return la(lane & 15, begin + i); });
    const Fragment b = LoadFragment([&](int i) { return lb(begin + i, lane & 15); });
    hi = Mma(a, b, hi, low);
  }
  return hi + low * (1.f / kResidualScale);
}

__device__ __forceinline__ int FragmentRow(int element) {
  return 2 * element + ((threadIdx.x & 31) >> 4);
}
}  // namespace crucible::prefill
