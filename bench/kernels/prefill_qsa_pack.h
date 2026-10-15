// The QSA prefill query pack before an earlier change (unchanged from engine/layers/kernels/prefill_qsa.h): the fp16 halves of
// the normalized fp32 query that QsaPrepKernel stores. The bench reference of bench/qsa_prep_half.hip,
// bench/prefill_qsa.hip and tests/qsa_prep_half_test.hip. The engine runs QsaPrefillQueryKernel since train 20.
#pragma once
#include "../../engine/layers/kernels/prefill_qsa.h"

namespace crucible::prefill {
__global__ void QsaPrefillPackKernel(QsaPrefill a) {
  const size_t i = (size_t)blockIdx.x * 256 + threadIdx.x;
  if (i >= (size_t)a.input.R * kQsaOutWidth) return;
  const float q = a.input.q[(i / 256) * 512 + i % 256];
  a.q_hi[i] = __float2half(q);
  if (a.q_lo) a.q_lo[i] = __float2half((q - __half2float(a.q_hi[i])) * kResidualScale);
}
}  // namespace crucible::prefill
