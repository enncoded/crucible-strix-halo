// Measurement only: counts the distinct routed experts of one MoE call so the
// spec harness can compare verify bytes with the bandwidth floor.
#pragma once

#include "../model/flashnext/geometry.h"

namespace crucible {

// acc[2R] += distinct experts among the R * kTopK selections; acc[2R+1] += 1.
__global__ void CountExpertsKernel(const int* __restrict__ sel, int R,
                                   unsigned long long* __restrict__ acc) {
  if (threadIdx.x != 0) return;
  int u = 0;
  for (int i = 0; i < R * kTopK; ++i) {
    bool seen = false;
    for (int j = 0; j < i && !seen; ++j) seen = sel[j] == sel[i];
    u += !seen;
  }
  atomicAdd(acc + 2 * R, (unsigned long long)u);
  atomicAdd(acc + 2 * R + 1, 1ull);
}

}  // namespace crucible
