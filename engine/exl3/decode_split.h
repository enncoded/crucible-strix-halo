// Decode split policy, shared by plan building and the front tile order.
#pragma once
#include <utility>
#include <vector>

namespace crucible {

// A fixed span and divisor keep reductions independent of activation row count.
inline int SplitFor(int A, int target) {
  for (int d = 1; d <= A; ++d)
    if (A % d == 0 && A / d <= target) return d;
  return A;
}

// One span target per dense decode group: the longest span that still gives
// enough waves to cover the GPU. io holds (I, O) per matrix. The loader calls
// this too, because the front tile order depends on each matrix's split.
inline int DecodeTarget(const std::vector<std::pair<int, int>>& io) {
  int target = 20;
  for (int t : {160, 80, 40, 20}) {
    long waves = 0;
    for (const auto& [I, O] : io) waves += (long)(O / 16) * SplitFor(I / 16, t);
    target = t;
    if (waves >= 900) break;
  }
  return target;
}

}  // namespace crucible
