// Partial last-layer pruning (idea from Gufo, gufo-org/gufo, MIT). CPU-visible, so the rule has a table test.
#pragma once
#include <algorithm>

namespace crucible {
// The first row of an R-row chunk whose last-layer output a reader needs: the head reads R-1 when predict is
// set, and the MTP warm-up reads rows from keep. It rounds down to a multiple of align; 0 (no pruning) when
// fewer than 9 rows would remain, since the bulk kernels need more than 8 rows.
constexpr int PruneFirstRow(int R, bool predict, int keep, int align) {
  const int needed = std::min(std::max(keep, 0), predict ? R - 1 : R);
  const int first = needed / align * align;
  return R - first > 8 ? first : 0;
}
}  // namespace crucible
