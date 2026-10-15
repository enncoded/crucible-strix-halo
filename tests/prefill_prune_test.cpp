// The first last-layer row that a pruned prefill chunk must compute (PruneFirstRow).
#include "../engine/runtime/prefill_prune.h"
#include <cassert>

using crucible::PruneFirstRow;

int main() {
  struct Case { int R; bool predict; int keep; int align; int first; };
  constexpr Case cases[] = {
    {4096, false, -5, 128, 0},       // keep < 0: every row is read
    {4096, false, 5000, 128, 0},     // keep > R without predict: no row is read, but fewer than 9 rows remain
    {4096, true, 5000, 128, 3968},   // keep > R with predict: the head row R-1 rounds down to 3968
    {1, true, 1, 128, 0},            // predict with R = 1
    {136, true, 136, 128, 0},        // R - first = 8: no pruning
    {137, true, 137, 128, 128},      // R - first = 9: prune from 128
    {4096, false, 3840, 128, 3840},  // an exact multiple of 128
    {4096, false, 3841, 128, 3840},  // one past it rounds down, never up
    {4096, false, 3839, 128, 3712},
    {4000, true, 3744, 128, 3712},   // the MTP warm-up of the 4,000-token proof
    {2500, true, 2500, 128, 2432},   // spec 0: only the head reads
  };
  for (const Case& c : cases) assert(PruneFirstRow(c.R, c.predict, c.keep, c.align) == c.first);
  static_assert(PruneFirstRow(4096, true, 3840, 128) == 3840);
}
