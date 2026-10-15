// Per-block dense work lists. A chain is one (column block, split) pair and
// always stays on one wave in tile order, so any chain-to-block assignment
// gives bitwise equal partial sums. Contiguous chain ranges per block keep
// every block within one chain of the mean, where strided items gave the
// low blocks every remainder item.
#pragma once
#include <vector>

#include "plans.h"

namespace crucible {

struct BalancedItems {
  GemvItem* items = nullptr;  // grouped by block
  int* offsets = nullptr;     // [blocks + 1]
  int blocks = 0, n = 0;
};

// Items never cross a job or split boundary and hold at most eight columns.
// front: items also stay inside one eight-column group (one front chunk).
inline std::vector<GemvItem> BalancedItemList(const std::vector<LinJob>& jobs, int blocks,
                                              std::vector<int>& offsets, bool front = false) {
  struct Chain { int job, s, col; };
  std::vector<Chain> chains;
  for (int j = 0; j < (int)jobs.size(); ++j)
    for (int s = 0; s < jobs[j].S; ++s)
      for (int c = 0; c < jobs[j].O / 16; ++c) chains.push_back({j, s, c});
  const long n = (long)chains.size();
  std::vector<GemvItem> items;
  offsets.assign(blocks + 1, 0);
  for (int b = 0; b < blocks; ++b) {
    offsets[b] = (int)items.size();
    const long c0 = n * b / blocks, c1 = n * (b + 1) / blocks;
    for (long c = c0; c < c1;) {
      const Chain& h = chains[c];
      int nb = 1;
      while (nb < 8 && c + nb < c1 && chains[c + nb].job == h.job && chains[c + nb].s == h.s &&
             chains[c + nb].col == h.col + nb && !(front && chains[c + nb].col % 8 == 0))
        ++nb;
      items.push_back({h.job, h.s, h.col, nb});
      c += nb;
    }
  }
  offsets[blocks] = (int)items.size();
  return items;
}

inline BalancedItems BuildBalancedItems(const std::vector<LinJob>& jobs, int blocks, bool front = false) {
  std::vector<int> offsets;
  const std::vector<GemvItem> items = BalancedItemList(jobs, blocks, offsets, front);
  return {Upload(items), Upload(offsets), blocks, (int)items.size()};
}

inline void FreeBalancedItems(BalancedItems& b) {
  if (b.items) HIP_OK(hipFree(b.items));
  if (b.offsets) HIP_OK(hipFree(b.offsets));
  b = {};
}

}  // namespace crucible
