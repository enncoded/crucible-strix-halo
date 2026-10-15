// LookupIndex must propose exactly what LookupDraft proposes over the same history (drafts change only speed,
// but the serve default and its measurements assume the same drafts at a fraction of the CPU time).
#include <cassert>
#include <cstdio>
#include <random>
#include <vector>
#include "../engine/runtime/draft_policy.h"

using crucible::LookupDraft;
using crucible::LookupIndex;

int main() {
  {
    LookupIndex index(3);
    assert(index.Empty() && index.Draft(4).empty());
    index.Append(std::vector<int>{1, 2, 3, 4, 5, 1, 2, 3});
    assert(!index.Empty() && index.Back() == 3 && index.Size() == 8);
    assert(index.Draft(4) == std::vector<int>({4, 5, 1, 2}));
    assert(index.Draft(0).empty());
    index.Append(std::vector<int>{9});
    assert(index.Draft(4).empty());
  }
  std::mt19937 rng(7);
  for (int trial = 0; trial < 4000; ++trial) {
    const int alphabet = 2 + int(rng() % 6), min_match = 1 + int(rng() % 9);
    LookupIndex index(min_match);
    std::vector<int> history;
    const int steps = int(rng() % 120);
    for (int step = 0; step < steps; ++step) {
      // Prompts arrive as one block, verified tokens in small groups.
      std::vector<int> tokens(step == 0 ? rng() % 300 : 1 + rng() % 8);
      for (auto& token : tokens) token = int(rng() % alphabet);
      history.insert(history.end(), tokens.begin(), tokens.end());
      index.Append(tokens);
      const int depth = int(rng() % 9);
      assert(index.Size() == history.size());
      assert(index.Draft(depth) == LookupDraft(history, depth, min_match));
    }
  }
  // Long runs of one token stress the 32-token match cap.
  LookupIndex runs(8);
  std::vector<int> same(5000, 11);
  runs.Append(same);
  assert(runs.Draft(7) == LookupDraft(same, 7, 8));
  std::puts("PASS: LookupIndex proposes the LookupDraft drafts");
}
