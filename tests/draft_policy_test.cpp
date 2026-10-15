#include <cassert>
#include <vector>
#include <random>
#include "../engine/runtime/draft_policy.h"

using crucible::LookupDraft;
std::vector<int> Brute(const std::vector<int>& h, int depth) {
  for (int len = std::min(32, (int)h.size() - 1); len >= 3; --len)
    for (int end = (int)h.size() - 1; end >= len; --end)
      if (std::equal(h.begin() + end - len, h.begin() + end, h.end() - len))
        return {h.begin() + end, h.begin() + std::min(end + depth, (int)h.size())};
  return {};
}
int main() {
  using crucible::ParseDraftSelection;
  using crucible::ValidDraftSelection;
  assert(ParseDraftSelection(" 0, 42,1939\n") == std::vector<int>({0, 42, 1939}));
  for (const auto* invalid : {"", "1,", "1,,2", "1junk", "1.5", "2147483648", "x"}) {
    bool caught = false;
    try { ParseDraftSelection(invalid); } catch (const std::invalid_argument&) { caught = true; }
    assert(caught);
  }
  assert(ValidDraftSelection({0, 1, 1939}, 1940));
  assert(!ValidDraftSelection({}, 1940));
  assert(!ValidDraftSelection({-1}, 1940));
  assert(!ValidDraftSelection({1940}, 1940));
  assert(!ValidDraftSelection({1, 1}, 1940));
  assert(!ValidDraftSelection({2, 1}, 1940));
  assert(LookupDraft({}, 4).empty());
  assert(LookupDraft({1, 2, 3}, 4).empty());
  assert(LookupDraft({1, 2, 3, 4, 5, 1, 2, 3}, 4) == std::vector<int>({4, 5, 1, 2}));
  assert(LookupDraft({1, 2, 3, 4, 5, 1, 2, 3}, 1) == std::vector<int>({4}));
  assert(LookupDraft({1, 2, 7, 1, 2}, 4).empty());
  assert(LookupDraft({1, 2, 3, 4, 1, 2, 3, 9, 1, 2, 3}, 2) == std::vector<int>({9, 1}));
  assert(LookupDraft({0, 1, 2, 3, 8, 1, 2, 3, 9, 0, 1, 2, 3}, 2) == std::vector<int>({8, 1}));
  assert(LookupDraft({1, 2, 3, 4, 1, 2, 3}, 0).empty());
  assert(LookupDraft({1, 2, 3, 4, 1, 2, 3}, 4, 4).empty());
  std::mt19937 rng(42);
  for (int trial = 0; trial < 10000; ++trial) {
    std::vector<int> history(rng() % 500);
    for (auto& token : history) token = rng() % 8;
    const int depth = 1 + rng() % 7;
    assert(LookupDraft(history, depth) == Brute(history, depth));
  }
}
