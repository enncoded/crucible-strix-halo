#include "../engine/runtime/slot_checkpoints.h"
#include <cassert>
#include <numeric>
#include <span>
using namespace crucible;

static std::vector<int> Tokens(size_t count, int first = 1) {
  std::vector<int> ids(count); std::iota(ids.begin(), ids.end(), first); return ids;
}
static std::span<const int> Below(const std::vector<int>& ids, size_t length) { return {ids.data(), length}; }
static SlotCheckpoint Point(const std::vector<int>& ids, size_t length, CheckpointKind kind, uint64_t owner) {
  return {std::vector<int>(ids.begin(), ids.begin() + std::ptrdiff_t(length)), {}, 0, false, kind, owner};
}

int main() {
  const auto prompt = Tokens(100);
  SlotCheckpoints c(2);
  // Three buffers: the turn start, the last grid point and the prompt end of one request.
  for (const auto& [length, kind] : {std::pair{20, CheckpointKind::Start}, {40, CheckpointKind::Grid},
                                     {60, CheckpointKind::End}}) {
    const auto victim = c.Victim(0, Below(prompt, size_t(length)), kind, 1);
    assert(victim);
    c.Store(0, *victim, Point(prompt, size_t(length), kind, 1));
  }
  // A later grid point of the same request replaces its earlier one.
  assert(*c.Victim(0, Below(prompt, 50), CheckpointKind::Grid, 1) == 1);
  c.Store(0, 1, Point(prompt, 50, CheckpointKind::Grid, 1));
  // The slot already holds that state: nothing to copy.
  assert(!c.Victim(0, Below(prompt, 60), CheckpointKind::Start, 2));
  // Same length, other key: not the same state. Under the invariant (every point is a prefix of
  // the slot's rows, Keep) this cannot happen; if it does, the stale point is the one to replace.
  auto stale = prompt; stale[7] += 1 << 20;
  assert(*c.Victim(0, Below(stale, 60), CheckpointKind::Start, 2) == 2);
  // The longest checkpoint that a prompt extends by at least one token.
  assert(*c.Best(0, prompt) == 2 && c.At(0, 2).hist.size() == 60);
  auto split = prompt; split[55] = -1;  // a token split inside the previous answer
  assert(c.At(0, *c.Best(0, split)).hist.size() == 50 && c.Common(0, split) == 50);
  assert(!c.Best(0, std::vector<int>(prompt.begin(), prompt.begin() + 20)));  // the last row must run
  assert(!c.Best(1, prompt) && c.Common(1, prompt) == 0);
  auto other = prompt; other[3] = -2;
  assert(!c.Best(0, other));
  // A new request: the shortest checkpoint leaves first.
  assert(*c.Victim(0, Below(prompt, 70), CheckpointKind::End, 2) == 0);
  // A restore at 50 keeps the rows below it; a reset or host restore keeps none.
  c.Keep(0, 50);
  assert(c.At(0, *c.Best(0, prompt)).hist.size() == 50);
  c.Keep(0, 0);
  assert(!c.Best(0, prompt) && *c.Victim(0, Below(prompt, 10), CheckpointKind::End, 3) == 0);
  bool threw = false;
  try { c.Store(0, 3, Point(prompt, 10, CheckpointKind::End, 3)); } catch (const std::invalid_argument&) { threw = true; }
  assert(threw);
  // Points compare cache keys: the same tokens under another key (another image) do not match.
  auto keyed = Point(prompt, 30, CheckpointKind::End, 4);
  keyed.key = keyed.hist;
  keyed.key[5] += 1 << 20;
  c.Store(1, 0, keyed);
  assert(!c.Best(1, prompt) && *c.Best(1, keyed.key) == 0 && c.At(1, 0).hist.size() == 30);
  auto stored = c.At(1, 0).key;
  stored.insert(stored.end(), prompt.begin() + 30, prompt.end());
  assert(c.Common(1, stored) == 30);
  c.Store(0, 0, Point(prompt, 10, CheckpointKind::End, 5));  // an empty key is the tokens
  assert(c.At(0, 0).key == c.At(0, 0).hist);
  threw = false;
  keyed.key.pop_back();
  try { c.Store(1, 1, keyed); } catch (const std::invalid_argument&) { threw = true; }
  assert(threw);
  return 0;
}
