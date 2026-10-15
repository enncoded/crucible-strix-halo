// one list of a sequence's fixed-size state. SequenceState checks the list's total
// against its FixedBytes() at construction, so a field that one list gains alone stops the engine at start.
#include "../engine/model/flashnext/fixed_state.h"
#include <cassert>
#include <map>
#include <vector>
using namespace crucible;

struct FakeState {
  std::map<int, float*> gstate, ghist, gpend;
  float* ple_ring = nullptr;
};

int main() {
  constexpr size_t pend = 4096;
  static float cells[4 * 36 + 1];
  FakeState state;
  for (int layer = 0; layer < 36; ++layer) {
    state.gstate[layer] = cells + layer;
    state.ghist[layer] = cells + 36 + layer;
    state.gpend[layer] = cells + 72 + layer;
  }
  state.ple_ring = cells + 144;
  // The visit order is the snapshot layout: GDN states, conv rings, pending rows, then the PLE ring.
  std::vector<const void*> order;
  VisitFixedState(state, pend, [&](const void* pointer, size_t) { order.push_back(pointer); });
  assert(order.size() == 109 && order.front() == cells && order[36] == cells + 36 && order[72] == cells + 72 &&
         order.back() == cells + 144);
  const size_t total = 36 * (kGdnStateBytes + kGdnHistBytes + pend) + kPleRingBytes;
  assert(FixedStateBytes(state, pend) == total && FixedStateMatches(state, pend, total));
  // A state with one buffer more or less than the formula fails the check.
  state.gpend.erase(35);
  assert(!FixedStateMatches(state, pend, total));
  return 0;
}
