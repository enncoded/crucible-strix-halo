// per slot, the snapshot whose rows [0, rows) the slot still holds bit for bit.
// A save of the slot shares those rows' host chunks (ConversationStateIO::Save with a SaveBase). Used on the device
// worker only. Weak: an evicted snapshot that nothing else holds is gone, and the next save copies everything.
#pragma once
#include <algorithm>
#include <array>
#include <cstddef>
#include <memory>

namespace crucible {
template<class Snapshot, size_t kSlots>
class SlotBases {
 public:
  struct Base { std::shared_ptr<const Snapshot> snapshot; size_t rows = 0; };
  // A host restore or a save from the slot: the slot holds the snapshot's rows except its pending replay rows (the
  // next pass writes them again).
  void Hold(int slot, const std::shared_ptr<const Snapshot>& snapshot) {
    const size_t length = snapshot->hist.size(), replay = size_t(std::max(snapshot->resume.replay, 0));
    slots_.at(size_t(slot)) = {snapshot, length > replay ? length - replay : 0};
  }
  // An open that resets the slot (the restore hook, before any restore).
  void Clear(int slot) { slots_.at(size_t(slot)) = {}; }
  // A slot checkpoint resume at `length` with `replay` pending rows. It writes the fixed state and the pool block of
  // its length, and the request replays the pending rows; the rows below them are the slot's own. SharedSpans
  // rounds the rows down to a whole pool block.
  void Limit(int slot, size_t length, int replay) {
    const size_t pending = size_t(std::max(replay, 0)), kept = length > pending ? length - pending : 0;
    auto& base = slots_.at(size_t(slot));
    base.rows = std::min(base.rows, kept);
  }
  Base Get(int slot) const {
    const auto& base = slots_.at(size_t(slot));
    return {base.snapshot.lock(), base.rows};
  }

 private:
  struct Weak { std::weak_ptr<const Snapshot> snapshot; size_t rows = 0; };
  std::array<Weak, kSlots> slots_{};
};
}  // namespace crucible
