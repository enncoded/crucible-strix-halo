// In-slot resume points (soak405, oracle-cache item 5). The device bytes live with the cache hooks; this header
// keeps which positions each slot can resume from, so the rules stay CPU-testable.
//
// A checkpoint holds the fixed-size state of a slot at one position: GDN state, conv ring, pending GDN rows, PLE
// ring, the compact raw ring and the open QSA pool block. KV rows below the position are not copied: the slot keeps
// them, because a stream only appends rows until something resets it or restores over it. So a checkpoint stays
// valid while its slot keeps every row below it.
//
// Every checkpoint resumes as a replay tail does (one small-row forward first, then bulk chunks from there):
// contract A, the G15 class (docs/SESSION_REPLAY.md). Grid-aligned prompt states of normal turns are host frontiers
// already, so the slot takes no checkpoint there.
#pragma once
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <stdexcept>
#include <vector>

namespace crucible {
enum class CheckpointKind { Start, Grid, End };
constexpr int kSlotCheckpoints = 3;

struct SlotCheckpoint {
  std::vector<int> hist;  // the tokens below the checkpoint
  std::vector<int> key;   // their cache key (empty: the tokens); image rows key by image, not by pad token IDs
  int replay = 0;         // pending GDN rows (a turn start on a restored tail)
  bool guarded = false;   // the state came from a guarded request's tail
  CheckpointKind kind = CheckpointKind::End;
  uint64_t owner = 0;     // the request that took it
};

class SlotCheckpoints {
 public:
  explicit SlotCheckpoints(int slots = 0) : held_(size_t(slots)) {}
  // The buffer for a new checkpoint whose cache key is `key`, or nullopt when the slot already holds that state. A
  // request's later grid point replaces its earlier one; else an empty buffer, else the shortest checkpoint leaves.
  // Invariant: every point is a prefix of the rows on its slot (Keep drops the others), so a point of the same length
  // has the same key. One with another key would be stale: it is the one to replace.
  std::optional<int> Victim(int slot, std::span<const int> key, CheckpointKind kind, uint64_t owner) const {
    const auto& held = Slot(slot);
    for (int i = 0; i < kSlotCheckpoints; ++i)
      if (held[i] && held[i]->key.size() == key.size())
        return std::equal(key.begin(), key.end(), held[i]->key.begin()) ? std::nullopt : std::optional<int>(i);
    if (kind == CheckpointKind::Grid)
      for (int i = 0; i < kSlotCheckpoints; ++i)
        if (held[i] && held[i]->owner == owner && held[i]->kind == CheckpointKind::Grid) return i;
    int victim = 0;
    for (int i = 0; i < kSlotCheckpoints; ++i) {
      if (!held[i]) return i;
      if (held[i]->hist.size() < held[victim]->hist.size()) victim = i;
    }
    return victim;
  }
  void Store(int slot, int index, SlotCheckpoint checkpoint) {
    if (index < 0 || index >= kSlotCheckpoints || checkpoint.hist.empty())
      throw std::invalid_argument("slot checkpoint needs a buffer index and tokens");
    if (checkpoint.key.empty()) checkpoint.key = checkpoint.hist;
    if (checkpoint.key.size() != checkpoint.hist.size())
      throw std::invalid_argument("slot checkpoint key and tokens differ in length");
    Slot(slot)[size_t(index)] = std::move(checkpoint);
  }
  // The longest checkpoint whose key `key` (a prompt's cache key) extends by at least one token: the last prompt row
  // must still run.
  std::optional<int> Best(int slot, std::span<const int> key) const {
    std::optional<int> best;
    const auto& held = Slot(slot);
    for (int i = 0; i < kSlotCheckpoints; ++i) {
      const auto& item = held[i];
      if (!item || item->key.size() >= key.size() || (best && item->key.size() <= held[*best]->key.size()))
        continue;
      if (std::equal(item->key.begin(), item->key.end(), key.begin())) best = i;
    }
    return best;
  }
  // The longest checkpoint length that the prompt key `key` repeats, for slot affinity (0: none).
  size_t Common(int slot, std::span<const int> key) const {
    const auto best = Best(slot, key);
    return best ? Slot(slot)[size_t(*best)]->hist.size() : 0;
  }
  // The slot keeps only its rows below `keep` (a restore at `keep`, or 0 for a reset or a host restore).
  void Keep(int slot, size_t keep) {
    for (auto& item : Slot(slot))
      if (item && item->hist.size() > keep) item.reset();
  }
  const SlotCheckpoint& At(int slot, int index) const {
    const auto& item = Slot(slot).at(size_t(index));
    if (!item) throw std::logic_error("slot checkpoint is empty");
    return *item;
  }

 private:
  using Held = std::array<std::optional<SlotCheckpoint>, kSlotCheckpoints>;
  Held& Slot(int slot) { return held_.at(size_t(slot)); }
  const Held& Slot(int slot) const { return held_.at(size_t(slot)); }
  std::vector<Held> held_;
};
}  // namespace crucible
