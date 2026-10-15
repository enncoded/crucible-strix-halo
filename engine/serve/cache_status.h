// Cache visibility for crucible_turn: the Oct 5 incident refused every large save, and no log showed it.
#pragma once
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>

namespace crucible {
// The latest save try. `refused` names why it was not saved: "budget" (larger than the whole cache), "host" (the
// 8 GiB host reserve, even after eviction), "protected" (only live sessions' resume points could make the room),
// "resident" (a copy of a state that its slot still holds, with no free room: the slot keeps it), "policy" (a state
// this cache never holds), or "memory" (allocation).
struct BatchCacheSave {
  size_t tokens = 0;
  bool saved = false;
  std::string refused;
};
struct BatchCacheStatus {
  size_t entries = 0, bytes = 0, evictions = 0;
  uint64_t attempts = 0;
  BatchCacheSave last;
  std::optional<uint64_t> host_available;  // MemAvailable when the status was read
};
}  // namespace crucible
