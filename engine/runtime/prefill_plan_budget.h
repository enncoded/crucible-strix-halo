#pragma once
#include <cstddef>

namespace crucible {
constexpr size_t kPrefillPlanBytes = size_t(1) << 30;
constexpr size_t kPrefillPlanCache = 16;

inline bool PrefillPlanBudgetExceeded(size_t entries, size_t cached, size_t added) {
  // Keep one oversized custom plan usable, but evict every other plan first.
  if (entries == 0) return false;
  return entries >= kPrefillPlanCache || added > kPrefillPlanBytes ||
         cached > kPrefillPlanBytes - added;
}
}  // namespace crucible
