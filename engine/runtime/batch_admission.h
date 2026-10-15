// Existing allocations already reduce MemAvailable. Reserve only future allocations here.
#pragma once
#include <cstdint>
#include <fstream>
#include <optional>
#include <sstream>
#include <string>

namespace crucible {
inline std::optional<uint64_t> AvailableHostMemory() {
  std::ifstream file("/proc/meminfo");
  for (std::string line; std::getline(file, line);) {
    if (!line.starts_with("MemAvailable:")) continue;
    std::istringstream input(line);
    std::string key, unit, extra;
    uint64_t kib;
    if (!(input >> key >> kib >> unit) || unit != "kB" || input >> extra ||
        kib > UINT64_MAX / 1024) return std::nullopt;
    return kib * 1024;
  }
  return std::nullopt;
}

inline std::string BatchAdmissionError(uint64_t available, uint64_t new_states,
                                        uint64_t unused_cache_budget = 0) {
  constexpr uint64_t kOsReserve = 8ull << 30;
  constexpr uint64_t kPlanReserve = 512ull << 20;
  // The frontend bounds 64 pending JSON requests plus event queues and samplers.
  constexpr uint64_t kHostReserve = 512ull << 20;
  constexpr uint64_t reserve = kOsReserve + kPlanReserve + kHostReserve;
  if (available <= reserve || new_states > available - reserve ||
      unused_cache_budget > available - reserve - new_states)
    return "batch states exceed the shared memory budget; reduce streams, context, or cache budget";
  return {};
}
}  // namespace crucible
