#pragma once
#include <cstdint>

namespace crucible {
struct MigrationWaitResult {
  bool available;
  int stable = 0, waited = 0;
};

// The global counter can keep changing because of unrelated processes.
// A startup check must not wait without a time limit.
template<class Reader, class Sleeper>
MigrationWaitResult WaitForMigration(Reader read, Sleeper pause) {
  constexpr int kStableSeconds = 3, kMaxWaitSeconds = 30;
  uint64_t previous = 0;
  MigrationWaitResult result{read(previous)};
  while (result.available && result.stable < kStableSeconds && result.waited < kMaxWaitSeconds) {
    pause(); ++result.waited;
    uint64_t current = 0;
    result.available = read(current);
    result.stable = result.available && current == previous ? result.stable+1 : 0;
    previous = current;
  }
  return result;
}
}  // namespace crucible
