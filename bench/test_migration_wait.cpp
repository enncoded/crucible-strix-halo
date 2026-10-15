#include <cassert>
#include <cstdint>
#include <vector>
#include "../engine/gfx1151/migration_wait.h"

crucible::MigrationWaitResult Run(const std::vector<int64_t>& counters) {
  std::size_t index = 0;
  int pauses = 0;
  const auto result = crucible::WaitForMigration([&](uint64_t& value) {
    assert(index < counters.size());
    const auto next = counters[index++];
    if (next < 0) return false;
    value = next; return true;
  },[&] { ++pauses; });
  assert(pauses == result.waited);
  return result;
}

int main() {
  auto result = Run({7,7,7,7});
  assert(result.available && result.stable == 3 && result.waited == 3);
  result = Run({1,2,3,4,4,4,4});
  assert(result.available && result.stable == 3 && result.waited == 6);
  result = Run({1,1,2,2,2,2});
  assert(result.available && result.stable == 3 && result.waited == 5);
  result = Run({-1});
  assert(!result.available && result.waited == 0);
  result = Run({1,1,-1});
  assert(!result.available && result.waited == 2 && result.stable == 0);
  uint64_t counter = 0;
  int pauses = 0;
  result = crucible::WaitForMigration([&](uint64_t& value) { value = counter++; return true; },
                                    [&] { ++pauses; });
  assert(result.available && result.stable == 0 && result.waited == 30 && pauses == 30);
}
