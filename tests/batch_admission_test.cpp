#include "../engine/runtime/batch_admission.h"
#include <cassert>
#include <limits>
int main() {
  using namespace crucible;
  constexpr uint64_t GiB = 1ull << 30;
  assert(BatchAdmissionError(12 * GiB, 2 * GiB).empty());
  assert(!BatchAdmissionError(8 * GiB, 1).empty());
  assert(!BatchAdmissionError(10 * GiB, 2 * GiB).empty());
  assert(!BatchAdmissionError(12 * GiB, 2 * GiB, 4 * GiB).empty());
  assert(BatchAdmissionError(12 * GiB, 2 * GiB, 1 * GiB).empty());
  assert(!BatchAdmissionError(std::numeric_limits<uint64_t>::max(),
                             std::numeric_limits<uint64_t>::max(), 1).empty());
  const auto available = AvailableHostMemory();
  assert(available && *available > 0);
}
