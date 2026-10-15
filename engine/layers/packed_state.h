// Kernel parameters own descriptor copies, so no asynchronous host upload is needed.
#pragma once
#include <cstddef>

namespace crucible {
template <class T, size_t N>
struct PackedStateArgs {
  T rows[N]{};
};
}  // namespace crucible
