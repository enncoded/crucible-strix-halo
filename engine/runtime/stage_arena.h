// One allocation for row buffers whose live ranges do not overlap.
#pragma once
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace crucible {
constexpr size_t kStageAlign = 256;

// A bump allocator over [base, base + size). With base == nullptr it only
// counts bytes (Measure), so one carve function sizes and fills a view.
struct StageView {
  char* base = nullptr;
  size_t size = SIZE_MAX, used = 0;
  template <class T> T* Take(size_t count) {
    const size_t offset = used;
    used = (used + count * sizeof(T) + kStageAlign - 1) / kStageAlign * kStageAlign;
    if (used > size) {
      std::fprintf(stderr, "stage arena overflow (%zu of %zu bytes)\n", used, size);
      std::abort();
    }
    return base ? reinterpret_cast<T*>(base + offset) : nullptr;
  }
};
template <class Carve> size_t StageBytes(Carve carve) {
  StageView view;
  carve(view);
  return view.used;
}

// Byte layout of two stage views over one arena. Shared: both views start at
// 0 and the arena is the larger view. Separate: the second view follows the
// first (the layout of two allocations).
struct StageLayout {
  size_t first = 0, second = 0, bytes = 0, second_offset = 0;
};
inline StageLayout StageArenaLayout(size_t first, size_t second, bool shared) {
  const size_t aligned = (first + kStageAlign - 1) / kStageAlign * kStageAlign;
  if (shared) return {first, second, std::max(first, second), 0};
  return {first, second, aligned + second, aligned};
}
}  // namespace crucible
