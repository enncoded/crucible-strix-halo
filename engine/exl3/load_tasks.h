// Weight load tasks. Kept free of HIP so the CPU tests can check the split.
#pragma once
#include <cstddef>
#include <cstdint>
#include <functional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace crucible {

struct LoadTask {
  size_t bytes;
  uint8_t* dst;
  const void* direct;                    // plain copy source, or
  std::function<void(uint8_t*)> fill;    // converts into the staging buffer
};

// Loader workers take whole tasks, so one large direct copy (the 39 GB 4.05
// PLE table) ran on one thread after the others finished. Pieces of at most
// `piece` bytes, in the same order, let every worker copy part of it. The
// bytes in the arena do not change: each piece copies its own source range.
inline std::vector<LoadTask> SplitDirectTasks(std::vector<LoadTask> tasks, size_t piece) {
  if (piece == 0) throw std::invalid_argument("load task piece size must be positive");
  std::vector<LoadTask> out;
  out.reserve(tasks.size());
  for (auto& t : tasks) {
    if (!t.direct || t.bytes <= piece) {
      out.push_back(std::move(t));
      continue;
    }
    const auto* src = static_cast<const uint8_t*>(t.direct);
    for (size_t o = 0; o < t.bytes; o += piece)
      out.push_back({std::min(piece, t.bytes - o), t.dst + o, src + o, nullptr});
  }
  return out;
}

}  // namespace crucible
