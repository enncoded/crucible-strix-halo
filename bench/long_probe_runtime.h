#pragma once
#include <algorithm>
#include <cmath>
#include <span>
#include <stdexcept>
#include <vector>

namespace crucible::longprobe {
struct Block {
  size_t start;
  int rows;
  bool operator==(const Block&) const = default;
};

inline void Validate(std::span<const int> input, std::span<const int> points, int max_rows, int vocab) {
  if (input.empty() || points.empty() || max_rows < 1 || vocab < 2 || points.back() != (int)input.size()-1)
    throw std::runtime_error("invalid clipped probe schedule");
  int previous = -1;
  for (const int position : points) {
    if (position <= previous || position < 0 || position >= (int)input.size())
      throw std::runtime_error("probe positions must be sorted, unique, and within the prompt");
    previous = position;
  }
}

inline int Greedy(std::span<const float> values) {
  if (values.empty() || std::any_of(values.begin(),values.end(),[](float value){ return !std::isfinite(value); }))
    throw std::runtime_error("missing or nonfinite probe logits");
  return int(std::max_element(values.begin(),values.end())-values.begin());
}

// Adapted from quality_probe (4bf0d90). Stops expose only existing last-row heads.
// Clipping changes bulk boundaries. The gate also captures the production schedule.
template<class Engine, class Copy, class Emit>
std::vector<Block> Capture(Engine& engine, std::span<const int> input, std::span<const int> points,
                          int max_rows, bool bulk, int vocab, Copy copy, Emit emit) {
  Validate(input,points,max_rows,vocab);
  engine.Reset();
  size_t offset = 0;
  std::vector<Block> blocks;
  for (const int point : points) {
    while (offset <= (size_t)point) {
      const int rows = std::min<size_t>(max_rows,size_t(point)+1-offset);
      const bool end = offset+rows == size_t(point)+1;
      const auto ids = bulk ? engine.Prefill(input.data()+offset,rows,"",end)
                            : engine.Forward(input.data()+offset,rows,0,rows);
      blocks.push_back({offset,rows});
      offset += rows;
      if (!end) continue;
      if (ids.size() != size_t(bulk ? 1 : rows)) throw std::runtime_error("wrong probe argmax count");
      const int first = bulk ? 0 : rows-1;
      const auto logits = copy(engine.logits()+size_t(first)*vocab,size_t(vocab));
      if (logits.size() != size_t(vocab) || Greedy(logits) != ids.back())
        throw std::runtime_error("copied probe logits disagree with the engine argmax");
      emit(point,ids.back(),logits);
    }
  }
  return blocks;
}
}  // namespace crucible::longprobe
