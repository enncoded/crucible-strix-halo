#pragma once

#include <cstdint>
#include <span>
#include <utility>
#include <vector>

namespace crucible {

// Keep generated row inputs alive until the queued device copy completes.
class BatchHostInputs {
 public:
  BatchHostInputs(const std::vector<int>& tokens, std::vector<int64_t> rows)
      : tokens_(tokens), rows_(std::move(rows)) {}
  std::span<const int> tokens() const { return tokens_; }
  std::span<const int64_t> rows() const { return rows_; }

 private:
  const std::span<const int> tokens_;
  const std::vector<int64_t> rows_;
};

}  // namespace crucible
