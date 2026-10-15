// CPU-visible contract for the trunk's recurrent state and speculative rings.
#pragma once
#include <array>
#include <cstddef>
#include <limits>
#include <string>

namespace crucible {
class RetainedPrefix {
 public:
  std::string Forward(int rows, int replay, int commit) {
    // The 16-slot PLE ring preserves nine old rows only when verification retains
    // at least one new row. GDN has no snapshot for rollback across committed rows.
    if (rows <= 0 || commit < 1 || commit > rows)
      return "commit must retain at least one row and cannot exceed the forward row count";
    if (replay < 0 || size_t(replay) != length_ - committed_ || size_t(replay) > pending_)
      return "replay must equal the retained pending prefix from the previous forward";
    if (size_t(rows) > std::numeric_limits<size_t>::max() - length_)
      return "forward length overflow";
    window_start_ = length_;
    committed_ = length_ + size_t(commit);
    length_ += size_t(rows);
    pending_ = size_t(rows - commit);
    return "";
  }
  std::string Truncate(size_t n) {
    if (n > length_) return "Truncate cannot grow token history";
    if (n == length_) return "";
    if (n == window_start_) return "zero-prefix rollback is unsupported; retain at least one verification row";
    if (n < committed_) return "Truncate cannot cross committed recurrent state";
    length_ = n;
    return "";
  }
  void Reset() { length_ = committed_ = window_start_ = pending_ = 0; }
  bool Clean() const { return !length_ && !committed_ && !window_start_ && !pending_; }
  // The state disk stores these four counters with a snapshot (state_disk.h).
  std::array<size_t, 4> Fields() const { return {length_, committed_, window_start_, pending_}; }
  static bool FromFields(const std::array<size_t, 4>& fields, RetainedPrefix& out) {
    const auto [length, committed, window_start, pending] = fields;
    if (committed > length || window_start > length || pending > length) return false;
    out.length_ = length, out.committed_ = committed, out.window_start_ = window_start, out.pending_ = pending;
    return true;
  }

 private:
  size_t length_ = 0, committed_ = 0, window_start_ = 0, pending_ = 0;
};
}  // namespace crucible
