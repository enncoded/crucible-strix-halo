// CPU-only proposal helpers. Verification, never a proposal, commits tokens.
#pragma once

#include <algorithm>
#include <charconv>
#include <cctype>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <unordered_map>
#include <string>
#include <string_view>
#include <vector>

namespace crucible {

inline bool ValidDraftSelection(const std::vector<int>& ids, int limit) {
  int previous = -1;
  for (int id : ids) {
    if (id <= previous || id >= limit) return false;
    previous = id;
  }
  return !ids.empty();
}

inline std::vector<int> ParseDraftSelection(const std::string& text) {
  std::vector<int> ids;
  for (size_t start = 0;;) {
    const size_t end = text.find(',', start);
    std::string_view field(text.data() + start, (end == std::string::npos ? text.size() : end) - start);
    while (!field.empty() && std::isspace((unsigned char)field.front())) field.remove_prefix(1);
    while (!field.empty() && std::isspace((unsigned char)field.back())) field.remove_suffix(1);
    int value = 0;
    const auto [next, error] = std::from_chars(field.data(), field.data() + field.size(), value);
    if (field.empty() || error != std::errc{} || next != field.data() + field.size())
      throw std::invalid_argument("draft selection must contain comma-separated integers");
    ids.push_back(value);
    if (end == std::string::npos) return ids;
    start = end + 1;
  }
}

// Prefer the longest matching suffix, then its latest occurrence. Only copy
// already-known tokens; verification handles mismatches exactly as for MTP.
inline std::vector<int> LookupDraft(const std::vector<int>& history, int depth,
                                    int min_match = 3, int max_match = 32) {
  const int n = static_cast<int>(history.size());
  if (depth <= 0 || min_match <= 0) return {};
  int best_length = min_match - 1, best_end = -1;
  for (int end = n - 1; end >= min_match; --end) {
    int length = 0;
    while (length < std::min(max_match, end) &&
           history[end - length - 1] == history[n - length - 1]) ++length;
    if (length <= best_length) continue;
    best_length = length;
    best_end = end;
    if (length == max_match) break;
  }
  if (best_end < 0) return {};
  return {history.begin() + best_end, history.begin() + best_end + std::min(depth, n - best_end)};
}

// LookupDraft over a growing history, with the same result: a full scan costs about 0.2 ms at 262K tokens, and
// batched serve looks up every speculative request twice per pass. Ends whose preceding min_match tokens hash
// alike are chained, latest first, so a lookup visits only ends that can match.
class LookupIndex {
 public:
  explicit LookupIndex(int min_match = 8) : min_(min_match) {
    if (min_match < 1) throw std::invalid_argument("lookup minimum must be positive");
  }
  void Append(std::span<const int> tokens) {
    for (const int token : tokens) {
      history_.push_back(token);
      // LookupDraft scans ends up to n - 1, so the newest position becomes an end when the next token arrives;
      // its key (the min_match tokens before it) is already known.
      const int end = int(history_.size()) - 1;
      if (end < min_) { previous_.push_back(-1); continue; }
      const auto [head, added] = heads_.try_emplace(Key(end), end);
      previous_.push_back(added ? -1 : head->second);
      head->second = end;
    }
  }
  std::vector<int> Draft(int depth, int max_match = 32) const {
    const int n = int(history_.size());
    if (depth <= 0 || n <= min_) return {};
    const auto head = heads_.find(Key(n));
    if (head == heads_.end()) return {};
    int best_length = min_ - 1, best_end = -1;
    // Latest end first and a strict improvement, as in LookupDraft. A hash collision only adds a short match.
    for (int end = head->second; end >= 0; end = previous_[end]) {
      int length = 0;
      while (length < std::min(max_match, end) && history_[end - length - 1] == history_[n - length - 1]) ++length;
      if (length <= best_length) continue;
      best_length = length;
      best_end = end;
      if (length == max_match) break;
    }
    if (best_end < 0) return {};
    return {history_.begin() + best_end, history_.begin() + best_end + std::min(depth, n - best_end)};
  }
  bool Empty() const { return history_.empty(); }
  int Back() const { return history_.back(); }
  size_t Size() const { return history_.size(); }

 private:
  // FNV-1a over the min_match tokens before `end`.
  uint64_t Key(int end) const {
    uint64_t hash = 1469598103934665603ull;
    for (int i = end - min_; i < end; ++i) hash = (hash ^ uint32_t(history_[i])) * 1099511628211ull;
    return hash;
  }
  int min_;
  std::vector<int> history_, previous_;
  std::unordered_map<uint64_t, int> heads_;
};

}  // namespace crucible
