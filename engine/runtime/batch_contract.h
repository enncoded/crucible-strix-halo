// CPU boundary checks run before any stream advances its retained prefix.
#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace crucible {
inline std::string BatchError(const std::vector<int>& streams, const std::vector<int>& tokens,
                              const std::vector<size_t>& lengths, const std::vector<int>& capacities,
                              int max_rows, int vocab) {
  if (capacities.size() != lengths.size()) return "batch capacities must match stream state";
  if (streams.empty() || streams.size() != tokens.size() || streams.size() > size_t(max_rows))
    return "batch needs one token per stream and a supported row count";
  std::vector<bool> seen(lengths.size());
  for (size_t r = 0; r < streams.size(); ++r) {
    const int s = streams[r];
    if (s < 0 || size_t(s) >= lengths.size()) return "batch stream ID is out of range";
    if (seen[s]) return "batch stream IDs must be unique";
    seen[s] = true;
    if (tokens[r] < 0 || tokens[r] >= vocab) return "batch token ID is out of range";
    if (capacities[s] <= 0 || lengths[s] >= size_t(capacities[s])) return "batch stream context is full";
  }
  return "";
}
inline std::string BatchError(const std::vector<int>& streams, const std::vector<int>& tokens,
                              const std::vector<size_t>& lengths, int ctx, int max_rows, int vocab) {
  return BatchError(streams, tokens, lengths, std::vector<int>(lengths.size(), ctx), max_rows, vocab);
}
}  // namespace crucible
