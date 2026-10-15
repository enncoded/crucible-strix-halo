// Validate all temporal spans before any request changes its recurrent prefix.
#pragma once
#include <set>
#include <vector>
#include "retained_prefix.h"
namespace crucible {
struct VerifySpan {
  int stream;
  std::vector<int> tokens;
  int replay=0,commit=1;
};
struct SpecPrefix {
  size_t length;
  int capacity;
  RetainedPrefix retained;
};
inline std::string SpecSpansError(const std::vector<VerifySpan>& spans,const std::vector<SpecPrefix>& prefixes,
                                  int projection_limit,int temporal_limit,int vocabulary) {
  if (spans.empty() || projection_limit<1 || temporal_limit<1 || vocabulary<1) return "invalid verification limits";
  int total=0;
  std::set<int> used;
  for (const auto& span:spans) {
    if (span.stream<0 || size_t(span.stream)>=prefixes.size() || !used.insert(span.stream).second)
      return "verification streams must be valid and distinct";
    if (span.tokens.empty() || span.tokens.size()>size_t(temporal_limit) || span.tokens.size()>size_t(projection_limit-total))
      return "verification spans exceed the row limits";
    total+=int(span.tokens.size());
    for (int token:span.tokens) if (token<0 || token>=vocabulary) return "verification token is out of range";
    const auto& source=prefixes[span.stream];
    if (source.capacity<=0 || source.length>size_t(source.capacity) || span.tokens.size()>size_t(source.capacity)-source.length)
      return "verification span exceeds its request context";
    auto retained=source.retained;
    if (const auto error=retained.Forward(int(span.tokens.size()),span.replay,span.commit); !error.empty()) return error;
  }
  return "";
}
} // namespace crucible
