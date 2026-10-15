// Growing conversations reuse complete stable chunks, not an unfinished generation suffix.
#pragma once
#include "draft_warmup.h"
#include <stdexcept>

namespace crucible {
struct ConversationAdmission {
  size_t frontier = 0, prompt_length = 0;
  bool exact_retry = false;
  static ConversationAdmission ForPrompt(size_t length, size_t boundary, int rows, int mtp_prefix,
                                         bool exact_retry = false) {
    if (rows < 1 || mtp_prefix < 0) throw std::invalid_argument("invalid conversation snapshot admission");
    const size_t warmup = mtp_prefix ? DraftWarmupStart(length, mtp_prefix) : length;
    const size_t stable = std::min(boundary ? boundary : length, warmup);
    return {stable / size_t(rows) * size_t(rows), length, exact_retry && mtp_prefix == 0};
  }
  bool Allows(size_t length) const {
    return length && (length == frontier || (exact_retry && length == prompt_length));
  }
  bool Any() const { return frontier || (exact_retry && prompt_length); }
};
}  // namespace crucible
