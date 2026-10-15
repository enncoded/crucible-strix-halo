// Keep user-facing decode defaults separate from fixed benchmark policies.
#pragma once
#include <cmath>
#include <stdexcept>
#include <string>

namespace crucible {
inline constexpr int kDefaultSpecDepth = 7;
inline constexpr float kDefaultDraftQ = 0.3f;
inline constexpr const char* kDefaultDraftHead = "64k";
inline constexpr int kDefaultMtpPrefix = 256;
// Prompt lookup (batched serve): copy up to `lookup` history tokens after the longest suffix match of at least
// `lookup_min` tokens. Verification commits only target tokens, so the output does not depend on it.
inline constexpr int kDefaultLookup = 0;
inline constexpr int kDefaultLookupMin = 8;

struct DecodePolicy {
  int spec = kDefaultSpecDepth;
  double draft_q = kDefaultDraftQ;
  std::string draft_head = kDefaultDraftHead;
  int mtp_prefix = kDefaultMtpPrefix;
  int draft_adapt = -1;  // -1 selects the built-in head's automatic adaptive policy.
  int lookup = kDefaultLookup;
  int lookup_min = kDefaultLookupMin;
  // 1: batched serve sizes each draft chain by expected tokens per ms (draft_value.h) instead of draft_q.
  int draft_value = 0;
};

inline void CheckDecodePolicy(const DecodePolicy& policy) {
  if (policy.spec < 0 || policy.spec > 7) throw std::runtime_error("spec must be in [0,8)");
  if (!std::isfinite(policy.draft_q) || policy.draft_q < 0 || policy.draft_q > 1)
    throw std::runtime_error("draft_q must be finite and in [0,1]");
  if (policy.draft_head != "full" && policy.draft_head != "64k")
    throw std::runtime_error("draft_head must be full or 64k");
  if (policy.mtp_prefix < 0 || policy.mtp_prefix > 262144)
    throw std::runtime_error("mtp_prefix must be in [0,262144]");
  if (policy.draft_adapt < -1 || policy.draft_adapt > 1 ||
      (policy.draft_adapt == 1 && policy.draft_head != "64k"))
    throw std::runtime_error("draft_adapt must be 0 or 1; 1 requires the 64k head");
  if (policy.lookup < 0 || policy.lookup > 7) throw std::runtime_error("lookup must be in [0,8)");
  if (policy.draft_value < 0 || policy.draft_value > 1) throw std::runtime_error("draft_value must be 0 or 1");
  if (policy.lookup_min < 1 || policy.lookup_min > 32) throw std::runtime_error("lookup_min must be in [1,32]");
}
}  // namespace crucible
