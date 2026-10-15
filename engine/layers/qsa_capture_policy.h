// Which QSA capture a layer call may write. CPU-visible, so the rule has
// a table test (tests/qsa_capture_policy_test.cpp).
#pragma once
#include <string_view>

namespace crucible {
enum class QsaCaptureKind { kIndexOnly, kFull, kRefused };

// A full capture records the fp32 query buffer as the attention query, so it
// needs the stored normalized query (q_normed). The prefill query kernel
// (the only prefill path since train 20) writes only the fp16 query halves and
// leaves the raw q projection in that buffer. index_only: the CRUCIBLE_QSA_CAPTURE_INDEX_ONLY text or null.
inline QsaCaptureKind QsaCaptureMode(const char* index_only, bool q_normed) {
  if (index_only && std::string_view(index_only) == "1") return QsaCaptureKind::kIndexOnly;
  return q_normed ? QsaCaptureKind::kFull : QsaCaptureKind::kRefused;
}

inline constexpr const char* kQsaFullCaptureRefusal =
    "a full QSA capture needs the normalized query, and prefill does not store it: "
    "set CRUCIBLE_QSA_CAPTURE_INDEX_ONLY=1 for an index-only capture";
}  // namespace crucible
