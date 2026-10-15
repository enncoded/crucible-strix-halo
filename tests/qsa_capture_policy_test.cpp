// Which QSA capture a layer call may write (QsaCaptureMode). The prefill
// query kernel leaves the raw q projection in the fp32 buffer, so a full
// capture there would record a query with no norm and no RoPE.
#include "../engine/layers/qsa_capture_policy.h"
#include <cassert>
#include <string>

using crucible::QsaCaptureKind;
using crucible::QsaCaptureMode;

int main() {
  struct Case { const char* index_only; bool q_normed; QsaCaptureKind kind; };
  constexpr Case cases[] = {
    {"1", true, QsaCaptureKind::kIndexOnly},   // index-only never reads q
    {"1", false, QsaCaptureKind::kIndexOnly},  // also in prefill
    {nullptr, true, QsaCaptureKind::kFull},    // stored normalized q (decode, batch)
    {"0", true, QsaCaptureKind::kFull},
    {nullptr, false, QsaCaptureKind::kRefused},  // prefill: full capture stops
    {"0", false, QsaCaptureKind::kRefused},
    {"", false, QsaCaptureKind::kRefused},
  };
  for (const Case& c : cases) assert(QsaCaptureMode(c.index_only, c.q_normed) == c.kind);
  // The refusal names the index-only capture, and no deleted switch.
  assert(std::string(crucible::kQsaFullCaptureRefusal).find("PREP_HALF") == std::string::npos);
  assert(std::string(crucible::kQsaFullCaptureRefusal).find("CRUCIBLE_QSA_CAPTURE_INDEX_ONLY=1") != std::string::npos);
  return 0;
}
