// Host checks must run before a token upload or a context-cache write.
#pragma once

#include <algorithm>
#include <cstddef>
#include <span>

namespace crucible {

inline bool ValidTokenIds(std::span<const int> ids, int vocab) {
  return std::all_of(ids.begin(), ids.end(), [vocab](int id) { return id >= 0 && id < vocab; });
}

inline bool ContextRowsFit(int pos, int rows, int ctx, int max_rows) {
  return pos >= 0 && rows > 0 && rows <= max_rows && rows <= ctx && pos <= ctx - rows;
}

inline bool RequestFitsContext(size_t prompt, int count, int ctx) {
  return prompt > 0 && count > 0 && ctx > 0 && prompt <= (size_t)ctx &&
         (size_t)(count - 1) <= (size_t)ctx - prompt;
}

// mtp_ctx: the positions of the slot's MTP cache.
inline int BoundedDraftDepth(int requested, int ctx, int trunk, int mtp, int catchup, int mtp_ctx) {
  if (trunk < 0 || trunk >= ctx || mtp < 0 || catchup <= 0 || catchup > mtp_ctx ||
      mtp > mtp_ctx - catchup) return 0;
  return std::max(0, std::min({requested, ctx - trunk - 1, mtp_ctx - mtp - catchup + 1}));
}
inline int BoundedDraftDepth(int requested, int ctx, int trunk, int mtp, int catchup) {
  return BoundedDraftDepth(requested, ctx, trunk, mtp, catchup, ctx);
}
// A step writes MTP positions mpos .. mpos + catchup + requested - 2 (catch-up pairs, then one per later draft).
// When they pass the slot's MTP cache, the MTP restarts at position 0 with an empty history. The MTP only
// proposes drafts and the trunk verifies them, so a restart changes drafts and speed, never the output.
inline bool MtpRestart(int mpos, int catchup, int requested, int mtp_ctx) {
  return mpos > 0 && catchup > 0 && mpos > mtp_ctx - catchup - std::max(requested, 1) + 1;
}

}  // namespace crucible
