// Batched serve with the full MTP window (mtp_prefix 0) keeps the MTP history in its cache snapshots
// The MTP only proposes drafts, so nothing here can change an output.
#pragma once
#include "../runtime/conversation_resume.h"
#include "../runtime/draft_warmup.h"
#include "serve_protocol.h"
#include <algorithm>
#include <limits>
#include <optional>

namespace crucible {
// A request that warms the MTP over its whole prompt. Its snapshots can carry the MTP rows. Every request keeps the
// one token key: an entry with MTP rows restores target state for any request, and a full-window request continues
// the MTP rows that it finds (an entry without them starts a fresh warm-up at the restored length).
inline bool BatchMtpHistory(const ServeRequest& r) {
  return r.spec > 0 && r.max_tokens > 1 && r.mtp_prefix == 0;
}
// The longest restore: every request stops one row early (the last prompt row gives the first token's logits).
// A bounded window that restores past its warm-up start warms the MTP from the restored length (Open: origin), as
// a tail restore does: drafts change, the output does not. Before an earlier change the restore stopped at the warm-up start, so
// an identical re-send whose last grid frontier lay in its last mtp_prefix rows prefilled the whole prompt.
inline size_t BatchRestoreLimit(const ServeRequest& r) {
  return r.ids.size() < 2 ? 0 : r.ids.size() - 1;
}
// What a slot's MTP cache holds after a full warm-up to `length` target rows: the pairs of rows [0, length - 1),
// each row's trunk stack with the token after it, at MTP positions equal to the rows. The carry buffer holds the
// stack of row length - 1, which pairs with the next prompt token.
struct MtpHeld {
  size_t length = 0;
  bool operator==(const MtpHeld&) const = default;
  int Mpos() const { return int(length) - 1; }
};
// The resume record form of a held state (ConversationStateIO saves rows [0, mpos) and the carry).
inline bool MtpResumeHeld(const ConversationResume& resume) {
  return resume.length > 0 && resume.have_carry && resume.mpos == int(resume.length) - 1 &&
         resume.pending_tokens.empty();
}
// The resume record form of target state only (no MTP rows, no carry).
inline bool MtpResumeEmpty(const ConversationResume& resume) {
  return resume.mpos == 0 && !resume.have_carry && resume.pending_tokens.empty();
}
inline std::optional<MtpHeld> MtpHeldOf(const ConversationResume& resume) {
  if (!MtpResumeHeld(resume)) return std::nullopt;
  return MtpHeld{resume.length};
}
// A finished request's MTP state: `mpos` consumed pairs (rows [0, mpos)) and `pending` pairs whose stacks wait in
// the pair buffer (rows [mpos, mpos + pending), each with the token after it). A tail keeps `logical` rows: the
// pairs of rows [mpos, logical - 1) go through the MTP (`flush`, at most one call), and the stack of row
// logical - 1 (pair buffer row `carry`) becomes the carry. None when the tail does not end inside the pending rows,
// or when the slot's MTP cache (mtp_rows positions) cannot hold MTP positions up to logical - 1.
struct MtpTailPlan {
  int flush = 0, carry = 0;
  bool operator==(const MtpTailPlan&) const = default;
};
inline std::optional<MtpTailPlan> MtpTailFlush(int mpos, size_t pending, size_t logical, int max_rows,
                                               int mtp_rows = std::numeric_limits<int>::max()) {
  if (mpos < 0 || !pending || logical <= size_t(mpos) || logical > size_t(mpos) + pending) return std::nullopt;
  if (logical - 1 > size_t(std::max(mtp_rows, 0))) return std::nullopt;
  const int carry = int(logical - 1 - size_t(mpos));
  if (carry > max_rows) return std::nullopt;
  return MtpTailPlan{carry, carry};
}
// Open: a full-window request continues the slot's MTP history only when it restored exactly that state.
inline bool MtpContinues(const ServeRequest& r, size_t cached, const std::optional<MtpHeld>& held) {
  return cached > 0 && BatchMtpHistory(r) && held && held->length == cached;
}
}  // namespace crucible
