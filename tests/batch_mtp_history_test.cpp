// MTP history in batched cache snapshots: keys, restore limits, the held MTP state,
// the tail flush, and the frontiers for the full window.
#include "../engine/serve/batch_mtp_history.h"
#include "../engine/serve/batch_conversation_cache.h"
#include <cassert>
#include <numeric>
using namespace crucible;

static ServeRequest Request(size_t length, int spec, int max_tokens, int mtp_prefix) {
  ServeRequest r;
  r.ids.resize(length);
  std::iota(r.ids.begin(), r.ids.end(), 1);
  r.spec = spec;
  r.max_tokens = max_tokens;
  r.mtp_prefix = mtp_prefix;
  return r;
}

static void Keys() {
  assert(BatchMtpHistory(Request(100, 7, 32, 0)));
  assert(!BatchMtpHistory(Request(100, 7, 32, 256)));
  assert(!BatchMtpHistory(Request(100, 0, 32, 0)));  // a plain request warms no MTP
  assert(!BatchMtpHistory(Request(100, 7, 1, 0)));   // one token runs as a plain step
}

static void Limits() {
  // Every request stops one row early; a bounded window no longer stops before its warm-up rows.
  assert(BatchRestoreLimit(Request(10000, 7, 32, 256)) == 9999);
  assert(BatchRestoreLimit(Request(10000, 7, 32, 0)) == 9999);
  assert(BatchRestoreLimit(Request(10000, 0, 32, 256)) == 9999);
  assert(BatchRestoreLimit(Request(10000, 7, 1, 256)) == 9999);
  assert(BatchRestoreLimit(Request(100, 7, 32, 256)) == 99);
  assert(BatchRestoreLimit(Request(1, 7, 32, 0)) == 0);
}

static void Held() {
  ConversationResume r;
  r.length = 8192;
  assert(MtpResumeEmpty(r) && !MtpResumeHeld(r) && !MtpHeldOf(r));
  r.mpos = 8191;
  assert(!MtpResumeHeld(r) && !MtpResumeEmpty(r));  // rows without the carry
  r.have_carry = true;
  assert(MtpResumeHeld(r) && MtpHeldOf(r) == MtpHeld{8192} && MtpHeldOf(r)->Mpos() == 8191);
  r.pending_tokens = {5};
  assert(!MtpResumeHeld(r));
  r.pending_tokens.clear();
  r.mpos = 8000;  // a gap: not the full history
  assert(!MtpResumeHeld(r));
  // Open continues only the exact restored state of a full-window request.
  const auto full = Request(9000, 7, 32, 0);
  assert(MtpContinues(full, 8192, MtpHeld{8192}));
  assert(!MtpContinues(full, 8192, MtpHeld{4096}));
  assert(!MtpContinues(full, 8192, std::nullopt));
  assert(!MtpContinues(full, 0, MtpHeld{0}));
  assert(!MtpContinues(Request(9000, 7, 32, 256), 8192, MtpHeld{8192}));
}

static void TailFlush() {
  // mpos 100, pending rows 100..104: a tail of 105 rows flushes rows 100..103 and keeps row 104 as the carry.
  assert(MtpTailFlush(100, 5, 105, 8) == (MtpTailPlan{4, 4}));
  // A stop inside the window keeps fewer rows.
  assert(MtpTailFlush(100, 5, 102, 8) == (MtpTailPlan{1, 1}));
  assert(MtpTailFlush(100, 5, 101, 8) == (MtpTailPlan{0, 0}));
  // Nine pending pairs (a packed row before the flush): the carry is pair buffer row 8.
  assert(MtpTailFlush(100, 9, 109, 8) == (MtpTailPlan{8, 8}));
  // The tail ends before the pending rows or after them, or there are none: no MTP history.
  assert(!MtpTailFlush(100, 5, 100, 8));
  assert(!MtpTailFlush(100, 5, 106, 8));
  assert(!MtpTailFlush(100, 0, 100, 8));
  assert(!MtpTailFlush(-1, 5, 3, 8));
  // the flush writes MTP positions mpos..logical - 2; a slot MTP cache of fewer positions keeps no history.
  assert(MtpTailFlush(100, 5, 105, 8, 104) == (MtpTailPlan{4, 4}));
  assert(!MtpTailFlush(100, 5, 105, 8, 103));
  assert(MtpTailFlush(100, 5, 101, 8, 100) == (MtpTailPlan{0, 0}));
  assert(!MtpTailFlush(100, 5, 101, 8, 99));
}

static void Frontiers() {
  // The full window warms up to the frontier, so it admits the same frontiers as target state. A window cuts them.
  for (const int window : {0, 256, 4096}) {
    for (const size_t length : {size_t(5000), size_t(12288), size_t(200000)}) {
      const size_t frontier = ConversationAdmission::ForPrompt(length, 0, 4096, window).frontier;
      const size_t stable = window ? DraftWarmupStart(length, window) : length;
      assert(frontier == stable / 4096 * 4096);
      if (!window) assert(frontier == ConversationAdmission::ForPrompt(length, 0, 4096, 0).frontier);
    }
  }
  assert(ConversationAdmission::ForPrompt(200000, 0, 4096, 0).frontier == 196608);
  assert(ConversationAdmission::ForPrompt(200000, 0, 4096, 256).frontier == 196608);
  assert(ConversationAdmission::ForPrompt(200000, 0, 4096, 4096).frontier == 192512);
  assert(ConversationAdmission::ForPrompt(12288, 0, 4096, 4096).frontier == 8192);
  assert(ConversationAdmission::ForPrompt(12288, 0, 4096, 0).frontier == 12288);
}

// The cache side: one token key for every request. An entry holds target state, or target state with the full MTP
// history; the key does not depend on the MTP window (a spec 0 turn's tail stays a replay hit for a full-window turn).
struct Snapshot { std::vector<int> hist; ConversationResume resume; };
static std::shared_ptr<const Snapshot> Make(const std::vector<int>& ids, size_t length, bool prompt, int mpos,
                                            bool carry, std::vector<int> pending = {}) {
  auto s = std::make_shared<Snapshot>();
  s->hist.assign(ids.begin(), ids.begin() + std::ptrdiff_t(length));
  s->resume.length = length;
  s->resume.prompt_only = prompt;
  s->resume.prompt_boundary = prompt ? ids.size() - 1 : 0;
  s->resume.mpos = mpos;
  s->resume.have_carry = carry;
  s->resume.pending_tokens = std::move(pending);
  return s;
}
static void Cache() {
  std::vector<int> ids(40);
  std::iota(ids.begin(), ids.end(), 1);
  ConversationPolicy policy;
  policy.prefill_rows = 8;
  const uint64_t room = uint64_t(9) << 30;
  BatchConversationCache<Snapshot> cache(64 * 8, {}, 8, 0);
  const auto capture = [&](size_t at, int mpos, bool carry, std::vector<int> pending = {}) {
    return cache.Capture(ids, policy, {at, CacheRetention::Session}, 64, room,
                         [&] { return Make(ids, at, true, mpos, carry, pending); });
  };
  const auto tail = [&](size_t length, int mpos, bool carry) {
    return cache.CaptureTail(std::span<const int>(ids.data(), length), policy, 64, room,
                             [&] { return Make(ids, length, false, mpos, carry); });
  };
  const auto engine_fault = [](auto save) {
    try { save(); assert(false); } catch (const std::runtime_error&) {}
  };
  // A prompt frontier with the full MTP history, and one with target state only.
  assert(capture(16, 15, true));
  assert(capture(8, 0, false));
  // MTP state with a gap, without the carry, or with pending pairs is an engine fault.
  engine_fault([&] { capture(24, 20, true); });
  engine_fault([&] { capture(24, 23, false); });
  engine_fault([&] { capture(24, 23, true, {7}); });
  engine_fault([&] { capture(24, 0, true); });
  const auto hit = cache.Find(ids, policy, 0, 39, 8);
  assert(hit && hit->snapshot->hist.size() == 16 && MtpHeldOf(hit->snapshot->resume) == MtpHeld{16});
  // A target-only save at a later frontier is found first; the MTP rows are optional content, not a key.
  assert(capture(24, 0, false));
  assert(cache.Find(ids, policy, 0, 39, 8)->snapshot->hist.size() == 24);
  assert(!MtpHeldOf(cache.Find(ids, policy, 0, 39, 8)->snapshot->resume));
  // Tails: the same two forms.
  assert(tail(30, 29, true));
  engine_fault([&] { tail(32, 30, true); });
  engine_fault([&] { tail(32, 31, false); });
  const auto replay = cache.FindTail(ids, policy);
  assert(replay && replay->snapshot->hist.size() == 30 && MtpHeldOf(replay->snapshot->resume) == MtpHeld{30});
  assert(tail(31, 0, false));
  assert(cache.FindTail(ids, policy)->snapshot->hist.size() == 31);
  assert(cache.HoldsTail(std::span<const int>(ids.data(), 30), policy, 0));
  // Speculation itself stays outside the batch key (serial serve keys with it).
  auto spec = policy;
  spec.spec = 7;
  assert(!cache.Capture(ids, spec, {32, CacheRetention::Session}, 64, room, [&] { return Make(ids, 32, true, 0, false); }));
}

// an identical re-send of a prompt whose last grid frontier lies in its last 256 rows (the default MTP window).
// The first send (max_tokens 1, no tail) saved the frontier; the re-send (spec on, window 256) must restore it, as a
// full-window request does. Sizes of the head-to-head case: a 32,849-token prompt, boundary 5 rows before its end.
static void Resend() {
  std::vector<int> ids(32849);
  std::iota(ids.begin(), ids.end(), 1);
  const size_t boundary = ids.size() - 5, rows = 4096;
  ConversationPolicy policy;
  policy.prefill_rows = int(rows);
  BatchConversationCache<Snapshot> cache(1 << 20, {}, 4, 0);
  const auto frontiers = BatchCacheFrontiers(ids.size(), boundary, int(rows));
  assert(frontiers.size() == 1 && frontiers[0].position == 32768 && !frontiers[0].speculative);
  assert(cache.Capture(ids, policy, frontiers[0], 64, uint64_t(9) << 30, [&] {
    auto s = std::make_shared<Snapshot>(*Make(ids, 32768, true, 0, false));
    s->resume.prompt_boundary = boundary;
    return std::shared_ptr<const Snapshot>(s);
  }));
  for (const auto& resend : {Request(ids.size(), 7, 320, 256), Request(ids.size(), 7, 320, 0),
                             Request(ids.size(), 0, 320, 256), Request(ids.size(), 7, 1, 256)}) {
    const size_t limit = BatchRestoreLimit(resend);
    assert(cache.PeekFind(ids, policy, boundary, limit, int(rows)) == 32768);
    const auto hit = cache.Find(ids, policy, boundary, limit, int(rows));
    assert(hit && hit->snapshot->hist.size() == 32768);
  }
}

int main() {
  Keys();
  Limits();
  Resend();
  Held();
  TailFlush();
  Frontiers();
  Cache();
  std::puts("batch_mtp_history_test: ok");
  return 0;
}
