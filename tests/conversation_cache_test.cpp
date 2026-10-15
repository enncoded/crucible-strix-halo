#include "../engine/runtime/conversation_cache.h"
#include "../engine/runtime/conversation_resume.h"
#include "../engine/runtime/conversation_admission.h"
#include <cassert>
#include <memory>
using namespace crucible;
struct Snapshot { int value; };
static std::shared_ptr<const Snapshot> Make(int value) { return std::make_shared<Snapshot>(Snapshot{value}); }

// Budget and host room together (G15 tails near the host reserve): evicted entries return host memory, so they
// count toward the reserve; a request that even an empty cache could not fit evicts nothing.
static void HostReserve() {
  const std::vector<int> a{1}, b{2}, c{3};
  PrefixCache<Snapshot> cache(100);
  assert(cache.Insert(a, 0, 30, Make(1)) && cache.Insert(b, 0, 30, Make(2)));
  assert(cache.ReserveHost(30, 1000 + 30, 1000) && cache.Bytes() == 60);           // enough memory: no eviction
  assert(cache.ReserveHost(30, 1000 + 10, 1000) && cache.Bytes() == 30 && cache.Find(b, 0));  // 20 short: LRU a
  assert(!cache.ReserveHost(30, 1000 - 10, 1000) && cache.Bytes() == 30 && cache.Find(b, 0));  // 40 short: keep b
  assert(!cache.ReserveHost(101, 1u << 30, 1000) && cache.Bytes() == 30);
  assert(cache.Insert(c, 0, 30, Make(3)));
  assert(cache.ReserveHost(50, 1000 + 20, 1000) && cache.Bytes() <= 30);  // 30 short: one entry leaves
}

// Superseded entries leave first; everything else keeps LRU order.
static void SupersededEvictionOrder() {
  const auto all = [](const Snapshot&) { return true; };
  const std::vector<int> root{1}, older{1, 2}, newer{1, 2, 3, 4}, other{5, 6}, fork{1, 2, 7, 8};
  {
    PrefixCache<Snapshot> cache(100);
    assert(cache.Insert(other, 0, 30, Make(1)));  // least recently used, not superseded
    assert(cache.Insert(older, 0, 30, Make(2)));
    assert(cache.Supersede(newer, 0, all) == 1);
    assert(cache.Reserve(60) && cache.Evictions() == 1 && cache.SupersededEvictions() == 1);
    assert(cache.Find(other, 0)->value == 1 && !cache.Find(newer, 0));
  }
  {
    PrefixCache<Snapshot> cache(100);
    assert(cache.Insert(older, 0, 30, Make(2)));
    assert(cache.Supersede(older, 0, all) == 0);                    // equal length is not a strict prefix
    assert(cache.Supersede(newer, 1, all) == 0);                    // another policy never matches
    assert(cache.Supersede(other, 0, all) == 0);                    // not a prefix
    assert(cache.Supersede(newer, 0, [](const Snapshot&) { return false; }) == 0);
    assert(cache.Insert(other, 0, 30, Make(1)));
    assert(cache.Reserve(60) && cache.SupersededEvictions() == 0);  // plain LRU: `older` leaves
    assert(!cache.Find(newer, 0) && cache.Find(other, 0));
  }
  {
    // A hit means the entry is useful again, as in a fork whose original branch continues.
    PrefixCache<Snapshot> cache(100);
    assert(cache.Insert(older, 0, 30, Make(2)));
    assert(cache.Supersede(fork, 0, all) == 1);
    assert(cache.Insert(fork, 0, 30, Make(3)));
    assert(cache.Find(newer, 0)->value == 2);
    assert(cache.Insert(other, 0, 30, Make(1)));
    assert(cache.Reserve(40) && cache.SupersededEvictions() == 0);  // LRU: `fork` leaves
    assert(cache.Find(fork, 0)->value == 2);
    assert(cache.Find(newer, 0)->value == 2);
  }
  {
    // Among superseded entries, the least recently used one leaves first.
    PrefixCache<Snapshot> cache(100);
    assert(cache.Insert(older, 0, 20, Make(2)));
    assert(cache.Insert(root, 0, 20, Make(0)));
    assert(cache.Insert(other, 0, 20, Make(1)));
    assert(cache.Supersede(newer, 0, all) == 2);
    assert(cache.Supersede(newer, 0, all) == 0);                    // counts new marks only
    assert(cache.Reserve(60) && cache.SupersededEvictions() == 1);  // `older` leaves before `root`
    assert(cache.Find(older, 0)->value == 0 && cache.Find(other, 0));
  }
  {
    // Replacing an entry with the same key clears its mark.
    PrefixCache<Snapshot> cache(100);
    assert(cache.Insert(older, 0, 30, Make(2)));
    assert(cache.Supersede(newer, 0, all) == 1);
    assert(cache.Insert(older, 0, 30, Make(4)));
    assert(cache.Insert(other, 0, 30, Make(1)));
    assert(cache.Find(older, 0)->value == 4);
    assert(cache.Reserve(60) && cache.SupersededEvictions() == 0 && cache.Find(older, 0));
  }
}

int main() {
  HostReserve();
  const auto small = ConversationAdmission::ForPrompt(4088, 4081, 4096, 256);
  assert(small.frontier == 0 && !small.Allows(4081) && !small.Allows(4088));
  const auto below = ConversationAdmission::ForPrompt(4351, 4344, 4096, 256);
  const auto minimum = ConversationAdmission::ForPrompt(4352, 4345, 4096, 256);
  assert(below.frontier == 0 && minimum.frontier == 4096 && minimum.Allows(4096));
  const auto middle = ConversationAdmission::ForPrompt(16363, 16356, 4096, 256);
  assert(middle.frontier == 12288 && middle.Allows(12288));
  for (size_t length : {4096, 8192, 16356, 16363}) assert(!middle.Allows(length));
  const auto long_prompt = ConversationAdmission::ForPrompt(30698, 30691, 4096, 256);
  assert(long_prompt.frontier == 28672 && long_prompt.Allows(28672));
  const auto exact = ConversationAdmission::ForPrompt(4088, 4081, 4096, 0, true);
  assert(exact.frontier == 0 && exact.Allows(4088) && !exact.Allows(4081));
  const auto capped_exact = ConversationAdmission::ForPrompt(4088, 4081, 4096, 256, true);
  assert(!capped_exact.Allows(4088));
  assert(ConversationRetainedRows(100, 98, 5) == 3);
  assert(ConversationRetainedRows(100, 100, 5) == 5);
  assert(ConversationRetainedRows(100, 95, 5) == 0);
  assert(ConversationRetainedRows(100, 101, 5) == 0);
  assert(ConversationRetainedRows(3, 2, 4) == 0);
  ConversationPolicy base;
  base.draft_selection = {1, 7}; base.adaptive = true;
  PrefixCache<Snapshot, ConversationPolicy> configured(90);
  const int policy_ids[] = {1, 2};
  configured.Insert(policy_ids, base, 30, std::make_shared<Snapshot>(Snapshot{1}));
  auto dense = base; dense.dense = true;
  auto selection = base; selection.draft_selection = {1, 8};
  auto prefill = base; prefill.prefill_rows = 64;
  auto margin = base; margin.margin_bits = 1;
  auto cap = base; cap.mtp_prefix = 256;
  auto automatic = base; automatic.head_auto = true;
  assert(configured.Find(policy_ids, base));
  assert(!configured.Find(policy_ids, dense));
  assert(!configured.Find(policy_ids, selection));
  assert(!configured.Find(policy_ids, prefill));
  assert(!configured.Find(policy_ids, margin));
  assert(!configured.Find(policy_ids, cap));
  assert(!configured.Find(policy_ids, automatic));
  PrefixCache<Snapshot> cache(100);
  const std::vector<int> a{1,2,3}, b{1,2,4}, longer{1,2,3,5};
  assert(!cache.Find(a, 0));
  assert(cache.Insert(a, 0, 30, std::make_shared<Snapshot>(Snapshot{1})));
  assert(cache.Find(longer, 0)->value == 1);
  assert(!cache.Find(b, 0));
  assert(!cache.Find(a, 1));
  assert(cache.Insert(longer, 0, 40, std::make_shared<Snapshot>(Snapshot{2})));
  assert(cache.Find(longer, 0)->value == 2);
  assert(cache.Find(longer, 0, 3)->value == 1);
  assert(!cache.Find(longer, 0, 2));
  ConversationResume pending{100, 70, 2, 7, true, {4, 5, 7}, {1.f}};
  const auto target = ConversationTargetResume(pending);
  assert(target.length == 100 && target.replay == 2 && target.next_id == 7);
  assert(target.mpos == 0 && !target.have_carry && target.pending_tokens.empty());
  assert(target.logits == pending.logits && pending.mpos == 70);
  {
    // Session replay (G15): the tail a finished request keeps, one rule for serial and batch serve.
    assert(!ConversationTailState(10, 10, 1, 0));   // no forwarded output row: serial saves no tail either
    assert(!ConversationTailState(13, 10, 5, 0));   // the stream is behind the published output: refuse
    const auto exact = ConversationTailState(14, 10, 5, 3);
    assert(exact && exact->length == 14 && exact->replay == 3);
    // A stop or limit inside an accepted window: base 12, 3 accepted (length 16), 5 published (logical 14).
    const auto cut = ConversationTailState(16, 10, 5, 3);
    assert(cut && cut->length == 14 && cut->replay == 1);
    assert(!ConversationTailState(16, 10, 5, 0));   // committed rows beyond the published output
    // Matching serial RunGen: keep = ConversationRetainedRows(length, logical, replay + 1), replay = keep - 1.
    for (size_t length = 10; length < 24; ++length) for (int replay = 0; replay < 8; ++replay) {
      const size_t logical = 10 + 6 - 1;
      const auto got = ConversationTailState(length, 10, 6, replay);
      const int keep = length == logical ? replay + 1 : ConversationRetainedRows(length, logical, size_t(replay) + 1);
      assert(bool(got) == (keep > 0));
      if (got) assert(got->length == logical && got->replay == keep - 1);
    }
  }
  ConversationResume bulk; bulk.length = 64; bulk.prompt_only = true;
  assert(ConversationPromptCompatible(bulk, 128, 0, 64));
  bulk.length = 48;
  assert(!ConversationPromptCompatible(bulk, 128, 0, 64));
  bulk.prompt_boundary = 48;
  assert(ConversationPromptCompatible(bulk, 128, 48, 64, true));
  // Canonical schedule (f390d73): without the opt-in split no chunk ends at the boundary, so an unaligned
  // prompt-end state cannot resume a longer prompt (with --exact-retry the hit would skip the zero grid).
  assert(!ConversationPromptCompatible(bulk, 128, 48, 64));
  assert(ConversationPromptCompatible(bulk, 48, 48, 64));
  {
    ConversationResume retry; retry.length = 21; retry.prompt_only = true; retry.prompt_boundary = 21;
    assert(ConversationPromptCompatible(retry, 40, 21, 8, true));
    assert(!ConversationPromptCompatible(retry, 40, 21, 8));
    assert(ConversationPromptCompatible(retry, 21, 21, 8));
    retry.length = 16;
    assert(ConversationPromptCompatible(retry, 40, 21, 8));
  }
  bulk.length = 112;
  assert(ConversationPromptCompatible(bulk, 128, 48, 64, true));
  assert(!ConversationPromptCompatible(bulk, 128, 48, 64));
  assert(!ConversationPromptCompatible(bulk, 128, 0, 64, true));
  bulk.prompt_only = false;
  assert(!ConversationPromptCompatible(bulk, 128, 48, 64, true));
  assert(ConversationPromptCompatible(bulk, 128, 48, 0));
  assert(cache.Find(a, 0)->value == 1);
  assert(cache.Insert(b, 0, 40, std::make_shared<Snapshot>(Snapshot{3})));
  assert(cache.Bytes() == 70 && cache.Evictions() == 1);
  assert(cache.Find(longer, 0)->value == 1);
  assert(!cache.Insert(b, 0, 101, std::make_shared<Snapshot>(Snapshot{9})));
  assert(cache.Find(b, 0)->value == 3);
  assert(cache.Insert(a, 0, 20, std::make_shared<Snapshot>(Snapshot{4})));
  assert(cache.Bytes() == 60 && cache.Find(a, 0)->value == 4);
  PrefixCache<Snapshot> off(0);
  assert(!off.Insert(a, 0, 1, std::make_shared<Snapshot>(Snapshot{0})));
  assert(!off.Find(a, 0));
  SupersededEvictionOrder();
}
