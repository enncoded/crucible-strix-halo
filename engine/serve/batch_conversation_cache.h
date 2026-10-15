// One device worker owns this LRU. Retention classes never weaken consumer compatibility.
#pragma once
#include "../runtime/conversation_cache.h"
#include "../runtime/conversation_admission.h"
#include "../runtime/conversation_resume.h"
#include "cache_status.h"
#include <chrono>
#include <cstdio>
#include <functional>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>

namespace crucible {
enum class CacheRetention { Session, Shared };
// `speculative`: a grid point past the stable boundary, which only a later prompt that repeats it can use.
struct CacheFrontier {
  size_t position = 0;
  CacheRetention retention = CacheRetention::Session;
  bool speculative = false;
};
enum class SnapshotMemoryAction { Allocate, Free };
struct SnapshotMemoryLedger {
  virtual ~SnapshotMemoryLedger() = default;
  virtual void Update(SnapshotMemoryAction action, size_t bytes) noexcept = 0;
};
// A shared system-and-tools frontier sits on the same fixed grid, so another session's chunks
// before it are identical and its restore stays exact (E6).
inline std::vector<CacheFrontier> BatchCacheFrontiers(size_t length, size_t boundary, int rows, size_t shared = 0) {
  std::vector<CacheFrontier> frontiers;
  const size_t own = ConversationAdmission::ForPrompt(length, boundary, rows, 0).frontier;
  const size_t common = shared ? ConversationAdmission::ForPrompt(length, std::min(shared, boundary ? boundary : length),
                                                                  rows, 0).frontier : 0;
  if (common && common != own) frontiers.push_back({common, CacheRetention::Shared});
  if (own) frontiers.push_back({own, common == own ? CacheRetention::Shared : CacheRetention::Session});
  // A grid point inside the generation prompt: the next turn repeats the header but its last token (the reasoning
  // marker), so that state saves one chunk of prefill (crucible-t13 soak S5). Grid states are exact for any prompt
  // that repeats them; the stable boundary only predicts which states a later prompt repeats.
  const size_t header = rows > 0 && length ? (length - 1) / size_t(rows) * size_t(rows) : 0;
  if (boundary && header > own && header > common) frontiers.push_back({header, CacheRetention::Session, true});
  return frontiers;
}
// A prompt that repeats part of a cached tail but not all of it (soak405). Counts only, never tokens or text.
// `shorter`: the prompt is not longer than the tail (an edit or a retry), not a turn that follows it.
struct TailDivergence { size_t tail = 0, common = 0; bool shorter = false; };
inline std::string SessionReplayDivergenceLine(size_t prompt, size_t restored, const TailDivergence& missed) {
  char line[160];
  std::snprintf(line, sizeof line, "session_replay %s prompt=%zu restored=%zu tail=%zu first_mismatch=%zu\n",
                missed.shorter ? "shorter" : "diverged", prompt, restored, missed.tail, missed.common);
  return line;
}
// The resume record of a prompt frontier at `cursor` for a request with stable boundary `boundary`. A header
// frontier passes the boundary; its own check then uses the whole prompt (Capture checks it).
inline ConversationResume BatchFrontierResume(size_t cursor, size_t boundary) {
  ConversationResume state;
  state.length = cursor;
  state.prompt_only = true;
  state.prompt_boundary = cursor > boundary ? 0 : boundary;
  return state;
}
// The MTP state that an entry may hold: none (target state only), or the MTP rows of the whole prefix with the carry
// (mpos = length - 1). A gap or pending pairs is a fault.
// The key stays the tokens: drafts change only speed, so they must not decide whether a later turn hits.
inline bool BatchCacheMtpState(const ConversationResume& resume) {
  const bool empty = resume.mpos == 0 && !resume.have_carry && resume.pending_tokens.empty();
  const bool held = resume.length > 0 && resume.have_carry && resume.mpos == int(resume.length) - 1 &&
                    resume.pending_tokens.empty();
  return empty || held;
}
inline bool BatchCacheConsumerCompatible(const ConversationResume& resume, size_t length,
                                         size_t boundary, size_t warmup_start, int rows) {
  const size_t stable = boundary ? boundary : length;
  return rows > 0 && resume.prompt_only && resume.length > 0 && resume.length <= stable &&
         resume.length <= warmup_start && resume.length <= length && resume.length % size_t(rows) == 0 &&
         ConversationPromptCompatible(resume, length, boundary, rows);
}

// The default MemAvailable that a save keeps free (the server's --host-reserve-mib).
inline constexpr uint64_t kDefaultHostReserve = uint64_t(8) << 30;
template<class Snapshot>
class BatchConversationCache {
  struct MemoryLease {
    const std::shared_ptr<SnapshotMemoryLedger> ledger;
    const size_t bytes;
    MemoryLease(std::shared_ptr<SnapshotMemoryLedger> ledger, size_t bytes) : ledger(std::move(ledger)), bytes(bytes) {
      if (this->ledger) this->ledger->Update(SnapshotMemoryAction::Allocate, bytes);
    }
    ~MemoryLease() { if (ledger) ledger->Update(SnapshotMemoryAction::Free, bytes); }
  };
 public:
  struct Entry {
    std::shared_ptr<const MemoryLease> memory;
    std::shared_ptr<const Snapshot> snapshot;
    CacheRetention retention;
    bool guarded = false;  // a tail of a guarded request (reasoning): a replay from it shows the client keeps it
    // Lineage for the replay diagnostic: this entry's number, and the number of the longest entry that its tokens
    // extended when it was saved (0: none). A tail's parent is the state that its request restored or passed.
    uint64_t serial = 0, parent = 0;
  };
  // `live`: sessions that can run at once (the slot count). Each keeps one protected resume point.
  // `probation`: a new entry leaves first and is never a protected resume point until a later request uses it or
  // its branch (a hit, a slot resume, a trusted tail, or a save that extends a used state). One-time requests (a Pi
  // compaction summary: one long prompt that never continues) then cannot push out live sessions. Speed only.
  explicit BatchConversationCache(size_t budget = 0, std::shared_ptr<SnapshotMemoryLedger> ledger = {},
                                  size_t live = kMaxRows, uint64_t host_reserve = kDefaultHostReserve,
                                  bool probation = true, double idle_seconds = kDefaultIdleSeconds)
      : ledger_(std::move(ledger)), cache_(budget), live_(std::max<size_t>(live, 1)), host_reserve_(host_reserve),
        probation_(probation), idle_seconds_(idle_seconds) {}
  // Tests set the wall clock (seconds); the default is the steady clock.
  void SetClock(std::function<double()> clock) { clock_now_ = std::move(clock); }
  uint64_t HostReserve() const { return host_reserve_; }
  std::shared_ptr<const Entry> Find(std::span<const int> ids, const ConversationPolicy& policy,
                                   size_t boundary, size_t warmup_start, int rows) {
    const size_t stable = boundary ? boundary : ids.size();
    return Confirm(cache_.FindEntry(ids, policy, std::min(stable, warmup_start), [&](const Entry& entry) {
      return BatchCacheConsumerCompatible(entry.snapshot->resume, ids.size(), boundary, warmup_start, rows);
    }));
  }
  // Session replay (G15): the longest generated tail this prompt repeats, leaving at least one row to predict.
  std::shared_ptr<const Entry> FindTail(std::span<const int> ids, const ConversationPolicy& policy) {
    if (ids.size() < 2) return {};
    return Confirm(cache_.FindEntry(ids, policy, ids.size() - 1,
                                    [](const Entry& entry) { return !entry.snapshot->resume.prompt_only; }));
  }
  // What Find and FindTail would restore, without a change to the cache (in-slot checkpoints compare against it).
  size_t PeekFind(std::span<const int> ids, const ConversationPolicy& policy, size_t boundary, size_t warmup_start,
                  int rows) const {
    const size_t stable = boundary ? boundary : ids.size();
    return cache_.Longest(ids, policy, std::min(stable, warmup_start), [&](const Entry& entry) {
      return BatchCacheConsumerCompatible(entry.snapshot->resume, ids.size(), boundary, warmup_start, rows);
    });
  }
  size_t PeekTail(std::span<const int> ids, const ConversationPolicy& policy) const {
    if (ids.size() < 2) return 0;
    return cache_.Longest(ids, policy, ids.size() - 1, [](const Entry& entry) { return !entry.snapshot->resume.prompt_only; });
  }
  // The newest tail saved after the restored entry is this conversation's last answer. A prompt that leaves it lost
  // replay by its own tokens: `common` is the first position where it differs. Compares one tail, from the restore on.
  std::optional<TailDivergence> MissedTail(std::span<const int> ids, const ConversationPolicy& policy,
                                           const Entry* restored) const {
    if (!restored) return {};
    std::span<const int> newest;
    uint64_t serial = 0;
    cache_.ForEach(policy, [&](std::span<const int> tail, const Entry& entry) {
      if (entry.parent == restored->serial && !entry.snapshot->resume.prompt_only && entry.serial > serial)
        newest = tail, serial = entry.serial;
    });
    const size_t start = restored->snapshot->hist.size(), limit = std::min(newest.size(), ids.size());
    if (!serial || start > limit) return {};
    const size_t common = size_t(std::mismatch(newest.begin() + start, newest.begin() + limit, ids.begin() + start).first -
                                 newest.begin());
    if (common >= newest.size()) return {};
    return TailDivergence{newest.size(), common, ids.size() <= newest.size()};
  }
  // A tail holds target state, and from a full-window request also the MTP rows (BatchCacheMtpState). Without them
  // the next turn starts a fresh MTP warm-up, as a prompt restore does.
  // A tail is speculative: the next prompt repeats it only when the client sends back every output token, and many
  // clients drop the reasoning. A speculative tail never pushes out a live session's resume point, its own included.
  // `trusted`: the request itself resumed from its branch's previous tail and its client keeps what this tail holds,
  // so this tail replaces the older entries of its branch, as before.
  // `resident`: a slot still holds this state, so the copy takes only free room or superseded entries; the slot
  // saves it later if it must leave the device (Release in batched_serve.h).
  // `hist` is the cache key. `tokens` is the snapshot's own history when the key differs (image rows key by image).
  template<class Create>
  bool CaptureTail(std::span<const int> hist, const ConversationPolicy& policy, size_t bytes, uint64_t available,
                   Create create, bool trusted = false, bool guarded = false, bool resident = false,
                   std::span<const int> tokens = {}) {
    if (tokens.empty()) tokens = hist;
    if (tokens.size() != hist.size()) throw std::invalid_argument("tail key and tokens differ in length");
    Try(hist.size());
    if (policy.spec != 0 || policy.mtp_prefix != 0 || policy.prefill_rows < 1 || hist.empty()) return Refuse("policy");
    if (!bytes || bytes > cache_.Budget()) return Refuse("budget");
    const uint64_t parent = Parent(hist, policy);  // before eviction can remove it
    const bool proven = ExtendsProven(hist, policy);  // also before eviction
    const auto marked = trusted ? Supersede(hist, policy, [](const Entry& entry) {
      return entry.retention == CacheRetention::Session;
    }) : std::set<const Item*>{};
    if (!Admit(bytes, available, resident, !trusted)) return Unmark(marked, refusal_);
    try {
      auto lease = std::make_shared<MemoryLease>(ledger_, bytes);
      auto snapshot = create();
      if (!snapshot || snapshot->hist.size() != hist.size() ||
          !std::equal(snapshot->hist.begin(), snapshot->hist.end(), tokens.begin()) ||
          snapshot->resume.prompt_only || snapshot->resume.length != hist.size() ||
          snapshot->resume.replay < 0 || snapshot->resume.replay > kMaxRows ||
          !BatchCacheMtpState(snapshot->resume))
        throw std::runtime_error("session replay needs a generated tail with target state or full MTP rows");
      auto entry = std::make_shared<Entry>(Entry{std::move(lease), std::move(snapshot), CacheRetention::Session, guarded,
                                                 ++serials_, parent});
      return Saved(cache_.Insert(hist, policy, bytes, std::move(entry)), trusted, trusted || proven);
    } catch (const std::bad_alloc&) {
      return Unmark(marked, "memory");
    }
  }
  // `ids` is the cache key; `tokens` as in CaptureTail.
  template<class Create>
  bool Capture(std::span<const int> ids, const ConversationPolicy& policy, const CacheFrontier& frontier,
               size_t bytes, uint64_t available, Create create, std::span<const int> tokens = {}) {
    if (tokens.empty()) tokens = ids;
    if (tokens.size() != ids.size()) throw std::invalid_argument("prompt key and tokens differ in length");
    Try(frontier.position);
    if (policy.spec != 0 || policy.mtp_prefix != 0 || policy.prefill_rows < 1 ||
        !frontier.position || frontier.position > ids.size() || frontier.position % size_t(policy.prefill_rows))
      return Refuse("policy");
    if (!bytes || bytes > cache_.Budget()) return Refuse("budget");
    // Older Session frontiers that this one extends leave first.
    // Shared frontiers stay in plain LRU: other sessions can still restore them.
    // A tail is never superseded by a prompt frontier: replay makes it the most valuable entry (G15).
    // A speculative frontier (past the boundary) is a speculative tail: a retry of this prompt, or a next turn that
    // drops the reasoning, cannot use it. It marks nothing until a hit confirms it.
    const auto marked = frontier.speculative ? std::set<const Item*>{} :
        Supersede(ids.first(frontier.position), policy, [](const Entry& entry) {
          return entry.retention == CacheRetention::Session && entry.snapshot->resume.prompt_only;
        });
    // Before Admit: the state that this frontier extends is superseded now, so Admit evicts it first.
    const bool proven = ExtendsProven(ids.first(frontier.position), policy);
    if (!Admit(bytes, available, false, frontier.speculative)) return Unmark(marked, refusal_);
    try {
      auto lease = std::make_shared<MemoryLease>(ledger_, bytes);
      auto snapshot = create();
      if (!snapshot || snapshot->hist.size() != frontier.position ||
          !std::equal(snapshot->hist.begin(), snapshot->hist.end(), tokens.begin()) ||
          !BatchCacheConsumerCompatible(snapshot->resume, ids.size(), snapshot->resume.prompt_boundary,
                                        ids.size(), policy.prefill_rows) ||
          snapshot->resume.length != frontier.position || snapshot->resume.replay ||
          !BatchCacheMtpState(snapshot->resume))
        throw std::runtime_error("batch capture needs a committed frontier with target state or full MTP rows");
      auto entry = std::make_shared<Entry>(Entry{std::move(lease), std::move(snapshot), frontier.retention, false,
                                                 ++serials_});
      // A prompt frontier holds client tokens, so it is a confirmed resume point; a speculative one is not yet.
      return Saved(cache_.Insert(ids.first(frontier.position), policy, bytes, std::move(entry)), !frontier.speculative,
                   proven);
    } catch (const std::bad_alloc&) {
      return Unmark(marked, "memory");
    }
  }
  // A slot resume copies nothing from the host, so it marks the session's newest host entry that the prompt extends
  // as used: that session stays live.
  void Touch(std::span<const int> ids, const ConversationPolicy& policy) {
    Item* best = nullptr;
    const double now = Now();
    cache_.ForEach([&](Item& item) {
      if (item.policy != policy || item.ids.size() > ids.size() || !std::equal(item.ids.begin(), item.ids.end(), ids.begin()))
        return;
      // Every host entry of the session is in use: the longest one is often the slot's copy, which is never a head,
      // so a head that only the copy refreshed would go stale while its session runs.
      item.active_at = now;
      if (!best || item.ids.size() > best->ids.size()) best = &item;
    });
    if (best) best->used = ++tick_, best->active = ++clock_, best->proven = true;
  }
  // Whether the entry with exactly this key has left probation.
  bool Proven(std::span<const int> ids, const ConversationPolicy& policy) {
    bool proven = false;
    cache_.ForEach([&](Item& item) { proven |= item.policy == policy && item.proven && std::ranges::equal(item.ids, ids); });
    return proven;
  }
  // The tail state that slot `slot` holds ("One copy"). A host entry of that state is
  // a copy: it leaves before any state that exists only on the host, and it is never a protected resume point.
  void SetResident(int slot, std::span<const int> hist, int replay) {
    resident_[slot] = {std::vector<int>(hist.begin(), hist.end()), replay};
  }
  void ClearResident(int slot) { resident_.erase(slot); }
  bool HoldsTail(std::span<const int> hist, const ConversationPolicy& policy, int replay) {
    bool held = false;
    cache_.ForEach([&](Item& item) {
      held |= item.policy == policy && !item.snapshot->snapshot->resume.prompt_only &&
              item.snapshot->snapshot->resume.replay == replay && std::ranges::equal(item.ids, hist);
    });
    return held;
  }
  // Host bytes of tail entries that a slot also holds.
  size_t CopyBytes() {
    size_t bytes = 0;
    cache_.ForEach([&](Item& item) { if (Copy(item)) bytes += item.bytes; });
    return bytes;
  }
  // The disk tier (state_disk.h): `spill` receives each evicted entry of the classes kStale to kWaiting (a state
  // that exists nowhere else) with its key. A superseded entry and a copy of a slot's state stay off the disk.
  // The writer holds a spilled entry in host memory until its write ends, so it frees nothing for the save that
  // evicted it. `room` is the host memory above the reserve that this save leaves, less the entries that the
  // hook already holds; the hook returns whether it holds this one.
  using Spill = std::function<bool(std::span<const int> key, const ConversationPolicy&,
                                   const std::shared_ptr<const Entry>&, int tier, uint64_t room)>;
  void SetSpill(Spill spill) {
    spill_ = std::move(spill);
    cache_.SetEvicted([victims = victims_](const Item& item, int tier) {
      if (tier >= kStale && tier <= kWaiting)
        victims->push_back({item.ids, item.policy, item.snapshot, tier, item.bytes});
    });
  }
  // Every entry with its key and superseded mark, the most recently used first (a stopping engine writes them to
  // disk in this order, so a flush that runs out of time loses the oldest).
  template<class Visit>
  void ForEachEntry(Visit visit) {
    std::vector<Item*> items;
    cache_.ForEach([&](Item& item) { items.push_back(&item); });
    std::stable_sort(items.begin(), items.end(), [](const Item* a, const Item* b) { return a->used > b->used; });
    for (Item* item : items) visit(std::span<const int>(item->ids), item->policy, item->snapshot, item->superseded);
  }
  size_t Bytes() const { return cache_.Bytes(); }
  size_t Budget() const { return cache_.Budget(); }
  size_t Evictions() const { return cache_.Evictions(); }
  size_t SupersededEvictions() const { return cache_.SupersededEvictions(); }
  // Rounds of saves and hits (those not on probation) after which a session with no new save or hit counts as idle.
  static constexpr uint64_t kIdleRounds = 16;
  // A resume point that no request used for this long (wall clock) is idle too (a review: with probation, the
  // session clock stops while only one-time requests run, so an abandoned session could stay protected for ever).
  static constexpr double kDefaultIdleSeconds = 1800;
  BatchCacheStatus Status() const { return {cache_.Entries(), cache_.Bytes(), cache_.Evictions(), attempts_, last_, {}}; }
 private:
  using Item = typename PrefixCache<Entry, ConversationPolicy>::Entry;
  void Try(size_t tokens) { ++attempts_; last_ = {tokens, false, {}}; }
  // The longest entry that `hist` extends. It runs once per tail save, as Supersede does.
  uint64_t Parent(std::span<const int> hist, const ConversationPolicy& policy) const {
    uint64_t parent = 0;
    size_t longest = 0;
    cache_.ForEach(policy, [&](std::span<const int> ids, const Entry& entry) {
      if (ids.size() > longest && ids.size() < hist.size() && std::equal(ids.begin(), ids.end(), hist.begin()))
        parent = entry.serial, longest = ids.size();
    });
    return parent;
  }
  // Whether `ids` extends a proven state. A save asks before Admit, which can evict that state: in a full cache a
  // session's next frontier evicts the frontier that it supersedes, and the new one then landed on probation.
  bool ExtendsProven(std::span<const int> ids, const ConversationPolicy& policy) {
    bool proven = false;
    cache_.ForEach([&](Item& item) {
      proven |= item.proven && item.policy == policy && item.ids.size() < ids.size() &&
                std::equal(item.ids.begin(), item.ids.end(), ids.begin());
    });
    return proven;
  }
  bool Refuse(const char* reason) { last_.refused = reason; return false; }
  // Marks the Session entries that `ids` extends, as PrefixCache::Supersede, and returns the new marks.
  template<class Accept>
  std::set<const Item*> Supersede(std::span<const int> ids, const ConversationPolicy& policy, Accept accept) {
    std::set<const Item*> marked;
    cache_.ForEach([&](Item& item) {
      if (item.superseded || item.policy != policy || item.ids.size() >= ids.size() ||
          !std::equal(item.ids.begin(), item.ids.end(), ids.begin()) || !accept(*item.snapshot)) return;
      item.superseded = true;
      marked.insert(&item);
    });
    return marked;
  }
  // A refused save keeps the marks of no older entry: a refused trusted tail must not demote its session's resume
  // point. Entries that a failed save already evicted are gone from the list, so only live items match.
  bool Unmark(const std::set<const Item*>& marked, const char* reason) {
    if (!marked.empty()) cache_.ForEach([&](Item& item) { if (marked.contains(&item)) item.superseded = false; });
    return Refuse(reason);
  }
  bool Copy(const Item& item) const {
    const auto& resume = item.snapshot->snapshot->resume;
    return !resume.prompt_only && std::any_of(resident_.begin(), resident_.end(), [&](const auto& held) {
      return held.second.second == resume.replay && std::ranges::equal(held.second.first, item.ids);
    });
  }
  // `proven`: the request itself continued its branch (a trusted tail). A save that extends a proven state is proven.
  bool Saved(bool saved, bool confirmed, bool proven) {
    // An eviction outside Admit gets no room. None happens today: Insert evicts only tiers 0 and 1, which never spill.
    HandVictims(0);
    if (!saved) return Refuse("budget");
    Item& item = cache_.Newest();
    item.used = ++tick_;
    item.confirmed = confirmed ? tick_ : 0;
    item.proven = proven;
    cache_.ForEach([&](Item& other) { item.proven |= other.proven && Extends(item, other); });
    // A save on probation does not move the session clock: one-time requests never make a live session idle.
    item.active = OnProbation(item) ? clock_ : ++clock_;
    item.active_at = Now();
    return last_.saved = true;
  }
  static bool Extends(const Item& longer, const Item& prefix) {
    return longer.policy == prefix.policy && prefix.ids.size() < longer.ids.size() &&
           std::equal(prefix.ids.begin(), prefix.ids.end(), longer.ids.begin());
  }
  // A hit shows the client repeated this state: it becomes its session's resume point, and the Session entries that
  // it extends leave first. A tail stays protected only from its first hit on.
  std::shared_ptr<const Entry> Confirm(Item* hit) {
    if (!hit) return {};
    hit->used = hit->confirmed = ++tick_;
    hit->active = ++clock_;
    hit->active_at = Now();
    hit->proven = true;
    cache_.ForEach([&](Item& item) {
      if (&item != hit && item.snapshot->retention == CacheRetention::Session && Extends(*hit, item))
        item.superseded = true;
    });
    return hit->snapshot;
  }
  // Eviction classes, first to leave first. A session's resume point is its newest
  // confirmed entry that no other confirmed entry extends. The `live_` most recently used of them are live; a resume
  // point not used in the last kIdleRounds * live_ saves and hits is idle.
  enum Tier : int {
    kKept = -1,      // the resume point of a live session: never leaves
    kSuperseded = 0,
    kCopy = 1,       // a copy of a slot's state
    kStale = 2,      // a session that no request used for idle_seconds_ (wall clock): it leaves before new states
    kTrial = 3,      // on probation (no later request used it yet) and speculative (no client prompt showed it)
    kProbation = 4,  // on probation, a client prompt's state (one-time requests)
    kIdle = 5,       // an idle session's entries
    kOverShare = 6,  // extra entries of a session above its fair share
    kExtra = 7,      // other extra entries
    kWaiting = 8,    // the resume point of a recent session beyond the live count (more sessions than slots)
  };
  bool OnProbation(const Item& item) const { return probation_ && !item.proven; }
  double Now() const {
    return clock_now_ ? clock_now_()
                      : std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
  }
  std::map<const Item*, int> Plan() {
    const double now = Now();
    std::vector<Item*> heads, confirmed;
    std::set<const Item*> copies;
    cache_.ForEach([&](Item& item) { if (Copy(item)) copies.insert(&item); });
    cache_.ForEach([&](Item& item) {
      if (!item.superseded && item.confirmed && !copies.contains(&item) && !OnProbation(item)) confirmed.push_back(&item);
    });
    for (Item* item : confirmed)
      if (std::none_of(confirmed.begin(), confirmed.end(), [&](const Item* other) { return Extends(*other, *item); }))
        heads.push_back(item);
    std::stable_sort(heads.begin(), heads.end(), [](const Item* a, const Item* b) { return a->used > b->used; });
    std::map<const Item*, bool> live;
    std::set<const Item*> idle, stale;
    for (size_t i = 0; i < heads.size(); ++i) {
      if (now - heads[i]->active_at > idle_seconds_) stale.insert(heads[i]);
      if (clock_ - heads[i]->active > kIdleRounds * live_ || stale.contains(heads[i])) idle.insert(heads[i]);
      live[heads[i]] = i < live_ && !idle.contains(heads[i]);
    }
    // Each other entry belongs to the longest resume point that it extends (none for shared prefixes).
    std::map<const Item*, const Item*> owner;
    std::map<const Item*, size_t> held;
    cache_.ForEach([&](Item& item) {
      const Item* best = nullptr;
      for (const Item* head : heads)
        if (Extends(item, *head) && (!best || head->ids.size() > best->ids.size())) best = head;
      owner[&item] = best;
      held[best ? best : &item] += item.bytes;
    });
    const size_t lives = std::max<size_t>(1, std::count_if(live.begin(), live.end(), [](const auto& h) { return h.second; }));
    const size_t share = cache_.Budget() / lives;
    std::map<const Item*, int> tier;
    cache_.ForEach([&](Item& item) {
      const Item* head = live.contains(&item) ? &item : owner[&item];
      if (item.superseded) tier[&item] = kSuperseded;
      else if (copies.contains(&item)) tier[&item] = kCopy;
      else if (head && stale.contains(head)) tier[&item] = kStale;
      else if (OnProbation(item)) tier[&item] = item.confirmed ? kProbation : kTrial;
      else if (live.contains(&item)) tier[&item] = live[&item] ? kKept : idle.contains(&item) ? kIdle : kWaiting;
      else if (head && idle.contains(head)) tier[&item] = kIdle;
      else tier[&item] = head && held[head] > share ? kOverShare : kExtra;
    });
    return tier;
  }
  // Hands this save's victims to the spill hook. The save leaves available + victims - bytes above zero; the
  // reserve takes its part, and each held victim takes its bytes from the rest. A victim that is not held frees its
  // memory when `victims` ends.
  void SpillVictims(uint64_t available, size_t bytes) {
    uint64_t have = available;
    for (const auto& victim : *victims_) have = victim.bytes > UINT64_MAX - have ? UINT64_MAX : have + victim.bytes;
    const uint64_t need = host_reserve_ + bytes;
    HandVictims(have > need ? have - need : 0);
  }
  void HandVictims(uint64_t room) {
    if (victims_->empty()) return;
    const auto victims = std::move(*victims_);
    victims_->clear();
    for (const auto& victim : victims)
      if (spill_(victim.key, victim.policy, victim.entry, victim.tier, room))
        room -= std::min<uint64_t>(room, victim.bytes);
  }
  // Budget room and the host reserve (default 8 GiB of MemAvailable) together (ConversationCache ReserveHost): evicted snapshots count
  // toward the reserve, and a save that cannot fit evicts nothing. A save that only live resume points could make
  // room for is refused as "protected". A copy of a slot's state takes only superseded entries ("resident"). A
  // speculative tail never takes a waiting session's resume point: with more sessions than slots, that state is the
  // next turn of a session that has no slot (tests/slot_copy_soak_test.cpp).
  bool Admit(size_t bytes, uint64_t available, bool resident = false, bool speculative = false) {
    const uint64_t reserve = host_reserve_;
    // No eviction needed: skip the plan.
    if (cache_.ReserveHost(bytes, available, reserve, [](const Item&) { return -1; })) return true;
    const auto plan = Plan();
    if (cache_.ReserveHost(bytes, available, reserve, [&](const Item& item) {
          const int tier = plan.at(&item);
          // A speculative save never takes a client prompt's state that waits for its session's next turn.
          if ((resident && tier > kSuperseded) || (speculative && (tier == kWaiting || tier == kProbation)))
            return int(kKept);
          return tier;
        })) {
      SpillVictims(available, bytes);
      return true;
    }
    if (resident) {
      refusal_ = "resident";
      return false;
    }
    // An empty cache would hold it: only live resume points stand in the way.
    refusal_ = available + cache_.Bytes() >= reserve + bytes ? "protected" : "host";
    return false;
  }
  std::shared_ptr<SnapshotMemoryLedger> ledger_;
  PrefixCache<Entry, ConversationPolicy> cache_;
  const size_t live_;
  const uint64_t host_reserve_;
  const bool probation_;
  const double idle_seconds_;
  std::function<double()> clock_now_;
  std::map<int, std::pair<std::vector<int>, int>> resident_;  // slot: tail tokens and pending replay rows
  uint64_t tick_ = 0;
  // Counts saves and hits that are not on probation (without probation: every save and hit, as tick_).
  uint64_t clock_ = 0;
  const char* refusal_ = "host";
  uint64_t attempts_ = 0, serials_ = 0;
  BatchCacheSave last_;
  struct Victim {
    std::vector<int> key;
    ConversationPolicy policy;
    std::shared_ptr<const Entry> entry;
    int tier;
    size_t bytes;
  };
  Spill spill_;
  // Evicted in the current Admit, waiting for SpillVictims (shared with the eviction hook, so no `this` capture).
  std::shared_ptr<std::vector<Victim>> victims_ = std::make_shared<std::vector<Victim>>();
};
}  // namespace crucible
