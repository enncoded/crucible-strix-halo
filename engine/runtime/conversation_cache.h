// Immutable entries own their KV bytes, so edits cannot mutate an older branch.
#pragma once
#include <algorithm>
#include <cstddef>
#include <iterator>
#include <list>
#include <memory>
#include <span>
#include <vector>
#include <cstdint>
#include <functional>

namespace crucible {
template <class Snapshot, class Policy = int>
class PrefixCache {
 public:
  explicit PrefixCache(size_t budget) : budget_(budget) {}
  // Batch retention fields (BatchConversationCache). Serial serve leaves them at zero.
  struct Entry {
    std::vector<int> ids;
    Policy policy;
    size_t bytes;
    std::shared_ptr<const Snapshot> snapshot;
    bool superseded = false;
    uint64_t confirmed = 0;  // when a client prompt last showed this state is its prefix; 0: speculative
    uint64_t used = 0;       // the last save or hit
    bool proven = false;     // a later request used this state or its branch (probation, BatchConversationCache)
    uint64_t active = 0;     // the session clock at the last save or hit that is not on probation
    double active_at = 0;    // the wall clock (seconds) at that save or hit
  };
  template<class Accept = decltype([](const Snapshot&) { return true; })>
  std::shared_ptr<const Snapshot> Find(std::span<const int> ids, const Policy& policy,
                                       size_t max_length = SIZE_MAX, Accept accept = {}) {
    Entry* found = FindEntry(ids, policy, max_length, accept);
    return found ? found->snapshot : nullptr;
  }
  // As Find, but returns the hit entry (now the most recent) for the caller's retention fields.
  template<class Accept = decltype([](const Snapshot&) { return true; })>
  Entry* FindEntry(std::span<const int> ids, const Policy& policy, size_t max_length = SIZE_MAX, Accept accept = {}) {
    auto best = entries_.end();
    for (auto it = entries_.begin(); it != entries_.end(); ++it) {
      if (it->policy != policy || it->ids.size() > ids.size() || it->ids.size() > max_length ||
          (best != entries_.end() && it->ids.size() <= best->ids.size())) continue;
      if (std::equal(it->ids.begin(), it->ids.end(), ids.begin()) && accept(*it->snapshot)) best = it;
    }
    if (best == entries_.end()) return nullptr;
    best->superseded = false;  // A hit shows the entry is still useful, for example on a fork that continues.
    entries_.splice(entries_.begin(), entries_, best);
    return &entries_.front();
  }
  // The length that FindEntry would return, without a change to the LRU order or the marks (0: none).
  template<class Accept = decltype([](const Snapshot&) { return true; })>
  size_t Longest(std::span<const int> ids, const Policy& policy, size_t max_length = SIZE_MAX, Accept accept = {}) const {
    size_t best = 0;
    for (const auto& entry : entries_)
      if (entry.policy == policy && entry.ids.size() <= ids.size() && entry.ids.size() <= max_length &&
          entry.ids.size() > best && std::equal(entry.ids.begin(), entry.ids.end(), ids.begin()) &&
          accept(*entry.snapshot))
        best = entry.ids.size();
    return best;
  }
  // The entry that the last successful Insert added.
  Entry& Newest() { return entries_.front(); }
  template<class Visit>
  void ForEach(Visit visit) { for (auto& entry : entries_) visit(entry); }
  // Read-only visit of every entry under `policy`, newest first: visit(ids, snapshot). A diagnostic never changes
  // what the cache keeps.
  template<class Visit>
  void ForEach(const Policy& policy, Visit visit) const {
    for (const auto& entry : entries_)
      if (entry.policy == policy) visit(std::span<const int>(entry.ids), *entry.snapshot);
  }
  bool Insert(std::span<const int> ids, const Policy& policy, size_t bytes,
              std::shared_ptr<const Snapshot> snapshot) {
    if (ids.empty() || !snapshot || !bytes || bytes > budget_) return false;
    for (auto it = entries_.begin(); it != entries_.end(); ++it) {
      if (it->policy == policy && it->ids.size() == ids.size() &&
          std::equal(it->ids.begin(), it->ids.end(), ids.begin())) {
        bytes_ -= it->bytes;
        entries_.erase(it);
        break;
      }
    }
    Reserve(bytes);
    entries_.push_front({std::vector<int>(ids.begin(), ids.end()), policy, bytes, std::move(snapshot)});
    bytes_ += bytes;
    return true;
  }
  // Marks the older frontiers that `ids` extends, so eviction removes them before any other entry.
  // Without session identity, a fork also marks its parent's frontier; a later hit clears that mark.
  template<class Accept>
  size_t Supersede(std::span<const int> ids, const Policy& policy, Accept accept) {
    size_t marked = 0;
    for (auto& entry : entries_) {
      if (entry.superseded || entry.policy != policy || entry.ids.size() >= ids.size() ||
          !std::equal(entry.ids.begin(), entry.ids.end(), ids.begin()) || !accept(*entry.snapshot)) continue;
      entry.superseded = true;
      ++marked;
    }
    return marked;
  }
  // Room for `bytes` in the budget and in host memory (`available` measured now, keep `reserve` free). Evicted
  // entries return their host memory, so they count. When even an empty cache would leave the reserve short, it
  // evicts nothing (a refused save keeps the older states). A restore that still holds an evicted snapshot can
  // delay that memory for one request.
  bool ReserveHost(size_t bytes, uint64_t available, uint64_t reserve) {
    return ReserveHost(bytes, available, reserve, SupersededFirst);
  }
  // `tier` orders the victims: a lower tier leaves first, least recently used first within a tier. A negative tier
  // never leaves. When the evictable entries cannot make the room, nothing is evicted.
  template<class Tier>
  bool ReserveHost(size_t bytes, uint64_t available, uint64_t reserve, Tier tier) {
    if (!bytes || bytes > budget_) return false;
    const uint64_t needed = reserve + bytes;
    const uint64_t short_by = needed > available ? needed - available : 0;
    const size_t over = bytes_ + bytes > budget_ ? bytes_ + bytes - budget_ : 0;
    const uint64_t free = std::max<uint64_t>(over, short_by);
    if (!free) return true;
    std::vector<std::pair<int, typename std::list<Entry>::iterator>> order;
    for (auto it = entries_.end(); it != entries_.begin();) {
      --it;
      const int rank = tier(static_cast<const Entry&>(*it));
      if (rank >= 0) order.emplace_back(rank, it);
    }
    std::stable_sort(order.begin(), order.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
    uint64_t room = 0;
    size_t victims = 0;
    while (room < free && victims < order.size()) room += order[victims++].second->bytes;
    if (room < free) return false;
    for (size_t i = 0; i < victims; ++i) Evict(order[i].second, order[i].first);
    return true;
  }
  // Call before allocating GPU snapshots to limit transient memory too.
  bool Reserve(size_t bytes) {
    if (!bytes || bytes > budget_) return false;
    return bytes_ <= budget_ - bytes || ReserveHost(bytes, UINT64_MAX, 0);
  }
  size_t Budget() const { return budget_; }
  size_t Bytes() const { return bytes_; }
  size_t Entries() const { return entries_.size(); }
  size_t Evictions() const { return evictions_; }
  size_t SupersededEvictions() const { return superseded_evictions_; }
  // Runs for each entry that eviction removes, with its tier, before it leaves (the disk tier, state_disk.h).
  void SetEvicted(std::function<void(const Entry&, int tier)> evicted) { evicted_ = std::move(evicted); }
 private:
  static int SupersededFirst(const Entry& entry) { return entry.superseded ? 0 : 1; }
  void Evict(typename std::list<Entry>::iterator victim, int tier) {
    if (evicted_) evicted_(*victim, tier);
    superseded_evictions_ += victim->superseded;
    bytes_ -= victim->bytes;
    entries_.erase(victim);
    ++evictions_;
  }
  size_t budget_, bytes_ = 0, evictions_ = 0, superseded_evictions_ = 0;
  std::list<Entry> entries_;
  std::function<void(const Entry&, int)> evicted_;
};
}  // namespace crucible
