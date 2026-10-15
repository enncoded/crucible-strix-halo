// The disk tier of batch serve: which disk states a request can restore, the stage
// step before an open, the spill of host evictions, and the flush of a stopping engine. CPU only; batched_serve.h
// does the device copies.
#pragma once
#include "batch_conversation_cache.h"
#include "../runtime/state_disk.h"

namespace crucible {
// The settings of the disk identity: the switches, the pack, the state format and the MTP slot rows. A saved state's
// MTP rows must fit the slot that restores it: another MTP bound is another directory.
inline std::vector<std::string> BatchStateDiskSettings(const std::string& checkpoint, int prefill_rows, bool vision,
                                                       int mtp_rows, char** environ_list) {
  auto settings = StateDiskSettings(environ_list, prefill_rows, vision);
  settings.push_back("pack=" + std::filesystem::path(checkpoint).lexically_normal().filename().string());
  settings.push_back("format=" + std::to_string(kStateVersion));
  settings.push_back("mtp_rows=" + std::to_string(mtp_rows));
  return settings;
}
// What a request can restore, as BatchPrefixCache::Lookup decides for the host cache.
struct DiskQuery {
  std::span<const int> key;  // the prompt's cache key
  ConversationPolicy policy;
  size_t boundary = 0;       // cache_prefix_tokens
  size_t limit = 0;          // the last prompt-frontier position that this request can use (Lookup's limit)
  int rows = 0;              // the prefill grid
  bool replay = false;       // session replay: generated tails are restorable too
};
// A generated tail leaves at least one prompt row to run; a prompt frontier passes the host cache's own check.
inline bool DiskStateFits(const StateMeta& meta, const DiskQuery& query) {
  const size_t prompt = query.key.size();
  if (!meta.resume.prompt_only) return query.replay && meta.key.size() + 1 <= prompt;
  const size_t stable = query.boundary ? query.boundary : prompt;
  return meta.key.size() <= std::min(stable, query.limit) &&
         BatchCacheConsumerCompatible(meta.resume, prompt, query.boundary, query.limit, query.rows);
}

template<class Snapshot>
class BatchDiskTier {
 public:
  using Ready = typename StateDisk<Snapshot>::ReadyState;
  BatchDiskTier(StateDiskOptions options, bool compact) : disk_(std::move(options)), compact_(compact) {}
  bool Enabled() const { return disk_.Enabled(); }
  StateDisk<Snapshot>& Disk() { return disk_; }

  // A host eviction (BatchConversationCache::SetSpill). Only queues: the writer thread does the disk work. `room`: the
  // host memory that the queued state may hold, less the reads that Stage reserved and the reader has not
  // allocated yet. Returns whether the writer holds the entry.
  template<class Entry>
  bool Spill(std::span<const int> key, const ConversationPolicy& policy, const std::shared_ptr<const Entry>& entry,
             uint64_t room = UINT64_MAX) {
    const uint64_t reading = disk_.UnallocatedReadBytes();
    room = room == UINT64_MAX ? room : room > reading ? room - reading : 0;
    return disk_.Spill(key, policy, compact_, entry->retention == CacheRetention::Shared, entry->guarded, entry,
                       *entry->snapshot, false, room).empty();
  }
  // Starts the read of the longest disk state for `query` when it is longer than `floor` (what the slots and the
  // host cache hold). `room`: bytes that read states may hold in host memory.
  StageStatus Stage(const DiskQuery& query, size_t floor, uint64_t room) {
    if (!Enabled()) return {};
    const auto meta = disk_.Best(query.key, query.policy, compact_, [&](const StateMeta& m) {
      return DiskStateFits(m, query);
    });
    if (!meta || meta->key.size() <= floor) return {};
    return disk_.Stage(*meta, room);
  }
  // The longest read state for `query`, if longer than `beat` (the host cache's state). Call Consume after the
  // restore.
  Ready Take(const DiskQuery& query, size_t beat) {
    if (!Enabled()) return {};
    auto ready = disk_.Ready(query.key, query.policy, compact_, [&](const StateMeta& m) { return DiskStateFits(m, query); });
    if (!ready.snapshot || ready.meta.key.size() <= beat) return {};
    return ready;
  }
  // The length that Take would give (0: none).
  size_t Peek(const DiskQuery& query) {
    const auto ready = Take(query, 0);
    return ready.snapshot ? ready.meta.key.size() : 0;
  }
  void Consume(const Ready& ready) { disk_.Consume(ready); }
  // A stopping engine: every host entry that is not superseded and is long enough goes to disk. Returns the count
  // queued; the caller then drains.
  template<class Cache>
  size_t FlushHost(Cache& cache) {
    size_t queued = 0;
    cache.ForEachEntry([&](std::span<const int> key, const ConversationPolicy& policy, const auto& entry, bool superseded) {
      if (superseded || key.size() < disk_.MinTokens()) return;
      queued += disk_.Spill(key, policy, compact_, entry->retention == CacheRetention::Shared, entry->guarded, entry,
                            *entry->snapshot, true).empty();
    });
    return queued;
  }
  // A stopping engine: one slot state, saved to host memory by the caller. Waits for its write, so only one such
  // state holds extra host memory at a time.
  bool FlushOne(std::span<const int> key, const ConversationPolicy& policy, bool guarded,
                const std::shared_ptr<const Snapshot>& snapshot, double seconds) {
    if (key.size() < disk_.MinTokens()) return false;
    const auto result = disk_.Spill(key, policy, compact_, false, guarded, snapshot, *snapshot, true);
    return (result.empty() || result == "queued") ? disk_.Drain(seconds) : result == "present";
  }
  bool Drain(double seconds) { return disk_.Drain(seconds); }
  // Why a stop flush of a slot state of `bytes` would write nothing ("present": its file is on disk; "daily": the
  // daily write budget), or "" when it would write. The caller then skips the device copy.
  std::string Skip(std::span<const int> key, const ConversationPolicy& policy, size_t bytes) {
    if (disk_.Present(key, policy, compact_)) return "present";
    if (disk_.OverDailyBudget(bytes)) return "daily";
    return {};
  }

 private:
  StateDisk<Snapshot> disk_;
  const bool compact_;
};
}  // namespace crucible
