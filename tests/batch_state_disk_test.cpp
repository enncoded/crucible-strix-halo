// The disk tier of batch serve (engine/serve/batch_state_disk.h): host evictions reach the disk, a new process finds
// them, and a request restores only a state that the host cache's own rules would accept.
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <numeric>
#include <vector>

std::mutex pinned_mutex;
std::map<uintptr_t, size_t> pinned;
template<class T> int hipHostMalloc(T** out, size_t n) {
  std::lock_guard lock(pinned_mutex);
  *out = static_cast<T*>(std::malloc(n));
  pinned[uintptr_t(*out)] = n;
  return 0;
}
int hipHostFree(void* p) {
  std::lock_guard lock(pinned_mutex);
  assert(pinned.erase(uintptr_t(p)) == 1);
  std::free(p);
  return 0;
}
constexpr int hipMemcpyDeviceToHost = 0, hipMemcpyHostToDevice = 1;
int hipMemcpy(void* out, const void* in, size_t n, int) { std::memcpy(out, in, n); return 0; }
int hipStreamSynchronize(int) { return 0; }
#define HIP_OK(expr) assert((expr) == 0)
#include "../engine/serve/batch_state_disk.h"
#include "support/disk_test_dir.h"
#include <chrono>
#include <thread>
#include <atomic>
#include <future>

using namespace crucible;
namespace fs = std::filesystem;

struct Snapshot {
  std::vector<int> hist;
  RetainedPrefix retained;
  ConversationResume resume;
  SnapshotBytes bytes;
  std::vector<std::pair<int, int>> qsa_layout;
};
using Tier = BatchDiskTier<Snapshot>;
using Cache = BatchConversationCache<Snapshot>;

fs::path Home() {
  static const fs::path home = [] {
    const DiskTestDir dir = PickDiskTestDir("batch_state_disk_test");
    // The spill must take O_DIRECT, as state_disk_test requires (a tmpfs root hides the disk path).
    assert(dir.direct);
    ::setenv("HOME", dir.path.c_str(), 1);
    return dir.path;
  }();
  return home;
}
StateDiskOptions Options() {
  StateDiskOptions options;
  options.root = Home() / "states";
  options.identity = std::string(64, '7');
  options.directory = std::string(16, '7') + "-" + std::string(16, '7') + "-" + std::string(16, '7');
  options.min_tokens = 12;
  options.log = [](const std::string&) {};
  return options;
}
ConversationPolicy Policy() {
  ConversationPolicy policy;
  policy.prefill_rows = 8;
  return policy;
}
std::vector<int> Tokens(int first, size_t count) {
  std::vector<int> ids(count);
  std::iota(ids.begin(), ids.end(), first);
  return ids;
}
std::shared_ptr<const Snapshot> Make(std::span<const int> hist, bool tail, size_t boundary) {
  auto s = std::make_shared<Snapshot>();
  s->hist.assign(hist.begin(), hist.end());
  s->resume.length = hist.size();
  s->resume.prompt_only = !tail;
  s->resume.prompt_boundary = tail ? 0 : boundary;
  s->bytes = SnapshotBytes(3000 + hist.size());
  s->bytes.ForEachPiece(0, s->bytes.size(), [&](const SnapshotChunk& chunk, size_t inside, size_t count, size_t at) {
    for (size_t i = 0; i < count; ++i) chunk.host[inside + i] = uint8_t(at + i + hist.size());
  });
  return s;
}
bool Capture(Cache& cache, const std::vector<int>& ids, size_t position) {
  return cache.Capture(ids, Policy(), {position, CacheRetention::Session}, 64, uint64_t(64) << 30,
                       [&] { return Make(std::span<const int>(ids).first(position), false, ids.size() - 1); });
}
bool Tail(Cache& cache, const std::vector<int>& ids, size_t length) {
  return cache.CaptureTail(std::span<const int>(ids).first(length), Policy(), 64, uint64_t(64) << 30,
                           [&] { return Make(std::span<const int>(ids).first(length), true, 0); });
}
DiskQuery Query(const std::vector<int>& prompt, bool replay = true) {
  return {prompt, Policy(), prompt.size() - 1, prompt.size() - 1, 8, replay};
}
StageStatus StageWait(Tier& tier, const DiskQuery& query, size_t floor) {
  for (int i = 0; i < 2000; ++i) {
    const auto status = tier.Stage(query, floor, UINT64_MAX);
    if (status.state != StageState::Reading) return status;
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
  }
  assert(false);
  return {};
}

// Stage reserves a read before the reader allocates it, so MemAvailable does not show those bytes
// yet. A spill's room leaves them out until the read state holds its memory.
void SpillRoomLeavesUnallocatedReads() {
  auto options = Options();
  options.root = Home() / "states-room";
  std::promise<void> release;
  std::shared_future<void> gate = release.get_future().share();
  std::atomic<int> waiting = 0;
  options.before_read = [&] { ++waiting; gate.wait(); };
  Tier tier(options, false);
  const auto x = Tokens(1, 40), y = Tokens(101, 40);
  const auto xs = Make(std::span<const int>(x).first(16), false, x.size() - 1);
  assert(tier.Disk().Spill(xs->hist, Policy(), false, false, false, xs, *xs).empty() && tier.Drain(30));
  assert(tier.Stage(Query(x), 0, UINT64_MAX).state == StageState::Reading);
  while (waiting == 0) std::this_thread::yield();
  const uint64_t pending = xs->bytes.size();
  assert(tier.Disk().UnallocatedReadBytes() == pending);
  const auto ys = Make(std::span<const int>(y).first(16), false, y.size() - 1);
  const auto entry = std::make_shared<const Cache::Entry>(Cache::Entry{{}, ys, CacheRetention::Session});
  const uint64_t need = ys->bytes.size();
  assert(!tier.Spill(std::span<const int>(y).first(16), Policy(), entry, pending + need - 1));  // refused: memory
  release.set_value();
  assert(StageWait(tier, Query(x), 0).state == StageState::Ready && tier.Disk().UnallocatedReadBytes() == 0);
  assert(tier.Spill(std::span<const int>(y).first(16), Policy(), entry, need) && tier.Drain(30));
}

int main() {
  const auto x = Tokens(1, 40), y = Tokens(101, 40), z = Tokens(201, 40), w = Tokens(301, 40);
  std::shared_ptr<const Snapshot> x_tail;
  {
    Tier tier(Options(), false);
    assert(tier.Enabled());
    Cache cache(64 * 3, {}, 2);
    cache.SetSpill([&](std::span<const int> key, const ConversationPolicy& policy,
                       const std::shared_ptr<const Cache::Entry>& entry, int, uint64_t room) {
      return tier.Spill(key, policy, entry, room);
    });
    // x's tail is confirmed by a hit (its 16-token frontier becomes superseded), then x idles while y and z run.
    assert(Capture(cache, x, 16) && Tail(cache, x, 21));
    const auto hit = cache.FindTail(Tokens(1, 30), Policy());
    assert(hit && hit->snapshot->hist.size() == 21);
    x_tail = hit->snapshot;
    assert(Capture(cache, y, 16) && Capture(cache, z, 16));  // the superseded frontier leaves, not to disk
    assert(tier.Drain(30) && tier.Disk().Entries() == 0);
    for (uint64_t i = 0; i < Cache::kIdleRounds * 2; ++i)
      assert(cache.Find(y, Policy(), 39, 39, 8) && cache.Find(z, Policy(), 39, 39, 8));
    assert(Capture(cache, w, 16));  // the idle tail leaves to disk
    assert(tier.Drain(30) && tier.Disk().Entries() == 1);
    // A stopping engine writes the host entries that are not superseded.
    const size_t before = tier.Disk().Entries();
    assert(tier.FlushHost(cache) >= 1 && tier.Drain(30) && tier.Disk().Entries() > before);
  }
  {
    // A stop flush skips superseded entries: v's frontier, superseded by its tail.
    auto options = Options();
    options.root = Home() / "states-flush";  // another identity in the same root would remove x's directory
    Tier tier(std::move(options), false);
    Cache cache(64 * 4, {}, 2);
    const auto v = Tokens(401, 40);
    assert(Capture(cache, v, 16) && Tail(cache, v, 21) && cache.FindTail(Tokens(401, 30), Policy()));
    assert(tier.FlushHost(cache) == 1 && tier.Drain(30) && tier.Disk().Entries() == 1);
    assert(!tier.Disk().Present(std::vector<int>(v.begin(), v.begin() + 16), Policy(), false));
    assert(tier.Disk().Present(std::vector<int>(v.begin(), v.begin() + 21), Policy(), false));
  }
  Tier tier(Options(), false);
  const auto prompt = Tokens(1, 30);  // x's tail (21 tokens) and 9 more
  // A floor at the state's length (a slot holds it) starts no read.
  assert(tier.Stage(Query(prompt), 21, UINT64_MAX).state == StageState::None);
  // Without session replay a tail is not restorable, and x's superseded frontier never went to disk.
  assert(tier.Stage(Query(prompt, false), 0, UINT64_MAX).state == StageState::None);
  const auto status = StageWait(tier, Query(prompt), 0);
  assert(status.state == StageState::Ready && status.tokens == 21);
  assert(!tier.Take(Query(prompt), 21).snapshot);  // the host cache holds as much: no disk restore
  const auto ready = tier.Take(Query(prompt), 16);
  assert(ready.snapshot && ready.meta.key.size() == 21 && !ready.meta.resume.prompt_only);
  assert(ready.snapshot->hist == x_tail->hist && ready.snapshot->bytes == x_tail->bytes);
  assert(ready.snapshot->resume.length == 21 && ready.snapshot->resume.next_id == x_tail->resume.next_id);
  tier.Consume(ready);
  assert(tier.Peek(Query(prompt)) == 0);
  // A prompt frontier fits only the host rules: y's 16-token frontier, but not past the request's limit.
  const auto y_prompt = Tokens(101, 30);
  assert(StageWait(tier, Query(y_prompt, false), 0).tokens == 16);
  auto short_limit = Query(y_prompt, false);
  short_limit.limit = 15;
  assert(tier.Stage(short_limit, 0, UINT64_MAX).state == StageState::None && tier.Peek(short_limit) == 0);
  assert(tier.Peek(Query(y_prompt)) == 16);
  // A prompt that leaves the state's tokens finds nothing.
  auto other = prompt;
  other[5] = 999;
  assert(tier.Stage(Query(other), 0, UINT64_MAX).state == StageState::None);
  // A stop flush skips the device copy of a slot state whose file is on disk, or that the daily budget refuses.
  assert(tier.Skip(x_tail->hist, Policy(), 3021) == "present");
  assert(tier.Skip(Tokens(501, 30), Policy(), 3030).empty());
  {
    auto small = Options();
    small.write_per_day = 1000;
    Tier limited(std::move(small), false);
    assert(limited.Skip(Tokens(501, 30), Policy(), 3030) == "daily");
  }
  SpillRoomLeavesUnallocatedReads();
  fs::remove_all(Home());
  {
    // the disk identity names the MTP slot rows, so a state saved with another MTP bound
    // is in another directory and never restores into a smaller slot.
    char* none[] = {nullptr};
    const auto a = BatchStateDiskSettings("/m/x/../pack-a", 4096, false, 33280, none);
    const auto b = BatchStateDiskSettings("/m/pack-a", 4096, false, 106496, none);
    assert(std::find(a.begin(), a.end(), "mtp_rows=33280") != a.end());
    assert(std::find(a.begin(), a.end(), "pack=pack-a") != a.end());
    assert(std::find(b.begin(), b.end(), "mtp_rows=106496") != b.end());
    assert(StateDiskIdentityFor("e", "c", a).directory != StateDiskIdentityFor("e", "c", b).directory);
    assert(StateDiskIdentityFor("e", "c", a).directory ==
           StateDiskIdentityFor("e", "c", BatchStateDiskSettings("/m/x/../pack-a", 4096, false, 33280, none)).directory);
  }
  std::puts("batch_state_disk_test passed");
}
