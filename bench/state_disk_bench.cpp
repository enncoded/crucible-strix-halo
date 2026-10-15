// Disk tier speed (engine/runtime/state_disk.h): write and read one synthetic state of N GiB through the real code path
// (O_DIRECT, parallel requests, chunk checksums), and the checksum rate alone. CPU only.
// Usage: state_disk_bench DIR [GIB=3.5] [THREADS=4]   (DIR must be inside $HOME; the bench removes its files)
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>

std::mutex pinned_mutex;
template<class T> int hipHostMalloc(T** out, size_t n) {
  *out = static_cast<T*>(std::aligned_alloc(4096, n));
  return *out ? 0 : 1;
}
int hipHostFree(void* p) { std::free(p); return 0; }
constexpr int hipMemcpyDeviceToHost = 0, hipMemcpyHostToDevice = 1;
int hipMemcpy(void* out, const void* in, size_t n, int) { std::memcpy(out, in, n); return 0; }
int hipStreamSynchronize(int) { return 0; }
#define HIP_OK(expr) ((void)(expr))
#include "../engine/runtime/state_disk.h"
#include <chrono>
#include <thread>

using namespace crucible;
struct Snapshot {
  std::vector<int> hist;
  RetainedPrefix retained;
  ConversationResume resume;
  SnapshotBytes bytes;
  std::vector<std::pair<int, int>> qsa_layout;
};
static double Now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

int main(int argc, char** argv) {
  if (argc < 2) { std::fprintf(stderr, "usage: state_disk_bench DIR [GIB] [THREADS]\n"); return 2; }
  const double gib = argc > 2 ? std::atof(argv[2]) : 3.5;
  const int threads = argc > 3 ? std::atoi(argv[3]) : 4;
  const size_t bytes = size_t(gib * double(uint64_t(1) << 30));
  StateDiskOptions options;
  options.root = argv[1];
  options.identity = std::string(64, 'b');
  options.directory = std::string(16, 'b') + "-" + std::string(16, 'b') + "-" + std::string(16, 'b');
  options.min_tokens = 1;
  options.io_threads = threads;
  options.budget = uint64_t(64) << 30;
  options.queue_bytes = bytes + 1;
  options.log = [](const std::string& line) { std::fprintf(stderr, "%s\n", line.c_str()); };
  auto snapshot = std::make_shared<Snapshot>();
  snapshot->hist.resize(100000);
  for (size_t i = 0; i < snapshot->hist.size(); ++i) snapshot->hist[i] = int(i * 2654435761u % 248000);
  snapshot->resume.length = snapshot->hist.size();
  snapshot->resume.replay = 1;
  snapshot->bytes = SnapshotBytes(bytes);
  uint64_t seed = 88172645463325252ull;
  for (const auto& chunk : snapshot->bytes.Chunks())
    for (size_t i = 0; i < SnapshotChunkPool::kChunkBytes; i += 8) {
      seed ^= seed << 13, seed ^= seed >> 7, seed ^= seed << 17;
      std::memcpy(chunk.host + i, &seed, 8);
    }
  // Checksum rate on one thread (the write and the read run it on every I/O thread).
  double start = Now();
  uint64_t sink = 0;
  for (const auto& chunk : snapshot->bytes.Chunks()) sink ^= StateChecksum(chunk.host, SnapshotChunkPool::kChunkBytes);
  const double sum_s = Now() - start;
  ConversationPolicy policy;
  policy.prefill_rows = 4096;
  {
    StateDisk<Snapshot> disk(options);
    start = Now();
    const auto result = disk.Spill(snapshot->hist, policy, false, false, false, snapshot, *snapshot);
    if (!result.empty() || !disk.Drain(600)) { std::fprintf(stderr, "spill failed: %s\n", result.c_str()); return 1; }
    const double write_s = Now() - start;
    std::printf("bytes=%zu threads=%d checksum_gbs=%.2f write_s=%.3f write_gbs=%.2f\n", bytes, threads,
                double(bytes) / sum_s / 1e9, write_s, double(bytes) / write_s / 1e9);
  }
  StateDisk<Snapshot> disk(options);  // a restart: the index comes from the file header
  const auto meta = disk.Best(snapshot->hist, policy, false, [](const StateMeta&) { return true; });
  if (!meta) { std::fprintf(stderr, "index rebuild failed\n"); return 1; }
  start = Now();
  StageStatus status;
  while ((status = disk.Stage(*meta, UINT64_MAX)).state == StageState::Reading) std::this_thread::sleep_for(std::chrono::milliseconds(1));
  const double read_s = Now() - start;
  const auto ready = disk.Ready(snapshot->hist, policy, false, [](const StateMeta&) { return true; });
  const bool same = ready.snapshot && ready.snapshot->bytes == snapshot->bytes && ready.snapshot->hist == snapshot->hist;
  std::printf("read_s=%.3f read_gbs=%.2f state=%s same_bytes=%d sink=%llu\n", read_s, double(bytes) / read_s / 1e9,
              StageName(status.state), int(same), static_cast<unsigned long long>(sink & 1));
  std::error_code error;
  std::filesystem::remove(disk.Directory() / meta->name, error);
  return same ? 0 : 1;
}
