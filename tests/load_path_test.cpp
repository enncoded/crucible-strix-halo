// CPU checks for the load path: validation reads no mapped payload (P1), and
// large direct copies split into pieces that any worker can take (P2).
#include "../engine/model/flashnext/weight_validation.h"
#include "../engine/exl3/load_tasks.h"
#include <sys/resource.h>
#include <sys/wait.h>
#include <atomic>
#include <cassert>
#include <chrono>
#include <filesystem>
#include <mutex>
#include <random>
#include <set>
#include <thread>

namespace fs = std::filesystem;

// One linear: lin.trellis I16 [8,8,48], lin.suh/svh F16 [128], lin.mul1 I32 [].
static fs::path WriteLinear(const fs::path& dir, uint32_t marker) {
  fs::create_directories(dir);
  const size_t trellis = 8 * 8 * 48 * 2, scale = 128 * 2;
  const size_t o1 = trellis, o2 = o1 + scale, o3 = o2 + scale, end = o3 + 4;
  std::string header = "{\"lin.trellis\":{\"dtype\":\"I16\",\"shape\":[8,8,48],\"data_offsets\":[0," +
      std::to_string(o1) + "]},\"lin.suh\":{\"dtype\":\"F16\",\"shape\":[128],\"data_offsets\":[" +
      std::to_string(o1) + "," + std::to_string(o2) + "]},\"lin.svh\":{\"dtype\":\"F16\",\"shape\":[128],"
      "\"data_offsets\":[" + std::to_string(o2) + "," + std::to_string(o3) + "]},\"lin.mul1\":{\"dtype\":"
      "\"I32\",\"shape\":[],\"data_offsets\":[" + std::to_string(o3) + "," + std::to_string(end) + "]}}";
  header.append((8 - header.size() % 8) % 8, ' ');
  std::vector<uint8_t> payload(end);
  for (size_t i = 0; i < o3; ++i) payload[i] = uint8_t(i * 7);
  std::memcpy(payload.data() + o3, &marker, 4);
  std::ofstream shard(dir / "shard.safetensors", std::ios::binary);
  const uint64_t size = header.size();
  shard.write(reinterpret_cast<const char*>(&size), 8);
  shard.write(header.data(), header.size());
  shard.write(reinterpret_cast<const char*>(payload.data()), payload.size());
  shard.close();
  std::ofstream(dir / "model.safetensors.index.json")
      << "{\"weight_map\":{\"lin.trellis\":\"shard.safetensors\",\"lin.suh\":\"shard.safetensors\","
         "\"lin.svh\":\"shard.safetensors\",\"lin.mul1\":\"shard.safetensors\"}}";
  return dir;
}

// Runs fn in a child process; returns its exit status (Fail() exits with 1).
template <class F> static int InChild(F fn) {
  std::fflush(nullptr);
  const pid_t pid = fork();
  assert(pid >= 0);
  if (pid == 0) {
    fn();
    std::_Exit(0);
  }
  int status = 0;
  waitpid(pid, &status, 0);
  return WIFEXITED(status) ? WEXITSTATUS(status) : 128;
}

static void ValidationReadsNoMappedPayload(const fs::path& root) {
  const fs::path good = WriteLinear(root / "good", 0x83DCD12D);
  {
    st::Checkpoint ck(good.string());
    {
      st::NoMappedReads guard(ck);
      assert(crucible::ValidateLinear(ck, "lin", 128, 128) == 3);
      uint32_t marker = 0;
      ck.ReadBytes("lin.mul1", &marker, sizeof(marker));
      assert(marker == 0x83DCD12D);
    }
    int16_t first[4];
    ck.ReadBytes("lin.suh", first, sizeof(first));
    // After validation the loader reads the mapping as before; both paths give the same bytes.
    assert(std::memcmp(first, ck.Data("lin.suh"), sizeof(first)) == 0);
    uint32_t mapped = 0;
    std::memcpy(&mapped, ck.Data("lin.mul1"), sizeof(mapped));
    assert(mapped == 0x83DCD12D);
  }
  // A mapped read under the guard is a bug: it brings back the 4 MB readahead per marker.
  assert(InChild([&] {
    st::Checkpoint ck(good.string());
    st::NoMappedReads guard(ck);
    (void)ck.Data("lin.mul1");
  }) == 1);
  // A wrong marker still fails through the pread path.
  const fs::path bad = WriteLinear(root / "bad", 0x12345678);
  assert(InChild([&] {
    st::Checkpoint ck(bad.string());
    st::NoMappedReads guard(ck);
    crucible::ValidateLinear(ck, "lin", 128, 128);
  }) == 1);
  // Reading more bytes than the tensor holds fails.
  assert(InChild([&] {
    st::Checkpoint ck(good.string());
    uint64_t wide = 0;
    ck.ReadBytes("lin.mul1", &wide, sizeof(wide));
  }) == 1);
}

// Many I32 scalars m0.mul1 .. m{count-1}.mul1 with value 1000 + i, one 256-byte tensor and one
// 0-byte tensor.
static fs::path WriteScalars(const fs::path& dir, int count) {
  fs::create_directories(dir);
  std::string header = "{";
  std::string index = "{\"weight_map\":{";
  for (int i = 0; i < count; ++i) {
    const std::string name = "m" + std::to_string(i) + ".mul1";
    header += "\"" + name + "\":{\"dtype\":\"I32\",\"shape\":[],\"data_offsets\":[" + std::to_string(4 * i) + "," +
              std::to_string(4 * i + 4) + "]},";
    index += "\"" + name + "\":\"shard.safetensors\",";
  }
  header += "\"big\":{\"dtype\":\"F16\",\"shape\":[128],\"data_offsets\":[" + std::to_string(4 * count) + "," +
            std::to_string(4 * count + 256) + "]},\"empty\":{\"dtype\":\"I32\",\"shape\":[0],\"data_offsets\":[" +
            std::to_string(4 * count + 256) + "," + std::to_string(4 * count + 256) + "]}}";
  index += "\"big\":\"shard.safetensors\",\"empty\":\"shard.safetensors\"}}";
  header.append((8 - header.size() % 8) % 8, ' ');
  std::vector<uint8_t> payload(4 * size_t(count) + 256, 0x5A);
  for (int i = 0; i < count; ++i) {
    const uint32_t value = 1000 + i;
    std::memcpy(payload.data() + 4 * i, &value, 4);
  }
  std::ofstream shard(dir / "shard.safetensors", std::ios::binary);
  const uint64_t size = header.size();
  shard.write(reinterpret_cast<const char*>(&size), 8);
  shard.write(header.data(), header.size());
  shard.write(reinterpret_cast<const char*>(payload.data()), payload.size());
  shard.close();
  std::ofstream(dir / "model.safetensors.index.json") << index;
  return dir;
}

static void SmallReadsPrefetchInParallel(const fs::path& root) {
  constexpr int kCount = 2000;
  const fs::path dir = WriteScalars(root / "scalars", kCount);
  st::Checkpoint ck(dir.string());
  auto names = ck.NamesEndingWith(".mul1");
  assert(names.size() == kCount);
  names.push_back("big");  // above the small-value limit: not prefetched
  names.push_back("empty");
  {
    st::NoMappedReads guard(ck);
    assert(ck.Prefetch(names, 16) == kCount + 1);
    assert(ck.Prefetch(names, 16) == 0);  // already held
    for (int i = 0; i < kCount; ++i) {
      uint32_t value = 0;
      ck.ReadBytes("m" + std::to_string(i) + ".mul1", &value, sizeof(value));
      assert(value == uint32_t(1000 + i));
    }
    uint8_t big[4];
    ck.ReadBytes("big", big, sizeof(big));  // still readable through pread
    assert(big[0] == 0x5A);
    uint32_t none = 7;
    ck.ReadBytes("empty", &none, 0);  // a held 0-byte value: nothing to copy
    assert(none == 7);
  }
  // No thread can start (process limit): the calling thread reads every value.
  assert(InChild([&] {
    st::Checkpoint fresh(dir.string());
    const rlimit none{0, 0};
    if (setrlimit(RLIMIT_NPROC, &none) != 0) std::_Exit(3);
    if (fresh.Prefetch(fresh.NamesEndingWith(".mul1"), 16) != kCount) std::_Exit(4);
    uint32_t value = 0;
    fresh.ReadBytes("m1234.mul1", &value, sizeof(value));
    if (value != 2234) std::_Exit(5);
  }) == 0);
  // A prefetched value keeps the size check.
  assert(InChild([&] {
    uint64_t wide = 0;
    ck.ReadBytes("m0.mul1", &wide, sizeof(wide));
  }) == 1);
  // The same bytes as the mapping.
  assert(std::memcmp(ck.Data("m1999.mul1"), "\xb7\x0b\x00\x00", 4) == 0);
}

static void LargeDirectTasksSplitAcrossWorkers() {
  constexpr size_t kPiece = 1 << 16;
  std::vector<uint8_t> big(10 * kPiece + 12345), small(1000), dst(big.size() + small.size() + 64, 0);
  std::mt19937 rng(7);
  for (auto& b : big) b = uint8_t(rng());
  for (auto& b : small) b = uint8_t(rng());
  std::vector<crucible::LoadTask> tasks;
  tasks.push_back({64, dst.data(), nullptr, [](uint8_t* stage) { std::memset(stage, 0xAB, 64); }});
  tasks.push_back({big.size(), dst.data() + 64, big.data(), nullptr});
  tasks.push_back({small.size(), dst.data() + 64 + big.size(), small.data(), nullptr});
  const auto pieces = crucible::SplitDirectTasks(std::move(tasks), kPiece);
  assert(pieces.size() == 1 + 11 + 1);
  assert(!pieces[0].direct && pieces[0].bytes == 64);
  size_t covered = 0;
  for (size_t i = 1; i <= 11; ++i) {
    assert(pieces[i].direct == big.data() + covered && pieces[i].dst == dst.data() + 64 + covered);
    assert(pieces[i].bytes == (i < 11 ? kPiece : 12345));
    covered += pieces[i].bytes;
  }
  assert(covered == big.size() && pieces[12].bytes == small.size());
  // The same scheduling as Loader::Run: workers take the next task index.
  std::atomic<size_t> next{0};
  std::mutex lock;
  std::set<std::thread::id> big_workers;
  auto work = [&] {
    std::vector<uint8_t> stage(kPiece);
    for (size_t i; (i = next++) < pieces.size();) {
      const auto& t = pieces[i];
      if (t.direct) {
        std::memcpy(stage.data(), t.direct, t.bytes);
        std::memcpy(t.dst, stage.data(), t.bytes);
        if (i >= 1 && i <= 11) {
          std::lock_guard<std::mutex> hold(lock);
          big_workers.insert(std::this_thread::get_id());
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
      } else {
        t.fill(stage.data());
        std::memcpy(t.dst, stage.data(), t.bytes);
      }
    }
  };
  std::vector<std::thread> workers;
  for (int i = 0; i < 4; ++i) workers.emplace_back(work);
  for (auto& w : workers) w.join();
  assert(big_workers.size() > 1);
  for (size_t i = 0; i < 64; ++i) assert(dst[i] == 0xAB);
  assert(std::memcmp(dst.data() + 64, big.data(), big.size()) == 0);
  assert(std::memcmp(dst.data() + 64 + big.size(), small.data(), small.size()) == 0);
  // A task at or below the piece size, and fill tasks, stay as they are.
  std::vector<crucible::LoadTask> one;
  one.push_back({kPiece, dst.data(), big.data(), nullptr});
  assert(crucible::SplitDirectTasks(std::move(one), kPiece).size() == 1);
}

int main() {
  const fs::path root = fs::temp_directory_path() / ("load_path_test." + std::to_string(getpid()));
  ValidationReadsNoMappedPayload(root);
  SmallReadsPrefetchInParallel(root);
  LargeDirectTasksSplitAcrossWorkers();
  fs::remove_all(root);
  std::puts("load path tests passed");
}
