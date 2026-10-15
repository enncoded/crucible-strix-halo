// NVMe tier for conversation states.
// A state that the host cache evicts, or that a stopping engine holds, goes to one file. A later request, also after
// a restart of the same binary, reads it back on a reader thread and restores the same bytes that Save wrote.
// The device worker never waits for the disk: a spill only queues, and a read runs before the request opens.
// Files hold state derived from prompts: directory 0700, files 0600, and logs carry counts and sizes only.
#pragma once
#include "conversation_resume.h"
#include "retained_prefix.h"
#include "sha256.h"
#include "snapshot_bytes.h"
#include "../switches.h"
#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <ctime>
#include <cstdio>
#include <deque>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <unistd.h>

namespace crucible {
inline constexpr size_t kStateAlign = 4096;  // O_DIRECT offsets, lengths and buffers
inline constexpr uint64_t kStateMagic = 0x3154415453435243ull;  // "CRCSTAT1"
inline constexpr uint32_t kStateVersion = 1;
inline constexpr size_t kStateFixedHeader = 48;
inline constexpr uint64_t kStateMaxHeader = uint64_t(256) << 20;

inline std::string StateSha256(const uint8_t* p, size_t n) {
  Sha256 hash;
  hash.Update(p, n);
  return hash.HexDigest();
}
inline uint64_t StateRoundUp(uint64_t n) { return (n + kStateAlign - 1) / kStateAlign * kStateAlign; }

// A fast check against torn or damaged files, not a cryptographic hash: four independent multiply-xor lanes.
inline uint64_t StateChecksum(const uint8_t* p, size_t n) {
  constexpr uint64_t kMul = 0x9E3779B97F4A7C15ull;
  uint64_t lane[4] = {0x243F6A8885A308D3ull, 0x13198A2E03707344ull, 0xA4093822299F31D0ull, 0x082EFA98EC4E6C89ull};
  size_t i = 0;
  for (; i + 32 <= n; i += 32)
    for (int k = 0; k < 4; ++k) {
      uint64_t word;
      std::memcpy(&word, p + i + 8 * k, 8);
      lane[k] = (lane[k] ^ word) * kMul;
      lane[k] ^= lane[k] >> 29;
    }
  uint64_t h = uint64_t(n) * kMul;
  for (const uint64_t v : lane) h = (h ^ v) * kMul, h ^= h >> 31;
  for (; i < n; ++i) h = (h ^ p[i]) * kMul, h ^= h >> 31;
  return h;
}

class StateBytesOut {
 public:
  void U8(uint8_t v) { out_.push_back(v); }
  void U32(uint32_t v) { Raw(&v, 4); }
  void U64(uint64_t v) { Raw(&v, 8); }
  void I32(int32_t v) { Raw(&v, 4); }
  void Str(const std::string& s) { U64(s.size()); Raw(s.data(), s.size()); }
  template<class T> void Vec(const std::vector<T>& v) { U64(v.size()); Raw(v.data(), v.size() * sizeof(T)); }
  void Ints(std::span<const int> v) { U64(v.size()); Raw(v.data(), v.size() * sizeof(int)); }
  void Raw(const void* p, size_t n) {
    if (!n) return;
    const auto* bytes = static_cast<const uint8_t*>(p);
    out_.insert(out_.end(), bytes, bytes + n);
  }
  std::vector<uint8_t>& Bytes() { return out_; }
 private:
  std::vector<uint8_t> out_;
};

class StateBytesIn {
 public:
  StateBytesIn(const uint8_t* p, size_t n) : p_(p), n_(n) {}
  uint8_t U8() { uint8_t v; Raw(&v, 1); return v; }
  uint32_t U32() { uint32_t v; Raw(&v, 4); return v; }
  uint64_t U64() { uint64_t v; Raw(&v, 8); return v; }
  int32_t I32() { int32_t v; Raw(&v, 4); return v; }
  std::string Str() { std::string s(Count(1), '\0'); Raw(s.data(), s.size()); return s; }
  template<class T> std::vector<T> Vec() { std::vector<T> v(Count(sizeof(T))); Raw(v.data(), v.size() * sizeof(T)); return v; }
  void Raw(void* out, size_t n) {
    if (!n) return;
    if (n > n_ - at_) throw std::runtime_error("state header overrun");
    std::memcpy(out, p_ + at_, n);
    at_ += n;
  }
  size_t At() const { return at_; }
 private:
  size_t Count(size_t each) {
    const uint64_t count = U64();
    if (count > (n_ - at_) / each) throw std::runtime_error("state header overrun");
    return size_t(count);
  }
  const uint8_t* p_;
  size_t n_, at_ = 0;
};

// The number of members of an aggregate. A new member of ConversationPolicy or ConversationResume fails the
// static_asserts below until WritePolicy and WriteResume store it: a field left out would let two states that
// differ in it share one file name.
struct StateAnyField { template<class T> operator T() const; };
template<class T, class... A>
constexpr size_t StateFieldCount() {
  if constexpr (requires { T{A{}..., StateAnyField{}}; }) return StateFieldCount<T, A..., StateAnyField>();
  else return sizeof...(A);
}
// Optional members of a later train. Stored when present.
template<class T> constexpr size_t StateKvFields() {
  return size_t(requires(T t) { t.kv_quant; }) + size_t(requires(T t) { t.kv_layout; });
}
template<class T> void WriteKvFields(StateBytesOut& out, const T& t) {
  if constexpr (requires { t.kv_quant; }) out.U64(uint64_t(t.kv_quant));
  if constexpr (requires { t.kv_layout; }) out.U64(uint64_t(t.kv_layout));
}
// The snapshot's own KV format fields, in order (a fake-quant state never restores into real storage).
template<class T> std::vector<uint64_t> SnapshotKvFields(const T& t) {
  std::vector<uint64_t> fields;
  if constexpr (requires { t.kv_quant; }) fields.push_back(uint64_t(t.kv_quant));
  if constexpr (requires { t.kv_layout; }) fields.push_back(uint64_t(t.kv_layout));
  return fields;
}
template<class T> bool SetSnapshotKvFields(const std::vector<uint64_t>& fields, T& t) {
  if (fields.size() != StateKvFields<T>()) return false;
  size_t at = 0;
  if constexpr (requires { t.kv_quant; }) t.kv_quant = decltype(t.kv_quant)(fields[at++]);
  if constexpr (requires { t.kv_layout; }) t.kv_layout = decltype(t.kv_layout)(fields[at++]);
  return true;
}
template<class T> void ReadKvFields(StateBytesIn& in, T& t) {
  if constexpr (requires { t.kv_quant; }) t.kv_quant = decltype(t.kv_quant)(in.U64());
  if constexpr (requires { t.kv_layout; }) t.kv_layout = decltype(t.kv_layout)(in.U64());
}
static_assert(StateFieldCount<ConversationPolicy>() == 15 + StateKvFields<ConversationPolicy>(),
              "a ConversationPolicy member changed: store it in WritePolicy and ReadPolicy (state_disk.h)");
static_assert(StateFieldCount<ConversationResume>() == 9,
              "a ConversationResume member changed: store it in WriteResume and ReadResume (state_disk.h)");

inline void WritePolicy(StateBytesOut& out, const ConversationPolicy& p) {
  for (const int v : {p.spec, p.chunk, p.prefill_rows, p.lookup, p.lookup_min, p.mtp_prefix}) out.I32(v);
  for (const bool v : {p.sampled, p.dense, p.adaptive, p.last_head, p.overlap, p.head_auto}) out.U8(v);
  out.U32(p.margin_bits);
  out.U32(p.confidence_bits);
  out.Vec(p.draft_selection);
  WriteKvFields(out, p);
}
inline ConversationPolicy ReadPolicy(StateBytesIn& in) {
  ConversationPolicy p;
  for (int* v : {&p.spec, &p.chunk, &p.prefill_rows, &p.lookup, &p.lookup_min, &p.mtp_prefix}) *v = in.I32();
  for (bool* v : {&p.sampled, &p.dense, &p.adaptive, &p.last_head, &p.overlap, &p.head_auto}) *v = in.U8() != 0;
  p.margin_bits = in.U32();
  p.confidence_bits = in.U32();
  p.draft_selection = in.Vec<int>();
  ReadKvFields(in, p);
  return p;
}
inline void WriteResume(StateBytesOut& out, const ConversationResume& r) {
  out.U64(r.length);
  for (const int v : {r.mpos, r.replay, r.next_id}) out.I32(v);
  out.U8(r.have_carry);
  out.Vec(r.pending_tokens);
  out.Vec(r.logits);
  out.U8(r.prompt_only);
  out.U64(r.prompt_boundary);
}
inline ConversationResume ReadResume(StateBytesIn& in) {
  ConversationResume r;
  r.length = size_t(in.U64());
  for (int* v : {&r.mpos, &r.replay, &r.next_id}) *v = in.I32();
  r.have_carry = in.U8() != 0;
  r.pending_tokens = in.Vec<int>();
  r.logits = in.Vec<float>();
  r.prompt_only = in.U8() != 0;
  r.prompt_boundary = size_t(in.U64());
  return r;
}

// What the index keeps of one file: the full key (compared, never only hashed) and the resume fields that decide
// whether a request can use it. `resume` has no logits.
struct StateMeta {
  std::vector<int> key;
  ConversationPolicy policy;
  bool compact = false, shared = false, guarded = false;
  ConversationResume resume;
  uint64_t payload = 0, file_bytes = 0, used = 0;
  std::string name;
  bool superseded = false;  // index only: a longer state on disk extends this one (its own branch moved on)
};
// One file's header, parsed.
struct StateHeader {
  StateMeta meta;
  std::string identity;
  std::vector<int> hist;
  std::array<size_t, 4> retained{};
  ConversationResume resume;  // with logits
  std::vector<std::pair<int, int>> layout;
  std::vector<uint64_t> kv;  // SnapshotKvFields
  std::vector<uint64_t> sums;
  uint64_t header_bytes = 0;
};

template<class Snapshot>
std::vector<uint8_t> BuildStateHeader(const std::string& identity, const StateMeta& meta, const Snapshot& snapshot,
                                      const std::vector<uint64_t>& sums) {
  StateBytesOut out;
  out.U64(kStateMagic);
  out.U32(kStateVersion);
  out.U32(0);
  out.U64(0);  // header bytes, patched below
  out.U64(0);  // used bytes, patched below
  out.U64(snapshot.bytes.size());
  out.U64(SnapshotChunkPool::kChunkBytes);
  out.Str(identity);
  WritePolicy(out, meta.policy);
  out.U8(meta.compact);
  out.U8(meta.shared);
  out.U8(meta.guarded);
  out.Ints(meta.key);
  out.Ints(snapshot.hist);
  for (const size_t v : snapshot.retained.Fields()) out.U64(v);
  WriteResume(out, snapshot.resume);
  out.U64(snapshot.qsa_layout.size());
  for (const auto& [layer, slots] : snapshot.qsa_layout) out.I32(layer), out.I32(slots);
  out.Vec(SnapshotKvFields(snapshot));
  out.Vec(sums);
  auto& bytes = out.Bytes();
  const uint64_t used = bytes.size() + 8, total = StateRoundUp(used);
  std::memcpy(bytes.data() + 16, &total, 8);
  std::memcpy(bytes.data() + 24, &used, 8);
  const uint64_t sum = StateChecksum(bytes.data(), bytes.size());
  out.U64(sum);
  bytes.resize(total, 0);
  return std::move(bytes);
}

// Throws on any damage. `bytes` holds at least the header's own length.
inline StateHeader ParseStateHeader(const uint8_t* bytes, size_t size) {
  StateBytesIn fixed(bytes, std::min(size, kStateFixedHeader));
  StateHeader header;
  if (fixed.U64() != kStateMagic || fixed.U32() != kStateVersion) throw std::runtime_error("not a state file");
  fixed.U32();
  header.header_bytes = fixed.U64();
  const uint64_t used = fixed.U64();
  header.meta.payload = fixed.U64();
  if (fixed.U64() != SnapshotChunkPool::kChunkBytes || used < kStateFixedHeader + 8 || used > header.header_bytes ||
      header.header_bytes > size || header.header_bytes % kStateAlign)
    throw std::runtime_error("state header bounds");
  uint64_t sum;
  std::memcpy(&sum, bytes + used - 8, 8);
  if (sum != StateChecksum(bytes, used - 8)) throw std::runtime_error("state header checksum");
  StateBytesIn in(bytes + kStateFixedHeader, used - 8 - kStateFixedHeader);
  header.identity = in.Str();
  header.meta.policy = ReadPolicy(in);
  header.meta.compact = in.U8() != 0;
  header.meta.shared = in.U8() != 0;
  header.meta.guarded = in.U8() != 0;
  header.meta.key = in.Vec<int>();
  header.hist = in.Vec<int>();
  for (size_t& v : header.retained) v = size_t(in.U64());
  header.resume = ReadResume(in);
  const uint64_t layers = in.U64();
  if (layers > 4096) throw std::runtime_error("state header layout");
  for (uint64_t i = 0; i < layers; ++i) {
    const int layer = in.I32();
    header.layout.emplace_back(layer, in.I32());
  }
  header.kv = in.Vec<uint64_t>();
  header.sums = in.Vec<uint64_t>();
  const uint64_t chunks = (header.meta.payload + SnapshotChunkPool::kChunkBytes - 1) / SnapshotChunkPool::kChunkBytes;
  if (header.sums.size() != chunks || header.hist.size() != header.meta.key.size())
    throw std::runtime_error("state header fields");
  header.meta.resume = header.resume;
  header.meta.resume.logits.clear();
  header.meta.file_bytes = header.header_bytes + StateRoundUp(header.meta.payload);
  return header;
}

// The process identity that a state belongs to ("Key"). Any change of the engine binary,
// the checkpoint metadata or an engine setting gives a new directory, so a new binary never reads old states.
struct StateDiskIdentity { std::string sha, directory; };
inline std::string StateSha(const std::string& text) {
  return StateSha256(reinterpret_cast<const uint8_t*>(text.data()), text.size());
}
inline StateDiskIdentity StateDiskIdentityFor(const std::string& binary_sha, const std::string& checkpoint_sha,
                                              std::vector<std::string> settings) {
  std::sort(settings.begin(), settings.end());
  std::string joined;
  for (const auto& item : settings) joined += item + "\n";
  const std::string settings_sha = StateSha(joined);
  const std::string text = "binary=" + binary_sha + "\ncheckpoint=" + checkpoint_sha + "\nsettings=" + settings_sha + "\n";
  return {StateSha(text), binary_sha.substr(0, 16) + "-" + checkpoint_sha.substr(0, 16) + "-" + settings_sha.substr(0, 16)};
}
// "" when the file cannot be read.
inline std::string FileSha256(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) return {};
  std::vector<char> data((std::istreambuf_iterator<char>(in)), {});
  return StateSha256(reinterpret_cast<const uint8_t*>(data.data()), data.size());
}
// The checkpoint identity of the gate records (tools/gates.py checkpoint_identity).
inline std::string CheckpointSha256(const std::filesystem::path& checkpoint) {
  std::string text;
  for (const char* name : {"config.json", "quantization_config.json", "model.safetensors.index.json"}) {
    const std::string sha = FileSha256(checkpoint / name);
    text += std::string(name) + ":" + (sha.empty() ? "-" : sha) + "\n";
  }
  // The weight files themselves, cheaply: name, size and modification time (a changed file gives a new identity).
  std::vector<std::string> weights;
  std::error_code error;
  for (const auto& item : std::filesystem::directory_iterator(checkpoint, error)) {
    const std::string name = item.path().filename();
    struct stat info{};
    if (!name.ends_with(".safetensors") || ::stat(item.path().c_str(), &info) != 0) continue;
    weights.push_back(name + ":" + std::to_string(info.st_size) + ":" + std::to_string(info.st_mtim.tv_sec) + "." +
                      std::to_string(info.st_mtim.tv_nsec));
  }
  std::sort(weights.begin(), weights.end());
  for (const auto& item : weights) text += item + "\n";
  return StateSha(text);
}
// Every CRUCIBLE_* switch of this process (numerics A/Bs and layouts change state bytes), the content of an
// ablation file and of the MTP sidecar, and the serve options that change state.
inline std::vector<std::string> StateDiskSettings(char** environ_list, int prefill_rows, bool vision) {
  std::vector<std::string> settings;
  for (char** item = environ_list; item && *item; ++item) {
    const std::string entry = *item;
    if (entry.rfind("CRUCIBLE_", 0) != 0) continue;
    settings.push_back(entry);
    if (entry.rfind("CRUCIBLE_ABLATE=", 0) == 0)
      settings.push_back("ablate_sha256=" + FileSha256(entry.substr(16)));
    // The sidecar's content, not only its path: a file that changes in place gives other MTP rows and drafts.
    if (entry.rfind("CRUCIBLE_MTP_SIDECAR=", 0) == 0)
      settings.push_back("mtp_sidecar_sha256=" + FileSha256(entry.substr(21)));
  }
  settings.push_back("prefill_rows=" + std::to_string(prefill_rows));
  settings.push_back(std::string("vision=") + (vision ? "1" : "0"));
  std::sort(settings.begin(), settings.end());
  return settings;
}

struct StateDiskOptions {
  std::filesystem::path root;  // parent of the identity directories; must be inside $HOME
  std::string identity;        // StateDiskIdentity::sha
  std::string directory;       // StateDiskIdentity::directory
  uint64_t budget = uint64_t(64) << 30;        // bytes of state files on disk
  size_t min_tokens = 16384;                   // smaller states are cheaper to prefill than to write and read
  uint64_t queue_bytes = uint64_t(8) << 30;    // evicted bytes that wait for the writer (held in host memory)
  uint64_t free_reserve = uint64_t(16) << 30;  // filesystem space that a spill always leaves free
  int io_threads = 8;                          // parallel requests of one read or write (the test machine: 8 reach 6.9 GB/s)
  double stage_seconds = 60;                   // a read state that no request uses within this time leaves memory
  std::function<uint64_t()> free_space;        // tests; default statvfs of the directory
  std::function<void(const std::string&)> log; // default stderr
  std::function<void()> before_write;          // tests: runs on the writer thread before each write
  std::function<void()> before_chunk;          // tests: runs on an I/O thread before each payload chunk
  std::function<void()> before_read;           // tests: runs on the reader thread before each read
  // tests: an errno for the payload pwrite or pread on `buffer` (0: none), as the kernel gives EFAULT for direct I/O
  // on pages that it cannot pin (hipHostMalloc memory on the test machine).
  std::function<int(const void* buffer)> io_fault;
  // SSD wear guard: bytes that the tier may write in one local calendar day, across restarts (the ledger file
  // `write-ledger` in `root`). Above it every spill is refused, stop flushes too. 0: the tier only reads.
  uint64_t write_per_day = uint64_t(300) << 30;
  std::function<std::string()> today;          // tests; default StateToday
};
enum class StageState { None, Reading, Ready, Failed };
struct StageStatus { StageState state = StageState::None; size_t tokens = 0; std::string reason; };
// Counts and bytes since the start of this process (refused_daily: spills that the daily write budget refused,
// present: spills of a state whose file is on disk already, so nothing was written).
struct StateDiskStats {
  uint64_t written = 0, refused = 0, reads = 0, failed = 0, evicted = 0;
  uint64_t written_bytes = 0, read_bytes = 0, present = 0, refused_daily = 0;
};
// The local calendar day, as YYYY-MM-DD.
inline std::string StateToday() {
  const std::time_t now = std::time(nullptr);
  std::tm local{};
  localtime_r(&now, &local);
  char text[16];
  std::strftime(text, sizeof text, "%F", &local);
  return text;
}
inline const char* StageName(StageState state) {
  switch (state) {
    case StageState::Reading: return "reading";
    case StageState::Ready: return "ready";
    case StageState::Failed: return "failed";
    default: return "none";
  }
}

// An aligned 8 MiB buffer for each payload chunk of a direct write or read (allocated on first use).
class StateBounce {
 public:
  uint8_t* Get() {
    if (!data_) data_.reset(static_cast<uint8_t*>(std::aligned_alloc(kStateAlign, SnapshotChunkPool::kChunkBytes)));
    if (!data_) throw std::bad_alloc();
    return data_.get();
  }
 private:
  struct Free { void operator()(uint8_t* p) const { std::free(p); } };
  std::unique_ptr<uint8_t, Free> data_;
};
inline int StateWriteAll(int fd, const uint8_t* p, size_t n, uint64_t offset) {
  while (n) {
    const ssize_t done = ::pwrite(fd, p, n, off_t(offset));
    if (done < 0 && errno == EINTR) continue;
    if (done <= 0) return done < 0 ? errno : EIO;
    p += done, n -= size_t(done), offset += uint64_t(done);
  }
  return 0;
}
inline int StateReadAll(int fd, uint8_t* p, size_t n, uint64_t offset) {
  while (n) {
    const ssize_t done = ::pread(fd, p, n, off_t(offset));
    if (done < 0 && errno == EINTR) continue;
    if (done <= 0) return done < 0 ? errno : EIO;
    p += done, n -= size_t(done), offset += uint64_t(done);
  }
  return 0;
}
// O_DIRECT where the filesystem supports it: page cache copies would double the memory traffic. Payload I/O uses only
// StateBounce buffers (malloc pages), never the pinned chunks.
inline int StateOpen(const std::filesystem::path& path, int flags, mode_t mode = 0600) {
  const int fd = ::open(path.c_str(), flags | O_DIRECT | O_CLOEXEC, mode);
  if (fd >= 0 || errno != EINVAL) return fd;
  return ::open(path.c_str(), flags | O_CLOEXEC, mode);
}
// The reason of a failed write step: "full", or "io errno=N" (the number only, for diagnosis).
inline std::string StateIoReason(int error) {
  return error == ENOSPC || error == EDQUOT ? "full" : "io errno=" + std::to_string(error);
}
// Runs io(index, bounce) for every index on up to `threads` threads; returns the first error ("" when all pass).
inline std::string StateParallel(size_t count, int threads, const std::function<std::string(size_t, StateBounce&)>& io) {
  std::atomic<size_t> next = 0;
  std::mutex mutex;
  std::string error;
  const auto run = [&] {
    StateBounce bounce;
    for (size_t index; (index = next++) < count;) {
      {
        std::lock_guard lock(mutex);
        if (!error.empty()) return;
      }
      std::string failed;
      try { failed = io(index, bounce); } catch (const std::bad_alloc&) { failed = "memory"; }
      if (failed.empty()) continue;
      std::lock_guard lock(mutex);
      if (error.empty()) error = failed;
      return;
    }
  };
  std::vector<std::thread> pool;
  for (int t = 1; t < std::min<int>(threads, int(count)); ++t) pool.emplace_back(run);
  run();
  for (auto& thread : pool) thread.join();
  return error;
}

template<class Snapshot>
class StateDisk {
 public:
  // A staged state that a request can restore.
  struct ReadyState { std::shared_ptr<const Snapshot> snapshot; StateMeta meta; };

  static_assert(StateFieldCount<Snapshot>() == 5 + StateKvFields<Snapshot>(),
                "a snapshot member changed: store it in BuildStateHeader and Read (state_disk.h)");
  explicit StateDisk(StateDiskOptions options) : options_(std::move(options)) {
    if (!options_.log) options_.log = [](const std::string& line) { std::fprintf(stderr, "%s\n", line.c_str()); };
    if (!options_.free_space) options_.free_space = [this] { return FreeSpace(); };
    if (!options_.today) options_.today = StateToday;
    options_.io_threads = std::clamp(options_.io_threads, 1, 16);
    if (!IdentityDirectory(options_.directory) || options_.identity.empty())
      throw std::invalid_argument("state disk identity is malformed");
    PrepareRoot();
    dir_ = options_.root / options_.directory;
    MakePrivateDirectory(dir_);
    lock_ = ::open((dir_ / ".lock").c_str(), O_RDWR | O_CREAT | O_CLOEXEC, 0600);
    if (lock_ < 0 || ::flock(lock_, LOCK_EX | LOCK_NB) != 0) {
      Log("state_disk off reason=locked");
      return;
    }
    const size_t removed = RemoveOtherIdentities();
    Rebuild();
    Unlink();
    enabled_ = true;
    day_ = options_.today();
    day_written_ = LedgerAdd(0);
    Log("state_disk start entries=" + std::to_string(index_.size()) + " bytes=" + std::to_string(bytes_) +
        " removed_dirs=" + std::to_string(removed) + " day_written=" + std::to_string(day_written_) +
        " write_per_day=" + std::to_string(options_.write_per_day));
    writer_ = std::thread([this] { WriterLoop(); });
    reader_ = std::thread([this] { ReaderLoop(); });
  }
  ~StateDisk() {
    stopping_ = true;  // a write in progress ends at its next chunk; its temporary file leaves
    {
      std::lock_guard lock(mutex_);
      stop_ = true;
    }
    work_.notify_all();
    if (writer_.joinable()) writer_.join();
    if (reader_.joinable()) reader_.join();
    if (lock_ >= 0) ::close(lock_);
  }
  StateDisk(const StateDisk&) = delete;
  StateDisk& operator=(const StateDisk&) = delete;

  bool Enabled() const { return enabled_; }
  const std::filesystem::path& Directory() const { return dir_; }
  size_t MinTokens() const { return options_.min_tokens; }

  // Queues a state for the writer. `keep` holds the snapshot memory until the write ends. Returns "" when queued,
  // else why not: off, short, present (already on disk), queued, memory (the held bytes would pass `room`, the host
  // memory above the reserve that the evicting save leaves, an earlier change), busy (the queue is full). `flush` (a stopping
  // engine) ignores the room and the queue limit, because those states are in memory already.
  std::string Spill(std::span<const int> key, const ConversationPolicy& policy, bool compact, bool shared, bool guarded,
                    std::shared_ptr<const void> keep, const Snapshot& snapshot, bool flush = false,
                    uint64_t room = UINT64_MAX) {
    if (!enabled_) return "off";
    if (key.size() < options_.min_tokens) return "short";
    if (snapshot.hist.size() != key.size()) return "policy";
    StateMeta meta;
    meta.key.assign(key.begin(), key.end());
    meta.policy = policy;
    meta.compact = compact, meta.shared = shared, meta.guarded = guarded;
    meta.resume = snapshot.resume;
    meta.resume.logits.clear();
    meta.payload = snapshot.bytes.size();
    meta.file_bytes = FileEstimate(key, snapshot);
    meta.name = Name(policy, compact, key);
    const char* refused = nullptr;
    {
      std::lock_guard lock(mutex_);
      RollDay();
      if (const auto found = index_.find(meta.name); found != index_.end()) {
        found->second.used = ++tick_;
        ++stats_.present;
        refused = "present";
      } else if (queued_names_.contains(meta.name)) {
        refused = "queued";
      } else if (day_written_ + queued_file_bytes_ + FileEstimate(meta) > options_.write_per_day) {
        ++stats_.refused_daily;
        refused = "daily";
      } else if (!flush && meta.payload > room) {
        refused = "memory";
      } else if (!flush && queued_bytes_ && queued_bytes_ + meta.payload > options_.queue_bytes) {
        refused = "busy";
      } else {
        queued_file_bytes_ += FileEstimate(meta);
        queued_bytes_ += meta.payload;
        queued_names_.insert(meta.name);
        jobs_.push_back({std::move(meta), std::move(keep), &snapshot});
      }
    }
    if (refused) {
      const std::string reason = refused;
      if (reason == "busy" || reason == "daily" || reason == "memory")
        Log("state_disk spill tokens=" + std::to_string(key.size()) + " bytes=" + std::to_string(snapshot.bytes.size()) +
            " result=refused reason=" + reason + " day_written=" + std::to_string(DayWritten()) +
            " write_per_day=" + std::to_string(options_.write_per_day));
      return refused;
    }
    work_.notify_all();
    return {};
  }
  // The longest state on disk whose key starts `key`, with the same policy and storage mode, that `accept` takes.
  template<class Accept>
  std::optional<StateMeta> Best(std::span<const int> key, const ConversationPolicy& policy, bool compact,
                                Accept accept) const {
    std::lock_guard lock(mutex_);
    const StateMeta* best = nullptr;
    for (const auto& [name, meta] : index_)
      if (Matches(meta, key, policy, compact) && (!best || meta.key.size() > best->key.size()) && accept(meta))
        best = &meta;
    if (!best) return std::nullopt;
    return *best;
  }
  // Starts or reports the read of `meta`'s file into host memory. `room`: bytes that staged states may hold in all.
  StageStatus Stage(const StateMeta& meta, uint64_t room) {
    StageStatus status;
    {
      std::lock_guard lock(mutex_);
      Expire();
      status.tokens = meta.key.size();
      if (const auto found = staged_.find(meta.name); found != staged_.end()) {
        status.state = found->second.state;
        status.reason = found->second.reason;
        if (status.state == StageState::Failed) Unstage(found);
        // A poll shows that the request still waits (for a slot): its read state stays in memory.
        if (status.state == StageState::Ready) found->second.since = std::chrono::steady_clock::now();
        return status;
      }
      const auto indexed = index_.find(meta.name);
      if (indexed == index_.end()) return {StageState::Failed, status.tokens, "missing"};
      if (staged_bytes_ + meta.payload > room) return {StageState::None, status.tokens, "memory"};
      indexed->second.used = ++tick_;
      Staged staged;
      staged.meta = indexed->second;
      staged_[meta.name] = std::move(staged);
      staged_bytes_ += meta.payload;
      reads_.push_back(meta.name);
      status.state = StageState::Reading;
    }
    work_.notify_all();
    return status;
  }
  // The longest read state that `key` starts with (as Best). It stays staged until Consume.
  template<class Accept>
  ReadyState Ready(std::span<const int> key, const ConversationPolicy& policy, bool compact, Accept accept) {
    std::lock_guard lock(mutex_);
    Expire();
    const Staged* best = nullptr;
    for (const auto& [name, staged] : staged_)
      if (staged.state == StageState::Ready && Matches(staged.meta, key, policy, compact) &&
          (!best || staged.meta.key.size() > best->meta.key.size()) && accept(staged.meta))
        best = &staged;
    if (!best) return {};
    return {best->snapshot, best->meta};
  }
  // The request restored `ready`: its host copy leaves (the file stays on disk).
  void Consume(const ReadyState& ready) {
    std::lock_guard lock(mutex_);
    if (const auto found = staged_.find(ready.meta.name); found != staged_.end() && found->second.state == StageState::Ready)
      Unstage(found);
  }
  // Waits until every queued spill is written or refused. False when `seconds` pass first.
  bool Drain(double seconds) {
    std::unique_lock lock(mutex_);
    return idle_.wait_for(lock, std::chrono::duration<double>(seconds), [this] { return jobs_.empty() && !writing_; });
  }
  uint64_t StagedBytes() const { std::lock_guard lock(mutex_); return staged_bytes_; }
  // Bytes of reads that Stage reserved and the reader has not allocated yet: MemAvailable does not show them, so a
  // spill's room leaves them out.
  uint64_t UnallocatedReadBytes() const {
    std::lock_guard lock(mutex_);
    uint64_t bytes = 0;
    for (const auto& [name, staged] : staged_)
      if (staged.state == StageState::Reading && !staged.allocated) bytes += staged.meta.payload;
    return bytes;
  }
  size_t Entries() const { std::lock_guard lock(mutex_); return index_.size(); }
  uint64_t Bytes() const { std::lock_guard lock(mutex_); return bytes_; }
  StateDiskStats Stats() const { std::lock_guard lock(mutex_); return stats_; }
  // Bytes that the tier wrote today (all engines that share the state root), as the ledger last showed them.
  uint64_t DayWritten() {
    std::lock_guard lock(mutex_);
    RollDay();
    return day_written_;
  }
  // A file with this full key, policy and storage mode is on disk: a spill would write nothing.
  bool Present(std::span<const int> key, const ConversationPolicy& policy, bool compact) const {
    const std::string name = Name(policy, compact, key);
    std::lock_guard lock(mutex_);
    return index_.contains(name);
  }
  // A spill would pass the daily write budget (the caller can skip the device copy of a stop flush).
  bool OverDailyBudget(size_t payload) {
    std::lock_guard lock(mutex_);
    RollDay();
    return day_written_ + queued_file_bytes_ + StateRoundUp(payload) + kStateAlign > options_.write_per_day;
  }

 private:
  struct Job { StateMeta meta; std::shared_ptr<const void> keep; const Snapshot* snapshot; };
  struct Staged {
    StateMeta meta;
    StageState state = StageState::Reading;
    std::shared_ptr<const Snapshot> snapshot;
    std::string reason;
    std::chrono::steady_clock::time_point since = std::chrono::steady_clock::now();
    bool allocated = false;  // the reader holds its memory (UnallocatedReadBytes)
  };
  // Three groups of 16 lower-case hex digits joined by '-' (StateDiskIdentity::directory).
  static bool IdentityDirectory(const std::string& name) {
    if (name.size() != 50) return false;
    for (size_t i = 0; i < name.size(); ++i) {
      const char c = name[i];
      if (i % 17 == 16 ? c != '-' : !((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'))) return false;
    }
    return true;
  }
  static double Now() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
  }
  void Log(const std::string& line) const { options_.log(line); }
  // The file size that a spill of `snapshot` writes, before its header exists: the key and the history, the
  // logits, the pending tokens, the layout, one checksum per chunk, and at most 512 bytes of fixed fields.
  static uint64_t FileEstimate(std::span<const int> key, const Snapshot& snapshot) {
    const size_t chunks = (snapshot.bytes.size() + SnapshotChunkPool::kChunkBytes - 1) / SnapshotChunkPool::kChunkBytes;
    return StateRoundUp(kStateFixedHeader + 512 + 4 * (key.size() + snapshot.hist.size()) +
                        4 * (snapshot.resume.logits.size() + snapshot.resume.pending_tokens.size()) +
                        8 * snapshot.qsa_layout.size() + 8 * chunks) + StateRoundUp(snapshot.bytes.size());
  }
  // A queued job's estimate (StateMeta::file_bytes holds it until the write).
  static uint64_t FileEstimate(const StateMeta& meta) { return meta.file_bytes; }
  // Holds mutex_. A new local day starts the day's totals at zero.
  void RollDay() {
    const std::string today = options_.today();
    if (today == day_) return;
    day_ = today;
    day_written_ = day_read_ = 0;
  }
  // Adds `bytes` to today's line of the ledger that all engines on this root share, and returns today's total. The
  // ledger holds one line "YYYY-MM-DD BYTES", mode 0600. An I/O error keeps this process's own count.
  uint64_t LedgerAdd(uint64_t bytes) {
    const std::string today = options_.today();
    const int fd = ::open((options_.root / "write-ledger").c_str(), O_RDWR | O_CREAT | O_CLOEXEC | O_NOFOLLOW, 0600);
    if (fd < 0) return bytes;
    struct Close { int fd; ~Close() { ::close(fd); } } closer{fd};
    if (::flock(fd, LOCK_EX) != 0) return bytes;
    char text[64] = {};
    const ssize_t got = ::pread(fd, text, sizeof text - 1, 0);
    uint64_t total = 0;
    char day[16] = {};
    unsigned long long stored = 0;
    if (got > 0 && std::sscanf(text, "%15s %llu", day, &stored) == 2 && today == day) total = stored;
    total += bytes;
    const std::string line = today + " " + std::to_string(total) + "\n";
    if (::ftruncate(fd, 0) != 0 || ::pwrite(fd, line.data(), line.size(), 0) != ssize_t(line.size())) return total;
    return total;
  }
  static bool Matches(const StateMeta& meta, std::span<const int> key, const ConversationPolicy& policy, bool compact) {
    return meta.compact == compact && meta.policy == policy && meta.key.size() <= key.size() &&
           std::equal(meta.key.begin(), meta.key.end(), key.begin());
  }
  std::string Name(const ConversationPolicy& policy, bool compact, std::span<const int> key) const {
    StateBytesOut out;
    out.Str(options_.identity);
    WritePolicy(out, policy);
    out.U8(compact);
    out.Ints(key);
    return StateSha256(out.Bytes().data(), out.Bytes().size()) + ".state";
  }
  // The root must sit inside the user's home. Components that this call creates get mode 0700.
  void PrepareRoot() {
    const char* home = crucible::SwitchText("HOME");
    if (!home || !*home || !options_.root.is_absolute()) throw std::invalid_argument("state disk needs HOME and an absolute directory");
    const auto base = std::filesystem::weakly_canonical(home);
    const auto root = std::filesystem::weakly_canonical(options_.root);
    const auto relative = root.lexically_relative(base);
    if (relative.empty() || relative == "." || *relative.begin() == "..")
      throw std::invalid_argument("the state directory must be inside the home directory");
    std::filesystem::path path = base;
    bool created = false;
    for (const auto& part : relative) {
      path /= part;
      created = ::mkdir(path.c_str(), 0700) == 0;
      if (!created && errno != EEXIST) throw std::runtime_error("cannot create the state directory");
    }
    // The root holds the ledger and every identity directory: this user's own directory, no symlink, mode 0700. An
    // existing root must be the tier's own folder (empty, or only the tier's files), so a chmod never changes a
    // shared folder such as ~/.cache.
    if (!created) {
      std::error_code error;
      for (const auto& item : std::filesystem::directory_iterator(root, error)) {
        const std::string name = item.path().filename();
        if (name != "write-ledger" && !(IdentityDirectory(name) && item.is_directory(error)))
          throw std::invalid_argument("the state directory must be a folder of its own (it holds other files)");
      }
      if (error) throw std::runtime_error("cannot read the state directory");
    }
    MakePrivateDirectory(root);
    options_.root = root;
  }
  static void MakePrivateDirectory(const std::filesystem::path& path) {
    if (::mkdir(path.c_str(), 0700) != 0 && errno != EEXIST) throw std::runtime_error("cannot create the state directory");
    struct stat info{};
    if (::lstat(path.c_str(), &info) != 0 || !S_ISDIR(info.st_mode) || info.st_uid != ::geteuid())
      throw std::runtime_error("the state directory is not a private directory of this user");
    if ((info.st_mode & 0777) != 0700 && ::chmod(path.c_str(), 0700) != 0)
      throw std::runtime_error("cannot set the state directory mode");
  }
  static bool StateFileName(const std::string& name) {
    return name.size() > 6 && (name.ends_with(".state") || name.find(".state.tmp.") != std::string::npos ||
                               name.find(".state.dead.") != std::string::npos);
  }
  // Directories of other identities (another binary, checkpoint or setting) leave at start, unless a live engine
  // holds their lock. Only state files, temporary files and the lock are removed; anything else stays.
  size_t RemoveOtherIdentities() {
    size_t removed = 0;
    std::error_code error;
    for (const auto& item : std::filesystem::directory_iterator(options_.root, error)) {
      const std::string name = item.path().filename();
      if (name == options_.directory || !item.is_directory(error) || item.is_symlink(error) ||
          !IdentityDirectory(name)) continue;
      const int fd = ::open((item.path() / ".lock").c_str(), O_RDWR | O_CLOEXEC | O_NOFOLLOW);
      if (fd >= 0 && ::flock(fd, LOCK_EX | LOCK_NB) != 0) { ::close(fd); continue; }
      for (const auto& file : std::filesystem::directory_iterator(item.path(), error))
        if (StateFileName(file.path().filename()) && file.is_regular_file(error))
          std::filesystem::remove(file.path(), error);
      std::filesystem::remove(item.path() / ".lock", error);
      if (fd >= 0) ::close(fd);
      removed += std::filesystem::remove(item.path(), error) ? 1 : 0;
    }
    return removed;
  }
  // The index comes from the file headers. Damaged files, files of another identity and temporary files leave.
  void Rebuild() {
    std::error_code error;
    std::vector<std::pair<std::filesystem::file_time_type, std::string>> order;
    for (const auto& item : std::filesystem::directory_iterator(dir_, error)) {
      const std::string name = item.path().filename();
      if (!StateFileName(name) || !item.is_regular_file(error)) continue;
      if (!name.ends_with(".state")) { std::filesystem::remove(item.path(), error); continue; }
      try {
        const auto header = ReadHeader(item.path());
        if (header.identity != options_.identity || header.meta.file_bytes != std::filesystem::file_size(item.path()) ||
            Name(header.meta.policy, header.meta.compact, header.meta.key) != name)
          throw std::runtime_error("foreign state file");
        StateMeta meta = header.meta;
        meta.name = name;
        Insert(std::move(meta));
        bytes_ += header.meta.file_bytes;
        order.emplace_back(item.last_write_time(error), name);
      } catch (const std::exception&) {
        std::filesystem::remove(item.path(), error);
      }
    }
    std::sort(order.begin(), order.end());
    for (const auto& [time, name] : order) index_[name].used = ++tick_;
    MakeRoom(0, nullptr);
  }
  StateHeader ReadHeader(const std::filesystem::path& path) const {
    const int fd = StateOpen(path, O_RDONLY);
    if (fd < 0) throw std::runtime_error("cannot open a state file");
    struct Close { int fd; ~Close() { ::close(fd); } } closer{fd};
    StateBounce first;
    if (StateReadAll(fd, first.Get(), kStateAlign, 0)) throw std::runtime_error("cannot read a state header");
    uint64_t header_bytes;
    std::memcpy(&header_bytes, first.Get() + 16, 8);
    if (header_bytes < kStateAlign || header_bytes > kStateMaxHeader || header_bytes % kStateAlign)
      throw std::runtime_error("state header bounds");
    if (header_bytes == kStateAlign) return ParseStateHeader(first.Get(), kStateAlign);
    std::unique_ptr<uint8_t, void (*)(void*)> all(static_cast<uint8_t*>(std::aligned_alloc(kStateAlign, header_bytes)), std::free);
    if (!all) throw std::bad_alloc();
    if (StateReadAll(fd, all.get(), header_bytes, 0)) throw std::runtime_error("cannot read a state header");
    return ParseStateHeader(all.get(), header_bytes);
  }
  uint64_t FreeSpace() const {
    struct statvfs info{};
    if (::statvfs(dir_.c_str(), &info) != 0) return 0;
    return uint64_t(info.f_bavail) * info.f_frsize;
  }
  static bool Extends(const StateMeta& longer, const StateMeta& meta) {
    return longer.key.size() > meta.key.size() && longer.policy == meta.policy && longer.compact == meta.compact &&
           std::equal(meta.key.begin(), meta.key.end(), longer.key.begin());
  }
  // Disk eviction: a state that a longer state of its own branch extends leaves first (it cannot be the best
  // match any more, unless it is a shared frontier), then the least recently used. `incoming`: the state that
  // needs the room (it counts as a longer state). Holds mutex_.
  // The bytes that MakeRoom would free for a write of `file_bytes` (0 when the budget has room).
  uint64_t EvictableBytes(uint64_t file_bytes, const StateMeta& incoming) {
    std::lock_guard lock(mutex_);
    if (bytes_ + file_bytes <= options_.budget) return 0;
    uint64_t others = 0;
    for (const auto& [name, meta] : index_) if (name != incoming.name) others += meta.file_bytes;
    return std::min(others, bytes_ + file_bytes - options_.budget);
  }
  // The superseded marks are kept up to date on each insert and removal (Insert, Remove), so a choice is one pass.
  bool MakeRoom(uint64_t need, const StateMeta* incoming) {
    while (bytes_ + need > options_.budget) {
      const StateMeta* victim = nullptr;
      bool victim_superseded = false;
      for (const auto& [name, meta] : index_) {
        if (incoming && name == incoming->name) continue;
        const bool superseded = !meta.shared && (meta.superseded || (incoming && Extends(*incoming, meta)));
        if (!victim || superseded > victim_superseded || (superseded == victim_superseded && meta.used < victim->used))
          victim = &meta, victim_superseded = superseded;
      }
      if (!victim) return false;
      Log("state_disk evict tokens=" + std::to_string(victim->key.size()) + " bytes=" + std::to_string(victim->file_bytes) +
          " superseded=" + (victim_superseded ? "1" : "0"));
      Remove(victim->name);
      ++stats_.evicted;
    }
    return true;
  }
  // Holds mutex_. The unlink of a large file can take tens of milliseconds, so a rename (metadata only) takes the
  // file out of the index's name space now, and Unlink removes it later without the lock.
  void Remove(const std::string& name) {
    const auto found = index_.find(name);
    if (found == index_.end()) return;
    const auto dead = dir_ / (name + ".dead." + std::to_string(++dead_serial_));
    if (::rename((dir_ / name).c_str(), dead.c_str()) == 0) doomed_.push_back(dead);
    bytes_ -= found->second.file_bytes;
    const StateMeta gone = std::move(found->second);
    index_.erase(found);
    // The states that only the removed one extended are not superseded any more.
    for (auto& [other_name, meta] : index_)
      if (meta.superseded && Extends(gone, meta))
        meta.superseded = std::any_of(index_.begin(), index_.end(), [&](const auto& o) { return Extends(o.second, meta); });
  }
  // Holds mutex_. Adds `meta` to the index and updates the superseded marks: one pass over the index.
  void Insert(StateMeta meta) {
    meta.superseded = false;
    for (auto& [other_name, other] : index_) {
      if (other_name == meta.name) continue;
      if (Extends(meta, other)) other.superseded = true;
      else if (Extends(other, meta)) meta.superseded = true;
    }
    const std::string name = meta.name;
    index_[name] = std::move(meta);
  }
  void Unlink() {
    std::vector<std::filesystem::path> doomed;
    {
      std::lock_guard lock(mutex_);
      doomed.swap(doomed_);
    }
    std::error_code error;
    for (const auto& path : doomed) std::filesystem::remove(path, error);
  }
  void Expire() {
    const auto now = std::chrono::steady_clock::now();
    for (auto it = staged_.begin(); it != staged_.end();)
      if (it->second.state == StageState::Ready &&
          std::chrono::duration<double>(now - it->second.since).count() > options_.stage_seconds)
        it = Unstage(it);
      else ++it;
  }
  typename std::map<std::string, Staged>::iterator Unstage(typename std::map<std::string, Staged>::iterator it) {
    staged_bytes_ -= it->second.meta.payload;
    return staged_.erase(it);
  }

  void WriterLoop() {
    for (;;) {
      Job job;
      {
        std::unique_lock lock(mutex_);
        work_.wait(lock, [this] { return stop_ || !jobs_.empty(); });
        if (stop_) {
          for (const auto& dropped : jobs_) queued_bytes_ -= dropped.meta.payload;
          jobs_.clear();
          queued_names_.clear();
          idle_.notify_all();
          return;
        }
        job = std::move(jobs_.front());
        jobs_.pop_front();
        writing_ = true;
      }
      if (options_.before_write) options_.before_write();
      const std::string result = Write(job);
      std::string line = "state_disk spill tokens=" + std::to_string(job.meta.key.size()) + " bytes=" +
                         std::to_string(job.meta.payload) + " " + result;
      {
        std::lock_guard lock(mutex_);
        queued_bytes_ -= job.meta.payload;
        queued_file_bytes_ -= FileEstimate(job.meta);
        queued_names_.erase(job.meta.name);
        ++(result.starts_with("ms=") ? stats_.written : stats_.refused);
        line += " day_written=" + std::to_string(day_written_) + " total_written=" + std::to_string(stats_.written_bytes);
      }
      job.keep.reset();
      Log(line);
      {
        std::lock_guard lock(mutex_);
        writing_ = false;
      }
      idle_.notify_all();
    }
  }
  // Returns "ms=<T> result=written" or "result=refused reason=<why>".
  std::string Write(const Job& job) {
    const double start = Now();
    const Snapshot& snapshot = *job.snapshot;
    const size_t chunks = snapshot.bytes.Chunks().size();
    const uint64_t header_bytes = BuildStateHeader(options_.identity, job.meta, snapshot,
                                                   std::vector<uint64_t>(chunks)).size();
    const uint64_t file_bytes = header_bytes + StateRoundUp(job.meta.payload);
    // Other engines on this root write to the same ledger: read it, so their writes count against the budget. Only
    // writes that run at the same time in other engines can still pass it (each counts when it ends).
    const uint64_t shared = LedgerAdd(0);
    {
      std::lock_guard lock(mutex_);
      RollDay();
      day_written_ = std::max(day_written_, shared);
      if (day_written_ + file_bytes > options_.write_per_day) {
        ++stats_.refused_daily;
        return "result=refused reason=daily";
      }
      if (file_bytes > options_.budget) return "result=refused reason=budget";
    }
    // The checks that can refuse run before eviction, so a refused write evicts nothing. The space of files that
    // eviction would free counts toward the free space (and files of earlier evictions leave first).
    Unlink();
    if (options_.free_space() + EvictableBytes(file_bytes, job.meta) < file_bytes + options_.free_reserve)
      return "result=refused reason=full";
    if (stopping_) return "result=refused reason=stopped";
    {
      std::lock_guard lock(mutex_);
      if (!MakeRoom(file_bytes, &job.meta)) return "result=refused reason=budget";
    }
    Unlink();
    const auto final_path = dir_ / job.meta.name;
    const auto temp = dir_ / (job.meta.name + ".tmp." + std::to_string(::getpid()));
    const int fd = StateOpen(temp, O_WRONLY | O_CREAT | O_EXCL);
    if (fd < 0) return "result=refused reason=" + StateIoReason(errno);
    std::vector<uint64_t> sums(chunks);
    std::string failed = StateParallel(chunks, options_.io_threads, [&](size_t index, StateBounce& bounce) -> std::string {
      if (options_.before_chunk) options_.before_chunk();
      if (stopping_) return "stopped";
      const SnapshotChunk& chunk = snapshot.bytes.Chunks()[index];
      const size_t length = std::min(SnapshotChunkPool::kChunkBytes, size_t(job.meta.payload) - index * SnapshotChunkPool::kChunkBytes);
      // Every chunk goes through the bounce buffer: direct I/O from the pinned chunk pages fails with EFAULT on the test machine
      // (Oct 9: every spill was refused). The padding is zero: the file never holds bytes of another state.
      const size_t padded = StateRoundUp(length);
      uint8_t* source = bounce.Get();
      std::memcpy(source, chunk.host, length);
      std::memset(source + length, 0, padded - length);
      // On the bounce buffer: the pinned chunk is read once (by the copy). The bytes are the same.
      sums[index] = StateChecksum(source, length);
      int error = options_.io_fault ? options_.io_fault(source) : 0;
      if (!error) error = StateWriteAll(fd, source, padded, header_bytes + index * SnapshotChunkPool::kChunkBytes);
      return error ? StateIoReason(error) : "";
    });
    if (failed.empty()) {
      const auto header = BuildStateHeader(options_.identity, job.meta, snapshot, sums);
      std::unique_ptr<uint8_t, void (*)(void*)> aligned(static_cast<uint8_t*>(std::aligned_alloc(kStateAlign, header.size())), std::free);
      if (!aligned) failed = "memory";
      else {
        std::memcpy(aligned.get(), header.data(), header.size());
        const int error = StateWriteAll(fd, aligned.get(), header.size(), 0);
        if (error) failed = StateIoReason(error);
      }
    }
    if (failed.empty() && ::fsync(fd) != 0) failed = StateIoReason(errno);
    ::close(fd);
    if (failed.empty() && ::rename(temp.c_str(), final_path.c_str()) != 0) failed = StateIoReason(errno);
    if (!failed.empty()) {
      std::error_code error;
      std::filesystem::remove(temp, error);
      return "result=refused reason=" + failed;
    }
    if (const int dir = ::open(dir_.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC); dir >= 0) { ::fsync(dir); ::close(dir); }
    StateMeta meta = job.meta;
    meta.file_bytes = file_bytes;
    const uint64_t day = LedgerAdd(file_bytes);
    {
      std::lock_guard lock(mutex_);
      day_written_ = std::max(day, day_written_ + file_bytes);
      stats_.written_bytes += file_bytes;
      if (const auto old = index_.find(meta.name); old != index_.end()) bytes_ -= old->second.file_bytes;
      meta.used = ++tick_;
      Insert(std::move(meta));
      bytes_ += file_bytes;
    }
    char text[64];
    std::snprintf(text, sizeof text, "ms=%.1f result=written", (Now() - start) * 1e3);
    return text;
  }

  void ReaderLoop() {
    for (;;) {
      std::string name;
      StateMeta meta;
      {
        std::unique_lock lock(mutex_);
        work_.wait(lock, [this] { return stop_ || !reads_.empty(); });
        if (stop_) return;
        name = reads_.front();
        reads_.pop_front();
        const auto found = staged_.find(name);
        if (found == staged_.end()) continue;
        meta = found->second.meta;
      }
      if (options_.before_read) options_.before_read();
      const double start = Now();
      std::string reason;
      std::shared_ptr<Snapshot> snapshot;
      try { snapshot = Read(meta, reason); } catch (const std::bad_alloc&) { reason = "memory"; }
      char text[224];
      std::snprintf(text, sizeof text, "state_disk read tokens=%zu bytes=%llu ms=%.1f result=%s%s%s", meta.key.size(),
                    static_cast<unsigned long long>(meta.payload), (Now() - start) * 1e3, snapshot ? "ok" : "failed",
                    snapshot ? "" : " reason=", snapshot ? "" : reason.c_str());
      {
        std::lock_guard lock(mutex_);
        ++(snapshot ? stats_.reads : stats_.failed);
        RollDay();
        if (snapshot) stats_.read_bytes += meta.payload, day_read_ += meta.payload;
        if (snapshot) std::snprintf(text + std::strlen(text), sizeof text - std::strlen(text), " day_read=%llu",
                                    static_cast<unsigned long long>(day_read_));
        // A damaged or foreign file leaves the disk; a read that only lacked memory or met an I/O error keeps it.
        if (!snapshot && reason != "memory" && !reason.starts_with("io")) Remove(name);
        if (const auto found = staged_.find(name); found != staged_.end()) {
          found->second.state = snapshot ? StageState::Ready : StageState::Failed;
          found->second.snapshot = std::move(snapshot);
          found->second.reason = reason;
          found->second.since = std::chrono::steady_clock::now();
          if (!found->second.snapshot) { staged_bytes_ -= found->second.meta.payload; found->second.meta.payload = 0; }
        }
      }
      Unlink();
      Log(text);
    }
  }
  // Reads and checks one file. The header must name this identity and the same key, policy and storage mode, and
  // every payload chunk must match its checksum.
  std::shared_ptr<Snapshot> Read(const StateMeta& meta, std::string& reason) {
    const auto path = dir_ / meta.name;
    StateHeader header;
    try { header = ReadHeader(path); } catch (const std::bad_alloc&) { throw; } catch (const std::exception&) {
      reason = "header";
      return nullptr;
    }
    std::error_code error;
    if (header.identity != options_.identity || header.meta.key != meta.key || !(header.meta.policy == meta.policy) ||
        header.meta.compact != meta.compact || header.meta.payload != meta.payload ||
        std::filesystem::file_size(path, error) != header.meta.file_bytes) {
      reason = "header";
      return nullptr;
    }
    auto snapshot = std::make_shared<Snapshot>();
    snapshot->hist = std::move(header.hist);
    if (!RetainedPrefix::FromFields(header.retained, snapshot->retained) || !SetSnapshotKvFields(header.kv, *snapshot)) {
      reason = "header";
      return nullptr;
    }
    snapshot->resume = std::move(header.resume);
    snapshot->qsa_layout = std::move(header.layout);
    snapshot->bytes = SnapshotBytes(size_t(meta.payload));
    {
      std::lock_guard lock(mutex_);
      if (const auto found = staged_.find(meta.name); found != staged_.end()) found->second.allocated = true;
    }
    const int fd = StateOpen(path, O_RDONLY);
    if (fd < 0) { reason = "io errno=" + std::to_string(errno); return nullptr; }
    const auto& chunks = snapshot->bytes.Chunks();
    reason = StateParallel(chunks.size(), options_.io_threads, [&](size_t index, StateBounce& bounce) -> std::string {
      const size_t length = std::min(SnapshotChunkPool::kChunkBytes, size_t(meta.payload) - index * SnapshotChunkPool::kChunkBytes);
      const size_t padded = StateRoundUp(length);
      uint8_t* target = chunks[index].host;
      // Through the bounce buffer, as the write: direct I/O into the pinned chunk pages fails with EFAULT.
      uint8_t* into = bounce.Get();
      int error = options_.io_fault ? options_.io_fault(into) : 0;
      if (!error) error = StateReadAll(fd, into, padded, header.header_bytes + index * SnapshotChunkPool::kChunkBytes);
      if (error) return "io errno=" + std::to_string(error);
      // On the bounce buffer, before the copy: the pinned chunk is written once and never read.
      if (StateChecksum(into, length) != header.sums[index]) return "checksum";
      std::memcpy(target, into, length);
      return "";
    });
    ::close(fd);
    return reason.empty() ? snapshot : nullptr;
  }

  StateDiskOptions options_;
  std::filesystem::path dir_;
  int lock_ = -1;
  bool enabled_ = false;
  mutable std::mutex mutex_;
  std::condition_variable work_, idle_;
  bool stop_ = false, writing_ = false;
  std::atomic<bool> stopping_ = false;
  std::map<std::string, StateMeta> index_;
  uint64_t bytes_ = 0, tick_ = 0, dead_serial_ = 0;
  std::vector<std::filesystem::path> doomed_;
  std::deque<Job> jobs_;
  std::set<std::string> queued_names_;
  uint64_t queued_bytes_ = 0, queued_file_bytes_ = 0;
  std::string day_;
  uint64_t day_written_ = 0, day_read_ = 0;
  std::map<std::string, Staged> staged_;
  std::deque<std::string> reads_;
  uint64_t staged_bytes_ = 0;
  StateDiskStats stats_;
  std::thread writer_, reader_;
};
}  // namespace crucible
