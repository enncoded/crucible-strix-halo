// Snapshot storage in fixed pinned chunks from one process-wide pool (oracle-cache item 8).
// A save takes chunks without hipHostMalloc or a zero fill once the pool is warm, and an evicted snapshot gives its
// chunks back. The chunk size bounds the memory that a snapshot holds beyond its own bytes.
#pragma once
#include <algorithm>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <mutex>
#include <new>
#include <optional>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

namespace crucible {
struct SnapshotChunk { uint8_t* host = nullptr; uint8_t* device = nullptr; };

class SnapshotChunkPool {
 public:
  static constexpr size_t kChunkBytes = size_t(8) << 20;
  static SnapshotChunkPool& Instance() {
    // Never destroyed: process exit returns the pages, and a destructor could run after the HIP runtime.
    static auto* pool = new SnapshotChunkPool;
    return *pool;
  }
  // Idle chunks stay pinned while all pinned chunks, in use and idle, fit in this many bytes (the conversation cache
  // budget, which admission already reserves). Chunks in use may pass it for a while (a disk write holds an evicted
  // snapshot, an earlier change); when they come back, the excess goes back to the system. Zero returns each released chunk at once.
  void SetPinnedLimit(size_t bytes) {
    std::vector<SnapshotChunk> excess;
    {
      std::lock_guard lock(mutex_);
      pinned_limit_ = bytes;
      blocked_ = false;
      excess = Trim();
    }
    wake_.notify_all();
    Free(excess);
  }
  size_t IdleBytes() const {
    std::lock_guard lock(mutex_);
    return idle_.size() * kChunkBytes;
  }
  // a save pins each new chunk on the calling thread (about 5.5 GB/s on the test machine), so a 3.5 GB tail save stopped
  // every decoder of the batch worker for 0.65 s. A helper thread pins idle chunks before the saves instead: it keeps
  // `bytes` idle, inside the pinned limit, and pins a chunk only when `may_pin` allows it (the host reserve). A pin
  // failure or a refusal stops it until the next call. Zero stops the growth; idle chunks stay, as evicted ones do.
  void Reserve(size_t bytes, std::function<bool()> may_pin = {}) {
    {
      std::lock_guard lock(mutex_);
      reserve_ = bytes;
      may_pin_ = std::move(may_pin);
      blocked_ = false;
      if (bytes && !helper_.joinable()) {
        stop_ = false;
        // Without a thread the saves pin on the caller, as before the reserve.
        try { helper_ = std::thread([this] { Fill(); }); } catch (const std::exception&) {}
      }
    }
    wake_.notify_all();
  }
  // Ends the helper and waits for its pin in progress: call it before the HIP runtime can go away.
  void StopReserve() {
    std::thread helper;
    {
      std::lock_guard lock(mutex_);
      stop_ = true;
      reserve_ = 0;
      helper = std::move(helper_);
    }
    wake_.notify_all();
    if (helper.joinable()) helper.join();
  }
  // The helper pins now or has chunks left to pin (tests wait for it).
  bool Filling() const {
    std::lock_guard lock(mutex_);
    return pinning_ || (helper_.joinable() && !stop_ && !blocked_ && Short());
  }
  SnapshotChunk Take() {
    {
      std::unique_lock lock(mutex_);
      if (!idle_.empty()) {
        const SnapshotChunk chunk = idle_.back();
        idle_.pop_back();
        lock.unlock();
        wake_.notify_all();  // the helper pins a replacement while the target stands
#if CRUCIBLE_DEBUG_NONFINITE & 2
        std::memset(chunk.host, 0xFF, kChunkBytes);  // a restore must read only bytes that a save wrote
#endif
        return chunk;
      }
    }
    const SnapshotChunk chunk = Pin();
#if CRUCIBLE_DEBUG_NONFINITE & 2
    std::memset(chunk.host, 0xFF, kChunkBytes);
#endif
    std::lock_guard lock(mutex_);
    ++pinned_;
    return chunk;
  }
  void Give(SnapshotChunk chunk) {
    std::vector<SnapshotChunk> excess;
    {
      std::lock_guard lock(mutex_);
      idle_.push_back(chunk);
      excess = Trim();
    }
    Free(excess);
  }

 private:
  static SnapshotChunk Pin() {
    SnapshotChunk chunk;
    if (hipHostMalloc(&chunk.host, kChunkBytes) != 0) throw std::bad_alloc();
#ifdef __HIPCC__
    void* device = nullptr;
    if (hipHostGetDevicePointer(&device, chunk.host, 0) != hipSuccess) {
      HIP_OK(hipHostFree(chunk.host));
      throw std::bad_alloc();
    }
    chunk.device = static_cast<uint8_t*>(device);
#else
    chunk.device = chunk.host;
#endif
    return chunk;
  }
  // Under the lock: the idle bytes are below the reserve and one more chunk fits the limit.
  bool Short() const {
    return idle_.size() * kChunkBytes < reserve_ && (pinned_ + 1) * kChunkBytes <= pinned_limit_;
  }
  // The helper thread: pins outside the lock, so Take, Give and IdleBytes on the device worker never wait for a pin.
  void Fill() {
    for (;;) {
      std::function<bool()> may_pin;
      {
        std::unique_lock lock(mutex_);
        wake_.wait(lock, [&] { return stop_ || (!blocked_ && Short()); });
        if (stop_) return;
        try {
          may_pin = may_pin_;
        } catch (const std::exception&) {
          blocked_ = true;
          continue;
        }
        pinning_ = true;
      }
      // No exception leaves the helper (it would end the process): a failed check or pin stops it until the next
      // Reserve or SetPinnedLimit.
      std::optional<SnapshotChunk> chunk;
      try {
        if (!may_pin || may_pin()) chunk = Pin();
      } catch (const std::exception&) {}
      std::vector<SnapshotChunk> excess;
      {
        std::lock_guard lock(mutex_);
        pinning_ = false;
        bool stop = !chunk;
        try {
          if (chunk) {
            idle_.push_back(*chunk);
            ++pinned_;
            chunk.reset();  // the pool holds it now
            excess = Trim();  // a Take on the worker pinned at the same time and passed the limit
          }
        } catch (const std::exception&) {
          stop = true;
        }
        if (stop) blocked_ = true;
      }
      if (chunk) HIP_OK(hipHostFree(chunk->host));  // the pool could not keep it
      Free(excess);
    }
  }
  // Under the lock: the idle chunks above the limit leave the pool.
  std::vector<SnapshotChunk> Trim() {
    std::vector<SnapshotChunk> excess;
    while (!idle_.empty() && pinned_ * kChunkBytes > pinned_limit_) {
      excess.push_back(idle_.back());
      idle_.pop_back();
      --pinned_;
    }
    return excess;
  }
  // Outside the lock: hipHostFree synchronizes the device, so a disk writer that frees hundreds of chunks must not
  // stop the device worker's Take, Give and IdleBytes.
  static void Free(const std::vector<SnapshotChunk>& chunks) {
    for (const auto& chunk : chunks) HIP_OK(hipHostFree(chunk.host));
  }
  mutable std::mutex mutex_;
  std::vector<SnapshotChunk> idle_;
  size_t pinned_limit_ = 0, pinned_ = 0;  // pinned_: chunks in use and idle
  // The reserve helper. Never joined at exit: the pool is never destroyed (StopReserve ends it).
  std::condition_variable wake_;
  std::thread helper_;
  std::function<bool()> may_pin_;
  size_t reserve_ = 0;
  bool stop_ = false, blocked_ = false, pinning_ = false;
};

// The bytes of one snapshot, in order across its chunks. Contents start unspecified (no fill): Save writes them all.
class SnapshotBytes {
 public:
  SnapshotBytes() = default;
  explicit SnapshotBytes(size_t size) { Resize(size); }
  SnapshotBytes(const SnapshotBytes& other) {
    Resize(other.size_);
    for (size_t index = 0; index < chunks_.size(); ++index)
      std::memcpy(chunks_[index].host, other.chunks_[index].host, Piece(index));
  }
  SnapshotBytes& operator=(const SnapshotBytes& other) {
    if (this != &other) *this = SnapshotBytes(other);
    return *this;
  }
  SnapshotBytes(SnapshotBytes&& other) noexcept
      : chunks_(std::exchange(other.chunks_, {})), size_(std::exchange(other.size_, 0)) {}
  SnapshotBytes& operator=(SnapshotBytes&& other) noexcept {
    if (this != &other) {
      Release();
      chunks_ = std::exchange(other.chunks_, {});
      size_ = std::exchange(other.size_, 0);
    }
    return *this;
  }
  ~SnapshotBytes() { Release(); }

  size_t size() const { return size_; }
  bool empty() const { return size_ == 0; }
  // Tests shorten a copy to model a corrupt snapshot.
  void pop_back() { if (size_) --size_; }
  const std::vector<SnapshotChunk>& Chunks() const { return chunks_; }
  // Calls f(chunk, offset_in_chunk, bytes, offset_in_range) for each chunk piece of [offset, offset + bytes).
  template<class F> void ForEachPiece(size_t offset, size_t bytes, F f) const {
    if (offset > size_ || bytes > size_ - offset) throw std::out_of_range("snapshot byte range");
    for (size_t done = 0; done < bytes;) {
      const size_t at = offset + done;
      const size_t index = at / SnapshotChunkPool::kChunkBytes, inside = at % SnapshotChunkPool::kChunkBytes;
      const size_t count = std::min(bytes - done, SnapshotChunkPool::kChunkBytes - inside);
      f(chunks_[index], inside, count, done);
      done += count;
    }
  }
  std::vector<uint8_t> Copy() const {
    std::vector<uint8_t> out(size_);
    ForEachPiece(0, size_, [&](const SnapshotChunk& chunk, size_t inside, size_t count, size_t at) {
      std::memcpy(out.data() + at, chunk.host + inside, count);
    });
    return out;
  }
  bool operator==(const SnapshotBytes& other) const {
    if (size_ != other.size_) return false;
    for (size_t index = 0; index < chunks_.size(); ++index)
      if (std::memcmp(chunks_[index].host, other.chunks_[index].host, Piece(index)) != 0) return false;
    return true;
  }

 private:
  size_t Piece(size_t index) const {
    return std::min(SnapshotChunkPool::kChunkBytes, size_ - index * SnapshotChunkPool::kChunkBytes);
  }
  void Resize(size_t size) {
    const size_t count = (size + SnapshotChunkPool::kChunkBytes - 1) / SnapshotChunkPool::kChunkBytes;
    chunks_.reserve(count);
    try {
      while (chunks_.size() < count) chunks_.push_back(SnapshotChunkPool::Instance().Take());
    } catch (...) {
      Release();
      throw;
    }
    size_ = size;
  }
  void Release() {
    for (const auto& chunk : chunks_) SnapshotChunkPool::Instance().Give(chunk);
    chunks_.clear();
    size_ = 0;
  }
  std::vector<SnapshotChunk> chunks_;
  size_t size_ = 0;
};

struct SnapshotPiece { const void* src; void* dst; size_t bytes; };
#ifdef __HIPCC__
// blockIdx.y picks a piece and x strides across it. A byte copy, so the saved bits equal the device bits.
__global__ void SnapshotCopyKernel(const SnapshotPiece* pieces, int first) {
  const SnapshotPiece piece = pieces[first + int(blockIdx.y)];
  const size_t start = size_t(blockIdx.x) * blockDim.x + threadIdx.x, stride = size_t(gridDim.x) * blockDim.x;
  if (((uintptr_t(piece.src) | uintptr_t(piece.dst) | piece.bytes) & 15) == 0) {
    const auto* src = static_cast<const uint4*>(piece.src);
    auto* dst = static_cast<uint4*>(piece.dst);
    for (size_t i = start; i < piece.bytes / 16; i += stride) dst[i] = src[i];
  } else {
    const auto* src = static_cast<const uint8_t*>(piece.src);
    auto* dst = static_cast<uint8_t*>(piece.dst);
    for (size_t i = start; i < piece.bytes; i += stride) dst[i] = src[i];
  }
}
#endif
// Copies the spans, in order, to (save) or from (restore) the snapshot chunks and waits for the copy. One kernel on
// the null stream moves every piece: on gfx1151 it reached 86 GB/s where one hipMemcpy per span reached 58 GB/s
// (bench/snapshot_copy_bench.hip). Null-stream order matches the hipMemcpy calls it replaces.
template<class Span>
void CopySnapshotSpans(const std::vector<Span>& spans, const SnapshotBytes& bytes, bool to_host) {
  std::vector<SnapshotPiece> pieces;
  size_t offset = 0;
  for (const auto& span : spans) {
    bytes.ForEachPiece(offset, span.bytes, [&](const SnapshotChunk& chunk, size_t inside, size_t count, size_t at) {
      auto* device = static_cast<uint8_t*>(const_cast<void*>(span.pointer)) + at;
      if (to_host) pieces.push_back({device, chunk.device + inside, count});
      else pieces.push_back({chunk.device + inside, device, count});
    });
    offset += span.bytes;
  }
  if (offset != bytes.size()) throw std::logic_error("snapshot spans do not cover the snapshot bytes");
#ifdef __HIPCC__
  // The piece table lives in pinned memory that the kernel reads; the wait below frees it for the next copy.
  static std::mutex mutex;
  static SnapshotPiece* table = nullptr;
  static SnapshotPiece* device_table = nullptr;
  static size_t capacity = 0;
  std::lock_guard lock(mutex);
  if (pieces.size() > capacity) {
    if (table) HIP_OK(hipHostFree(table));
    table = device_table = nullptr;
    capacity = 0;
    const size_t wanted = std::max<size_t>(pieces.size(), 4096);
    if (hipHostMalloc(&table, wanted * sizeof(SnapshotPiece)) != hipSuccess) throw std::bad_alloc();
    void* mapped = nullptr;
    HIP_OK(hipHostGetDevicePointer(&mapped, table, 0));
    device_table = static_cast<SnapshotPiece*>(mapped);
    capacity = wanted;
  }
  std::copy(pieces.begin(), pieces.end(), table);
  constexpr int kMaxGridY = 65535;
  for (size_t first = 0; first < pieces.size(); first += kMaxGridY) {
    const unsigned count = unsigned(std::min<size_t>(kMaxGridY, pieces.size() - first));
    hipLaunchKernelGGL(SnapshotCopyKernel, dim3(16, count), dim3(256), 0, 0, device_table, int(first));
    HIP_OK(hipGetLastError());
  }
#else
  for (const auto& piece : pieces)
    HIP_OK(hipMemcpy(piece.dst, piece.src, piece.bytes, to_host ? hipMemcpyDeviceToHost : hipMemcpyHostToDevice));
#endif
  HIP_OK(hipStreamSynchronize(0));
}
}  // namespace crucible
