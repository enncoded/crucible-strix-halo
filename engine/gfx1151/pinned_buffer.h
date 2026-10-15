#pragma once
#include <algorithm>
#include "common.h"

namespace crucible {

// Fixed storage avoids pageable-transfer registration during decode. The caller
// must complete stream work before it reuses or destroys this buffer.
template <typename T, size_t N>
class PinnedBuffer {
 public:
  PinnedBuffer() { HIP_OK(hipHostMalloc(&data_, sizeof(T) * N)); }
  ~PinnedBuffer() { HIP_OK(hipHostFree(data_)); }
  PinnedBuffer(const PinnedBuffer&) = delete;
  PinnedBuffer& operator=(const PinnedBuffer&) = delete;
  T* data() { return data_; }

 private:
  T* data_ = nullptr;
};

// Drafts can return before their input DMA completes. Protect host storage
// with a copy event rather than synchronizing the whole compute stream.
template <typename T, size_t N>
class PinnedInputBuffer {
 public:
  PinnedInputBuffer() { HIP_OK(hipEventCreateWithFlags(&copied_, hipEventDisableTiming)); }
  ~PinnedInputBuffer() {
    Wait();
    HIP_OK(hipEventDestroy(copied_));
  }
  PinnedInputBuffer(const PinnedInputBuffer&) = delete;
  PinnedInputBuffer& operator=(const PinnedInputBuffer&) = delete;

  template <typename Fill>
  void FillTo(T* device, size_t count, Fill fill, hipStream_t stream = 0) {
    fill(Begin(count));
    Commit(device, count, stream);
  }

  // Split FillTo for a fill that runs on another thread: Begin returns the
  // storage once the previous copy is done; Commit queues the copy.
  T* Begin(size_t count) {
    if (count > N) {
      std::fprintf(stderr, "pinned input capacity exceeded\n");
      std::abort();
    }
    Wait();
    return storage_.data();
  }
  void Commit(T* device, size_t count, hipStream_t stream = 0) {
    HIP_OK(hipMemcpyAsync(device, storage_.data(), sizeof(T) * count, hipMemcpyHostToDevice, stream));
    HIP_OK(hipEventRecord(copied_, stream));
    pending_ = true;
  }

  void CopyTo(T* device, const T* source, size_t count, hipStream_t stream = 0) {
    FillTo(device, count, [&](T* target) { std::copy_n(source, count, target); }, stream);
  }

  // The values of the last FillTo, until the next one.
  const T* data() { return storage_.data(); }

 private:
  void Wait() {
    if (pending_) HIP_OK(hipEventSynchronize(copied_));
    pending_ = false;
  }
  PinnedBuffer<T, N> storage_;
  hipEvent_t copied_{};
  bool pending_ = false;
};

}  // namespace crucible
