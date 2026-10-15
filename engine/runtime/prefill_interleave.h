// Decode passes between the layers of a bulk prefill chunk (interleave design).
// CPU-visible: the engine, the batch session and the CPU tests share these rules.
#pragma once
#include "../switches.h"
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <span>
#include <string>

namespace crucible {

// run() executes between layers; it must leave the chunk's live data (pf_H_ and the prefilling sequence) intact.
struct PrefillBetween {
  int every = 0;
  std::function<void()> run;
};

// Layer 0 consumes tok_ (embed) and layer 1 consumes rows_ (PLE), so the first pass can follow layer 1. No pass
// follows the last layer: the chunk then only packs H and runs its head. Passes sit half an interval off the chunk
// edges (after layers every/2, every/2 + every, ...): the decode step between chunks then acts as one more pass, so
// the gap across a chunk boundary matches the gaps inside it (every 8: 0.65 s at the edge became the longest gap).
constexpr bool InterleaveAfterLayer(int layer, int every, int layers) {
  return every > 0 && layer >= 1 && layer + 1 < layers && (layer + 1 + every / 2) % every == 0;
}

// Between layers the chunk keeps only its padded trunk stacks. A chunk of max_rows or fewer rows, or a tight
// layout, uses H_, which a decode pass overwrites.
constexpr bool InterleaveAllowed(int rows, int max_rows, bool padded) { return rows > max_rows && padded; }

constexpr int InterleavePasses(int every, int layers) {
  int passes = 0;
  for (int layer = 0; layer < layers; ++layer) passes += InterleaveAfterLayer(layer, every, layers);
  return passes;
}

constexpr uint64_t kFnvBasis = 1469598103934665603ull;
inline uint64_t Fnv1a(const void* data, size_t bytes, uint64_t hash = kFnvBasis) {
  const auto* p = static_cast<const unsigned char*>(data);
  for (size_t i = 0; i < bytes; ++i) hash = (hash ^ p[i]) * 1099511628211ull;
  return hash;
}
inline uint64_t Fnv1a(std::span<const int> ids, uint64_t hash = kFnvBasis) {
  return Fnv1a(ids.data(), ids.size_bytes(), hash);
}

// CRUCIBLE_INTERLEAVE_CHECK=PATH (analysis only; it synchronizes and copies logits): appends one line per
// verified row with a fully accepted prefix, and one per bulk prefill chunk:
//   row <prompt fnv> <position> <input token> <logits fnv>
//   chunk <prefix fnv> <pos0> <rows> <stacks fnv> <final logits fnv or 0>
// tools/interleave_check.py compares two files key by key; equal keys must give equal hashes.
class InterleaveCheck {
 public:
  explicit InterleaveCheck(const char* path = crucible::SwitchText("CRUCIBLE_INTERLEAVE_CHECK"))
      : file_(path && *path ? std::fopen(path, "a") : nullptr) {}
  ~InterleaveCheck() { if (file_) std::fclose(file_); }
  InterleaveCheck(const InterleaveCheck&) = delete;
  InterleaveCheck& operator=(const InterleaveCheck&) = delete;
  explicit operator bool() const { return file_ != nullptr; }
  void Row(uint64_t prompt, uint64_t position, int token, uint64_t logits) {
    if (!file_) return;
    // Engine and session append to one file: each line is flushed whole.
    std::fprintf(file_, "row %016llx %llu %d %016llx\n", (unsigned long long)prompt,
                 (unsigned long long)position, token, (unsigned long long)logits);
    std::fflush(file_);
  }
  void Chunk(uint64_t prefix, uint64_t pos0, int rows, uint64_t stacks, uint64_t logits) {
    if (file_) {
      std::fprintf(file_, "chunk %016llx %llu %d %016llx %016llx\n", (unsigned long long)prefix,
                   (unsigned long long)pos0, rows, (unsigned long long)stacks, (unsigned long long)logits);
      std::fflush(file_);
    }
  }

 private:
  std::FILE* file_;
};

}  // namespace crucible
