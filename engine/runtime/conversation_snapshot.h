// Included after Engine. Snapshots never refer to mutable engine storage.
#pragma once
#include "conversation_resume.h"
#include "../layers/cache_storage.h"
#include "../gfx1151/pinned_allocator.h"
#include "snapshot_bytes.h"
#include <memory>
#include <stdexcept>

struct ConversationSnapshot {
  std::vector<int> hist;
  RetainedPrefix retained;
  ConversationResume resume;
  SnapshotBytes bytes;
  std::vector<std::pair<int, int>> qsa_layout;
};
struct ConversationStateIO {
  static void Validate(Engine& e, const SequenceState& state, const ConversationResume& r,
                       size_t length, float* stk, float* carry, bool require_mtp = true) {
    const bool owned = std::any_of(e.sequences_.begin(), e.sequences_.end(),
                                  [&](const auto& s) { return s.get() == &state; });
    if (!owned || !stk || !carry) throw std::runtime_error("snapshot needs an owned stream and request stacks");
    if (r.length != length || length > size_t(state.capacity) || r.mpos < 0 || r.mpos > state.mtp_capacity)
      throw std::runtime_error("invalid conversation snapshot position");
    if (require_mtp && r.mpos > 0 && !state.qc.contains(kLayers))
      throw std::runtime_error("snapshot MTP position needs an allocated MTP cache");
    if (!r.logits.empty() && r.logits.size() != kVocab)
      throw std::runtime_error("snapshot target logit shape changed");
  }
  static std::vector<std::pair<int, int>> Layout(const SequenceState& state, bool include_mtp = true) {
    std::vector<std::pair<int, int>> layout;
    for (const auto& [layer, cache] : state.qc) {
      if (layer == kLayers && !include_mtp) continue;
      if (cache.raw_slots != 0 && cache.raw_slots != kQsaRawSlots)
        throw std::runtime_error("unsupported raw cache layout");
      layout.emplace_back(layer, cache.raw_slots);
    }
    return layout;
  }
  static std::vector<DeviceSpan> Spans(const SequenceState& state, size_t n, int mpos,
                                       float* stk, float* carry, bool include_mtp = true) {
    std::vector<DeviceSpan> spans;
    const auto add = [&](void* p, size_t bytes) { if (bytes) spans.push_back({p, bytes}); };
    for (const auto& span : state.FixedSpans()) add(const_cast<void*>(span.pointer), span.bytes);
    for (const auto& [l, c] : state.qc) {
      if (l == kLayers && !include_mtp) continue;
      const size_t length = l == kLayers ? size_t(mpos) : n;
      add(c.k, length * 512 * sizeof(__half));
      add(c.v, length * 512 * sizeof(__half));
      add(c.ik_raw, QsaRawSnapshotBytes(c.raw_slots, length));
      add(c.ik_pool, length ? QsaPoolFloats(int(length)) * sizeof(float) : 0);
    }
    add(stk, sizeof(float) * (kMaxRows + 1) * kWide);
    add(carry, sizeof(float) * kWide);
    return spans;
  }
  // The MTP rows that a restore allocates the MTP cache for (the snapshot holds them; the stream does not yet).
  static size_t MtpBytes(const SequenceState& state, size_t n) {
    return n * 2048 + QsaRawSnapshotBytes(state.compact_state ? kQsaRawSlots : 0, n) + QsaPoolFloats(int(n)) * sizeof(float);
  }
  // The bytes that Save copies for `rows` rows of this stream, with MTP rows up to `rows` when `mtp`, also before the
  // stream has its MTP cache.
  static size_t SaveBytes(const SequenceState& state, size_t rows, bool mtp) {
    rows = std::min(rows, size_t(state.capacity));
    const size_t mpos = mtp ? std::min(rows, size_t(state.mtp_capacity)) : 0;
    size_t bytes = 0;
    for (const auto& span : Spans(state, rows, int(mpos), nullptr, nullptr)) bytes += span.bytes;
    if (mpos && !state.qc.contains(kLayers)) bytes += MtpBytes(state, mpos);
    return bytes;
  }
  static size_t Bytes(Engine& e, const SequenceState& state, const ConversationResume& r,
                      float* stk, float* carry) {
    Validate(e, state, r, state.hist.size(), stk, carry);
    size_t bytes = sizeof(ConversationSnapshot) + state.hist.size() * sizeof(int) * 2;
    bytes += r.logits.size() * sizeof(float) + r.pending_tokens.size() * sizeof(int);
    bytes += Layout(state).size() * sizeof(std::pair<int, int>);
    for (const auto& span : Spans(state, r.length, r.mpos, stk, carry)) bytes += span.bytes;
    return bytes;
  }
  static std::shared_ptr<const ConversationSnapshot> Save(Engine& e, const SequenceState& state,
      const ConversationResume& r, float* stk, float* carry) {
    Validate(e, state, r, state.hist.size(), stk, carry);
    auto snapshot = std::make_shared<ConversationSnapshot>();
    snapshot->hist = state.hist;
    snapshot->retained = state.retained;
    snapshot->resume = r;
    snapshot->qsa_layout = Layout(state);
    const auto spans = Spans(state, r.length, r.mpos, stk, carry);
    size_t total = 0;
    for (const auto& span : spans) total += span.bytes;
    snapshot->bytes = SnapshotBytes(total);
    // Publish only a complete snapshot before decode or a following request can overwrite its source: the copy waits.
    CopySnapshotSpans(spans, snapshot->bytes, true);
    return snapshot;
  }
  static ConversationResume Restore(Engine& e, SequenceState& state, const ConversationSnapshot& snapshot,
                                    float* stk, float* carry) {
    Validate(e, state, snapshot.resume, snapshot.hist.size(), stk, carry, false);
    const bool mtp = std::any_of(snapshot.qsa_layout.begin(), snapshot.qsa_layout.end(),
                               [](const auto& item) { return item.first == kLayers; });
    auto trunk_layout = snapshot.qsa_layout;
    std::erase_if(trunk_layout, [](const auto& item) { return item.first == kLayers; });
    if (trunk_layout != Layout(state, false) || (snapshot.resume.mpos > 0 && !mtp))
      throw std::runtime_error("snapshot cache layout or MTP presence changed");
    if (mtp) {
      auto expected = Layout(state, false);
      expected.emplace_back(kLayers, state.compact_state ? kQsaRawSlots : 0);
      if (expected != snapshot.qsa_layout) throw std::runtime_error("snapshot raw cache layout changed");
    }
    size_t total = 0;
    for (const auto& span : Spans(state, snapshot.hist.size(), snapshot.resume.mpos, stk, carry, mtp))
      total += span.bytes;
    if (mtp && !state.qc.contains(kLayers) && snapshot.resume.mpos > 0) total += MtpBytes(state, snapshot.resume.mpos);
    if (total != snapshot.bytes.size()) throw std::runtime_error("snapshot state layout changed");
    if (mtp && !state.qc.contains(kLayers)) e.EnsureMtpCache(state, 0);
    const auto spans = Spans(state, snapshot.hist.size(), snapshot.resume.mpos, stk, carry, mtp);
    state.Reset();
    CopySnapshotSpans(spans, snapshot.bytes, false);
    state.hist = snapshot.hist;
    state.retained = snapshot.retained;
    if (!snapshot.resume.logits.empty()) {
      const PinnedVector<float> logits(snapshot.resume.logits.begin(), snapshot.resume.logits.end());
      HIP_OK(hipMemcpy(e.logits_, logits.data(), sizeof(float) * kVocab, hipMemcpyHostToDevice));
    }
    // RunGen rebuilds the request's proposal head before restore. Another request's
    // observed suffix must not change proposal probabilities or mapping ownership.
    return snapshot.resume;
  }
};
