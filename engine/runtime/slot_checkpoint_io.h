// Included after Engine, as conversation_snapshot.h. Device copies for in-slot checkpoints (slot_checkpoints.h).
//
// After position p a stream changes only its fixed-size state and, per QSA layer, the compact raw ring and the pool
// block that holds p (rows up to p - 1 can share a four-row pool group with later rows). Every other row below p is
// written once. So these spans at p, with the rows that the slot keeps, are the bytes that a full snapshot at p
// restores: ConversationStateIO::Spans writes the same raw ring and the same pool floats up to the block of p.
#pragma once
#include "conversation_snapshot.h"
#include "slot_checkpoints.h"

struct SlotCheckpointIO {
  static constexpr size_t kPoolBlockFloats = 4096;  // QsaPoolFloats: 4096 floats per 128 positions
  // The trunk's QSA layers only: a tail and a checkpoint hold target state; a speculative request warms its MTP rows
  // again after either restore.
  static std::vector<DeviceSpan> Spans(const SequenceState& state, size_t position) {
    std::vector<DeviceSpan> spans = state.FixedSpans();
    for (const auto& [layer, cache] : state.qc) {
      if (layer == kLayers) continue;
      if (cache.raw_slots) spans.push_back({cache.ik_raw, size_t(cache.raw_slots) * kQsaRawDim * sizeof(float)});
      if (position % 128)
        spans.push_back({cache.ik_pool + position / 128 * kPoolBlockFloats, kPoolBlockFloats * sizeof(float)});
    }
    return spans;
  }
  // The largest checkpoint of a stream (a position inside a pool block).
  static size_t Bytes(const SequenceState& state) {
    size_t bytes = 0;
    for (const auto& span : Spans(state, 1)) bytes += span.bytes;
    return bytes;
  }
  // One device buffer per checkpoint. Copies run on the null stream, in order with the forwards around them.
  struct Buffer {
    uint8_t* device = nullptr;
    size_t bytes = 0, position = 0;  // the spans' layout depends on the position
    crucible::RetainedPrefix retained;
    explicit Buffer(size_t size) : bytes(size) {
      HIP_OK(hipMalloc(&device, size));
#if CRUCIBLE_DEBUG_NONFINITE & 2
      HIP_OK(hipMemset(device, 0xFF, size));  // a restore must read only bytes that Save wrote
#endif
    }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    ~Buffer() { if (device) HIP_OK(hipFree(device)); }
  };
  static void Save(const SequenceState& state, Buffer& buffer) {
    size_t offset = 0;
    for (const auto& span : Spans(state, state.hist.size())) {
      if (offset + span.bytes > buffer.bytes) throw std::logic_error("slot checkpoint buffer is too small");
      HIP_OK(hipMemcpyAsync(buffer.device + offset, span.pointer, span.bytes, hipMemcpyDeviceToDevice, 0));
      offset += span.bytes;
    }
    buffer.retained = state.retained;
    buffer.position = state.hist.size();
  }
  // The stream must still hold every row below the checkpoint: the session drops a slot's checkpoints when a reset
  // or a host restore writes the slot (SlotCheckpoints::Keep).
  static void Restore(SequenceState& state, const Buffer& buffer, const crucible::SlotCheckpoint& point) {
    const size_t length = point.hist.size();
    if (buffer.position != length || state.hist.size() < length ||
        !std::equal(point.hist.begin(), point.hist.end(), state.hist.begin()))
      throw std::logic_error("slot checkpoint rows are no longer on the slot");
    size_t offset = 0;
    for (const auto& span : Spans(state, length)) {
      HIP_OK(hipMemcpyAsync(const_cast<void*>(span.pointer), buffer.device + offset, span.bytes,
                            hipMemcpyDeviceToDevice, 0));
      offset += span.bytes;
    }
    state.hist = point.hist;
    state.retained = buffer.retained;
  }
};
