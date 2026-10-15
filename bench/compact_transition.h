#pragma once
#include <algorithm>
#include <stdexcept>
#include <utility>

namespace crucible {
// The compared stream-one continuation must be the first restore into that stream.
// It performs the lazy MTP promotion; a second restore into a present cache would
// overwrite wrong promoted content and hide it.
template<class Engine, class Save, class Continue, class Serial, class HasMtp>
auto CompactAcrossStreams(Engine& engine, Save save, Continue continuation,
                          Serial serial, HasMtp has_mtp) {
  engine.ConfigureServeStreams(2);
  engine.SelectStream(0);
  const auto snapshot = save();
  const auto reference = continuation(snapshot);
  engine.SelectStream(1);
  serial();
  if (has_mtp()) throw std::runtime_error("promotion coverage requires an absent destination MTP cache");
  const auto actual = continuation(snapshot);
  if (!has_mtp()) throw std::runtime_error("snapshot restore did not promote the destination MTP cache");
  return std::make_pair(reference, actual);
}

// A snapshot saved right after a restore must hold the restored bytes, not only the same size.
template<class Snapshot>
void RequireSameSnapshot(const Snapshot& saved, const Snapshot& audit) {
  if (audit.hist != saved.hist || audit.qsa_layout != saved.qsa_layout ||
      audit.resume.length != saved.resume.length || audit.resume.mpos != saved.resume.mpos ||
      audit.bytes.size() != saved.bytes.size() || !(audit.bytes == saved.bytes))
    throw std::runtime_error("restored snapshot content differs");
}
}  // namespace crucible
