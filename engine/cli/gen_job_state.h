// gen --jobs runs several captures after one model load. Before each job, every
// piece of per-request state must equal its value after construction; a field
// that a reset leaves dirty stops the process.
#pragma once
#include <cstddef>
#include <string>
#include <vector>

namespace crucible {
struct GenJobState {
  size_t history = 0;               // token history of the active sequence
  bool retained_clean = true;       // RetainedPrefix counters
  size_t sequences = 1;             // gen uses one sequence; batch modes add more
  bool active_first = true;         // the active sequence is sequence 0
  size_t batch_streams = 0;
  bool verification_spans = false;  // a packed verification pass is open
  bool sequence_zero = true;        // KV, QSA index, GDN state, pending rows, PLE ring, MTP cache
  bool scratch_zero = true;         // gen trunk stacks and carry rows
  bool draft_selection = false;     // a restricted or dense draft head is loaded
  bool draft_last_only = true;
  bool draft_margin_reset = true;
  bool overlap_head = false;        // the asynchronous verification head exists
  bool profile = false;
  size_t profile_entries = 0, profile_marks = 0;
  bool expert_counts_zero = true;
  bool expert_counting = false;
};

// Names every field that differs from a fresh engine, or returns an empty list.
inline std::vector<std::string> GenJobStateIssues(const GenJobState& s) {
  std::vector<std::string> issues;
  const auto check = [&](bool clean, const char* name) { if (!clean) issues.emplace_back(name); };
  check(s.history == 0, "token history");
  check(s.retained_clean, "retained prefix");
  check(s.sequences == 1 && s.active_first, "sequence set");
  check(s.batch_streams == 0, "batch streams");
  check(!s.verification_spans, "verification spans");
  check(s.sequence_zero, "sequence device state");
  check(s.scratch_zero, "gen stack scratch");
  check(!s.draft_selection && s.draft_last_only, "draft head selection");
  check(s.draft_margin_reset, "draft margin");
  check(!s.overlap_head, "overlap head");
  check(!s.profile && s.profile_entries == 0 && s.profile_marks == 0, "profile counters");
  check(s.expert_counts_zero && !s.expert_counting, "expert counters");
  return issues;
}

inline std::string JoinIssues(const std::vector<std::string>& issues) {
  std::string text;
  for (const auto& issue : issues) text += (text.empty() ? "" : ", ") + issue;
  return text;
}
}  // namespace crucible
