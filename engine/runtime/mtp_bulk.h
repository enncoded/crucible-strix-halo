// Bulk MTP warm-up: the MTP cache rows of a prompt chunk in one pass of the prefill
// kernels instead of 8-row MTP calls. The MTP only proposes drafts, so its rows may differ in bits from the 8-row
// path; outputs do not change.
#pragma once
#include <cstddef>
#include <cstdlib>
#include <vector>

namespace crucible {
// Fewer pairs keep the 8-row path. The default window (256 rows) never reaches it, so default drafts do not change.
inline constexpr int kMtpBulkMinPairs = 1024;
inline bool MtpBulkFits(int pairs, int launch_rows) {
  return pairs >= kMtpBulkMinPairs && pairs <= launch_rows;
}
// The warm-up pairs of one prompt chunk [at, at + rows) from row `start_row` on. Pair j is the trunk stack of prompt
// row at + start_row + j - carry (the carry is the stack of the row before the chunk) with the next prompt token.
struct MtpChunkPairs {
  int count = 0;       // pairs
  int first_row = 0;   // the chunk row of the first stack that is not the carry
  size_t token = 0;    // the prompt index of pair 0's token
};
inline MtpChunkPairs MtpPairsOf(size_t at, int rows, int start_row, bool have_carry) {
  if (start_row >= rows) return {};
  return {rows - start_row - 1 + int(have_carry), start_row, at + size_t(start_row) + (have_carry ? 0 : 1)};
}
// Groups of linear jobs that fit the shared input (fp16) and partial (fp32) workspaces, in order. A job that does not
// fit alone returns an empty plan (the caller keeps the 8-row path).
template <class Job, class Fits>
std::vector<std::vector<Job>> PackJobs(const std::vector<Job>& jobs, size_t input_cap, size_t partial_cap, Fits same) {
  std::vector<std::vector<Job>> groups;
  size_t input = 0, partial = 0;
  for (const Job& job : jobs) {
    const size_t in = size_t(job.R) * size_t(job.I), out = size_t(job.R) * size_t(job.O);
    if (in > input_cap || out > partial_cap) return {};
    if (groups.empty() || input + in > input_cap || partial + out > partial_cap || !same(groups.back().front(), job)) {
      groups.emplace_back();
      input = partial = 0;
    }
    groups.back().push_back(job);
    input += in;
    partial += out;
  }
  return groups;
}
}  // namespace crucible
