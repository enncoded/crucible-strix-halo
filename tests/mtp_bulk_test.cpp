// Bulk MTP warm-up bookkeeping (engine/runtime/mtp_bulk.h): the pairs of a chunk equal those of the 8-row loop in RunPrompt,
// and the job groups fit the shared workspaces.
#include "../engine/runtime/mtp_bulk.h"
#include <cassert>
#include <cstdio>
#include <vector>
using namespace crucible;

// The 8-row loop of RunPrompt (engine/runtime/prompt.h): the stack and token of each pair, -1 for the carry.
static std::vector<std::pair<long, size_t>> Loop(size_t at, int rows, int start_row, bool have_carry) {
  std::vector<std::pair<long, size_t>> pairs;
  if (start_row >= rows) return pairs;
  const int warm = rows - start_row, count = warm - 1 + int(have_carry);
  for (int p = 0; p < count; p += 8)
    for (int j = 0; j < std::min(8, count - p); ++j) {
      const long stack = p + j == 0 && have_carry ? -1 : long(at) + start_row + p + j - int(have_carry);
      pairs.push_back({stack, at + size_t(start_row + p + j + (have_carry ? 0 : 1))});
    }
  return pairs;
}
static std::vector<std::pair<long, size_t>> Bulk(size_t at, int rows, int start_row, bool have_carry) {
  const auto c = MtpPairsOf(at, rows, start_row, have_carry);
  std::vector<std::pair<long, size_t>> pairs;
  for (int j = 0; j < c.count; ++j) {
    const long stack = j == 0 && have_carry ? -1 : long(at) + c.first_row + j - int(have_carry);
    pairs.push_back({stack, c.token + size_t(j)});
  }
  return pairs;
}

struct Job { int R, I, O, front; };

int main() {
  for (const size_t at : {size_t(0), size_t(4096), size_t(196608)})
    for (const int rows : {1, 9, 1024, 4096})
      for (const int start : {0, 3, 1000, 4095, 5000})
        for (const bool carry : {false, true}) {
          if (at == 0 && carry) continue;
          assert(Bulk(at, rows, start, carry) == Loop(at, rows, start, carry));
        }
  assert(MtpPairsOf(0, 4096, 0, false).count == 4095);
  assert(MtpPairsOf(4096, 4096, 0, true).count == 4096);
  assert(MtpPairsOf(4096, 4096, 4096, true).count == 0);
  // The default window never takes the bulk path; a full chunk does, up to the launch rows.
  assert(!MtpBulkFits(256, 4096) && !MtpBulkFits(1023, 4096));
  assert(MtpBulkFits(1024, 4096) && MtpBulkFits(4096, 4096));
  assert(!MtpBulkFits(4097, 4096));
  // fc_embedding and four fc_hidden streams at 4096 rows: 52M fp16 inputs do not fit a 42M workspace in one group.
  const std::vector<Job> jobs{{4096, 2560, 2560, 0}, {4096, 2560, 2560, 0}, {4096, 2560, 2560, 0},
                              {4096, 2560, 2560, 0}, {4096, 2560, 2560, 0}};
  const auto same = [](const Job& a, const Job& b) { return a.front == b.front; };
  const auto groups = PackJobs(jobs, size_t(4096) * 10240, size_t(4096) * 16384, same);
  assert(groups.size() == 2 && groups[0].size() == 4 && groups[1].size() == 1);
  assert(PackJobs(jobs, size_t(2048) * 2560, size_t(1) << 40, same).empty());
  const std::vector<Job> mixed{{64, 128, 128, 0}, {64, 128, 128, 2}, {64, 128, 128, 2}};
  assert(PackJobs(mixed, 1 << 20, 1 << 20, same).size() == 2);
  std::puts("mtp_bulk_test: ok");
  return 0;
}
