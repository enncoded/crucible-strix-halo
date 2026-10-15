// Bulk prefill launches that merge grid chunks (prefill-moe lesson 1).
#pragma once
#include <algorithm>
#include <cerrno>
#include <cstddef>
#include <cstdlib>

namespace crucible {
// Most rows in one bulk prefill launch. Every layer's GDN and QSA kernels still
// run in kPrefillCoreRows parts; the GEMMs (MoE experts, dense projections)
// see all rows of the launch, so each decoded expert weight serves more rows.
constexpr int kMaxPrefillLaunchRows = 32768;
// Rows of one GDN or QSA core part inside a launch (engine.hip kPrefillCoreRows). A merge needs a grid of whole
// parts: the core scratch holds min(grid, parts) rows, and a launch cuts its parts from its own start.
constexpr int kPrefillLaunchCoreRows = 2048;
// Rows of a launch when CRUCIBLE_PF_LAUNCH_ROWS is unset: two chunks of the 4,096-row grid. Bitwise equal to one
// chunk a launch (1K to 128K prompts), and 55 ms (32K) to 78 ms (128K) less wall per 4,096 rows for 1,677 MiB of
// stage arena. CRUCIBLE_PF_LAUNCH_ROWS=4096 turns it off. Only the 4,096-row grid merges by default: no other grid was
// measured (a 2048-row grid keeps one chunk a launch unless the variable merges it).
constexpr int kDefaultPrefillLaunchRows = 8192;
constexpr int kDefaultPrefillLaunchGrid = 4096;

// Rows of the bulk launch at prompt position `at` (prompt end `end`). The grid
// schedule takes min(step, end - at) rows a chunk. A launch merges whole grid
// chunks while it stays within `launch` rows and its end stays at or before
// `stop` (the first position that must end a chunk: a snapshot frontier or
// the MTP warm-up start). A partial chunk is always the last one of a
// launch, so a launch ends only on grid chunk ends and the GDN and QSA core
// cuts stay where the grid schedule puts them.
inline int PrefillLaunchRows(size_t at, size_t end, int step, int launch, size_t stop) {
  int rows = int(std::min<size_t>(size_t(step), end - at));
  while (rows % step == 0 && at + size_t(rows) < end) {
    const int next = int(std::min<size_t>(size_t(step), end - at - size_t(rows)));
    if (rows + next > launch || at + size_t(rows + next) > stop) break;
    rows += next;
  }
  return rows;
}

// An engine's launch rows; engines without merged launches take one grid chunk.
template <class E> int EngineLaunchRows(const E& engine) {
  if constexpr (requires { engine.PrefillLaunchRows(); }) return engine.PrefillLaunchRows();
  else return engine.PrefillRows();
}

// The launch rows of CRUCIBLE_PF_LAUNCH_ROWS (`value`, nullptr when unset) for
// grid chunks of `grid` rows: unset gives kDefaultPrefillLaunchRows for the
// grid kDefaultPrefillLaunchGrid and the grid itself for every other grid; a
// set value must be a multiple of the grid in [grid, kMaxPrefillLaunchRows],
// and above the grid only for a grid of whole core parts. Returns -1 for an
// invalid value, and 0 without bulk prefill (grid 0).
inline int PrefillLaunchSetting(const char* value, int grid) {
  if (grid <= 0) return 0;
  if (!value) return grid == kDefaultPrefillLaunchGrid ? kDefaultPrefillLaunchRows : grid;
  errno = 0;
  char* tail = nullptr;
  const long rows = std::strtol(value, &tail, 10);
  if (!*value || *tail || errno || rows < grid || rows > kMaxPrefillLaunchRows || rows % grid) return -1;
  if (rows > grid && grid % kPrefillLaunchCoreRows) return -1;
  return int(rows);
}
}  // namespace crucible
