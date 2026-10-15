#include <cassert>
#include <stdexcept>
#include <limits>
#include "../engine/runtime/draft_head_policy.h"

using crucible::DraftHeadPolicy;

void Step(DraftHeadPolicy& policy, double restricted_ms, double full_ms) {
  policy.Observe(2, policy.UseFull() ? full_ms : restricted_ms);
}

int main() {
  DraftHeadPolicy slower_full;
  for (int i = 0; i < 8; ++i) Step(slower_full, 40, 50);
  assert(!slower_full.UseFull());
  assert(slower_full.Steps(false) == 4 && slower_full.Steps(true) == 4);

  DraftHeadPolicy faster_full;
  for (int i = 0; i < 8; ++i) Step(faster_full, 50, 40);
  assert(faster_full.UseFull());
  for (int i = 0; i < 24; ++i) Step(faster_full, 50, 40);
  assert(!faster_full.UseFull());  // probe the other head after an exploit window
  for (int i = 0; i < 4; ++i) Step(faster_full, 50, 40);
  assert(faster_full.UseFull());

  DraftHeadPolicy almost_equal;
  for (int i = 0; i < 8; ++i) Step(almost_equal, 40.5, 40);
  assert(!almost_equal.UseFull());  // hysteresis retains the incumbent

  DraftHeadPolicy changing;
  for (int i = 0; i < 8; ++i) Step(changing, 40, 50);
  for (int i = 0; i < 84; ++i) Step(changing, 60, 30);
  assert(changing.IncumbentFull());

  for (double invalid : {0.0, -1.0, std::numeric_limits<double>::infinity(),
                         std::numeric_limits<double>::quiet_NaN()}) {
    bool caught = false;
    try { changing.Observe(1, invalid); } catch (const std::invalid_argument&) { caught = true; }
    assert(caught);
  }
  bool caught = false;
  try { changing.Observe(0, 40); } catch (const std::invalid_argument&) { caught = true; }
  assert(caught);
  assert(changing.Steps(false) + changing.Steps(true) == 92);
}
