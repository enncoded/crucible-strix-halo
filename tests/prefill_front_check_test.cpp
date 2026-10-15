// CPU-only check: the MoE prefill GEMMs must reject front-order weights.
#include "../engine/runtime/prefill_front_check.h"
#include <cassert>
#include <string>

using namespace crucible;
int main() {
  assert(PrefillMoeFrontIssue({}).empty());
  assert(PrefillMoeFrontIssue({{"L0.sg", 0}, {"L0.su", 0}, {"L0.sd", 0}}).empty());
  assert(PrefillMoeFrontIssue({{"L0.sg", 0}, {"L3.sd", 6}}) == " L3.sd(front 6)");
  assert(PrefillMoeFrontIssue({{"L1.sg", 1}, {"L1.su", 2}}) == " L1.sg(front 1) L1.su(front 2)");
  // Negative splits are invalid layouts, so they are rejected as well.
  assert(PrefillMoeFrontIssue({{"L2.su", -1}}) == " L2.su(front -1)");
  return 0;
}
