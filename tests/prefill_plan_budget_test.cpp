#include "../engine/runtime/prefill_plan_budget.h"
#include <cassert>
#include <limits>

int main() {
  using namespace crucible;
  assert(!PrefillPlanBudgetExceeded(0, 0, 651ULL << 20));
  assert(!PrefillPlanBudgetExceeded(1, 651ULL << 20, 160ULL << 20));
  assert(PrefillPlanBudgetExceeded(1, 651ULL << 20, 651ULL << 20));
  assert(!PrefillPlanBudgetExceeded(15, kPrefillPlanBytes - 16, 16));
  assert(PrefillPlanBudgetExceeded(16, 0, 16));
  assert(PrefillPlanBudgetExceeded(1, kPrefillPlanBytes, 1));
  assert(!PrefillPlanBudgetExceeded(0, 0, kPrefillPlanBytes + 1));
  assert(!PrefillPlanBudgetExceeded(0, 0, std::numeric_limits<size_t>::max()));
  assert(PrefillPlanBudgetExceeded(1, 16, kPrefillPlanBytes + 1));
  assert(PrefillPlanBudgetExceeded(1, 16, std::numeric_limits<size_t>::max()));
  assert(PrefillPlanBudgetExceeded(1, kPrefillPlanBytes + 1, 16));
  assert(!PrefillPlanBudgetExceeded(0, 0, kPrefillPlanBytes));
}
