// a development branch T2: with the padded bulk buffer, the tight stack buffer H_ holds only one packed verify pass.
#include "../engine/runtime/prefill_rows.h"
#include <cassert>

int main() {
  static_assert(TightStackRows(8192, true, 16) == 16);   // 8,192-row launches: 335 MB less
  static_assert(TightStackRows(4096, true, 16) == 16);
  static_assert(TightStackRows(8192, false, 16) == 8192);  // CRUCIBLE_PF_PAD_H=0: bulk launches run in H_
  static_assert(TightStackRows(16, true, 16) == 16 && TightStackRows(16, false, 16) == 16);  // no prefill
  assert(TightStackRows(8, true, 16) == 8);
  return 0;
}
