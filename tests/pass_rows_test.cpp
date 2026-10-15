// CPU checks of the packed verify pass capability: a pass holds kPackedRows
// rows only when every 16-row static group runs through the dense ring.
#include "../engine/runtime/pass_rows.h"
#include <cassert>
#include <iostream>

using namespace crucible;
int main() {
  static_assert(kPackedRows == 16 && kMaxRows == 8);
  // Front order or a chunk-aligned engine order through the ring; expert
  // groups (no destination) never run the dense ring.
  static_assert(PackedGroupFits({true, true, false}));
  static_assert(PackedGroupFits({true, false, true}));
  static_assert(!PackedGroupFits({true, false, false}));
  static_assert(!PackedGroupFits({false, true, true}));
  assert(MaxPassRows(true) == kPackedRows);
  assert(MaxPassRows(false) == kMaxRows);
  std::cout << "pass rows CPU checks PASS\n";
}
