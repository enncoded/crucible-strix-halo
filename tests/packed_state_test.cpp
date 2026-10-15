#include "../engine/layers/packed_state.h"
#include <cassert>
#include <type_traits>

struct StateArgs { const int* state; int position; int row; };
int main() {
  static_assert(std::is_trivially_copyable_v<crucible::PackedStateArgs<StateArgs, 8>>);
  const int state[] = {17, 23, 42, 65, 91, 103, 133, 199};
  const int streams[] = {7, 1, 5, 0, 6, 2, 4, 3};
  for (int count = 1; count <= 8; ++count) {
    crucible::PackedStateArgs<StateArgs, 8> args{};
    for (int row = 0; row < count; ++row)
      args.rows[row] = {state + streams[row], streams[row] * 11, row};
    const auto copy = args;
    args.rows[0] = {nullptr, -1, -1};
    for (int row = 0; row < count; ++row) {
      assert(copy.rows[row].state == state + streams[row]);
      assert(copy.rows[row].position == streams[row] * 11);
      assert(copy.rows[row].row == row);
    }
    for (int row = count; row < 8; ++row) assert(copy.rows[row].state == nullptr);
  }
}
