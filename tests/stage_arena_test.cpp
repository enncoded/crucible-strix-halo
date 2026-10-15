#include "../engine/runtime/stage_arena.h"
#include <cassert>
#include <vector>

int main() {
  using namespace crucible;
  // Takes are 256-byte aligned and counted without a base.
  const auto carve = [](StageView& v) { v.Take<float>(10); v.Take<char>(1); v.Take<double>(64); };
  assert(StageBytes(carve) == 256 + 256 + 512);
  StageView none;
  assert(none.Take<int>(0) == nullptr && none.used == 0);
  // A real view returns pointers inside the arena, in order and without overlap.
  std::vector<char> arena(2048);
  StageView view{arena.data(), arena.size()};
  float* a = view.Take<float>(10);
  char* b = view.Take<char>(1);
  double* c = view.Take<double>(64);
  assert(reinterpret_cast<char*>(a) == arena.data());
  assert(b == arena.data() + 256);
  assert(reinterpret_cast<char*>(c) == arena.data() + 512 && view.used == 1024);
  // Shared views both start at 0; separate views follow each other.
  const auto shared = StageArenaLayout(1000, 3000, true);
  assert(shared.bytes == 3000 && shared.second_offset == 0);
  const auto separate = StageArenaLayout(1000, 3000, false);
  assert(separate.bytes == 1024 + 3000 && separate.second_offset == 1024);
  assert(StageArenaLayout(5000, 0, true).bytes == 5000);
  return 0;
}
