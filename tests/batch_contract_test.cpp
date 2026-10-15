#include "../engine/runtime/batch_contract.h"

#include <cassert>
#include <vector>

int main() {
  using crucible::BatchError;
  const std::vector<size_t> lengths{0, 14, 23};
  assert(BatchError({1, 2}, {7, 8}, lengths, 32, 8, 100).empty());
  assert(BatchError({2, 0}, {99, 0}, lengths, 32, 8, 100).empty());
  assert(!BatchError({}, {}, lengths, 32, 8, 100).empty());
  assert(!BatchError({0}, {}, lengths, 32, 8, 100).empty());
  assert(!BatchError({0, 0}, {1, 2}, lengths, 32, 8, 100).empty());
  assert(!BatchError({-1}, {1}, lengths, 32, 8, 100).empty());
  assert(!BatchError({3}, {1}, lengths, 32, 8, 100).empty());
  assert(!BatchError({0}, {-1}, lengths, 32, 8, 100).empty());
  assert(!BatchError({0}, {100}, lengths, 32, 8, 100).empty());
  assert(!BatchError({2}, {1}, lengths, 23, 8, 100).empty());
  assert(BatchError({2}, {1}, lengths, 24, 8, 100).empty());
  assert(!BatchError({0, 1, 2}, {1, 2, 3}, lengths, 32, 2, 100).empty());
  assert(!BatchError({2}, {1}, lengths, std::vector<int>{32, 32, 23}, 8, 100).empty());
  assert(BatchError({2}, {1}, lengths, std::vector<int>{1, 15, 24}, 8, 100).empty());
  assert(!BatchError({0}, {1}, lengths, std::vector<int>{0, 15, 24}, 8, 100).empty());
  assert(!BatchError({0}, {1}, lengths, std::vector<int>{32}, 8, 100).empty());
}
