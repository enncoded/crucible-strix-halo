// Snapshot vectors use locked pages because the GPU must not map movable host pages.
#pragma once
#include <cstddef>
#include <limits>
#include <new>
#include <vector>

namespace crucible {
template<class T> struct PinnedAllocator {
  using value_type = T;
  PinnedAllocator() = default;
  template<class U> PinnedAllocator(const PinnedAllocator<U>&) noexcept {}
  T* allocate(size_t count) {
    if (count > std::numeric_limits<size_t>::max() / sizeof(T)) throw std::bad_array_new_length();
    T* data = nullptr;
    if (hipHostMalloc(&data, count * sizeof(T)) != 0) throw std::bad_alloc();
    return data;
  }
  void deallocate(T* data, size_t) noexcept { HIP_OK(hipHostFree(data)); }
  template<class U> bool operator==(const PinnedAllocator<U>&) const noexcept { return true; }
};
template<class T> using PinnedVector = std::vector<T, PinnedAllocator<T>>;
}  // namespace crucible
