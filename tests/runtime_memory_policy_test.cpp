#include "../engine/gfx1151/runtime_memory_policy.h"
#include <cassert>
#include <cstring>
#include <initializer_list>

int main() {
  for (const char* name : {"HSA_USERPTR_FOR_PAGED_MEM", "HSA_USE_SVM", "GPU_PINNED_MIN_XFER_SIZE"})
    assert(unsetenv(name) == 0);
  assert(crucible::SetRuntimeMemoryDefaults());
  assert(std::strcmp(std::getenv("HSA_USERPTR_FOR_PAGED_MEM"), "0") == 0);
  assert(std::strcmp(std::getenv("HSA_USE_SVM"), "0") == 0);
  assert(std::strcmp(std::getenv("GPU_PINNED_MIN_XFER_SIZE"), "65536") == 0);
  assert(setenv("HSA_USE_SVM", "1", 1) == 0);
  assert(setenv("GPU_PINNED_MIN_XFER_SIZE", "1", 1) == 0);
  assert(setenv("HSA_USERPTR_FOR_PAGED_MEM", "1", 1) == 0);
  assert(crucible::SetRuntimeMemoryDefaults());
  assert(std::strcmp(std::getenv("HSA_USE_SVM"), "1") == 0);
  assert(std::strcmp(std::getenv("GPU_PINNED_MIN_XFER_SIZE"), "1") == 0);
  assert(std::strcmp(std::getenv("HSA_USERPTR_FOR_PAGED_MEM"), "1") == 0);
}
