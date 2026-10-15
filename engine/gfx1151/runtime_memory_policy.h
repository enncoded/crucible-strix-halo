#pragma once
#include <cstdlib>

namespace crucible {
inline bool SetRuntimeMemoryDefaults() {
  // Migrating an SVM range stops every KFD queue, including decode queues.
  if (setenv("HSA_USERPTR_FOR_PAGED_MEM", "0", 0) != 0 ||
      setenv("HSA_USE_SVM", "0", 0) != 0) return false;
  // CLR measures this threshold in MiB. Stage pageable uploads, rather than
  // pin movable pages. Keep explicit user settings for controlled probes.
  return setenv("GPU_PINNED_MIN_XFER_SIZE", "65536", 0) == 0;
}
}  // namespace crucible
