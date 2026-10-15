// The GDN core launch for a.R = 1..8 rows (decode, verify forwards and short
// prefill chunks). GdnRowsKernel, a multi-row variant behind
// the GDN rows switch, was bitwise equal but gave no measurable gain
// an earlier change deleted it (git history has it).
#pragma once

#include "ops.h"

namespace crucible {

// state_cache as LayerGroups.
inline void LaunchGdnCore(const GdnArgs& a, int state_cache, hipStream_t stream = 0) {
  if (state_cache == 6) GdnKernel<4, true><<<48, 256, 0, stream>>>(a);
  else if (state_cache == 4) GdnKernel<4><<<48, 256, 0, stream>>>(a);
  else GdnKernel<0><<<48, 256, 0, stream>>>(a);
}

}  // namespace crucible
