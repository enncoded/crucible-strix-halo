// The GDN lanes path before the v conv moved into the scan: the bench reference of
// bench/gdn_vscan.hip and the PREFILL_LANES mode of bench/prefill_gdn.hip. The engine runs
// LaunchGdnLanes (vscan) since train 20; same values bit for bit.
#pragma once
#include "../../engine/layers/kernels/prefill_gdn.h"

namespace crucible::prefill {
inline void LaunchGdnLanesRef(GdnPrefill a, hipStream_t stream = nullptr) {
  ValidateGdnPrefill(a);
  GdnLanesPrepKernel<<<dim3(16+24,a.chunks*64),256,0,stream>>>(a);
  GdnScanLanesKernel<true><<<48*4,128,0,stream>>>(a);
  GdnLanesGateKernel<<<(a.input.R*48+7)/8,256,0,stream>>>(a);
  GdnPrefillHistoryKernel<<<40,256,0,stream>>>(a);
}
}  // namespace crucible::prefill
