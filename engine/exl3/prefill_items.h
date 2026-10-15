// Input-transform item arithmetic shared by the prefill kernels and the CPU tests.
#pragma once

#ifdef __HIPCC__
#define PREFILL_ITEMS_HD __host__ __device__
#else
#define PREFILL_ITEMS_HD
#endif

namespace crucible {
// Item w of the list that PrefillLinear::Build wrote for jobs of R rows and I
// columns each: jobs in order, rows in order, then 128-column blocks.
struct PrefillInIndex { int job, r, blk; };
PREFILL_ITEMS_HD inline PrefillInIndex PrefillInItemAt(int R, int I, int w) {
  const int blocks = I / 128, per = R * blocks;
  const int j = w / per, rem = w - j * per;
  return {j, rem / blocks, rem % blocks};
}

// Fused output-kernel classes of a packed prefill group of n jobs with EXL3
// widths K. The fused kernels take K3..5 or K6 jobs, not both (a K6 branch
// next to K3..5 makes that code spill), so a mixed group (the 4.05 QSA input:
// q, k, v K6, the indexer K4) runs as two fused launches (split).
// fusable: packed and every O % 128 == 0.
struct PrefillFusedClasses { bool fused, k6, split; };
PREFILL_ITEMS_HD inline PrefillFusedClasses PrefillFusedClassesOf(const int* K, int n, bool fusable) {
  int six = 0;
  for (int i = 0; i < n; ++i) six += K[i] == 6;
  if (!fusable) return {false, false, false};
  return {true, six == n, six > 0 && six < n};
}
}  // namespace crucible
