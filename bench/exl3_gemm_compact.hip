// Compile only compact expert inputs and skipped inactive row tiles.
#define EXL3_LEGACY_ONLY
#define EXL3_PACKED_ONLY
#define EXL3_COMPACT_ONLY
#include "exl3_gemm.hip"
