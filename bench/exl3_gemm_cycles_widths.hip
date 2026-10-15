// Matched decoder ratios for the3.05 and4.05 ceiling estimates.
#define main OriginalCycleProbeMain
#include "exl3_gemm_cycles.hip"
#undef main
int main() {
  uint32_t* out;HIP_CHECK(hipMalloc(&out,160*128*4));
  std::puts("# Matched K3/K4/K5/K6. 80 SIMD32s. No cold weights or activation loads.");
  Point("LDS warmup",DecodeStage<3,true>,out);
  Point("decode K3 LDS",DecodeStage<3,true>,out);
  Point("decode K4 LDS",DecodeStage<4,true>,out);
  Point("decode K5 LDS",DecodeStage<5,true>,out);
  Point("decode K6 LDS",DecodeStage<6,true>,out);
  Point("decode K3 registers",DecodeStage<3,false>,out);
  Point("decode K4 registers",DecodeStage<4,false>,out);
  Point("decode K5 registers",DecodeStage<5,false>,out);
  Point("decode K6 registers",DecodeStage<6,false>,out);
  HIP_CHECK(hipFree(out));
}
