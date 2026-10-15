// Calibrate the VALU counter with the same WMMA loop at N and2N iterations.
#define main WmmaRateMain
#include "wmma_rate.hip"
#undef main

int main() {
  float* out=nullptr;
  const auto allocated=hipMalloc(&out,1<<20);
  if(allocated!=hipSuccess) {
    std::fprintf(stderr,"WMMA calibration allocation failed: %s\n",hipGetErrorString(allocated));return 1;
  }
  for(int step=0;step<11;++step) {
    const int iterations=step<3?4096:(step&1?4096:8192);
    Kern<2,4,0><<<80,256>>>(out,iterations);
    const auto complete=hipDeviceSynchronize();
    if(complete!=hipSuccess) {
      std::fprintf(stderr,"WMMA calibration failed: %s\n",hipGetErrorString(complete));
      (void)hipFree(out);return 1;
    }
    std::printf("step=%d iterations=%d blocks=80 threads=256 wmmas_per_wave=%d\n",step+1,iterations,iterations*8);
  }
  const auto freed=hipFree(out);
  if(freed!=hipSuccess) {
    std::fprintf(stderr,"WMMA calibration cleanup failed: %s\n",hipGetErrorString(freed));return 1;
  }
}
