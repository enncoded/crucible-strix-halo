// Reuse the engine-shaped fixture and its independent codec/Mid oracle.
#define main KnownMidMain
#include "prefill_moe_mid.hip"
#undef main
#include "kernels/exl3_gemm_loader.h"

template<int STRIDE> bool LoaderCase(Fixture& f,bool quality,const std::vector<float>& g,
    const std::vector<float>& u,const gm::ColdCache& cold,double base) {
  int resident=0;hipDeviceProp_t prop{};HIP_OK(hipGetDeviceProperties(&prop,0));
  if(quality)HIP_OK(hipOccupancyMaxActiveBlocksPerMultiprocessor(&resident,ProbeMoeLoaderKernel<6,4,6,STRIDE>,320,0));
  else HIP_OK(hipOccupancyMaxActiveBlocksPerMultiprocessor(&resident,ProbeMoeLoaderKernel<6,3,5,STRIDE>,320,0));
  std::printf("loader stride=%d resident-CTAs/WGP=%d total-occupancy=%.1f%% compute-occupancy=%.1f%%\n",STRIDE,resident,100.*resident*320/prop.maxThreadsPerMultiProcessor,100.*resident*256/prop.maxThreadsPerMultiProcessor);
  auto run=[&]{if(quality)ProbeMoeLoaderKernel<6,4,6,STRIDE><<<256,320>>>(f.a);
               else ProbeMoeLoaderKernel<6,3,5,STRIDE><<<256,320>>>(f.a);};
  f.pg.Zero();f.pu.Zero();run();CheckLaunch();
  const auto cg=f.pg.Get(),cu=f.pu.Get();
  const bool same=std::memcmp(g.data(),cg.data(),g.size()*4)==0&&std::memcmp(u.data(),cu.data(),u.size()*4)==0;
  std::printf("loader stride=%d complete-G/U-bitwise=%d\n",STRIDE,(int)same);
  if(!same)return false;
  f.Mid();CheckLaunch();if(!Oracle(f))return false;
  const double ms=ColdTime(run,cold,"loader-body");
  const double flops=4.*f.R*kSlots*kHidden*kMoeInter;
  std::printf("pack=%d R=%d stride=%d gemm=%.4fms baseline=%.4fms saving/4K/48layers=%.3fms %.3fTFLOPS %.1f%%-49TF\n",
      quality?405:305,f.R,STRIDE,ms,base,(base-ms)*48,flops/ms*1e-9,flops/ms*1e-9/49*100);
  return true;
}
int main(int argc,char** argv) {
  if(argc>3)return 2;
  char* end=nullptr;const long R=argc>1?std::strtol(argv[1],&end,10):4096;
  if(R<16||R>4096||(argc>1&&(!*argv[1]||*end)))return 2;
  if(argc>2&&std::strcmp(argv[2],"305")&&std::strcmp(argv[2],"405"))return 2;
  const bool quality=argc>2&&std::strcmp(argv[2],"405")==0;
  Fixture f((int)R,quality);f.Plan();f.Input();f.Gemm();f.Mid();CheckLaunch();
  if(!Oracle(f))return 1;
  const auto cold=gm::MakeColdCache();const double base=ColdTime([&]{f.Gemm();},cold,"baseline-body");
  const auto g=f.pg.Get(),u=f.pu.Get();
  const bool ok=LoaderCase<8>(f,quality,g,u,cold,base)&&LoaderCase<12>(f,quality,g,u,cold,base);
  cold.Free();return ok?0:1;
}
