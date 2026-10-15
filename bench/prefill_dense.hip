// Dense fp16 prefill GEMM y[R][O] = x[R][2560] W[O][2560]^T: Big versus Fast
// kernels, bitwise check and time. Usage: prefill_dense R
#include "prefill_common.h"
#include "../engine/layers/prefill_hc.h"
using namespace crucible;
using namespace prefill_bench;

int main(int argc,char** argv) {
  const int R=argc>1 ? std::atoi(argv[1]) : 4096;
  if (R<1 || R>8192) return 2;
  Buffer<__half> x(Half(Random((size_t)R*kHidden,.5f,7)));
  int bad=0;
  for (int O : {96, 512, 513, 12800}) {
    Buffer<__half> W(Half(Random((size_t)O*kHidden,.02f,331+O)));
    Buffer<float> y1((size_t)R*O),y2((size_t)R*O),y3((size_t)R*O);
    const dim3 g((R+127)/128,(O+127)/128);
    auto big=[&] { PrefillDenseBigKernel<<<g,256>>>(W.p,O,x.p,y1.p,O,R); };
    auto f2=[&] { PrefillDenseFastKernel<2><<<g,256>>>(W.p,O,x.p,y2.p,O,R); };
    auto f4=[&] { PrefillDenseFastKernel<4><<<g,256>>>(W.p,O,x.p,y3.p,O,R); };
    big(); f2(); f4(); CheckLaunch();
    const bool s2=y2.Get()==y1.Get(), s4=y3.Get()==y1.Get();
    bad+=!s2+!s4;
    const double gf=2.0*R*O*kHidden*1e-6;
    const double tb=Time(big),t2=Time(f2),t4=Time(f4);
    std::printf("R=%d O=%d big %.1f us (%.1f TF) fast2 %.1f us (%.1f TF) bitwise=%d fast4 %.1f us (%.1f TF) bitwise=%d\n",
      R,O,tb,gf/tb,t2,gf/t2,(int)s2,t4,gf/t4,(int)s4);
  }
  return bad ? 9 : 0;
}
