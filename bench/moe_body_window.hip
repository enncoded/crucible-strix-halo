// Complete partial comparisons precede every timing. This is not gate evidence.
#if defined(CRUCIBLE_MOE_WINDOW_PROBE) && CRUCIBLE_MOE_WINDOW_PROBE
#error "The benchmark reference must use the original MoE body"
#endif
#ifndef CRUCIBLE_MOE_WINDOW_PROBE
#define CRUCIBLE_MOE_WINDOW_PROBE 0
#endif
#define main CodecOracleMain
#include "exl3_wide_codecs.hip"
#undef main
#include "exl3_moe_mid_cold.h"
#include "../engine/exl3/kernels/exl3_moe_window.h"
#include "kernels/exl3_moe_lut.h"
#include "moe_pair_stats.h"
#include "moe_partial_check.h"
#include "../engine/layers/prefill_moe.h"
#if CRUCIBLE_MOE_WINDOW_PROBE
#error "The benchmark reference must use the original MoE body"
#endif
#include <cstring>
#include <numeric>

__global__ void FillPartial(float* output,size_t cells,uint32_t poison) {
  const size_t index=(size_t)blockIdx.x*blockDim.x+threadIdx.x;
  if(index<cells)reinterpret_cast<uint32_t*>(output)[index]=poison;
}
void PoisonPartial(Buffer<float>& output,uint32_t poison) {
  FillPartial<<<(output.n+255)/256,256>>>(output.p,output.n,poison);CheckLaunch();
}
__global__ void MakeMoeLut(__half* lut) {
  const uint32_t state=blockIdx.x*blockDim.x+threadIdx.x;
  if(state<65536)lut[state]=__low2half(Mul1Pair(Mul1Product<kMulTrue16>(state),0));
}
template<int K,int RM,int WINDOW,bool HALF,bool SKIP,bool LUT> __global__ void WindowCheck(
    Exl3GemmJob j,const Exl3GemmItem* items,int n,const __half* lut) {
  __shared__ uint32_t ws[kGemmPairWords];
  if((int)blockIdx.x>=n)return;
  if constexpr(LUT)MoeLutBody<K,RM,HALF,SKIP>(j,items[blockIdx.x],ws,lut);
  else MoeWindowBody<K,RM,WINDOW,HALF,SKIP>(j,items[blockIdx.x],ws);
}
template<int K,int RM,int WINDOW,bool HALF,bool SKIP=true,bool LUT=false> bool CheckPoint(
    const std::vector<uint32_t>& weights,int M,const __half* lut=nullptr) {
  const auto hx=Half(Random((size_t)M*I,.1f,19));
  std::vector<__half> packed(hx.size());
  for(int r=0;r<M;++r)for(int i=0;i<I;++i)
    packed[((size_t)(i/16)*M+r)*16+i%16]=hx[(size_t)r*I+i];
  Buffer<uint32_t> w(weights);Buffer<__half> x(packed);
  Buffer<float> ref((size_t)M*O),got(ref.n);
  PoisonPartial(ref,crucible_bench::PartialPoison<false,false>());
  PoisonPartial(got,crucible_bench::PartialPoison<HALF,true>());
  std::vector<Exl3GemmItem> items;
  for(int m=0;m<M;m+=16*RM)items.push_back({0,0,m,0});
  Buffer<Exl3GemmItem> di(items);
  const Exl3GemmJob j{w.p,x.p,ref.p,nullptr,I,O,M,1,K,M};
  WideCheck<K,RM,2,false><<<items.size(),128>>>(j,di.p,items.size());CheckLaunch();
  auto candidate=j;candidate.partial=got.p;
  WindowCheck<K,RM,WINDOW,HALF,SKIP,LUT><<<items.size(),128>>>(candidate,di.p,items.size(),lut);CheckLaunch();
  const auto rh=ref.Get(),gh=got.Get();
  const std::vector<int> counts{M};
  bool same=crucible_bench::PartialCovered<false>(rh,counts,O,crucible_bench::PartialPoison<false,false>())&&
      crucible_bench::PartialCovered<HALF>(gh,counts,O,crucible_bench::PartialPoison<HALF,true>());
  if constexpr(HALF) {
    for(size_t i=0;i<rh.size();++i) {
      const float v=rh[i];
      const auto expected=__float2half(std::fabs(v)>65504.f?std::copysign(65504.f,v):v);
      __half observed;std::memcpy(&observed,reinterpret_cast<const unsigned char*>(gh.data())+i*2,2);
      same=same&&__half_as_ushort(observed)==__half_as_ushort(expected);
    }
  } else same=same&&std::memcmp(rh.data(),gh.data(),rh.size()*4)==0;
  std::printf("window-check K=%d M=%d RM=%d window=%d half=%d skip=%d lut=%d bitwise=%d\n",K,M,RM,WINDOW,HALF,SKIP,LUT,same);
  return same;
}
template<int K> int CheckWidth() {
  Buffer<__half> lut(65536);MakeMoeLut<<<256,256>>>(lut.p);CheckLaunch();
  std::mt19937 rng(31);std::vector<uint32_t> w((size_t)I*O*K/32);
  for(auto& v:w)v=rng();int bad=0;
  for(int M:{1,15,16,17,31,33,63,65,80,96,127,129,256}) {
#define CHECK(R,W,H) bad+=!CheckPoint<K,R,W,H>(w,M)
    CHECK(2,1,false);CHECK(4,1,false);CHECK(6,1,false);
    CHECK(6,2,false);CHECK(6,3,false);CHECK(6,1,true);CHECK(6,2,true);
#undef CHECK
#define LIVE_CHECK(R,W,H) bad+=!CheckPoint<K,R,W,H,false>(w,M)
    LIVE_CHECK(1,1,false);LIVE_CHECK(2,2,false);LIVE_CHECK(3,2,false);
    LIVE_CHECK(4,2,false);LIVE_CHECK(5,2,false);LIVE_CHECK(6,2,false);LIVE_CHECK(6,2,true);
#undef LIVE_CHECK
#define LUT_CHECK(R,H) bad+=!CheckPoint<K,R,1,H,true,true>(w,M,lut.p)
    LUT_CHECK(2,false);LUT_CHECK(4,false);LUT_CHECK(6,false);LUT_CHECK(6,true);
#undef LUT_CHECK
  }
  return bad;
}

template<int K,int WINDOW,bool HALF,int MODE> __device__ __forceinline__ void WindowDispatch(
    const Exl3GemmJob& j,const Exl3GemmItem& it,uint32_t* ws,const __half* lut) {
  if constexpr(MODE==4)MoeLutBody<K,6,HALF>(j,it,ws,lut);
  else if constexpr(MODE==0)MoeWindowBody<K,6,WINDOW,HALF>(j,it,ws);
  else if constexpr(MODE==3)MoeWindowBody<K,6,WINDOW,HALF,false>(j,it,ws);
  else if constexpr(MODE==2&&WINDOW==3)MoeLiveWindowBody<K,6,HALF>(j,it,ws);
  else {
    const int live=min(6,(j.M-it.m0+15)/16);
#define LIVE(R) case R: if constexpr(MODE==1) PrefillWideBody<K,R,2,true,false,true,HALF>(j,it,ws); \
    else if constexpr(MODE==5)MoeLutBody<K,R,HALF,false>(j,it,ws,lut); \
    else MoeWindowBody<K,R,(R<WINDOW?R:WINDOW),HALF,false>(j,it,ws); break
    switch(live){LIVE(1);LIVE(2);LIVE(3);LIVE(4);LIVE(5);LIVE(6);}
#undef LIVE
  }
}
template<int WINDOW,bool QUALITY,bool HALF,int MODE=0> __global__ void WindowQueue(
    const Exl3GemmJob* jobs,const Exl3GemmItem* items,const int* count,const __half* lut) {
  __shared__ uint32_t ws[kGemmPairWords];
  const int n=PrefillUniform(*count);
  for(int i=blockIdx.x;i<n;i+=gridDim.x) {
    const Exl3GemmItem it=PrefillUniform(items[i]);
    const Exl3GemmJob j=PrefillUniform(jobs[it.job]);
    if constexpr(QUALITY) {
      if(j.K==4)WindowDispatch<4,WINDOW,HALF,MODE>(j,it,ws,lut);
      else WindowDispatch<6,WINDOW,HALF,MODE>(j,it,ws,lut);
    } else {
      if(j.K==3)WindowDispatch<3,WINDOW,HALF,MODE>(j,it,ws,lut);
      else WindowDispatch<5,WINDOW,HALF,MODE>(j,it,ws,lut);
    }
    __syncthreads();
  }
}
template<class F> double Measure(F run,const gm::ColdCache& cold) {
  // Warm the current path, not the preceding variant, before cold samples.
  for(int i=0;i<20;++i){gm::Flush(&cold);run();CheckLaunch();}
  bench::EventTimer t;std::vector<double> ms;
  for(int i=0;i<7;++i){gm::Flush(&cold);t.Begin();run();ms.push_back(t.EndMs());}
  return bench::Median(ms);
}
template<class A,class B> crucible_bench::PairResult MeasurePair(A baseline,B candidate,
    const gm::ColdCache& cold) {
  bench::Sampler sampler(10);sampler.Start();
  const auto start=std::chrono::steady_clock::now();
  do {
    gm::Flush(&cold);baseline();CheckLaunch();
    gm::Flush(&cold);candidate();CheckLaunch();
  } while(std::chrono::steady_clock::now()-start<std::chrono::seconds(2));
  bench::EventTimer timer;
  auto sample=[&](auto run) {
    gm::Flush(&cold);timer.Begin();run();const double ms=timer.EndMs();CheckLaunch();return ms;
  };
  std::vector<crucible_bench::PairSample> samples;
  for(int round=0;round<8;++round) {
    const crucible_bench::PairSample ms={sample(baseline),sample(candidate),sample(candidate),sample(baseline)};
    samples.push_back(ms);
    std::printf("pair-sample round=%d ms=%.5f,%.5f,%.5f,%.5f\n",round,ms[0],ms[1],ms[2],ms[3]);
  }
  sampler.Stop();std::printf("pair-hardware %s\n",sampler.Summary().c_str());
  return crucible_bench::PairStats(samples);
}
template<bool QUALITY,bool HALF,bool PAIRED=false> bool Rates(bool down) {
  constexpr int R=4096,RM=6,IN=2560,INTER=640;
  Buffer<__half> lut(65536);MakeMoeLut<<<256,256>>>(lut.p);CheckLaunch();
  const int width=down?INTER:IN,out=down?IN:INTER,mats=down?1:2;
  const size_t total=(size_t)R*kSlots;
  std::mt19937 rng(7);std::vector<int> count(kExperts+1),expert(kExperts);
  std::iota(expert.begin(),expert.end(),0);
  for(int r=0;r<R;++r)for(int k=0;k<kTopK;++k) {
    std::swap(expert[k],expert[k+rng()%(kExperts-k)]);++count[expert[k]];
  }
  count[kExperts]=R;
  const size_t words=(size_t)IN*INTER*(QUALITY?4:3)/32;
  const size_t shared=(size_t)IN*INTER*(QUALITY?6:5)/32;
  std::vector<uint32_t> hw(words*kExperts*mats+shared*mats);
  for(auto& v:hw)v=rng();Buffer<uint32_t> weights(hw);
  Buffer<__half> x(Half(Random(total*width,.05f,11))),x2(Half(Random(total*width,.05f,13)));
  Buffer<float> y(total*out),y2(total*out);
  std::vector<Exl3GemmJob> jobs;std::vector<Exl3GemmItem> items;size_t base=0,wp=0;
  for(int e=0;e<=kExperts;++e) {
    const int K=e==kExperts?(QUALITY?6:5):(QUALITY?4:3),M=count[e];
    for(int m=0;m<mats;++m) {
      jobs.push_back({weights.p+wp,(m?x2.p:x.p)+base*width,(m?y2.p:y.p)+base*out,
          nullptr,width,out,M,1,K,M});wp+=(size_t)IN*INTER*K/32;
    }
    for(int row=0;row<M;row+=RM*16)for(int b=0;b<out/16;b+=8)
      for(int m=0;m<mats;++m)items.push_back({e*mats+m,0,row,b});
    base+=M;
  }
  Buffer<Exl3GemmJob> dj(jobs);Buffer<Exl3GemmItem> di(items);
  Buffer<int> n(std::vector<int>{(int)items.size()});
  auto reference=[&]{LaunchPrefillMoeGemm({RM,true},dj.p,di.p,n.p,QUALITY,HALF);};
  PoisonPartial(y,crucible_bench::PartialPoison<HALF,false>());
  PoisonPartial(y2,crucible_bench::PartialPoison<HALF,false>());
  reference();CheckLaunch();const auto ry=y.Get(),ry2=y2.Get();
  const bool covered=crucible_bench::PartialCovered<HALF>(ry,count,out,crucible_bench::PartialPoison<HALF,false>())&&
      (mats==1||crucible_bench::PartialCovered<HALF>(ry2,count,out,crucible_bench::PartialPoison<HALF,false>()));
  if(!covered){std::fprintf(stderr,"reference has an unwritten live cell or changed padding\n");return false;}
  const auto cold=gm::MakeColdCache();const double baseline=Measure(reference,cold);
  const double flops=2.*mats*total*width*out;
  std::printf("base pack=%d projection=%s half=%d ms=%.4f TFLOPS=%.3f roofline=%.1f%%\n",
      QUALITY?405:305,down?"down":"gateup",HALF,baseline,flops/baseline*1e-9,flops/baseline*1e-9/49*100);
  bool ok=true;
  auto variant=[&](auto wc,auto modec) {
    constexpr int W=decltype(wc)::value,MODE=decltype(modec)::value;
    auto run=[&]{WindowQueue<W,QUALITY,HALF,MODE><<<2048,128>>>(dj.p,di.p,n.p,lut.p);};
    PoisonPartial(y,crucible_bench::PartialPoison<HALF,true>());
    PoisonPartial(y2,crucible_bench::PartialPoison<HALF,true>());
    run();CheckLaunch();const auto cy=y.Get(),cy2=y2.Get();
    const bool same=crucible_bench::PartialMatches<HALF>(ry,cy,count,out)&&
        (mats==1||crucible_bench::PartialMatches<HALF>(ry2,cy2,count,out));
    ok=ok&&same;
    double ms=0;crucible_bench::PairResult pair{};
    if(same) {
      if constexpr(PAIRED) {
        std::printf("pair-begin pack=%d projection=%s window=%d mode=%d half=%d\n",
            QUALITY?405:305,down?"down":"gateup",W,MODE,HALF);
        pair=MeasurePair(reference,run,cold);ms=pair.candidate;
      } else ms=Measure(run,cold);
    }
    if constexpr(PAIRED)
      std::printf("paired pack=%d projection=%s window=%d mode=%d half=%d bitwise=%d baseline_ms=%.4f ms=%.4f delta_ms=%.4f delta_range=%.4f:%.4f saving48=%.3f TFLOPS=%.3f roofline=%.1f%%\n",
          QUALITY?405:305,down?"down":"gateup",W,MODE,HALF,same,pair.baseline,ms,pair.delta,
          pair.minimum_delta,pair.maximum_delta,pair.delta*48,ms?flops/ms*1e-9:0,ms?flops/ms*1e-9/49*100:0);
    else std::printf("window pack=%d projection=%s window=%d mode=%d half=%d bitwise=%d ms=%.4f saving48=%.3f TFLOPS=%.3f roofline=%.1f%%\n",
          QUALITY?405:305,down?"down":"gateup",W,MODE,HALF,same,ms,(baseline-ms)*48,
          ms?flops/ms*1e-9:0,ms?flops/ms*1e-9/49*100:0);
    std::fflush(stdout);
  };
#define VARIANT(W,M) variant(std::integral_constant<int,W>{},std::integral_constant<int,M>{})
  if constexpr(PAIRED) {
    VARIANT(2,1);VARIANT(3,2);VARIANT(2,4);VARIANT(2,5);
  } else {
    VARIANT(1,0);VARIANT(2,0);VARIANT(3,0);VARIANT(2,1);VARIANT(2,2);VARIANT(3,2);
    VARIANT(2,3);VARIANT(3,3);VARIANT(2,4);VARIANT(2,5);
  }
#undef VARIANT
  cold.Free();return ok;
}
int main(int argc,char** argv) {
  if(argc==2&&std::strcmp(argv[1],"check")==0) {
    const int bad=CheckWidth<3>()+CheckWidth<4>()+CheckWidth<5>()+CheckWidth<6>();
    std::printf("window failures=%d\n",bad);return bad?1:0;
  }
  if(argc!=3&&argc!=4)return 2;
  if(argc==4&&std::strcmp(argv[3],"paired"))return 2;
  const bool paired=argc==4;
  const bool down=std::strcmp(argv[2],"down")==0;
  if(!down&&std::strcmp(argv[2],"gateup"))return 2;
  if(!std::strcmp(argv[1],"305")) {
    if(paired)return (down?Rates<false,true,true>(true):Rates<false,false,true>(false))?0:1;
    return (down?Rates<false,true>(true):Rates<false,false>(false))?0:1;
  }
  if(!std::strcmp(argv[1],"405")) {
    if(paired)return (down?Rates<true,true,true>(true):Rates<true,false,true>(false))?0:1;
    return (down?Rates<true,true>(true):Rates<true,false>(false))?0:1;
  }
  return 2;
}
