// Measure H row padding before changing any production reader.
#include "prefill_common.h"
#include "../engine/layers/prefill_hc.h"
#include "../engine/layers/prefill_moe.h"
using namespace crucible;
using namespace prefill_bench;

__global__ void FillH(float* H,int rows,int stride) {
  const size_t i=(size_t)blockIdx.x*256+threadIdx.x;
  if (i>=(size_t)rows*kWide) return;
  H[(i/kWide)*stride+i%kWide]=(float)((int)(i%257)-128)*.001f;
}
__global__ void FillValues(float* x,size_t n) {
  const size_t i=(size_t)blockIdx.x*256+threadIdx.x;
  if (i<n) x[i]=(float)((int)(i%127)-63)*.01f;
}
__global__ void FillHalfValues(__half* x,size_t n) {
  const size_t i=(size_t)blockIdx.x*256+threadIdx.x;
  if (i<n) x[i]=__float2half((float)((int)(i%127)-63)*.01f);
}

// The production combine's arithmetic and wave shape, with one H stride.
__global__ void __launch_bounds__(256) CombineStride(PrefillMoe a,const float* weights,
    const float* post,float* H,int ldh) {
  const int w=blockIdx.x*8+(threadIdx.x>>5),lane=threadIdx.x&31;
  if (w>=a.dispatch.rows*20) return;
  const int r=w/20,blk=w%20; float y[4]={};
  for (int k=0;k<kSlots;++k) {
    const int pair=r*kTopK+k;
    const int e=k<kTopK ? a.dispatch.selected[pair] : kExperts;
    const int local=k<kTopK ? a.inverse[pair]-a.dispatch.offsets[e] : r;
    float v[4]; LinOutBlock(a.down[e],local,blk,v);
    const float weight=weights[r*kSlots+k];
#pragma unroll
    for (int i=0;i<4;++i) y[i]+=weight*v[i];
  }
  const int c=blk*128+lane*4;
#pragma unroll
  for (int s=0;s<kStreams;++s) {
    const float gate=post[r*kStreams+s];
#pragma unroll
    for (int i=0;i<4;++i) H[(size_t)r*ldh+s*kHidden+c+i]+=gate*y[i];
  }
}

int main() {
  constexpr int R=4096,M=kRank+4;
  Buffer<float> H((size_t)R*(kWide+512)),before((size_t)R*kWide),pd((size_t)R*kSlots*kHidden);
  FillValues<<<(pd.n+255)/256,256>>>(pd.p,pd.n);
  Buffer<__half> svh(std::vector<__half>(kHidden,__float2half(1.f)));
  Buffer<int> selected(std::vector<int>(R*kTopK,0)),inverse(R*kTopK),offsets(std::vector<int>(kExperts+1,0));
  std::vector<int> inv(R*kTopK); for (int i=0;i<R*kTopK;++i) inv[i]=i;
  HIP_OK(hipMemcpy(inverse.p,inv.data(),inv.size()*sizeof(int),hipMemcpyHostToDevice));
  std::vector<LinJob> jobs(kExperts+1);
  jobs[0].partial=pd.p; jobs[0].svh=svh.p; jobs[0].R=R*kTopK; jobs[0].O=kHidden; jobs[0].S=1;
  jobs[kExperts]=jobs[0]; jobs[kExperts].partial=pd.p+(size_t)R*kTopK*kHidden; jobs[kExperts].R=R;
  Buffer<LinJob> down(jobs);
  Buffer<__half> hpd(pd.n);
  FillHalfValues<<<(hpd.n+255)/256,256>>>(hpd.p,hpd.n);
  jobs[0].partial=reinterpret_cast<float*>(hpd.p);
  jobs[kExperts].partial=reinterpret_cast<float*>(hpd.p+(size_t)R*kTopK*kHidden);
  Buffer<LinJob> hdown(jobs);
  Buffer<float> weights(std::vector<float>(R*kSlots,.09f)),post(std::vector<float>(R*kStreams,.5f));
  PrefillMoe a{}; a.dispatch.rows=R; a.dispatch.selected=selected.p; a.dispatch.offsets=offsets.p;
  a.inverse=inverse.p; a.down=down.p;
  Buffer<__half> fn(std::vector<__half>((size_t)M*kWide,__float2half(.01f)));
  Buffer<float> dots((size_t)R*M*kStreams),ss(R*kStreams);
  FillH<<<(R*kWide+255)/256,256>>>(before.p,R,kWide);
  PrefillMoeCombineKernel<false><<<(R*20+7)/8,256>>>(a,weights.p,post.p,before.p);
  FillH<<<(R*kWide+255)/256,256>>>(H.p,R,kWide);
  CombineStride<<<(R*20+7)/8,256>>>(a,weights.p,post.p,H.p,kWide);
  CheckLaunch();
  const auto expected=before.Get(),got=H.Get();
  if (std::memcmp(expected.data(),got.data(),expected.size()*sizeof(float))) return 1;
  const double combine_bytes=(double)R*(kSlots*kHidden+2*kWide)*sizeof(float);
  Buffer<__half> up(std::vector<__half>((size_t)kWide*kRank,__float2half(.01f)));
  Buffer<__half> thi(std::vector<__half>((size_t)R*kRank,__float2half(.1f)));
  Buffer<__half> tlo(std::vector<__half>((size_t)R*kRank,__float2half(.001f)));
  Buffer<float> norm(std::vector<float>(kWide,1.f)),mixed((size_t)R*kHidden);
  for (int rep=0;rep<4;++rep) for (int index=0;index<6;++index) {
    constexpr int pads[]={0,32,64,128,256,512};
    const int pad=pads[(rep&1) ? 5-index : index];
    const int stride=kWide+pad;
    FillH<<<(R*kWide+255)/256,256>>>(H.p,R,stride); CheckLaunch();
    const auto combine=[&] { CombineStride<<<(R*20+7)/8,256>>>(a,weights.p,post.p,H.p,stride); };
    const auto half_combine=[&] {
      auto ha=a; ha.down=hdown.p;
      PrefillMoeCombineKernel<true><<<(R*20+7)/8,256>>>(ha,weights.p,post.p,H.p,stride);
    };
    const auto hc=[&] { PrefillHcDotsWsKernel<64,0,true,2,2,false,4><<<kStreams*((R+63)/64),288>>>(
        fn.p,M,H.p,dots.p,ss.p,R,stride); };
    hc(); CheckLaunch();
    const auto mix=[&] { PrefillHcMixWideKernel<true><<<dim3(kHidden/16,(R+127)/128),256>>>(
        up.p,norm.p,thi.p,tlo.p,ss.p,H.p,mixed.p,R,nullptr,stride); };
    const double us=Time(combine,16),halfus=Time(half_combine,16),hcus=Time(hc,16),mixus=Time(mix,16);
    const double gbps=combine_bytes/us/1e3;
    const double half_bytes=(double)R*(kSlots*kHidden*2+2*kWide*4),half_gbps=half_bytes/halfus/1e3;
    const double dots_tflops=2.*R*kStreams*M*kHidden/hcus/1e6;
    const double mix_tflops=4.*R*kWide*kRank/mixus/1e6;
    std::printf("rep=%d pad=%d combine_us=%.3f combine_GBs=%.1f %.2f%% DRAM half_combine_us=%.3f half_combine_GBs=%.1f %.2f%% DRAM hc_dots_us=%.3f %.2f%% WMMA hc_mix_us=%.3f %.2f%% WMMA\n",
                rep,pad,us,gbps,gbps/245.5*100,halfus,half_gbps,half_gbps/245.5*100,
                hcus,dots_tflops/49*100,mixus,mix_tflops/49*100);
  }
  return 0;
}
