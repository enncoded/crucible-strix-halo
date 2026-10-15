// Isolate small B-only LDS stages without changing any WMMA chain.
// R4096 useful work is 53.687 GFLOPs. Its 49 TFLOP/s floor is 1095.66 us.
#include "prefill_common.h"
#include "../engine/layers/prefill_hc.h"
#include <cstring>
using namespace crucible;
using namespace prefill_bench;
using StageU4=uint32_t __attribute__((ext_vector_type(4)));

__device__ __forceinline__ prefill::H16 StageFragment(const _Float16* source) {
  prefill::H16 result;
  auto* dst=reinterpret_cast<StageU4*>(&result);
  const auto* src=reinterpret_cast<const StageU4*>(source);
  dst[0]=src[0]; dst[1]=src[1];
  return result;
}

__device__ __forceinline__ void StageEpilogue(const prefill::F8* hi,const prefill::F8* lo,
    const float* w1,const float* ss,const float* H,float* mixed,__half* xh,int r0,int d,int R,int ldh) {
  float w[4];
#pragma unroll
  for (int s=0;s<4;++s) w[s]=w1[s*kHidden+d];
#pragma unroll
  for (int z=0;z<8;++z) {
    const int r=r0+prefill::FragmentRow(z);
    if (r>=R) continue;
    float v[4];
#pragma unroll
    for (int s=0;s<4;++s) {
      const float g=hi[s][z]+lo[s][z]*(1.f/prefill::kResidualScale);
      const float inv=rsqrtf(ss[r*4+s]/kHidden+kEps);
      v[s]=Sigmoid(g)*H[(size_t)r*ldh+s*kHidden+d]*inv*w[s];
    }
    const float x=RoundH(((v[0]+v[1])+(v[2]+v[3]))*.25f);
    if (mixed) mixed[(size_t)r*kHidden+d]=x;
    if (xh) xh[(size_t)r*kHidden+d]=__float2half(x);
  }
}

template<int STEP>
__global__ void __launch_bounds__(256) MixBStage(const __half* up,const float* w1,const __half* thi,
    const __half* tlo,const float* ss,const float* H,float* mixed,int R,__half* xh,int ldh) {
  constexpr int pieces=4*16*STEP/8,loads=(pieces+255)/256;
  __shared__ __attribute__((aligned(16))) _Float16 bs[2][64][STEP+8];
  const int t=threadIdx.x,lane=t&31,wave=t>>5;
  const int r0=blockIdx.y*128+wave*16,d0=blockIdx.x*16,d=d0+(lane&15),ra=r0+(lane&15);
  StageU4 next[loads];
  const auto load=[&](int k0) {
#pragma unroll
    for (int i=0;i<loads;++i) {
      const int p=t+i*256,row=p/(STEP/8),k8=p%(STEP/8)*8;
      if (p<pieces) next[i]=*reinterpret_cast<const StageU4*>(
          up+((size_t)(row/16)*kHidden+d0+row%16)*kRank+k0+k8);
    }
  };
  const auto store=[&](int buf) {
#pragma unroll
    for (int i=0;i<loads;++i) {
      const int p=t+i*256;
      if (p<pieces) *reinterpret_cast<StageU4*>(&bs[buf][p/(STEP/8)][p%(STEP/8)*8])=next[i];
    }
  };
  prefill::F8 hi[4]={},lo[4]={};
  load(0); store(0); __syncthreads();
#pragma unroll 1
  for (int k0=0,buf=0;k0<kRank;k0+=STEP,buf^=1) {
    const bool more=k0+STEP<kRank;
    if (more) load(k0+STEP);
#pragma unroll 2
    for (int kk=0;kk<STEP;kk+=16) {
      const int k=k0+kk;
      const prefill::H16 ah=ra<R ? PrefillLoad16(thi+(size_t)ra*kRank+k) : prefill::H16{};
      const prefill::H16 al=ra<R ? PrefillLoad16(tlo+(size_t)ra*kRank+k) : prefill::H16{};
#pragma unroll
      for (int s=0;s<4;++s) {
        const prefill::H16 b=StageFragment(&bs[buf][s*16+(lane&15)][kk]);
        hi[s]=__builtin_amdgcn_wmma_f32_16x16x16_f16_w32(ah,b,hi[s]);
        lo[s]=__builtin_amdgcn_wmma_f32_16x16x16_f16_w32(al,b,lo[s]);
      }
    }
    if (more) store(buf^1);
    __syncthreads();
  }
  StageEpilogue(hi,lo,w1,ss,H,mixed,xh,r0,d,R,ldh);
}

__global__ void __launch_bounds__(256) MixNoStage(const __half* up,const float* w1,const __half* thi,
    const __half* tlo,const float* ss,const float* H,float* mixed,int R,__half* xh,int ldh) {
  const int lane=threadIdx.x&31,wave=threadIdx.x>>5;
  const int r0=blockIdx.y*128+wave*16,d=blockIdx.x*16+(lane&15),ra=r0+(lane&15);
  if (r0>=R) return;
  prefill::F8 hi[4]={},lo[4]={};
#pragma unroll 1
  for (int k=0;k<kRank;k+=16) {
    const prefill::H16 ah=ra<R ? PrefillLoad16(thi+(size_t)ra*kRank+k) : prefill::H16{};
    const prefill::H16 al=ra<R ? PrefillLoad16(tlo+(size_t)ra*kRank+k) : prefill::H16{};
#pragma unroll
    for (int s=0;s<4;++s) {
      const prefill::H16 b=PrefillLoad16(up+((size_t)s*kHidden+d)*kRank+k);
      hi[s]=__builtin_amdgcn_wmma_f32_16x16x16_f16_w32(ah,b,hi[s]);
      lo[s]=__builtin_amdgcn_wmma_f32_16x16x16_f16_w32(al,b,lo[s]);
    }
  }
  StageEpilogue(hi,lo,w1,ss,H,mixed,xh,r0,d,R,ldh);
}

template<int STEP> void Measure(int R,int stride,bool half_only,const Buffer<__half>& up,
    const Buffer<float>& w1,const Buffer<__half>& thi,const Buffer<__half>& tlo,const Buffer<float>& ss,
    const Buffer<float>& H,const std::vector<float>& expected,const std::vector<__half>& expected_half) {
  Buffer<float> y(expected.size()); Buffer<__half> yh(expected_half.size());
  const auto run=[&] {
    if constexpr (STEP==0) MixNoStage<<<dim3(kHidden/16,(R+127)/128),256>>>(up.p,w1.p,thi.p,tlo.p,ss.p,
                                             H.p,half_only ? nullptr : y.p,R,yh.p,stride);
    else MixBStage<STEP><<<dim3(kHidden/16,(R+127)/128),256>>>(up.p,w1.p,thi.p,tlo.p,ss.p,
                                             H.p,half_only ? nullptr : y.p,R,yh.p,stride);
  };
  run(); CheckLaunch();
  const auto got=y.Get(); const auto gh=yh.Get();
  if (std::memcmp(got.data(),expected.data(),got.size()*sizeof(float)) ||
      std::memcmp(gh.data(),expected_half.data(),gh.size()*sizeof(__half))) {
    std::fprintf(stderr,"FAIL: R%d STEP%d half-only%d\n",R,STEP,half_only); std::exit(1);
  }
  const double us=Time(run,16),flops=4.*R*kWide*kRank;
  std::printf("R%d half-only%d STEP%d %.3f us %.2f%% useful WMMA roofline\n",R,half_only,STEP,us,flops/(us*49e6)*100);
}

int main() {
  std::puts("STEP0 is the original global-load kernel with unroll one. STEP16/32/64 stage only B.");
  Buffer<__half> up(Half(Random(kWide*kRank,.04f,1)));
  Buffer<float> w1(Random(kWide,.1f,2));
  for (int R : {33,129,513,1024,4096}) {
    constexpr int stride=kWide+64;
    Buffer<float> H(Random((size_t)R*stride,.3f,3));
    Buffer<__half> thi(Half(Random((size_t)R*kRank,.1f,4))),tlo(Half(Random((size_t)R*kRank,.1f,5)));
    Buffer<float> ss(std::vector<float>(R*4,float(kHidden)));
    for (bool half_only : {false,true}) {
      Buffer<float> y((size_t)R*kHidden); Buffer<__half> yh((size_t)R*kHidden);
      const auto baseline=[&] { PrefillHcMixWideKernel<true><<<dim3(kHidden/16,(R+127)/128),256>>>(up.p,w1.p,
                                      thi.p,tlo.p,ss.p,H.p,half_only ? nullptr : y.p,R,yh.p,stride); };
      baseline(); CheckLaunch(); const auto expected=y.Get(); const auto expected_half=yh.Get();
      const double us=Time(baseline,16),flops=4.*R*kWide*kRank;
      std::printf("R%d half-only%d baseline %.3f us %.2f%% useful WMMA roofline\n",R,half_only,us,flops/(us*49e6)*100);
      Measure<0>(R,stride,half_only,up,w1,thi,tlo,ss,H,expected,expected_half);
      Measure<16>(R,stride,half_only,up,w1,thi,tlo,ss,H,expected,expected_half);
      Measure<32>(R,stride,half_only,up,w1,thi,tlo,ss,H,expected,expected_half);
      Measure<64>(R,stride,half_only,up,w1,thi,tlo,ss,H,expected,expected_half);
    }
  }
  std::puts("PASS: staged HC operands and both output formats match bit for bit");
}
