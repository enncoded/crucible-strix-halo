#pragma once
#include "../../engine/layers/prefill_moe.h"

namespace crucible {
// Eight NB1 compute waves cover the same128 columns as four NB2 waves.
// Two loader waves publish the next A/raw-B tile while compute uses this tile.
// One CTA barrier per K step prevents reuse before every consumer finishes.
template<int K,int RM,int STRIDE> __device__ __forceinline__ void LoaderFill(
    const Exl3GemmJob& j,const Exl3GemmItem& it,int step,uint32_t* xs,uint32_t* raw) {
  using U4=uint32_t __attribute__((ext_vector_type(4)));
  constexpr int T=8*K;
  const int lt=threadIdx.x-256,A=j.I/16;
  const uint32_t bytes=(uint32_t)j.Xrows*j.I*2;
  const auto xrs=StreamRsrc(reinterpret_cast<const uint32_t*>(j.x),bytes);
  const auto brs=StreamRsrc(j.tiles,(uint32_t)(j.O/16)*A*T*4);
  for(int v=lt;v<RM*16*2;v+=64) {
    const int row=it.m0+v/2;
    const uint32_t off=(row<j.M?(uint32_t)row*32:bytes)+(v&1)*16+(uint32_t)step*j.Xrows*32;
    reinterpret_cast<U4*>(xs+(v/2)*STRIDE+(v&1)*4)[0]=__builtin_amdgcn_raw_buffer_load_b128(xrs,off,0,0);
  }
  for(int v=lt;v<8*T;v+=64) {
    const int col=it.b0+v/T;
    const uint32_t off=((uint32_t)col*A+step)*T*4+(v%T)*4;
    raw[v]=col*16<j.O?__builtin_amdgcn_raw_buffer_load_b32(brs,off,0,0):0;
  }
}
template<int RM> __device__ __forceinline__ void LoaderStore(const Exl3GemmJob& j,
    const Exl3GemmItem& it,int b,int lane,GemmF8* acc) {
  const auto ps=StreamRsrc(reinterpret_cast<const uint32_t*>(j.partial),(uint32_t)j.M*j.O*4);
  const uint32_t po=((uint32_t)(it.m0+lane/16)*j.O+b*16+(lane&15))*4;
#pragma unroll
  for(int r=0;r<RM;++r) {
    if(it.m0+r*16>=j.M)break;
#pragma unroll
    for(int z=0;z<8;++z)__builtin_amdgcn_raw_buffer_store_b32(
        __float_as_uint(acc[r][z]),ps,po+(uint32_t)(r*16+2*z)*j.O*4,0,0);
  }
}
template<int K,int RM,int STRIDE> __device__ __forceinline__ void LoaderBody(
    const Exl3GemmJob& j,const Exl3GemmItem& it,uint32_t* ws) {
  using U4=uint32_t __attribute__((ext_vector_type(4)));
  constexpr int T=8*K,APLANE=RM*16*STRIDE,BPLANE=8*T;
  static_assert(STRIDE==8||STRIDE==12);
  const int lane=threadIdx.x&31,wave=__builtin_amdgcn_readfirstlane(threadIdx.x>>5);
  const int b=it.b0+wave,A=j.I/16;
  uint32_t* xs=ws+8*kGemmWaveWords;uint32_t* raw=xs+2*APLANE;
  GemmF8 acc[RM]={};const LaneWindow<K> win(lane);
  if(wave>=8)LoaderFill<K,RM,STRIDE>(j,it,0,xs,raw);
  GemmGroupSync();
  for(int step=0;step<A;++step) {
    if(wave>=8) {
      if(step+1<A)LoaderFill<K,RM,STRIDE>(j,it,step+1,
          xs+((step+1)&1)*APLANE,raw+((step+1)&1)*BPLANE);
    } else if(b*16<j.O) {
      const auto* rp=raw+(step&1)*BPLANE+wave*T;
      __half2 p[4];DecodeLane<K,kMulTrue16>(rp[win.d0],rp[(win.d0+1)%T],
          K>=5?rp[(win.d0+2)%T]:0,win.shift,p);
      const GemmH16 bf=GemmBFragment<true>(p,ws+wave*kGemmWaveWords,lane);
#pragma unroll
      for(int r=0;r<RM;++r) {
        if(it.m0+r*16>=j.M)break;
        GemmH16 af;auto* dst=reinterpret_cast<U4*>(&af);
        const auto* src=reinterpret_cast<const U4*>(xs+(step&1)*APLANE+(r*16+(lane&15))*STRIDE);
        dst[0]=src[0];dst[1]=src[1];
        acc[r]=__builtin_amdgcn_wmma_f32_16x16x16_f16_w32(af,bf,acc[r]);
        asm volatile("" : : "v"(acc[r]) : "memory");
      }
    }
    GemmGroupSync();
  }
  if(wave<8&&b*16<j.O)LoaderStore<RM>(j,it,b,lane,acc);
}
template<int RM,int KG,int KS,int STRIDE> __global__ void __launch_bounds__(320) ProbeMoeLoaderKernel(PrefillMoe a) {
  constexpr int words=8*kGemmWaveWords+2*RM*16*STRIDE+2*8*8*KS;
  __shared__ __attribute__((aligned(16))) uint32_t ws[words];
  const int count=PrefillUniform(a.item_counts[0]);
  for(int p=blockIdx.x;p<count;p+=gridDim.x) {
    const auto it=PrefillUniform(a.i_gu[p]);const auto j=PrefillUniform(a.g_gu[it.job]);
    if(j.K==KG)LoaderBody<KG,RM,STRIDE>(j,it,ws);
    else if(j.K==KS)LoaderBody<KS,RM,STRIDE>(j,it,ws);
    GemmGroupSync();
  }
}
}
