// Standalone cached-codebook probe. Production and dense bodies do not use it.
#pragma once
#include "../../engine/exl3/kernels/exl3_gemm_pair.h"

namespace crucible {

template<int K> __device__ __forceinline__ void MoeLutDecode(
    uint32_t w0,uint32_t w1,uint32_t w2,int shift,__half2 (&p)[4],const __half* lut) {
  const uint32_t a0=__builtin_amdgcn_alignbit(w0,w1,32-shift);
  const uint32_t a1=__builtin_amdgcn_alignbit(w1,w2,32-shift);
  uint32_t states[8];
#pragma unroll
  for(int i=0;i<8;++i) {
    const int end=i*K+16;
    states[i]=end<=32?a0>>(32-end):__builtin_amdgcn_alignbit(a0,a1,64-end);
  }
#pragma unroll
  for(int z=0;z<4;++z)
    p[z]=__halves2half2(lut[states[2*z]&0xffffu],lut[states[2*z+1]&0xffffu]);
}

// Keep the production A/B order and accumulator chains. Only decode changes.
template<int K,int RM,bool HALF=false,bool SKIP=true> __device__ __forceinline__ void MoeLutBody(
    const Exl3GemmJob& j,const Exl3GemmItem& it,uint32_t* ws,const __half* lut) {
  constexpr int NB=2,T=8*K;
  const int lane=threadIdx.x&31,wave=__builtin_amdgcn_readfirstlane(threadIdx.x>>5);
  const int b=it.b0+wave*NB;
  if(b*16>=j.O)return;
  const int A=j.I/16;
  const LaneWindow<K> win(lane);
  __amdgpu_buffer_rsrc_t rs[NB];
#pragma unroll
  for(int n=0;n<NB;++n)rs[n]=StreamRsrc(j.tiles+(size_t)(b+n)*A*T,A*T*4);
  GemmF8 acc[NB][RM]={};
  const uint32_t xbytes=(uint32_t)j.Xrows*j.I*2;
  const auto xrs=StreamRsrc(reinterpret_cast<const uint32_t*>(j.x),xbytes);
  const int astep=j.Xrows*32;
  uint32_t xo[RM];
#pragma unroll
  for(int r=0;r<RM;++r) {
    const int row=it.m0+r*16+(lane&15);
    xo[r]=row<j.M?(uint32_t)row*32:xbytes;
  }
  uint32_t* w=ws+wave*kGemmWaveWords;
  const int o0=win.d0*4,o1=((win.d0+1)%T)*4,o2=((win.d0+2)%T)*4;
  uint32_t raw[NB][3];
#pragma unroll
  for(int n=0;n<NB;++n) {
    raw[n][0]=__builtin_amdgcn_raw_buffer_load_b32(rs[n],o0,0,0);
    raw[n][1]=__builtin_amdgcn_raw_buffer_load_b32(rs[n],o1,0,0);
    raw[n][2]=K>=5?__builtin_amdgcn_raw_buffer_load_b32(rs[n],o2,0,0):0;
  }
  for(int a=0;a<A;++a) {
    GemmH16 afr[RM];
#pragma unroll
    for(int r=0;r<RM;++r) {
      if(SKIP&&it.m0+r*16>=j.M)break;
      afr[r]=GemmAFragment<false>(xrs,xo[r],a,nullptr,0,astep);
    }
    asm volatile("" ::: "memory");
    const int next=(a+1)*T*4;
#pragma unroll
    for(int n=0;n<NB;++n) {
      const uint32_t w0=raw[n][0],w1=raw[n][1],w2=raw[n][2];
      raw[n][0]=__builtin_amdgcn_raw_buffer_load_b32(rs[n],o0+next,0,0);
      raw[n][1]=__builtin_amdgcn_raw_buffer_load_b32(rs[n],o1+next,0,0);
      if constexpr(K>=5)raw[n][2]=__builtin_amdgcn_raw_buffer_load_b32(rs[n],o2+next,0,0);
      __half2 p[4];
      MoeLutDecode<K>(w0,w1,w2,win.shift,p,lut);
      const GemmH16 bf=GemmBFragment<true>(p,w,lane);
#pragma unroll
      for(int r=0;r<RM;++r) {
        if(SKIP&&it.m0+r*16>=j.M)break;
        acc[n][r]=__builtin_amdgcn_wmma_f32_16x16x16_f16_w32(afr[r],bf,acc[n][r]);
      }
    }
  }
  constexpr uint32_t bytes=HALF?2:4;
  const auto ps=StreamRsrc(reinterpret_cast<const uint32_t*>(j.partial),(uint32_t)j.M*j.O*bytes);
#pragma unroll
  for(int n=0;n<NB;++n) {
    const uint32_t po=((uint32_t)(it.m0+lane/16)*j.O+(b+n)*16+(lane&15))*bytes;
#pragma unroll
    for(int r=0;r<RM;++r) {
      if(SKIP&&it.m0+r*16>=j.M)break;
#pragma unroll
      for(int z=0;z<8;++z) {
        const uint32_t offset=po+(uint32_t)(r*16+2*z)*j.O*bytes;
        if constexpr(HALF) {
          const float v=acc[n][r][z];
          const __half h=__float2half(fabsf(v)>65504.f?copysignf(65504.f,v):v);
          __builtin_amdgcn_raw_buffer_store_b16(__half_as_ushort(h),ps,offset,0,0);
        } else __builtin_amdgcn_raw_buffer_store_b32(__float_as_uint(acc[n][r][z]),ps,offset,0,0);
      }
    }
  }
}

}  // namespace crucible
