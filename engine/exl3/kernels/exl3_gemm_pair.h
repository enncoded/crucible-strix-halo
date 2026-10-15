#pragma once
#include "exl3_gemm.h"
namespace crucible {
constexpr int kGemmPairWords=2*kGemmWeightWords;
__device__ __forceinline__ void GemmNativeStore(const __half2 p[4],uint32_t* w,int lane) {
  const int q=lane&3,c=lane/4;
  uint32_t v[4];
#pragma unroll
  for(int z=0;z<4;++z)v[z]=*reinterpret_cast<const uint32_t*>(&p[z]);
  w[c*kGemmBStride+q]=v[0];w[c*kGemmBStride+q+4]=v[1];
  w[(c+8)*kGemmBStride+q]=v[2];w[(c+8)*kGemmBStride+q+4]=v[3];
}
__device__ __forceinline__ GemmH16 GemmStoredFragment(const uint32_t* w,int lane) {
  GemmH16 b;auto* bp=reinterpret_cast<uint32_t*>(&b);
#pragma unroll
  for(int z=0;z<8;++z)bp[z]=w[(lane&15)*kGemmBStride+z];
  return b;
}
// Two column tiles share one LDS publication and one reuse fence. This keeps
// the per-output K order but halves wave fences relative to separate exchanges.
// The four-wave queue has 8704 scratch bytes; this body needs6144.
template<int K,int RM,bool ORDER=true> __device__ __forceinline__ void GemmPairBody(
    const Exl3GemmJob& j,const Exl3GemmItem& it,uint32_t* ws) {
  constexpr int NB=2,T=8*K;
  const int lane=threadIdx.x&31,wave=__builtin_amdgcn_readfirstlane(threadIdx.x>>5);
  const int b=it.b0+wave*NB,A=j.I/16;
  if(b*16>=j.O)return;
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
  const int o0=win.d0*4,o1=((win.d0+1)%T)*4,o2=((win.d0+2)%T)*4;
  uint32_t raw[NB][3];
#pragma unroll
  for(int n=0;n<NB;++n) {
    raw[n][0]=__builtin_amdgcn_raw_buffer_load_b32(rs[n],o0,0,0);
    raw[n][1]=__builtin_amdgcn_raw_buffer_load_b32(rs[n],o1,0,0);
    raw[n][2]=K==5?__builtin_amdgcn_raw_buffer_load_b32(rs[n],o2,0,0):0;
  }
  for(int a=0;a<A;++a) {
    GemmH16 afr[RM];
#pragma unroll
    for(int r=0;r<RM;++r) {
      if(it.m0+r*16>=j.M)break;
      afr[r]=GemmAFragment<false>(xrs,xo[r],a,nullptr,0,astep);
    }
    if constexpr(ORDER)asm volatile("":::"memory");
    const int next=(a+1)*T*4;
#pragma unroll
    for(int n=0;n<NB;++n) {
      const uint32_t w0=raw[n][0],w1=raw[n][1],w2=raw[n][2];
      raw[n][0]=__builtin_amdgcn_raw_buffer_load_b32(rs[n],o0+next,0,0);
      raw[n][1]=__builtin_amdgcn_raw_buffer_load_b32(rs[n],o1+next,0,0);
      if constexpr(K==5)raw[n][2]=__builtin_amdgcn_raw_buffer_load_b32(rs[n],o2+next,0,0);
      __half2 p[4];DecodeLane<K,kMulTrue16>(w0,w1,w2,win.shift,p);
      GemmNativeStore(p,ws+(wave*NB+n)*kGemmWaveWords,lane);
    }
    GemmWaveSync();
#pragma unroll
    for(int n=0;n<NB;++n) {
      const GemmH16 bf=GemmStoredFragment(ws+(wave*NB+n)*kGemmWaveWords,lane);
#pragma unroll
      for(int r=0;r<RM;++r) {
        if(it.m0+r*16>=j.M)break;
        acc[n][r]=__builtin_amdgcn_wmma_f32_16x16x16_f16_w32(afr[r],bf,acc[n][r]);
      }
    }
    GemmWaveSync();
  }
  const auto ps=StreamRsrc(reinterpret_cast<const uint32_t*>(j.partial),(uint32_t)j.M*j.O*4);
#pragma unroll
  for(int n=0;n<NB;++n) {
    const uint32_t po=((uint32_t)(it.m0+lane/16)*j.O+(b+n)*16+(lane&15))*4;
#pragma unroll
    for(int r=0;r<RM;++r) {
      if(it.m0+r*16>=j.M)break;
#pragma unroll
      for(int z=0;z<8;++z)__builtin_amdgcn_raw_buffer_store_b32(__float_as_uint(acc[n][r][z]),
          ps,po+(uint32_t)(r*16+2*z)*j.O*4,0,0);
    }
  }
}
} // namespace crucible
