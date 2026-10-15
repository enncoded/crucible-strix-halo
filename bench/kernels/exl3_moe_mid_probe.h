#pragma once
#include "../../engine/layers/prefill_moe.h"

namespace crucible {
// PrefillWideBody's arithmetic and ordering, with a compact private sink.
// Keeping its accumulators inside this function avoids live G and U arrays.
template<int K,int RM,bool CHECK=false> __device__ __forceinline__ void ProbeMoeSinkBody(
    const Exl3GemmJob& j,const Exl3GemmItem& it,uint32_t* ws,float* sink) {
  constexpr int NB=2,T=8*K;
  const int lane=threadIdx.x&31,wave=__builtin_amdgcn_readfirstlane(threadIdx.x>>5);
  const int b=it.b0+wave*NB,A=j.I/16;
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
      if(it.m0+r*16>=j.M)break;
      afr[r]=GemmAFragment<false>(xrs,xo[r],a,nullptr,0,astep);
    }
    asm volatile("":::"memory");
    const int next=(a+1)*T*4;
#pragma unroll
    for(int n=0;n<NB;++n) {
      const uint32_t w0=raw[n][0],w1=raw[n][1],w2=raw[n][2];
      raw[n][0]=__builtin_amdgcn_raw_buffer_load_b32(rs[n],o0+next,0,0);
      raw[n][1]=__builtin_amdgcn_raw_buffer_load_b32(rs[n],o1+next,0,0);
      if constexpr(K>=5)raw[n][2]=__builtin_amdgcn_raw_buffer_load_b32(rs[n],o2+next,0,0);
      __half2 p[4];DecodeLane<K,kMulTrue16>(w0,w1,w2,win.shift,p);
      const GemmH16 bf=GemmBFragment<true>(p,w,lane);
#pragma unroll
      for(int r=0;r<RM;++r) {
        if(it.m0+r*16>=j.M)break;
        acc[n][r]=__builtin_amdgcn_wmma_f32_16x16x16_f16_w32(afr[r],bf,acc[n][r]);
      }
    }
  }
  const auto ps=StreamRsrc(reinterpret_cast<const uint32_t*>(sink),RM*16*128*4);
#pragma unroll
  for(int n=0;n<NB;++n) {
    const uint32_t po=((lane/16)*128+(wave*NB+n)*16+(lane&15))*4;
#pragma unroll
    for(int r=0;r<RM;++r) {
      if(it.m0+r*16>=j.M)break;
#pragma unroll
      for(int z=0;z<8;++z) {
        __builtin_amdgcn_raw_buffer_store_b32(__float_as_uint(acc[n][r][z]),ps,po+(r*16+2*z)*128*4,0,0);
        if constexpr(CHECK) {
          const int row=it.m0+r*16+2*z+lane/16;
          if(row<j.M)j.partial[(size_t)row*j.O+(b+n)*16+(lane&15)]=acc[n][r][z];
        }
      }
    }
  }
}

template<int RM> __device__ __forceinline__ void ProbeMoeMid(
    PrefillMoe a,Exl3GemmItem it,int M,const float* ring) {
  const int wave=threadIdx.x>>5,lane=threadIdx.x&31,e=it.job/2,c=it.b0*16+lane*4;
  // All GEMM values are dead before these scales enter the register file.
  const auto g=a.gu[it.job],u=a.gu[it.job+1],d=a.down[e];
  const float* rg=ring;const float* ru=ring+RM*16*128;
  for(int lr=wave;lr<min(RM*16,M-it.m0);lr+=4) {
    float gv[4],uv[4],v[4];
#pragma unroll
    for(int i=0;i<4;++i){gv[i]=0.f+rg[lr*128+lane*4+i];uv[i]=0.f+ru[lr*128+lane*4+i];}
    Wht128(gv);Wht128(uv);
#pragma unroll
    for(int i=0;i<4;++i) {
      gv[i]*=kHadNorm*__half2float(g.svh[c+i]);
      uv[i]*=kHadNorm*__half2float(u.svh[c+i]);
      v[i]=RoundH(RoundH(Silu(gv[i])*uv[i])*__half2float(d.suh[c+i]));
    }
    Wht128(v);
#pragma unroll
    for(int i=0;i<4;++i)d.xh[PackedIndex(it.m0+lr,c+i,d.R)]=__float2half(v[i]*kHadNorm);
  }
}
// Each persistent block owns one ring slot and one G/U pair at a time.
// There are no inter-block claims, waits, counter resets, or slot collisions.
template<int RM,int KG,int KS,bool CHECK=false> __global__ void __launch_bounds__(128) ProbeMoeMidKernel(
    PrefillMoe a,float* ring) {
  __shared__ uint32_t ws[kGemmWeightWords];
  const int count=PrefillUniform(*a.item_counts)/2;
  float* slot=ring+(size_t)blockIdx.x*RM*16*128*2;
  for(int p=blockIdx.x;p<count;p+=gridDim.x) {
    const auto it=PrefillUniform(a.i_gu[p*2]);
#pragma clang loop unroll(disable)
    for(int which=0;which<2;++which) {
      const auto j=PrefillUniform(a.g_gu[it.job+which]);
      float* sink=slot+which*RM*16*128;
      if(j.K==KG)ProbeMoeSinkBody<KG,RM,CHECK>(j,it,ws,sink);
      else if(j.K==KS)ProbeMoeSinkBody<KS,RM,CHECK>(j,it,ws,sink);
    }
    __builtin_amdgcn_fence(__ATOMIC_RELEASE,"workgroup");
    __syncthreads();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE,"workgroup");
    ProbeMoeMid<RM>(a,it,PrefillUniform(a.g_gu[it.job].M),slot);
    __syncthreads();
  }
}
// The existing FX publication protocol, but each G claim owns a complete
// G/U pair. Its private scratch slot needs no cross-block retirement.
template<int RM,int KG,int KS> __global__ void __launch_bounds__(128) ProbeMoeFxMidKernel(
    PrefillMoe a,float* ring,int lead) {
  constexpr int tile_words=2*16*136/2;
  __shared__ __attribute__((aligned(16))) uint32_t lds[tile_words];
  __shared__ int claim;
  auto* tile=reinterpret_cast<__half (*)[16][136]>(lds);
  float* slot=ring+(size_t)blockIdx.x*RM*16*128*2;
  const int t=threadIdx.x,T=a.tile_rows;
  const int ng=PrefillUniform(a.item_counts[0])/2,nx=PrefillUniform(a.fx_count[2]);
  for(;;) {
    if(t==0)claim=atomicAdd(a.fx_count+1,1);
    __syncthreads();const int p=PrefillUniform(claim);__syncthreads();
    Exl3GemmItem it{};int M=0,need=nx;
    if(p<ng) {
      it=PrefillUniform(a.i_gu[p*2]);M=PrefillUniform(a.g_gu[it.job].M);
      need=min(nx,a.fx_xoff[it.job>>1]+2*min((M+15)/16,(it.m0+T)/16));
    }
    const int limit=min(nx,need+lead);
    for(;;) {
      if(t==0)claim=__hip_atomic_load(a.fx_count,__ATOMIC_RELAXED,__HIP_MEMORY_SCOPE_AGENT)<limit
                       ?atomicAdd(a.fx_count,1):nx;
      __syncthreads();const int x=PrefillUniform(claim);__syncthreads();
      if(x>=nx)break;
      PrefillFxTransform(a,x,tile,threadIdx.x);
    }
    if(p>=ng)return;
    const int target=2*((min(M,it.m0+T)-it.m0+15)/16);
    if(t==0)while(__hip_atomic_load(a.fx_ready+p/5,__ATOMIC_RELAXED,__HIP_MEMORY_SCOPE_AGENT)<target)
      __builtin_amdgcn_s_sleep(1);
    __syncthreads();__builtin_amdgcn_fence(__ATOMIC_ACQUIRE,"agent");
#pragma clang loop unroll(disable)
    for(int which=0;which<2;++which) {
      const auto j=PrefillUniform(a.g_gu[it.job+which]);
      float* sink=slot+which*RM*16*128;
      if(j.K==KG)ProbeMoeSinkBody<KG,RM>(j,it,lds,sink);
      else if(j.K==KS)ProbeMoeSinkBody<KS,RM>(j,it,lds,sink);
    }
    __builtin_amdgcn_fence(__ATOMIC_RELEASE,"workgroup");__syncthreads();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE,"workgroup");
    ProbeMoeMid<RM>(a,it,M,slot);__syncthreads();
  }
}
}
