#pragma once
#include "../../engine/exl3/prefill_linear.h"

// Packed A shares LDS with the existing B exchange. All waves must participate.
namespace crucible {
template<int K,int RM,int NB,bool EARLY_A=true,bool SKIP=true,bool ORDER=SKIP,
         bool STAGE_A=false,int A_STRIDE=8,int WAVES=4> __device__ __forceinline__ void GemmStageBody(const Exl3GemmJob& j,
    const Exl3GemmItem& it, uint32_t* ws, GemmF8* keep=nullptr) {
  constexpr int T=8*K;
  const int lane=threadIdx.x&31, wave=__builtin_amdgcn_readfirstlane(threadIdx.x>>5);
  const int b=it.b0+wave*NB;
  if (b*16>=j.O) return;
  const int A=j.I/16;
  const LaneWindow<K> win(lane);
  __amdgpu_buffer_rsrc_t rs[NB];
#pragma unroll
  for (int n=0; n<NB; ++n) rs[n]=StreamRsrc(j.tiles+(size_t)(b+n)*A*T,A*T*4);
  GemmF8 acc[NB][RM]={};
  const uint32_t xbytes=(uint32_t)j.Xrows*j.I*2;
  const auto xrs=StreamRsrc(reinterpret_cast<const uint32_t*>(j.x),xbytes);
  const int astep=j.Xrows*32;
  uint32_t xo[RM];
#pragma unroll
  for (int r=0; r<RM; ++r) {
    const int row=it.m0+r*16+(lane&15);
    xo[r]=row<j.M ? (uint32_t)row*32 : xbytes;
  }
  using U4=uint32_t __attribute__((ext_vector_type(4)));
  static_assert(WAVES==4 || WAVES==8);
  uint32_t* xs=ws+WAVES*kGemmWaveWords;
  uint32_t sx[(RM+WAVES-1)/WAVES];
  if constexpr(STAGE_A) {
#pragma unroll
    for(int z=0;z<(RM+WAVES-1)/WAVES;++z) {
      const int row=it.m0+(threadIdx.x>>1)+z*WAVES*16;
      sx[z]=(row<j.M?(uint32_t)row*32:xbytes)+(threadIdx.x&1)*16;
    }
  }
  uint32_t* w=ws+wave*kGemmWaveWords;
  const int o0=win.d0*4, o1=((win.d0+1)%T)*4, o2=((win.d0+2)%T)*4;
  uint32_t raw[NB][3];
#pragma unroll
  for (int n=0; n<NB; ++n) {
    raw[n][0]=__builtin_amdgcn_raw_buffer_load_b32(rs[n],o0,0,0);
    raw[n][1]=__builtin_amdgcn_raw_buffer_load_b32(rs[n],o1,0,0);
    raw[n][2]=K>=5 ? __builtin_amdgcn_raw_buffer_load_b32(rs[n],o2,0,0) : 0;
  }
  for (int a=0; a<A; ++a) {
    if constexpr(STAGE_A) {
#pragma unroll
      for(int z=0;z<(RM+WAVES-1)/WAVES;++z) {
        const int i=threadIdx.x+z*WAVES*32;
        if(i<RM*32)reinterpret_cast<U4*>(xs+(i>>1)*A_STRIDE+(i&1)*4)[0]=
            __builtin_amdgcn_raw_buffer_load_b128(xrs,sx[z],a*astep,0);
      }
      GemmGroupSync();
    }
    GemmH16 afr[RM];
    auto load_a=[&] {
      if constexpr(!STAGE_A) {
#pragma unroll
      for (int r=0; r<RM; ++r) {
        if (SKIP && it.m0+r*16>=j.M) break;
        afr[r]=GemmAFragment<false>(xrs,xo[r],a,nullptr,0,astep);
      }
      // Keeps the A loads ahead of the weight refills (and, with EARLY_A, the
      // decode). Dense full tiles schedule better without it.
      if constexpr (ORDER) asm volatile("" ::: "memory");
      }
    };
    if constexpr (EARLY_A) load_a();
    const int next=(a+1)*T*4;
    GemmH16 bfs[EARLY_A ? 1 : NB];
    if constexpr (!EARLY_A) {
      // Exl3GemmBody's order: decode and exchange first, then A, then refill.
      uint32_t wv[NB][3];
#pragma unroll
      for (int n=0; n<NB; ++n) {
        wv[n][0]=raw[n][0]; wv[n][1]=raw[n][1]; wv[n][2]=raw[n][2];
        __half2 p[4];
        DecodeLane<K,kMulTrue16>(wv[n][0],wv[n][1],wv[n][2],win.shift,p);
        bfs[n]=GemmBFragment<true>(p,w,lane);
      }
      load_a();
    }
#pragma unroll
    for (int n=0; n<NB; ++n) {
      const uint32_t w0=raw[n][0], w1=raw[n][1], w2=raw[n][2];
      raw[n][0]=__builtin_amdgcn_raw_buffer_load_b32(rs[n],o0+next,0,0);
      raw[n][1]=__builtin_amdgcn_raw_buffer_load_b32(rs[n],o1+next,0,0);
      if constexpr (K>=5) raw[n][2]=__builtin_amdgcn_raw_buffer_load_b32(rs[n],o2+next,0,0);
      GemmH16 bf;
      if constexpr (EARLY_A) {
        __half2 p[4];
        DecodeLane<K,kMulTrue16>(w0,w1,w2,win.shift,p);
        bf=GemmBFragment<true>(p,w,lane);
      } else {
        bf=bfs[n];
      }
#pragma unroll
      for (int r=0; r<RM; ++r) {
        if (SKIP && it.m0+r*16>=j.M) break;
        if constexpr(STAGE_A) {
          GemmH16 af;auto* ap=reinterpret_cast<U4*>(&af);
          const auto* sp=reinterpret_cast<const U4*>(xs+(r*16+(lane&15))*A_STRIDE);
          ap[0]=sp[0];ap[1]=sp[1];
          acc[n][r]=__builtin_amdgcn_wmma_f32_16x16x16_f16_w32(af,bf,acc[n][r]);
#ifdef EXL3_STAGE_ORDER_A
          // Stop the scheduler from keeping all unrolled LDS A reads live at once.
          asm volatile("" : : "v"(acc[n][r]) : "memory");
#endif
        } else {
          acc[n][r]=__builtin_amdgcn_wmma_f32_16x16x16_f16_w32(afr[r],bf,acc[n][r]);
        }
      }
    }
    if constexpr(STAGE_A)GemmGroupSync();
  }
  if (keep) {
#pragma unroll
    for (int n=0; n<NB; ++n)
#pragma unroll
      for (int r=0; r<RM; ++r) keep[n*RM+r]=acc[n][r];
    return;
  }
  const auto ps=StreamRsrc(reinterpret_cast<const uint32_t*>(j.partial),(uint32_t)j.M*j.O*4);
#pragma unroll
  for (int n=0; n<NB; ++n) {
    const uint32_t po=((uint32_t)(it.m0+lane/16)*j.O+(b+n)*16+(lane&15))*4;
#pragma unroll
    for (int r=0; r<RM; ++r) {
      if (SKIP && it.m0+r*16>=j.M) break;
#pragma unroll
      for (int z=0; z<8; ++z)
        __builtin_amdgcn_raw_buffer_store_b32(__float_as_uint(acc[n][r][z]),ps,
                                              po+(uint32_t)(r*16+2*z)*j.O*4,0,0);
    }
  }
}

} // namespace crucible
