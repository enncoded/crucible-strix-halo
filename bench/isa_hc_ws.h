// Bench copy of prefill-glue a38b9f7 PrefillHcDotsWsKernel.
// POLICY 0 keeps the source schedule. POLICY 1 bounds the B load window.
// POLICY 2 also prevents the next B fragment from crossing current WMMAs.
#pragma once
#ifndef ISA_HC_REFERENCE
#error "Set ISA_HC_REFERENCE to the quoted path of a pinned engine/layers/prefill_hc.h"
#endif
#include ISA_HC_REFERENCE

namespace crucible {
template<int D=2,int POLICY=0,int RT=4,int FNSTRIDE=kWide,bool FULL_ROWS=false,int FIXED_M=0>
__global__ void __launch_bounds__(288) IsaHcWsKernel(const __half* fn,int M_input,
    const float* H,float* dots,float* ss,int R,int ldh) {
  const int M=FIXED_M ? FIXED_M : M_input;
  typedef _Float16 H4 __attribute__((ext_vector_type(4)));
  constexpr int STEP=64,LW=2,kPad=STEP+8,kI=STEP/32,kSteps=kHidden/STEP;
  constexpr int G=8/LW,TPR=8/G,ROWS=16*RT,NP=RT/2;
  static_assert(kSteps%D==0 && RT%2==0);
  __shared__ __attribute__((aligned(16))) _Float16 ahi[2][ROWS][kPad];
  const int t=threadIdx.x,lane=t&31,wave=t>>5;
  const int r0=(blockIdx.x>>2)*ROWS,s=blockIdx.x&3;
  if (wave>=kHcWsCompute) {
    const int u=t-kHcWsCompute*32,sr=u/TPR,p=u%TPR;
    bool live[NP]; const float* src[NP];
#pragma unroll
    for (int pp=0;pp<NP;++pp) {
      live[pp]=FULL_ROWS || r0+sr+32*pp<R;
      src[pp]=H+(size_t)(r0+sr+32*pp)*ldh+s*kHidden+4*G*p;
    }
    float sq[NP][G]={};
    float4 v[D][NP][kI][G];
    auto load=[&](float4 (&w)[NP][kI][G],int step) {
#pragma unroll
      for (int pp=0;pp<NP;++pp)
#pragma unroll
        for (int i=0;i<kI;++i)
#pragma unroll
          for (int h=0;h<G;++h)
            w[pp][i][h]=live[pp] ? *reinterpret_cast<const float4*>(src[pp]+step*STEP+i*32+4*h) : float4{};
    };
    auto store=[&](const float4 (&w)[NP][kI][G],int buf) {
#pragma unroll
      for (int pp=0;pp<NP;++pp)
#pragma unroll
        for (int i=0;i<kI;++i)
#pragma unroll
          for (int h=0;h<G;++h) {
            const float x[4]={w[pp][i][h].x,w[pp][i][h].y,w[pp][i][h].z,w[pp][i][h].w};
            H4 hi4;
#pragma unroll
            for (int e=0;e<4;++e) {
              sq[pp][h]+=x[e]*x[e];
              hi4[e]=(_Float16)x[e];
            }
            *reinterpret_cast<H4*>(&ahi[buf][sr+32*pp][4*G*p+32*i+4*h])=hi4;
          }
    };
#pragma unroll
    for (int d=0;d<D;++d) load(v[d],d);
    store(v[0],0);
    if (D<kSteps) load(v[0],D);
    PrefillLdsSync();
#pragma unroll 1
    for (int s0=0;s0<kSteps;s0+=D)
#pragma unroll
      for (int d=0;d<D;++d) {
        const int step=s0+d;
        if (step+1<kSteps) {
          store(v[(d+1)%D],(step+1)&1);
          if (step+1+D<kSteps) load(v[(d+1)%D],step+1+D);
        }
        PrefillLdsSync();
      }
#pragma unroll
    for (int pp=0;pp<NP;++pp) {
      float q[G];
#pragma unroll
      for (int h=0;h<G;++h) q[h]=sq[pp][h];
#pragma unroll
      for (int w=1;w<G;w<<=1)
#pragma unroll
        for (int h=0;h<G;h+=2*w) q[h]=q[h]+q[h+w];
      float a=q[0];
#pragma unroll
      for (int w=1;w<TPR;w<<=1) a+=__shfl_xor(a,w);
      if (p==0 && live[pp]) ss[(r0+sr+32*pp)*kStreams+s]=a;
    }
    return;
  }
  const __half* bp[3];
#pragma unroll
  for (int j=0;j<3;++j) {
    const int n=min((wave+kHcWsCompute*j)*16+(lane&15),M-1);
    bp[j]=fn+(size_t)n*FNSTRIDE+s*kHidden;
  }
  prefill::F8 hi[RT][3]={};
  prefill::H16 b[3];
#pragma unroll
  for (int j=0;j<3;++j) b[j]=PrefillLoad16(bp[j]);
  PrefillLdsSync();
#pragma unroll 1
  for (int step=0;step<kSteps;++step) {
    const int buf=step&1;
#pragma unroll
    for (int kb=0;kb<STEP;kb+=16) {
      const int k=step*STEP+kb,kn=k+16<kHidden ? k+16 : k;
      prefill::H16 bn[3];
#pragma unroll
      for (int j=0;j<3;++j) bn[j]=PrefillLoad16(bp[j]+kn);
      if constexpr (POLICY==2) __builtin_amdgcn_sched_barrier(0);
      prefill::H16 ah[RT];
#pragma unroll
      for (int i=0;i<RT;++i)
        ah[i]=*reinterpret_cast<const prefill::H16*>(&ahi[buf][i*16+(lane&15)][kb]);
#pragma unroll
      for (int j=0;j<3;++j)
#pragma unroll
        for (int i=0;i<RT;++i)
          hi[i][j]=__builtin_amdgcn_wmma_f32_16x16x16_f16_w32(ah[i],b[j],hi[i][j]);
#pragma unroll
      for (int j=0;j<3;++j) b[j]=bn[j];
      // A scheduler fence does not change memory semantics or WMMA order.
      if constexpr (POLICY!=0) __builtin_amdgcn_sched_barrier(0);
    }
    PrefillLdsSync();
  }
#pragma unroll
  for (int j=0;j<3;++j) {
    const int n=(wave+kHcWsCompute*j)*16+(lane&15);
    if (n>=M) continue;
#pragma unroll
    for (int i=0;i<RT;++i)
#pragma unroll
      for (int z=0;z<8;++z) {
        const int r=r0+i*16+prefill::FragmentRow(z);
        if (FULL_ROWS || r<R) dots[((size_t)r*M+n)*4+s]=hi[i][j][z];
      }
  }
}
}  // namespace crucible
