// Prefill HC and dense GEMM kernels that only benches launch: the first
// unfused HC stages and the HC dots and dense GEMM variants that the engine kernels replaced.
#pragma once
#include "../../engine/layers/prefill_hc.h"

namespace crucible {

template<class B> __device__ __forceinline__ prefill::F8 PrefillPackedProduct(int K,
    const __half* hi,const __half* lo,int ld,int offset,int R,int r0,B load_b) {
  const int r=r0+(threadIdx.x&15);
  prefill::F8 a={},l={};
#pragma unroll 1
  for (int k=0;k<K;k+=16) {
    const prefill::H16 ah=r<R ? PrefillLoad16(hi+(size_t)r*ld+offset+k) : prefill::H16{};
    const prefill::H16 al=r<R ? PrefillLoad16(lo+(size_t)r*ld+offset+k) : prefill::H16{};
    const prefill::H16 b=load_b(k);
    a=__builtin_amdgcn_wmma_f32_16x16x16_f16_w32(ah,b,a);
    l=__builtin_amdgcn_wmma_f32_16x16x16_f16_w32(al,b,l);
  }
  return a+l*(1.f/prefill::kResidualScale);
}

constexpr int kHcPad=kHcStep+8;
constexpr int kHcLStep=32, kHcLPad=kHcLStep+8, kHcCols=21*16;
constexpr int kDenseStep=32, kDensePad=kDenseStep+8;

__global__ void __launch_bounds__(256) PrefillHcSquaresKernel(const float* H,float* ss,int R) {
  const int w=blockIdx.x*8+(threadIdx.x>>5), lane=threadIdx.x&31;
  if (w>=R*kStreams) return;
  const float* h=H+(size_t)w*kHidden;
  float a=0;
  for (int d=lane;d<kHidden;d+=32) a+=h[d]*h[d];
  a=WaveSum(a); if (lane==0) ss[w]=a;
}

__global__ void __launch_bounds__(128) PrefillHcDotsKernel(const __half* fn,int M,
    const __half* hi,const __half* lo,float* dots,int R) {
  const int lane=threadIdx.x&31,wave=threadIdx.x>>5;
  const int r0=blockIdx.x*16,m0=blockIdx.y*64+wave*16,s=blockIdx.z;
  if (m0>=M) return;
  const int n=m0+(lane&15);
  const auto out=PrefillPackedProduct(kHidden,hi,lo,kWide,s*kHidden,R,r0,
    [&](int k) { return n<M ? PrefillLoad16(fn+(size_t)n*kWide+s*kHidden+k) : prefill::H16{}; });
#pragma unroll
  for (int z=0;z<8;++z) {
    const int r=r0+prefill::FragmentRow(z);
    if (r<R && n<M) dots[((size_t)r*M+n)*4+s]=out[z];
  }
}

__global__ void __launch_bounds__(256) PrefillHcMidKernel(const float* dots,int M,
    const float* ss,float* t,float* post,int R) {
  const int r=blockIdx.x,j=threadIdx.x;
  if (r>=R) return;
  float inv[4];
#pragma unroll
  for (int s=0;s<4;++s) inv[s]=rsqrtf(ss[r*4+s]/kHidden+kEps);
  for (int m=j;m<kRank;m+=256) {
    const float* d=dots+((size_t)r*M+m)*4;float a=0;
#pragma unroll
    for (int s=0;s<4;++s) a+=inv[s]*d[s];
    t[r*kRank+m]=Silu(a*.25f);
  }
  if (j<4 && post) {
    const float* d=dots+((size_t)r*M+kRank+j)*4;float a=0;
#pragma unroll
    for (int s=0;s<4;++s) a+=inv[s]*d[s];
    post[r*4+j]=2.f*Sigmoid(a*.25f);
  }
}

__global__ void __launch_bounds__(128) PrefillHcMixKernel(const __half* up,const float* w1,
    const __half* hi,const __half* lo,const float* ss,const float* H,float* mixed,int R) {
  __shared__ float gates[4][16][16];
  const int lane=threadIdx.x&31,s=threadIdx.x>>5,r0=blockIdx.x*16,d0=blockIdx.y*16;
  const auto out=PrefillPackedProduct(kRank,hi,lo,kRank,0,R,r0,
    [&](int k) { return PrefillLoad16(up+((size_t)s*kHidden+d0+(lane&15))*kRank+k); });
#pragma unroll
  for (int z=0;z<8;++z) gates[s][prefill::FragmentRow(z)][lane&15]=out[z];
  __syncthreads();
  for (int i=threadIdx.x;i<256;i+=128) {
    const int r=r0+i/16,d=d0+i%16;if (r>=R) continue;
    float v[4];
#pragma unroll
    for (int st=0;st<4;++st) {
      const size_t at=((size_t)r*4+st)*kHidden+d;
      const float inv=rsqrtf(ss[r*4+st]/kHidden+kEps);
      v[st]=Sigmoid(gates[st][i/16][i%16])*H[at]*inv*w1[st*kHidden+d];
    }
    mixed[(size_t)r*kHidden+d]=RoundH(((v[0]+v[1])+(v[2]+v[3]))*.25f);
  }
}

// Split variant: 4 waves own 12 column tiles each block, two blocks per row
// group, so twice as many blocks fill the 40 CUs.
__global__ void __launch_bounds__(128) PrefillHcDotsSplitKernel(const __half* fn,int M,
    const float* H,float* dots,float* ss,int R) {
  // The 8-half pad spreads the 16 A rows over distinct LDS banks.
  __shared__ __attribute__((aligned(16))) _Float16 ahi[kHcRows][kHcPad], alo[kHcRows][kHcPad];
  const int t=threadIdx.x,lane=t&31,wave=t>>5,r0=blockIdx.x*kHcRows,s=blockIdx.y,c0=blockIdx.z*12;
  const int sr=t>>2,sk=(t&3)*4;
  const bool live=r0+sr<R;
  const float* src=H+(size_t)(r0+sr)*kWide+s*kHidden+sk;
  float sq=0;
  prefill::F8 hi[2][3]={},lo[2][3]={};
  float4 next[4];
#pragma unroll
  for (int i=0;i<4;++i) next[i]=live ? *reinterpret_cast<const float4*>(src+i*16) : float4{};
#pragma unroll 1
  for (int k0=0;k0<kHidden;k0+=kHcStep) {
    __syncthreads();
#pragma unroll
    for (int i=0;i<4;++i) {
      const float v[4]={next[i].x,next[i].y,next[i].z,next[i].w};
#pragma unroll
      for (int e=0;e<4;++e) {
        sq+=v[e]*v[e];
        const _Float16 h=(_Float16)v[e];
        ahi[sr][sk+i*16+e]=h;
        alo[sr][sk+i*16+e]=(_Float16)((v[e]-(float)h)*prefill::kResidualScale);
      }
    }
    __syncthreads();
    if (k0+kHcStep<kHidden) {
#pragma unroll
      for (int i=0;i<4;++i) next[i]=live ? *reinterpret_cast<const float4*>(src+k0+kHcStep+i*16) : float4{};
    }
#pragma unroll
    for (int kb=0;kb<kHcStep;kb+=16) {
      prefill::H16 ah[2],al[2];
#pragma unroll
      for (int i=0;i<2;++i) {
        ah[i]=*reinterpret_cast<const prefill::H16*>(&ahi[i*16+(lane&15)][kb]);
        al[i]=*reinterpret_cast<const prefill::H16*>(&alo[i*16+(lane&15)][kb]);
      }
#pragma unroll
      for (int j=0;j<3;++j) {
        const int n=(c0+wave*3+j)*16+(lane&15);
        if ((c0+wave*3+j)*16>=M) continue;
        const prefill::H16 b=n<M ? PrefillLoad16(fn+(size_t)n*kWide+s*kHidden+k0+kb) : prefill::H16{};
#pragma unroll
        for (int i=0;i<2;++i) {
          hi[i][j]=__builtin_amdgcn_wmma_f32_16x16x16_f16_w32(ah[i],b,hi[i][j]);
          lo[i][j]=__builtin_amdgcn_wmma_f32_16x16x16_f16_w32(al[i],b,lo[i][j]);
        }
      }
    }
  }
  sq+=__shfl_xor(sq,1); sq+=__shfl_xor(sq,2);
  if ((t&3)==0 && live && blockIdx.z==0) ss[(r0+sr)*kStreams+s]=sq;
#pragma unroll
  for (int j=0;j<3;++j) {
    const int n=(c0+wave*3+j)*16+(lane&15);
    if ((c0+wave*3+j)*16>=M || n>=M) continue;
#pragma unroll
    for (int i=0;i<2;++i) {
      const prefill::F8 out=hi[i][j]+lo[i][j]*(1.f/prefill::kResidualScale);
#pragma unroll
      for (int z=0;z<8;++z) {
        const int r=r0+i*16+prefill::FragmentRow(z);
        if (r<R) dots[((size_t)r*M+n)*4+s]=out[z];
      }
    }
  }
}

// Variant of PrefillHcDotsFusedKernel: B rows past M are clamped to M-1 (never
// stored), so the three B loads of a k block issue together without branches.
template<bool UNROLL>
__global__ void __launch_bounds__(256) PrefillHcDotsGroupKernel(const __half* fn,int M,
    const float* H,float* dots,float* ss,int R) {
  __shared__ __attribute__((aligned(16))) _Float16 ahi[kHcRows][kHcPad], alo[kHcRows][kHcPad];
  const int t=threadIdx.x,lane=t&31,wave=t>>5,r0=blockIdx.x*kHcRows,s=blockIdx.y;
  const int sr=t>>3,sk=(t&7)*4;
  const bool live=r0+sr<R;
  const float* src=H+(size_t)(r0+sr)*kWide+s*kHidden+sk;
  const int tiles=wave<(M+15)/16-16 ? 3 : 2;
  const __half* bp[3];
#pragma unroll
  for (int j=0;j<3;++j) bp[j]=fn+(size_t)min((wave+8*j)*16+(lane&15),M-1)*kWide+s*kHidden;
  float sq=0;
  prefill::F8 hi[2][3]={},lo[2][3]={};
  float4 next[2];
#pragma unroll
  for (int i=0;i<2;++i) next[i]=live ? *reinterpret_cast<const float4*>(src+i*32) : float4{};
#pragma unroll 1
  for (int k0=0;k0<kHidden;k0+=kHcStep) {
    __syncthreads();
#pragma unroll
    for (int i=0;i<2;++i) {
      const float v[4]={next[i].x,next[i].y,next[i].z,next[i].w};
#pragma unroll
      for (int e=0;e<4;++e) {
        sq+=v[e]*v[e];
        const _Float16 h=(_Float16)v[e];
        ahi[sr][sk+i*32+e]=h;
        alo[sr][sk+i*32+e]=(_Float16)((v[e]-(float)h)*prefill::kResidualScale);
      }
    }
    __syncthreads();
    if (k0+kHcStep<kHidden) {
#pragma unroll
      for (int i=0;i<2;++i) next[i]=live ? *reinterpret_cast<const float4*>(src+k0+kHcStep+i*32) : float4{};
    }
    auto step=[&](int kb) {
      prefill::H16 b[3];
#pragma unroll
      for (int j=0;j<3;++j) b[j]=PrefillLoad16(bp[j]+k0+kb);
      prefill::H16 ah[2],al[2];
#pragma unroll
      for (int i=0;i<2;++i) {
        ah[i]=*reinterpret_cast<const prefill::H16*>(&ahi[i*16+(lane&15)][kb]);
        al[i]=*reinterpret_cast<const prefill::H16*>(&alo[i*16+(lane&15)][kb]);
      }
#pragma unroll
      for (int j=0;j<3;++j) {
        if (j>=tiles) continue;
#pragma unroll
        for (int i=0;i<2;++i) {
          hi[i][j]=__builtin_amdgcn_wmma_f32_16x16x16_f16_w32(ah[i],b[j],hi[i][j]);
          lo[i][j]=__builtin_amdgcn_wmma_f32_16x16x16_f16_w32(al[i],b[j],lo[i][j]);
        }
      }
    };
    if constexpr (UNROLL) {
#pragma unroll
      for (int kb=0;kb<kHcStep;kb+=16) step(kb);
    } else {
#pragma unroll 1
      for (int kb=0;kb<kHcStep;kb+=16) step(kb);
    }
  }
  sq+=__shfl_xor(sq,1); sq+=__shfl_xor(sq,2); sq+=__shfl_xor(sq,4);
  if ((t&7)==0 && live) ss[(r0+sr)*kStreams+s]=sq;
#pragma unroll
  for (int j=0;j<3;++j) {
    const int n=(wave+8*j)*16+(lane&15);
    if (j>=tiles || n>=M) continue;
#pragma unroll
    for (int i=0;i<2;++i) {
      const prefill::F8 out=hi[i][j]+lo[i][j]*(1.f/prefill::kResidualScale);
#pragma unroll
      for (int z=0;z<8;++z) {
        const int r=r0+i*16+prefill::FragmentRow(z);
        if (r<R) dots[((size_t)r*M+n)*4+s]=out[z];
      }
    }
  }
}

__global__ void __launch_bounds__(256) PrefillHcDotsLdsKernel(const __half* fn,int M,
    const float* H,float* dots,float* ss,int R) {
  __shared__ __attribute__((aligned(16))) _Float16 ahi[kHcRows][kHcLPad], alo[kHcRows][kHcLPad];
  __shared__ __attribute__((aligned(16))) _Float16 bs[kHcCols][kHcLPad];
  constexpr int kBParts=kHcCols*kHcLStep/8, kBLoads=(kBParts+255)/256;
  const int t=threadIdx.x,lane=t&31,wave=t>>5,r0=blockIdx.x*kHcRows,s=blockIdx.y;
  const int sr=t>>3,sk=(t&7)*4;
  const bool live=r0+sr<R;
  const float* src=H+(size_t)(r0+sr)*kWide+s*kHidden+sk;
  const __half* bsrc[kBLoads];
#pragma unroll
  for (int i=0;i<kBLoads;++i) {
    const int idx=min(t+i*256,kBParts-1),row=min(idx>>2,M-1);
    bsrc[i]=fn+(size_t)row*kWide+s*kHidden+(idx&3)*8;
  }
  const int tiles=wave<(M+15)/16-16 ? 3 : 2;
  float sq=0;
  prefill::F8 hi[2][3]={},lo[2][3]={};
  float4 na=live ? *reinterpret_cast<const float4*>(src) : float4{};
  uint4 nb[kBLoads];
#pragma unroll
  for (int i=0;i<kBLoads;++i) nb[i]=*reinterpret_cast<const uint4*>(bsrc[i]);
#pragma unroll 1
  for (int k0=0;k0<kHidden;k0+=kHcLStep) {
    __syncthreads();
    {
      const float v[4]={na.x,na.y,na.z,na.w};
#pragma unroll
      for (int e=0;e<4;++e) {
        sq+=v[e]*v[e];
        const _Float16 h=(_Float16)v[e];
        ahi[sr][sk+e]=h;
        alo[sr][sk+e]=(_Float16)((v[e]-(float)h)*prefill::kResidualScale);
      }
    }
#pragma unroll
    for (int i=0;i<kBLoads;++i) {
      const int idx=t+i*256;
      if (idx<kBParts) *reinterpret_cast<uint4*>(&bs[idx>>2][(idx&3)*8])=nb[i];
    }
    __syncthreads();
    if (k0+kHcLStep<kHidden) {
      na=live ? *reinterpret_cast<const float4*>(src+k0+kHcLStep) : float4{};
#pragma unroll
      for (int i=0;i<kBLoads;++i) nb[i]=*reinterpret_cast<const uint4*>(bsrc[i]+k0+kHcLStep);
    }
#pragma unroll
    for (int kb=0;kb<kHcLStep;kb+=16) {
      prefill::H16 ah[2],al[2];
#pragma unroll
      for (int i=0;i<2;++i) {
        ah[i]=*reinterpret_cast<const prefill::H16*>(&ahi[i*16+(lane&15)][kb]);
        al[i]=*reinterpret_cast<const prefill::H16*>(&alo[i*16+(lane&15)][kb]);
      }
#pragma unroll
      for (int j=0;j<3;++j) {
        if (j>=tiles) continue;
        const prefill::H16 b=*reinterpret_cast<const prefill::H16*>(&bs[(wave+8*j)*16+(lane&15)][kb]);
#pragma unroll
        for (int i=0;i<2;++i) {
          hi[i][j]=__builtin_amdgcn_wmma_f32_16x16x16_f16_w32(ah[i],b,hi[i][j]);
          lo[i][j]=__builtin_amdgcn_wmma_f32_16x16x16_f16_w32(al[i],b,lo[i][j]);
        }
      }
    }
  }
  sq+=__shfl_xor(sq,1); sq+=__shfl_xor(sq,2); sq+=__shfl_xor(sq,4);
  if ((t&7)==0 && live) ss[(r0+sr)*kStreams+s]=sq;
#pragma unroll
  for (int j=0;j<3;++j) {
    const int n=(wave+8*j)*16+(lane&15);
    if (j>=tiles || n>=M) continue;
#pragma unroll
    for (int i=0;i<2;++i) {
      const prefill::F8 out=hi[i][j]+lo[i][j]*(1.f/prefill::kResidualScale);
#pragma unroll
      for (int z=0;z<8;++z) {
        const int r=r0+i*16+prefill::FragmentRow(z);
        if (r<R) dots[((size_t)r*M+n)*4+s]=out[z];
      }
    }
  }
}

__global__ void __launch_bounds__(128) PrefillDenseLdsKernel(const __half* W,int O,
    const __half* x,float* y,int ldy,int R) {
  __shared__ __attribute__((aligned(16))) _Float16 as[2][64][kDensePad], bs[2][64][kDensePad];
  const int t=threadIdx.x,lane=t&31,wave=t>>5;
  const int r0=blockIdx.x*64,n0=blockIdx.y*64;
  const __half* ap[2]; const __half* bp[2];
#pragma unroll
  for (int i=0;i<2;++i) {
    const int idx=t+i*128,row=idx>>2,part=(idx&3)*8;
    ap[i]=x+(size_t)min(r0+row,R-1)*kHidden+part;
    bp[i]=W+(size_t)min(n0+row,O-1)*kHidden+part;
  }
  uint4 na[2],nb[2];
#pragma unroll
  for (int i=0;i<2;++i) { na[i]=*reinterpret_cast<const uint4*>(ap[i]); nb[i]=*reinterpret_cast<const uint4*>(bp[i]); }
  auto store=[&](int buf) {
#pragma unroll
    for (int i=0;i<2;++i) {
      const int idx=t+i*128,row=idx>>2,part=(idx&3)*8;
      *reinterpret_cast<uint4*>(&as[buf][row][part])=na[i];
      *reinterpret_cast<uint4*>(&bs[buf][row][part])=nb[i];
    }
  };
  store(0);
  __syncthreads();
  prefill::F8 acc[2][2]={};
  const int wr=(wave>>1)*32,wc=(wave&1)*32;
#pragma unroll 1
  for (int k0=0;k0<kHidden;k0+=kDenseStep) {
    const int buf=(k0/kDenseStep)&1;
    if (k0+kDenseStep<kHidden) {
#pragma unroll
      for (int i=0;i<2;++i) {
        na[i]=*reinterpret_cast<const uint4*>(ap[i]+k0+kDenseStep);
        nb[i]=*reinterpret_cast<const uint4*>(bp[i]+k0+kDenseStep);
      }
    }
#pragma unroll
    for (int kb=0;kb<kDenseStep;kb+=16) {
      prefill::H16 af[2],bf[2];
#pragma unroll
      for (int i=0;i<2;++i) {
        af[i]=*reinterpret_cast<const prefill::H16*>(&as[buf][wr+i*16+(lane&15)][kb]);
        bf[i]=*reinterpret_cast<const prefill::H16*>(&bs[buf][wc+i*16+(lane&15)][kb]);
      }
#pragma unroll
      for (int i=0;i<2;++i)
#pragma unroll
        for (int j=0;j<2;++j) acc[i][j]=__builtin_amdgcn_wmma_f32_16x16x16_f16_w32(af[i],bf[j],acc[i][j]);
    }
    if (k0+kDenseStep<kHidden) store(buf^1);
    __syncthreads();
  }
#pragma unroll
  for (int i=0;i<2;++i)
#pragma unroll
    for (int j=0;j<2;++j) {
      const int n=n0+wc+j*16+(lane&15);
      if (n>=O) continue;
#pragma unroll
      for (int z=0;z<8;++z) {
        const int r=r0+wr+i*16+prefill::FragmentRow(z);
        if (r<R) y[(size_t)r*ldy+n]=acc[i][j][z];
      }
    }
}

// HC and PLE inputs already lie on fp16 boundaries.
__global__ void __launch_bounds__(128) PrefillDenseKernel(const __half* W,int O,
    const __half* x,float* y,int ldy,int R) {
  const int lane=threadIdx.x&31,wave=threadIdx.x>>5;
  const int r0=blockIdx.x*16,n0=blockIdx.y*64+wave*16,n=n0+(lane&15),r=r0+(lane&15);
  if (n0>=O) return;
  prefill::F8 acc={};
#pragma unroll 1
  for (int k=0;k<kHidden;k+=16) {
    const auto a=r<R ? PrefillLoad16(x+(size_t)r*kHidden+k) : prefill::H16{};
    const auto b=n<O ? PrefillLoad16(W+(size_t)n*kHidden+k) : prefill::H16{};
    acc=__builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a,b,acc);
  }
#pragma unroll
  for (int z=0;z<8;++z) {
    const int row=r0+prefill::FragmentRow(z);
    if (row<R && n<O) y[(size_t)row*ldy+n]=acc[z];
  }
}

}  // namespace crucible
