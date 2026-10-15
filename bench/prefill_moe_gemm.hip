// MoE expert gate/up GEMM in isolation (random top-10 routing, cold random
// K3 weights): the engine kernel, then copies of its body that remove one
// cost at a time, to find what bounds the kernel (prefill_moe_gemm R d), and
// the separate input transform + gate/up GEMM versus the fused queue kernel
// (prefill_moe_gemm R).
#include "prefill_common.h"
#include "../engine/layers/prefill_moe.h"
#include "../engine/exl3/kernels/exl3_gemm_pair.h"
#include "kernels/exl3_gemm_stage.h"
using namespace crucible;
using namespace prefill_bench;

// Exl3GemmBody<3,RM,LDS,!STAGE_X,PF1,REGISTER_A,PACKED_X,SKIP_PAD_RM,STRICT_A_ORDER>
// with knockouts: D&1 no trellis decode, D&2 no A loads, D&4 no weight loads,
// D&8 no B exchange through LDS, D&16 no WMMA, D&32 no output store.
template<int RM,int D> __device__ __forceinline__ void DiagBody(const Exl3GemmJob& j,
    const Exl3GemmItem& it,uint32_t* ws) {
  constexpr int K=3,T=8*K;
  using U4=uint32_t __attribute__((ext_vector_type(4)));
  const int lane=threadIdx.x&31,wave=__builtin_amdgcn_readfirstlane(threadIdx.x>>5);
  const int b=it.b0+wave;
  if (b*16>=j.O) return;
  const int A=j.I/16;
  const LaneWindow<K> win(lane);
  const auto rs=StreamRsrc(j.tiles+(size_t)b*A*T,A*T*4);
  GemmF8 acc[RM]={};
  const uint32_t xbytes=(uint32_t)j.Xrows*j.I*2;
  const auto xrs=StreamRsrc(reinterpret_cast<const uint32_t*>(j.x),xbytes);
  const int astep=j.Xrows*32;
  uint32_t xo[RM];
#pragma unroll
  for (int r=0; r<RM; ++r) {
    const int row=it.m0+r*16+(lane&15);
    xo[r]=row<j.M ? (uint32_t)row*32 : xbytes;
  }
  uint32_t* w=ws+wave*kGemmWaveWords;
  const int o0=win.d0*4,o1=((win.d0+1)%T)*4;
  uint32_t r0=__builtin_amdgcn_raw_buffer_load_b32(rs,o0,0,0);
  uint32_t r1=__builtin_amdgcn_raw_buffer_load_b32(rs,o1,0,0);
  for (int a=0; a<A; ++a) {
    const uint32_t w0=r0,w1=r1;
    const int next=(a+1)*T*4;
    __half2 p[4];
    if constexpr ((D&1)!=0) {
#pragma unroll
      for (int z=0; z<4; ++z) { const uint32_t v=(w0^(w1<<z))&0x37ff37ffu; p[z]=*reinterpret_cast<const __half2*>(&v); }
    } else {
      DecodeLane<K,kMulTrue16>(w0,w1,0,win.shift,p);
    }
    GemmH16 bf;
    if constexpr ((D&8)!=0) {
      uint32_t* bp=reinterpret_cast<uint32_t*>(&bf);
#pragma unroll
      for (int z=0; z<8; ++z) bp[z]=*reinterpret_cast<const uint32_t*>(&p[z&3]);
    } else {
      bf=GemmBFragment<true>(p,w,lane);
    }
    GemmH16 afr[RM];
#pragma unroll
    for (int r=0; r<RM; ++r) {
      if (it.m0+r*16>=j.M) break;
      if constexpr ((D&2)!=0) {
        const uint32_t v=((uint32_t)(a+r)*0x00010001u)&0x33ff33ffu;
        auto* ap=reinterpret_cast<uint32_t*>(&afr[r]);
#pragma unroll
        for (int z=0; z<8; ++z) ap[z]=v;
      } else {
        afr[r]=GemmAFragment<false>(xrs,xo[r],a,nullptr,0,astep);
      }
    }
    asm volatile("" ::: "memory");
    if constexpr ((D&4)!=0) {
      r0=w0*0x9E3779B9u+(uint32_t)a; r1=w1^r0;
    } else {
      r0=__builtin_amdgcn_raw_buffer_load_b32(rs,o0+next,0,0);
      r1=__builtin_amdgcn_raw_buffer_load_b32(rs,o1+next,0,0);
    }
#pragma unroll
    for (int r=0; r<RM; ++r) {
      if (it.m0+r*16>=j.M) break;
      if constexpr ((D&16)!=0) {
        const auto* ap=reinterpret_cast<const _Float16*>(&afr[r]);
        const auto* bq=reinterpret_cast<const _Float16*>(&bf);
        acc[r][0]+=(float)ap[0]*(float)bq[0];
      } else {
        acc[r]=__builtin_amdgcn_wmma_f32_16x16x16_f16_w32(afr[r],bf,acc[r]);
      }
    }
  }
  if constexpr ((D&32)!=0) {
    float sum=0.f;
#pragma unroll
    for (int r=0; r<RM; ++r) sum+=acc[r][0]+acc[r][7];
    if (sum!=1234.5f) return;
  }
  const auto ps=StreamRsrc(reinterpret_cast<const uint32_t*>(j.partial),(uint32_t)j.M*j.O*4);
  const uint32_t po=((uint32_t)(it.m0+lane/16)*j.O+b*16+(lane&15))*4;
#pragma unroll
  for (int r=0; r<RM; ++r) {
    if (it.m0+r*16>=j.M) break;
#pragma unroll
    for (int z=0; z<8; ++z)
      __builtin_amdgcn_raw_buffer_store_b32(__float_as_uint(acc[r][z]),ps,po+(uint32_t)(r*16+2*z)*j.O*4,0,0);
  }
}
__global__ void WhtCheckKernel(const float* in,int* bad) {
  float a[4],b[4];
  for (int k=0; k<4; ++k) a[k]=b[k]=in[blockIdx.x*128+threadIdx.x*4+k];
  Wht128(a); Wht128Dpp(b);
  for (int k=0; k<4; ++k) if (__float_as_uint(a[k])!=__float_as_uint(b[k])) atomicAdd(bad,1);
}
// NB column tiles per wave: each A fragment feeds NB WMMAs. An item covers
// 64 * NB columns (b0 steps by 4 * NB).
template<int RM,int NB,int D=0> __device__ __forceinline__ void WideBody(const Exl3GemmJob& j,
    const Exl3GemmItem& it,uint32_t* ws) {
  constexpr int K=3,T=8*K;
  const int lane=threadIdx.x&31,wave=__builtin_amdgcn_readfirstlane(threadIdx.x>>5);
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
  uint32_t* w=ws+wave*kGemmWaveWords;
  const int o0=win.d0*4,o1=((win.d0+1)%T)*4;
  uint32_t r0[NB],r1[NB];
#pragma unroll
  for (int n=0; n<NB; ++n) { r0[n]=__builtin_amdgcn_raw_buffer_load_b32(rs[n],o0,0,0); r1[n]=__builtin_amdgcn_raw_buffer_load_b32(rs[n],o1,0,0); }
  for (int a=0; a<A; ++a) {
    GemmH16 afr[RM];
#pragma unroll
    for (int r=0; r<RM; ++r) {
      if (it.m0+r*16>=j.M) break;
      if constexpr ((D&2)!=0) {
        const uint32_t v=((uint32_t)(a+r)*0x00010001u)&0x33ff33ffu;
        auto* ap=reinterpret_cast<uint32_t*>(&afr[r]);
#pragma unroll
        for (int z=0; z<8; ++z) ap[z]=v;
      } else afr[r]=GemmAFragment<false>(xrs,xo[r],a,nullptr,0,astep);
    }
    asm volatile("" ::: "memory");
    const int next=(a+1)*T*4;
#pragma unroll
    for (int n=0; n<NB; ++n) {
      const uint32_t w0=r0[n],w1=r1[n];
      if constexpr ((D&4)!=0) { r0[n]=w0*0x9E3779B9u+(uint32_t)a; r1[n]=w1^r0[n]; }
      else {
        r0[n]=__builtin_amdgcn_raw_buffer_load_b32(rs[n],o0+next,0,0);
        r1[n]=__builtin_amdgcn_raw_buffer_load_b32(rs[n],o1+next,0,0);
      }
      __half2 p[4];
      if constexpr ((D&1)!=0) {
#pragma unroll
        for (int z=0; z<4; ++z) { const uint32_t v=(w0^(w1<<z))&0x37ff37ffu; p[z]=*reinterpret_cast<const __half2*>(&v); }
      } else DecodeLane<K,kMulTrue16>(w0,w1,0,win.shift,p);
      GemmH16 bf;
      if constexpr ((D&8)!=0) {
        uint32_t* bp=reinterpret_cast<uint32_t*>(&bf);
#pragma unroll
        for (int z=0; z<8; ++z) bp[z]=*reinterpret_cast<const uint32_t*>(&p[z&3]);
      } else bf=GemmBFragment<true>(p,w,lane);
#pragma unroll
      for (int r=0; r<RM; ++r) {
        if (it.m0+r*16>=j.M) break;
        if constexpr ((D&16)!=0) {
          const auto* ap=reinterpret_cast<const _Float16*>(&afr[r]);
          const auto* bq=reinterpret_cast<const _Float16*>(&bf);
          acc[n][r][0]+=(float)ap[0]*(float)bq[0];
        } else acc[n][r]=__builtin_amdgcn_wmma_f32_16x16x16_f16_w32(afr[r],bf,acc[n][r]);
      }
    }
  }
  const auto ps=StreamRsrc(reinterpret_cast<const uint32_t*>(j.partial),(uint32_t)j.M*j.O*4);
#pragma unroll
  for (int n=0; n<NB; ++n) {
    const uint32_t po=((uint32_t)(it.m0+lane/16)*j.O+(b+n)*16+(lane&15))*4;
#pragma unroll
    for (int r=0; r<RM; ++r) {
      if (it.m0+r*16>=j.M) break;
#pragma unroll
      for (int z=0; z<8; ++z)
        __builtin_amdgcn_raw_buffer_store_b32(__float_as_uint(acc[n][r][z]),ps,po+(uint32_t)(r*16+2*z)*j.O*4,0,0);
    }
  }
}
template<int RM,int NB,int D=0> __global__ void __launch_bounds__(128) WideKernel(const Exl3GemmJob* jobs,
    const Exl3GemmItem* items,const int* count) {
  __shared__ uint32_t ws[kGemmWeightWords];
  const int n=PrefillUniform(*count);
  for (int i=blockIdx.x; i<n; i+=gridDim.x) {
    const auto it=PrefillUniform(items[i]); const auto j=PrefillUniform(jobs[it.job]);
    WideBody<RM,NB,D>(j,it,ws);
    __syncthreads();
  }
}
// The engine body (PrefillWideBody) with and without the A order clobber.
template<int RM,bool ORDER> __global__ void __launch_bounds__(128) EngWideKernel(const Exl3GemmJob* jobs,
    const Exl3GemmItem* items,const int* count) {
  __shared__ uint32_t ws[kGemmWeightWords];
  const int n=PrefillUniform(*count);
  for (int i=blockIdx.x; i<n; i+=gridDim.x) {
    const auto it=PrefillUniform(items[i]); const auto j=PrefillUniform(jobs[it.job]);
    PrefillWideBody<3,RM,2,true,true,ORDER>(j,it,ws);
    __syncthreads();
  }
}
// Row bounds are invariant across K. Dispatch the live count once so the
// inner loop has no repeated skip checks and never reserves ghost fragments.
template<int MAX_RM,bool ORDER> __global__ void __launch_bounds__(128) LiveKernel(
    const Exl3GemmJob* jobs,const Exl3GemmItem* items,const int* count) {
  __shared__ uint32_t ws[kGemmWeightWords];
  const int n=PrefillUniform(*count);
  for (int i=blockIdx.x; i<n; i+=gridDim.x) {
    const auto it=PrefillUniform(items[i]); const auto j=PrefillUniform(jobs[it.job]);
    const int live=min(MAX_RM,(j.M-it.m0+15)/16);
#define LIVE(R) case R: if constexpr (MAX_RM>=R) PrefillWideBody<3,R,2,true,false,ORDER>(j,it,ws); break
    switch (live) { LIVE(1); LIVE(2); LIVE(3); LIVE(4); LIVE(5); LIVE(6); }
#undef LIVE
    __syncthreads();
  }
}
template<int RM,bool ORDER> __global__ void __launch_bounds__(128) PairKernel(
    const Exl3GemmJob* jobs,const Exl3GemmItem* items,const int* count) {
  __shared__ uint32_t ws[kGemmPairWords];
  const int n=PrefillUniform(*count);
  for(int i=blockIdx.x;i<n;i+=gridDim.x) {
    const auto it=PrefillUniform(items[i]);const auto j=PrefillUniform(jobs[it.job]);
    GemmPairBody<3,RM,ORDER>(j,it,ws);
    __syncthreads();
  }
}
template<int RM,int STRIDE=8> __global__ void __launch_bounds__(128) StageWideKernel(
    const Exl3GemmJob* jobs,const Exl3GemmItem* items,const int* count) {
  __shared__ uint32_t ws[kGemmWeightWords+RM*16*STRIDE];
  const int n=PrefillUniform(*count);
  for(int i=blockIdx.x;i<n;i+=gridDim.x) {
    const auto it=PrefillUniform(items[i]);const auto j=PrefillUniform(jobs[it.job]);
    GemmStageBody<3,RM,2,true,true,true,true,STRIDE>(j,it,ws);
    __syncthreads();
  }
}
// Engine body with A staged through LDS once per block (STAGE_X).
template<int RM> __global__ void __launch_bounds__(128) StageKernel(const Exl3GemmJob* jobs,
    const Exl3GemmItem* items,const int* count) {
  __shared__ uint32_t ws[kGemmWeightWords+RM*16*kGemmAStride];
  const int n=PrefillUniform(*count);
  for (int i=blockIdx.x; i<n; i+=gridDim.x) {
    const auto it=PrefillUniform(items[i]); const auto j=PrefillUniform(jobs[it.job]);
    Exl3GemmBody<3,RM,true,true,1,false,true,true,false>(j,it,ws);
    __syncthreads();
  }
}

template<int RM,int D> __global__ void __launch_bounds__(128) DiagKernel(const Exl3GemmJob* jobs,
    const Exl3GemmItem* items,const int* count) {
  __shared__ uint32_t ws[kGemmWeightWords];
  const int n=PrefillUniform(*count);
  for (int i=blockIdx.x; i<n; i+=gridDim.x) {
    const auto it=PrefillUniform(items[i]); const auto j=PrefillUniform(jobs[it.job]);
    DiagBody<RM,D>(j,it,ws);
    __syncthreads();
  }
}

template<class Run> void Diagnose(Run run,Buffer<Exl3GemmJob>& dj,Buffer<Exl3GemmItem>& di,Buffer<int>& dn,
    const std::vector<Exl3GemmJob>& jobs,Buffer<__half>& xg,Buffer<__half>& xu,Buffer<float>& pg,size_t total) {
  const auto ref=pg.Get();
  run("diag0",[&] { DiagKernel<8,0><<<2048,128>>>(dj.p,di.p,dn.p); });
  if (pg.Get()!=ref) std::printf("diag0 differs from the engine kernel\n");
  run("nodecode",[&] { DiagKernel<8,1><<<2048,128>>>(dj.p,di.p,dn.p); });
  run("noA",[&] { DiagKernel<8,2><<<2048,128>>>(dj.p,di.p,dn.p); });
  run("noW",[&] { DiagKernel<8,4><<<2048,128>>>(dj.p,di.p,dn.p); });
  run("noBx",[&] { DiagKernel<8,8><<<2048,128>>>(dj.p,di.p,dn.p); });
  run("nowmma",[&] { DiagKernel<8,16><<<2048,128>>>(dj.p,di.p,dn.p); });
  run("nostore",[&] { DiagKernel<8,32><<<2048,128>>>(dj.p,di.p,dn.p); });
  {
    // Every expert reads the same A rows, so A stays in L2: the DRAM part of
    // the A cost.
    auto hot=jobs;
    for (auto& j:hot) j.x=(j.x>=xu.p && j.x<xu.p+total*kHidden) ? xu.p : xg.p;
    Buffer<Exl3GemmJob> hj(hot);
    run("hotA",[&] { DiagKernel<8,0><<<2048,128>>>(hj.p,di.p,dn.p); });
    run("hotA+nost",[&] { DiagKernel<8,32><<<2048,128>>>(hj.p,di.p,dn.p); });
  }
  run("dec+Bx",[&] { DiagKernel<8,9><<<2048,128>>>(dj.p,di.p,dn.p); });
  run("onlyWMMA",[&] { DiagKernel<8,15><<<2048,128>>>(dj.p,di.p,dn.p); });
  run("onlyA",[&] { DiagKernel<8,29><<<2048,128>>>(dj.p,di.p,dn.p); });
  run("onlyW",[&] { DiagKernel<8,27><<<2048,128>>>(dj.p,di.p,dn.p); });
  run("onlyDec",[&] { DiagKernel<8,22><<<2048,128>>>(dj.p,di.p,dn.p); });
}

// Separate transform + GEMM (engine path) versus PrefillMoeFxKernel, with
// the real plan kernels; outputs must be bitwise equal.
void Fused(int R,int RM,std::mt19937& rng,const std::vector<int>& count,const uint32_t* tiles,size_t words) {
  const int rows=R*kSlots, routes=R*kTopK;
  std::vector<int> sel(routes),off(kExperts+1,0),pairs;
  {
    std::vector<int> ex(kExperts); std::iota(ex.begin(),ex.end(),0);
    for (int r=0; r<R; ++r) {
      for (int k=0; k<kTopK; ++k) std::swap(ex[k],ex[k+rng()%(kExperts-k)]);
      for (int k=0; k<kTopK; ++k) sel[r*kTopK+k]=ex[k];
    }
  }
  std::vector<int> cnt(kExperts,0);
  for (int v:sel) ++cnt[v];
  for (int e=0; e<kExperts; ++e) {
    off[e+1]=off[e]+cnt[e];
    for (int i=0; i<routes; ++i) if (sel[i]==e) pairs.push_back(i);
  }
  std::vector<int> ids(routes); for (int i=0; i<routes; ++i) ids[i]=pairs[i]/kTopK;
  Buffer<int> dsel(sel),dcnt(cnt),doff(off),dpairs(pairs),dids(ids),tc(kExperts*((routes+255)/256));
  Buffer<__half> xh(Half(Random((size_t)R*kHidden,.5f,21)));
  std::vector<float> sv=Random((size_t)(kExperts+1)*3*kHidden,.004f,23);
  for (auto& v:sv) v=(v<0 ? -1.f : 1.f)*(.012f+std::fabs(v));
  Buffer<__half> suh(Half(sv)),svh((size_t)(kExperts+1)*3*kHidden);
  std::vector<ExpertPtrs> ep(kExperts);
  auto lin=[&](int e,int w) {
    Lin l; l.I=kHidden; l.O=kMoeInter; l.K=3; l.tiles=const_cast<uint32_t*>(tiles)+(size_t)(2*e+(w&1))*words;
    l.suh=suh.p+(size_t)(e*3+w)*kHidden; l.svh=svh.p+(size_t)(e*3+w)*kHidden; return l;
  };
  for (int e=0; e<kExperts; ++e)
    for (int w=0; w<3; ++w) { const Lin l=lin(e,w); ep[e].tiles[w]=l.tiles; ep[e].suh[w]=l.suh; ep[e].svh[w]=l.svh; }
  Buffer<ExpertPtrs> dep(ep);
  const Lin sg=lin(kExperts,0),su=lin(kExperts,1),sd=lin(kExperts,2);
  const size_t items=((rows+31)/32+kExperts+1)*40;
  Buffer<LinJob> gu((kExperts+1)*2),down(kExperts+1);
  Buffer<Exl3GemmJob> g_gu((kExperts+1)*2),g_down(kExperts+1);
  Buffer<Exl3GemmItem> i_gu(items),i_down(items);
  Buffer<int> ic(2),inv(routes),fx_items(2*((rows+15)/16+kExperts+1)),fx_uoff(kExperts+1),fx_xoff(kExperts+1);
  Buffer<int> fx_ready((rows+15)/16+kExperts+1),fx_count(3);
  Buffer<float> pg((size_t)rows*kMoeInter),pu((size_t)rows*kMoeInter),pd((size_t)rows*kHidden);
  Buffer<__half> xg((size_t)rows*kHidden),xu((size_t)rows*kHidden),xd((size_t)rows*kMoeInter);
  PrefillMoe a{};
  a.dispatch={dsel.p,tc.p,dcnt.p,doff.p,dpairs.p,dids.p,R};
  a.gu=gu.p; a.down=down.p; a.g_gu=g_gu.p; a.g_down=g_down.p; a.i_gu=i_gu.p; a.i_down=i_down.p;
  a.item_counts=ic.p; a.inverse=inv.p; a.tile_rows=16*RM; a.packed=true;
  a.pg=pg.p; a.pu=pu.p; a.pd=pd.p; a.xg=xg.p; a.xu=xu.p; a.xd=xd.p; a.x=nullptr; a.xh=xh.p;
  a.fx_items=fx_items.p; a.fx_uoff=fx_uoff.p; a.fx_xoff=fx_xoff.p; a.fx_ready=fx_ready.p; a.fx_count=fx_count.p;
  auto plan=[&] { HIP_OK(hipMemsetAsync(ic.p,0,2*sizeof(int),0)); PrefillMoePlanKernel<<<1,512>>>(a,dep.p,sg,su,sd); };
  auto xform=[&] { PrefillMoeInputPackedKernel<<<dim3(40,(rows+15)/16),256>>>(a); };
  auto gemm=[&] { LaunchPrefillMoeGemm({RM,true},a.g_gu,a.i_gu,a.item_counts); };
  auto fplan=[&] { HIP_OK(hipMemsetAsync(ic.p,0,2*sizeof(int),0)); PrefillMoeFxPlanKernel<<<1,512>>>(a,dep.p,sg,su,sd); };
  auto fused=[&] { fplan(); LaunchPrefillMoeFx(RM,a); };
  plan(); xform(); gemm(); CheckLaunch();
  std::fprintf(stderr,"separate path done\n");
  const auto rg=pg.Get(),ru=pu.Get(); const auto rxg=xg.Get(),rxu=xu.Get();
  const double t_plan=Time(plan,7),t_x=Time(xform,7),t_g=Time(gemm,7);
  const double t_sep=Time([&] { plan(); xform(); gemm(); },7);
  pg.Zero(); pu.Zero(); xg.Zero(); xu.Zero();
  fplan(); CheckLaunch();
  {
    const auto c=fx_count.Get(); const auto n=ic.Get();
    std::fprintf(stderr,"fx plan: X items %d, G items %d, down items %d\n",c[2],n[0],n[1]);
  }
  LaunchPrefillMoeFx(RM,a); CheckLaunch();
  const bool same=pg.Get()==rg && pu.Get()==ru && xg.Get()==rxg && xu.Get()==rxu;
  const double t_f=Time(fused,7),t_fp=Time(fplan,7);
  pg.Zero(); fused(); CheckLaunch();
  const bool again=pg.Get()==rg;
  std::printf("R=%d rm=%d plan=%.1f xform=%.1f gemm=%.1f sep=%.1f | fxplan=%.1f fused=%.1f us bitwise=%d repeat=%d\n",
              R,RM,t_plan,t_x,t_g,t_sep,t_fp,t_f,(int)same,(int)again);
}

int main(int argc,char** argv) {
  const int R=argc>1 ? std::atoi(argv[1]) : 4096;
  if (R<16 || R>8192) return 2;
  const int RM=R>=4096 ? 8 : R>=2048 ? 4 : 2, Tm=16*RM;
  // Random top-10 routing; the shared expert takes all R rows.
  std::vector<int> count(kExperts+1,0);
  std::mt19937 rng(7);
  std::vector<int> ex(kExperts); std::iota(ex.begin(),ex.end(),0);
  for (int r=0; r<R; ++r) {
    for (int k=0; k<kTopK; ++k) std::swap(ex[k],ex[k+rng()%(kExperts-k)]);
    for (int k=0; k<kTopK; ++k) ++count[ex[k]];
  }
  count[kExperts]=R;
  const size_t words=(size_t)kHidden*kMoeInter*3/32, total=(size_t)R*kSlots;
  std::vector<uint32_t> tw(words*2*(kExperts+1));
  for (auto& v:tw) v=rng();
  Buffer<uint32_t> tiles(tw);
  // "down" as the third argument: I = 640, O = 2560, one matrix per expert.
  const bool dn_mode=argc>3 && argv[3][0]=='d';
  const int I_=dn_mode ? kMoeInter : kHidden, O_=dn_mode ? kHidden : kMoeInter, mats=dn_mode ? 1 : 2;
  Buffer<__half> xg(Half(Random(total*kHidden,.05f,11))),xu(Half(Random(total*kHidden,.05f,13)));
  Buffer<float> pg(total*kHidden),pu(total*kHidden);
  std::vector<Exl3GemmJob> jobs; std::vector<Exl3GemmItem> items;
  double flops=0;
  size_t off=0;
  for (int e=0; e<=kExperts; ++e) {
    const int M=count[e];
    for (int w=0; w<2; ++w)
      jobs.push_back({tiles.p+(size_t)(2*e+w)*words,(w ? xu.p : xg.p)+off*kHidden,
                      (w ? pu.p : pg.p)+off*O_,nullptr,I_,O_,M,1,3,M});
    for (int m=0; m<M; m+=Tm)
      for (int b=0; b<O_/64; ++b) for (int w=0; w<mats; ++w) items.push_back({2*e+w,0,m,b*4});
    flops+=2.*mats*M*kHidden*kMoeInter;
    off+=M;
  }
  Buffer<Exl3GemmJob> dj(jobs); Buffer<Exl3GemmItem> di(items);
  Buffer<int> dn(std::vector<int>{(int)items.size()});
  // The engine kernel takes packed items of 64 * kMoeNb columns.
  std::vector<Exl3GemmItem> eitems;
  for (int e=0; e<=kExperts; ++e)
    for (int m=0; m<count[e]; m+=Tm)
      for (int b=0; b<O_/16; b+=4*kMoeNb) for (int w=0; w<mats; ++w) eitems.push_back({2*e+w,0,m,b});
  Buffer<Exl3GemmItem> dei(eitems); Buffer<int> den(std::vector<int>{(int)eitems.size()});
  auto run=[&](const char* name,auto launch) {
    launch(); CheckLaunch();
    const double us=Time(launch,7);
    std::printf("%-10s R=%d rm=%d items=%zu us=%.1f tflops=%.2f\n",name,R,RM,items.size(),us,flops/us*1e-6);
  };
  {
    Buffer<float> in(Random(4096*128,1.f,5)); Buffer<int> bad(1);
    WhtCheckKernel<<<4096,32>>>(in.p,bad.p); CheckLaunch();
    std::printf("wht128 dpp mismatches %d\n",bad.Get()[0]);
  }
  run("engine",[&] { LaunchPrefillMoeGemm({RM,true},dj.p,dei.p,den.p); });
  {
    const auto e=pg.Get(); pg.Zero();
    run("old",[&] { DiagKernel<8,0><<<2048,128>>>(dj.p,di.p,dn.p); });
    if (RM==8) std::printf("engine equals the old body: %d\n",(int)(pg.Get()==e));
  }
  if (argc>2 && argv[2][0]=='w') {
    // Item shapes: RM row tiles per unit, NB column tiles per wave.
    const auto ref=pg.Get();
    auto shape=[&](const char* name,int rm,int nb,auto kern) {
      std::vector<Exl3GemmItem> it2;
      for (int e=0; e<=kExperts; ++e)
        for (int m=0; m<count[e]; m+=16*rm)
          for (int b=0; b<O_/16; b+=4*nb) for (int w=0; w<mats; ++w) it2.push_back({2*e+w,0,m,b});
      Buffer<Exl3GemmItem> d2(it2); Buffer<int> n2(std::vector<int>{(int)it2.size()});
      pg.Zero();
      run(name,[&] { kern(d2.p,n2.p); });
      std::printf("%s bitwise=%d\n",name,(int)(pg.Get()==ref));
    };
#define SHAPE(rm,nb) shape("w" #rm "x" #nb,rm,nb,[&](auto* it,auto* n) { WideKernel<rm,nb><<<2048,128>>>(dj.p,it,n); })
#define ENG(rm,o) shape("e" #rm "o" #o,rm,2,[&](auto* it,auto* n) { EngWideKernel<rm,o><<<2048,128>>>(dj.p,it,n); })
#define LIVE(rm,o) shape("live" #rm "o" #o,rm,2,[&](auto* it,auto* n) { LiveKernel<rm,o><<<2048,128>>>(dj.p,it,n); })
    if (R>=4096) { ENG(6,true); ENG(6,false); LIVE(6,false); LIVE(6,true); }
    else if (R>=2048) { LIVE(4,false); LIVE(4,true); }
    else { LIVE(2,false); LIVE(2,true); }
#undef LIVE
#define PAIR(rm,o) shape("pair" #rm "o" #o,rm,2,[&](auto* it,auto* n) { PairKernel<rm,o><<<2048,128>>>(dj.p,it,n); })
    if(R>=4096) { PAIR(6,true);PAIR(6,false); }
    else if(R>=2048) { PAIR(4,true);PAIR(4,false); }
    else { PAIR(2,true);PAIR(2,false); }
#undef PAIR
#define STAGE(rm,s) shape("stage" #rm "s" #s,rm,2,[&](auto* it,auto* n) { StageWideKernel<rm,s><<<2048,128>>>(dj.p,it,n); })
    if(R>=4096) { STAGE(6,8);STAGE(6,12);STAGE(8,8);STAGE(8,12); }
    else if(R>=2048) { STAGE(4,8);STAGE(4,12); }
    else { STAGE(2,8);STAGE(2,12); }
#undef STAGE
#define KO(d) shape("ko" #d,6,2,[&](auto* it,auto* n) { WideKernel<6,2,d><<<2048,128>>>(dj.p,it,n); })
    if (R>=4096) { SHAPE(6,2); KO(1); KO(2); KO(4); KO(8); KO(16); KO(9); KO(15); KO(29); KO(27); KO(22); }
#undef KO
    else if (R>=2048) { SHAPE(4,2); ENG(4,true); ENG(4,false); }
    else {
#define KO2(d) shape("ko" #d,2,2,[&](auto* it,auto* n) { WideKernel<2,2,d><<<2048,128>>>(dj.p,it,n); })
      SHAPE(2,2); KO2(1); KO2(2); KO2(4); KO2(16); KO2(15); KO2(27); KO2(22);
#undef KO2
    }
#undef ENG
#undef SHAPE
    return 0;
  }
  if (argc>2 && argv[2][0]=='d' && RM==8) Diagnose(run,dj,di,dn,jobs,xg,xu,pg,total);
  else Fused(R,RM,rng,count,tiles.p,words);
}
