// HC prefill stage timings: packed baseline kernels versus fused kernels.
#include "prefill_common.h"
#include "kernels/prefill_hc.h"
using namespace crucible;
using namespace prefill_bench;

namespace crucible {
// PrefillHcMixWideKernel with a pipelined k loop (the PrefillDenseFastKernel
// structure): RW x CW waves, each owning 16 rows by 16 columns of all four
// streams, share 32-k stages of t (high and low) and up in double-buffered
// LDS, with DEPTH stages of global loads in flight in registers. Each
// accumulator sees the same WMMA chain and the epilogue is unchanged, so
// mixed and xh are bitwise equal to PrefillHcMixWideKernel. Negative: R4096
// 2221 us (rw4 cw2) vs 2073 us for the tall kernel; the WMMA phase and the
// loads stop overlapping (no WMMA 1278 us, no WMMA and no H 638 us).
// PH loads the epilogue's H values before the k loop. DIAG (bench only):
// 1 skips the H loads, 2 skips the WMMAs.
constexpr int kMixStep=32, kMixPad=kMixStep+8;
template<int RW,int CW,int DEPTH=2,bool PH=false,int DIAG=0>
__global__ void __launch_bounds__(RW*CW*32) PrefillHcMixLdsKernel(const __half* up,const float* w1,
    const __half* thi,const __half* tlo,const float* ss,const float* H,float* mixed,int R,
    __half* xh=nullptr) {
  constexpr int kSteps=kRank/kMixStep, T=RW*CW*32, NA=2*16*RW*4, NB=4*16*CW*4;
  constexpr int na=NA/T, nb=NB/T;
  static_assert(kSteps%DEPTH==0 && NA%T==0 && NB%T==0);
  __shared__ __attribute__((aligned(16))) _Float16 as[2][2][16*RW][kMixPad], bs[2][4][16*CW][kMixPad];
  const int t=threadIdx.x,lane=t&31,wave=t>>5;
  const int rb=blockIdx.y*16*RW,cb=blockIdx.x*16*CW;
  const int wr=(wave/CW)*16,wc=(wave%CW)*16;
  typedef unsigned U4 __attribute__((ext_vector_type(4)));
  const __half* ap[na]; const __half* bp[nb];
  int ao[na],bo[nb];
#pragma unroll
  for (int i=0;i<na;++i) {
    const int idx=t+i*T,hl=idx/(16*RW*4),rem=idx%(16*RW*4),row=rem>>2,part=(rem&3)*8;
    // Rows past R load row R-1; their outputs are never stored.
    ap[i]=(hl ? tlo : thi)+(size_t)min(rb+row,R-1)*kRank+part;
    ao[i]=(hl*16*RW+row)*kMixPad+part;
  }
#pragma unroll
  for (int j=0;j<nb;++j) {
    const int idx=t+j*T,s=idx/(16*CW*4),rem=idx%(16*CW*4),col=rem>>2,part=(rem&3)*8;
    bp[j]=up+((size_t)s*kHidden+cb+col)*kRank+part;
    bo[j]=(s*16*CW+col)*kMixPad+part;
  }
  _Float16* const a0=&as[0][0][0][0];
  _Float16* const b0=&bs[0][0][0][0];
  constexpr int abuf=2*16*RW*kMixPad,bbuf=4*16*CW*kMixPad;
  U4 ra[DEPTH][na],rbv[DEPTH][nb];
#pragma unroll
  for (int i=0;i<na;++i) *reinterpret_cast<U4*>(a0+ao[i])=*reinterpret_cast<const U4*>(ap[i]);
#pragma unroll
  for (int j=0;j<nb;++j) *reinterpret_cast<U4*>(b0+bo[j])=*reinterpret_cast<const U4*>(bp[j]);
#pragma unroll
  for (int d=0;d<DEPTH;++d) {
#pragma unroll
    for (int i=0;i<na;++i) ra[d][i]=*reinterpret_cast<const U4*>(ap[i]+(d+1)*kMixStep);
#pragma unroll
    for (int j=0;j<nb;++j) rbv[d][j]=*reinterpret_cast<const U4*>(bp[j]+(d+1)*kMixStep);
  }
  const int r0=rb+wr,d=cb+wc+(lane&15);
  float hv[PH ? 8 : 1][4];
  if constexpr (PH && !(DIAG&1)) {
#pragma unroll
    for (int z=0;z<8;++z) {
      const int r=min(r0+prefill::FragmentRow(z),R-1);
#pragma unroll
      for (int s=0;s<4;++s) hv[z][s]=H[((size_t)r*4+s)*kHidden+d];
    }
  }
  PrefillLdsSync();
  prefill::F8 hi[4]={},lo[4]={};
#pragma unroll 1
  for (int s0=0;s0<kSteps;s0+=DEPTH)
#pragma unroll
    for (int dd=0;dd<DEPTH;++dd) {
      const int step=s0+dd,buf=step&1;
      // Slot dd holds stage step + 1; refill it with stage step + 1 + DEPTH.
      if (step+1<kSteps) {
#pragma unroll
        for (int i=0;i<na;++i) *reinterpret_cast<U4*>(a0+(buf^1)*abuf+ao[i])=ra[dd][i];
#pragma unroll
        for (int j=0;j<nb;++j) *reinterpret_cast<U4*>(b0+(buf^1)*bbuf+bo[j])=rbv[dd][j];
        if (step+1+DEPTH<kSteps) {
#pragma unroll
          for (int i=0;i<na;++i) ra[dd][i]=*reinterpret_cast<const U4*>(ap[i]+(step+1+DEPTH)*kMixStep);
#pragma unroll
          for (int j=0;j<nb;++j) rbv[dd][j]=*reinterpret_cast<const U4*>(bp[j]+(step+1+DEPTH)*kMixStep);
        }
      }
      if constexpr (!(DIAG&2)) {
#pragma unroll
        for (int kb=0;kb<kMixStep;kb+=16) {
          const prefill::H16 ah=*reinterpret_cast<const prefill::H16*>(&as[buf][0][wr+(lane&15)][kb]);
          const prefill::H16 al=*reinterpret_cast<const prefill::H16*>(&as[buf][1][wr+(lane&15)][kb]);
#pragma unroll
          for (int s=0;s<4;++s) {
            const prefill::H16 b=*reinterpret_cast<const prefill::H16*>(&bs[buf][s][wc+(lane&15)][kb]);
            hi[s]=__builtin_amdgcn_wmma_f32_16x16x16_f16_w32(ah,b,hi[s]);
            lo[s]=__builtin_amdgcn_wmma_f32_16x16x16_f16_w32(al,b,lo[s]);
          }
        }
      }
      PrefillLdsSync();
    }
  if (r0>=R) return;
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
      float h;
      if constexpr (DIAG&1) h=1.f;
      else if constexpr (PH) h=hv[z][s];
      else h=H[((size_t)r*4+s)*kHidden+d];
      v[s]=Sigmoid(g)*h*inv*w[s];
    }
    const float x=RoundH(((v[0]+v[1])+(v[2]+v[3]))*.25f);
    mixed[(size_t)r*kHidden+d]=x;
    if (xh) xh[(size_t)r*kHidden+d]=__float2half(x);
  }
}
// Tall mix with up in LDS: the block stages the whole k range of its 16
// columns (4 streams x 16 x 320 halves, 41 KB) once, then each wave runs the
// tall loop with t from global and up from LDS, so the hot loop issues 2
// instead of 6 global fragment loads per 8 WMMAs. Same WMMA chains and
// epilogue: bitwise equal to PrefillHcMixWideKernel<true>.
constexpr int kMixBPad=kRank+8;
template<int WAVES,int UNROLL=2>
__global__ void __launch_bounds__(WAVES*32) PrefillHcMixTallLdsKernel(const __half* up,const float* w1,
    const __half* thi,const __half* tlo,const float* ss,const float* H,float* mixed,int R,
    __half* xh=nullptr) {
  __shared__ __attribute__((aligned(16))) _Float16 bs[4][16][kMixBPad];
  typedef unsigned U4 __attribute__((ext_vector_type(4)));
  const int t=threadIdx.x,lane=t&31,wave=t>>5;
  const int c0=blockIdx.x*16;
  for (int i=t;i<4*16*kRank/8;i+=WAVES*32) {
    const int row=i/(kRank/8),part=(i%(kRank/8))*8,s=row>>4,col=row&15;
    *reinterpret_cast<U4*>(&bs[s][col][part])=
        *reinterpret_cast<const U4*>(up+((size_t)s*kHidden+c0+col)*kRank+part);
  }
  __syncthreads();
  const int r0=blockIdx.y*16*WAVES+wave*16;
  const int d=c0+(lane&15);
  if (r0>=R) return;
  const int ra=r0+(lane&15);
  prefill::F8 hi[4]={},lo[4]={};
#pragma unroll UNROLL
  for (int k=0;k<kRank;k+=16) {
    const prefill::H16 ah=ra<R ? PrefillLoad16(thi+(size_t)ra*kRank+k) : prefill::H16{};
    const prefill::H16 al=ra<R ? PrefillLoad16(tlo+(size_t)ra*kRank+k) : prefill::H16{};
#pragma unroll
    for (int s=0;s<4;++s) {
      const prefill::H16 b=*reinterpret_cast<const prefill::H16*>(&bs[s][lane&15][k]);
      hi[s]=__builtin_amdgcn_wmma_f32_16x16x16_f16_w32(ah,b,hi[s]);
      lo[s]=__builtin_amdgcn_wmma_f32_16x16x16_f16_w32(al,b,lo[s]);
    }
  }
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
      v[s]=Sigmoid(g)*H[((size_t)r*4+s)*kHidden+d]*inv*w[s];
    }
    const float x=RoundH(((v[0]+v[1])+(v[2]+v[3]))*.25f);
    mixed[(size_t)r*kHidden+d]=x;
    if (xh) xh[(size_t)r*kHidden+d]=__float2half(x);
  }
}
}  // namespace crucible

int main(int argc,char** argv) {
  const int R=argc>1 ? std::atoi(argv[1]) : 1024, M=kRank+4;
  if (R<1 || R>8192) return 2;
  Buffer<float> H(Random((size_t)R*kWide,.4f,311));
  Buffer<__half> fn(Half(Random((size_t)M*kWide,.02f,313)));
  Buffer<__half> up(Half(Random((size_t)kWide*kRank,.02f,317)));
  auto ws=Random(kWide,.05f,319); for (auto& v:ws) v+=1.f;
  Buffer<float> w1(ws),dots((size_t)R*M*4),ss(R*4),t((size_t)R*kRank),post(R*4),mixed((size_t)R*kHidden);
  Buffer<__half> hi((size_t)R*kWide),lo((size_t)R*kWide),thi((size_t)R*kRank),tlo((size_t)R*kRank);
  auto squares=[&] { PrefillHcSquaresKernel<<<(R*4+7)/8,256>>>(H.p,ss.p,R); };
  auto pack=[&] { PrefillPackKernel<<<(R*kWide+255)/256,256>>>(H.p,hi.p,lo.p,R*kWide); };
  auto dot=[&] { PrefillHcDotsKernel<<<dim3((R+15)/16,(M+63)/64,4),128>>>(fn.p,M,hi.p,lo.p,dots.p,R); };
  auto mid=[&] { PrefillHcMidKernel<<<R,256>>>(dots.p,M,ss.p,t.p,post.p,R); };
  auto tpack=[&] { PrefillPackKernel<<<(R*kRank+255)/256,256>>>(t.p,thi.p,tlo.p,R*kRank); };
  auto mix=[&] { PrefillHcMixKernel<<<dim3((R+15)/16,kHidden/16),128>>>(up.p,w1.p,thi.p,tlo.p,ss.p,H.p,mixed.p,R); };
  auto old=[&] { squares(); pack(); dot(); mid(); tpack(); mix(); };
  old(); CheckLaunch();
  const auto d0=dots.Get(),s0=ss.Get(),p0=post.Get(),m0=mixed.Get();
  auto fdot=[&] { PrefillHcDotsFusedKernel<64><<<dim3((R+31)/32,4),256>>>(fn.p,M,H.p,dots.p,ss.p,R); };
  auto fmid=[&] { PrefillHcMidPackKernel<<<R,256>>>(dots.p,M,ss.p,thi.p,tlo.p,post.p,R); };
  auto fmix=[&] { PrefillHcMixWideKernel<false><<<dim3(kHidden/32,(R+63)/64),256>>>(up.p,w1.p,thi.p,tlo.p,ss.p,H.p,mixed.p,R); };
  auto ldot=[&] { PrefillHcDotsLdsKernel<<<dim3((R+31)/32,4),256>>>(fn.p,M,H.p,dots.p,ss.p,R); };
  auto fused=[&] { fdot(); fmid(); fmix(); };
  dots.Zero(); ss.Zero(); post.Zero(); mixed.Zero();
  fused(); CheckLaunch();
  const auto de=Compare(dots.Get(),d0),se=Compare(ss.Get(),s0),pe=Compare(post.Get(),p0),me=Compare(mixed.Get(),m0);
  std::printf("R=%d fused vs packed: dots=%.3g ss=%.3g post=%.3g mixed=%.3g mixed_max_scaled=%.3g\n",
    R,de.rel_l2,se.rel_l2,pe.rel_l2,me.rel_l2,me.max_scaled);
  std::printf("packed us: squares=%.1f pack=%.1f dots=%.1f mid=%.1f tpack=%.1f mix=%.1f total=%.1f\n",
    Time(squares),Time(pack),Time(dot),Time(mid),Time(tpack),Time(mix),Time(old));
  std::printf("fused us: dots=%.1f midpack=%.1f mix=%.1f total=%.1f\n",Time(fdot),Time(fmid),Time(fmix),Time(fused));
  if (argc>2 && argv[2][0]=='p') {
    fdot(); CheckLaunch();
    const auto fd=dots.Get(),fs=ss.Get();
    bool all=true;
    for (int step : {128, 256}) {
      auto pd=[&] {
        if (step==128) PrefillHcDotsFusedKernel<128><<<dim3((R+31)/32,4),256>>>(fn.p,M,H.p,dots.p,ss.p,R);
        else PrefillHcDotsFusedKernel<256><<<dim3((R+31)/32,4),256>>>(fn.p,M,H.p,dots.p,ss.p,R);
      };
      dots.Zero(); ss.Zero(); pd(); CheckLaunch();
      const bool same=dots.Get()==fd && ss.Get()==fs;
      all=all && same;
      std::printf("step%d dots: bitwise=%d us=%.1f fused_us=%.1f\n",step,(int)same,Time(pd),Time(fdot));
    }
    return all ? 0 : 8;
  }
  if (argc>2 && argv[2][0]=='w') {
    // Wave-role dots: bitwise dots and squares against the fused kernel.
    fdot(); CheckLaunch();
    const auto fd=dots.Get(),fs=ss.Get();
    auto f128=[&] { PrefillHcDotsFusedKernel<128><<<dim3((R+31)/32,4),256>>>(fn.p,M,H.p,dots.p,ss.p,R); };
    bool all=true;
    for (int step : {64, 128}) {
      auto wd=[&] {
        constexpr int T=(kHcWsCompute+4)*32;
        if (step==64) PrefillHcDotsWsKernel<64><<<dim3((R+31)/32,4),T>>>(fn.p,M,H.p,dots.p,ss.p,R);
        else PrefillHcDotsWsKernel<128><<<dim3((R+31)/32,4),T>>>(fn.p,M,H.p,dots.p,ss.p,R);
      };
      dots.Zero(); ss.Zero(); wd(); CheckLaunch();
      const bool same=dots.Get()==fd && ss.Get()==fs;
      all=all && same;
      std::printf("R=%d ws%d dots: bitwise=%d us=%.1f fused128_us=%.1f\n",R,step,(int)same,Time(wd),Time(f128));
    }
    if (argc>3) {
      const int nb=4*((R+31)/32);
      auto run=[&](const char* name,auto launch) {
        dots.Zero(); ss.Zero(); launch(); CheckLaunch();
        const bool same=dots.Get()==fd && ss.Get()==fs;
        std::printf("R=%d %s bitwise=%d us=%.1f\n",R,name,(int)same,Time(launch));
        all=all && same;
      };
#define WS(STEP,LW,D) run("swap s" #STEP " lw" #LW " d" #D,[&] { PrefillHcDotsWsKernel<STEP,0,true,LW,D><<<nb,(kHcWsCompute+LW)*32>>>(fn.p,M,H.p,dots.p,ss.p,R); })
      WS(64,2,4); WS(64,4,8); WS(128,2,2); WS(32,4,16);
      std::printf("R=%d lw2d4 Hcached us=%.1f fncached us=%.1f\n",R,
        Time([&] { PrefillHcDotsWsKernel<64,2,true,2,4><<<nb,(kHcWsCompute+2)*32>>>(fn.p,M,H.p,dots.p,ss.p,R); }),
        Time([&] { PrefillHcDotsWsKernel<64,4,true,2,4><<<nb,(kHcWsCompute+2)*32>>>(fn.p,M,H.p,dots.p,ss.p,R); }));
#undef WS
    }
    return all ? 0 : 8;
  }
  if (argc>2 && argv[2][0]=='n') {
    // No-lo dots (approved numerics): retune step, loader waves and ring
    // depth; bitwise against the engine's LO=false launch.
    const int nb=4*((R+31)/32);
    auto eng=[&] { PrefillHcDotsWsKernel<64,0,true,2,4,false><<<nb,(kHcWsCompute+2)*32>>>(fn.p,M,H.p,dots.p,ss.p,R); };
    dots.Zero(); ss.Zero(); eng(); CheckLaunch();
    const auto nd=dots.Get(),ns=ss.Get();
    const auto e=Compare(nd,d0);
    std::printf("R=%d nolo vs split dots rel_l2=%.3g max_scaled=%.3g\n",R,e.rel_l2,e.max_scaled);
    bool all=true;
    auto run=[&](const char* name,auto launch) {
      dots.Zero(); ss.Zero(); launch(); CheckLaunch();
      const bool same=dots.Get()==nd && ss.Get()==ns;
      std::printf("R=%d %s bitwise=%d us=%.1f\n",R,name,(int)same,Time(launch));
      all=all && same;
    };
    run("split s64 lw2 d4 (engine before)",[&] { PrefillHcDotsWsKernel<64,0,true,2,4,true><<<nb,(kHcWsCompute+2)*32>>>(fn.p,M,H.p,dots.p,ss.p,R); });
#define NL(STEP,LW,D) run("nolo s" #STEP " lw" #LW " d" #D,[&] { PrefillHcDotsWsKernel<STEP,0,true,LW,D,false><<<nb,(kHcWsCompute+LW)*32>>>(fn.p,M,H.p,dots.p,ss.p,R); })
    NL(64,2,4); NL(64,2,2); NL(64,2,1); NL(64,4,2); NL(64,4,4); NL(64,4,8);
    NL(128,2,1); NL(128,2,2); NL(128,4,2); NL(32,2,8); NL(32,4,8); NL(32,4,16);
#undef NL
    // RT=4: 64-row blocks read fn half as often.
    const int nb4=4*((R+63)/64);
#define NL4(STEP,LW,D) run("nolo rt4 s" #STEP " lw" #LW " d" #D,[&] { PrefillHcDotsWsKernel<STEP,0,true,LW,D,false,4><<<nb4,(kHcWsCompute+LW)*32>>>(fn.p,M,H.p,dots.p,ss.p,R); })
    NL4(64,2,1); NL4(64,2,2); NL4(64,4,1); NL4(64,4,2); NL4(64,4,4); NL4(128,4,1); NL4(128,4,2); NL4(32,4,4); NL4(32,4,8);
#undef NL4
    std::printf("R=%d nolo rt4 s64 lw4 d2 fncached us=%.1f Hcached us=%.1f\n",R,
      Time([&] { PrefillHcDotsWsKernel<64,4,true,4,2,false,4><<<nb4,(kHcWsCompute+4)*32>>>(fn.p,M,H.p,dots.p,ss.p,R); }),
      Time([&] { PrefillHcDotsWsKernel<64,2,true,4,2,false,4><<<nb4,(kHcWsCompute+4)*32>>>(fn.p,M,H.p,dots.p,ss.p,R); }));
    std::printf("R=%d nolo lw2d4 Hcached us=%.1f fncached us=%.1f nowmma us=%.1f\n",R,
      Time([&] { PrefillHcDotsWsKernel<64,2,true,2,4,false><<<nb,(kHcWsCompute+2)*32>>>(fn.p,M,H.p,dots.p,ss.p,R); }),
      Time([&] { PrefillHcDotsWsKernel<64,4,true,2,4,false><<<nb,(kHcWsCompute+2)*32>>>(fn.p,M,H.p,dots.p,ss.p,R); }),
      Time([&] { PrefillHcDotsWsKernel<64,1,true,2,4,false><<<nb,(kHcWsCompute+2)*32>>>(fn.p,M,H.p,dots.p,ss.p,R); }));
    return all ? 0 : 8;
  }
  if (argc>2 && argv[2][0]=='L') {
    // LDS-pipelined mix: bitwise mixed and xh against the tall engine kernel.
    fused(); CheckLaunch();
    Buffer<__half> xh0((size_t)R*kHidden),xh1((size_t)R*kHidden);
    auto tm=[&] { PrefillHcMixWideKernel<true><<<dim3(kHidden/16,(R+127)/128),256>>>(up.p,w1.p,thi.p,tlo.p,ss.p,H.p,mixed.p,R,xh0.p); };
    mixed.Zero(); tm(); CheckLaunch();
    const auto fm=mixed.Get(); const auto fx=xh0.Get();
    bool all=true;
    auto run=[&](const char* name,auto launch,bool check) {
      mixed.Zero(); xh1.Zero(); launch(); CheckLaunch();
      const bool same=mixed.Get()==fm && xh1.Get()==fx;
      if (check) all=all && same;
      std::printf("R=%d %-22s bitwise=%d us=%.1f\n",R,name,(int)same,Time(launch));
    };
    std::printf("R=%d tall us=%.1f\n",R,Time(tm));
#define MIXL(RW,CW,DEPTH,PH,DIAG) run("lds rw" #RW " cw" #CW " d" #DEPTH " ph" #PH " diag" #DIAG,[&] { \
      PrefillHcMixLdsKernel<RW,CW,DEPTH,PH,DIAG><<<dim3(kHidden/(16*CW),(R+16*RW-1)/(16*RW)),RW*CW*32>>>( \
          up.p,w1.p,thi.p,tlo.p,ss.p,H.p,mixed.p,R,xh1.p); },DIAG==0)
    MIXL(4,2,2,false,0); MIXL(8,1,2,false,0); MIXL(8,2,2,false,0); MIXL(2,4,2,false,0);
    MIXL(4,4,2,false,0); MIXL(4,2,1,false,0); MIXL(4,2,2,true,0); MIXL(4,2,1,true,0);
    MIXL(4,2,2,false,1); MIXL(4,2,2,false,2); MIXL(4,2,2,false,3);
#undef MIXL
#define MIXT(W,U) run("tall-lds w" #W " u" #U,[&] { \
      PrefillHcMixTallLdsKernel<W,U><<<dim3(kHidden/16,(R+16*W-1)/(16*W)),W*32>>>( \
          up.p,w1.p,thi.p,tlo.p,ss.p,H.p,mixed.p,R,xh1.p); },true)
    MIXT(8,2); MIXT(16,2); MIXT(8,4); MIXT(16,4); MIXT(4,2);
#undef MIXT
    return all ? 0 : 9;
  }
  if (argc>2 && argv[2][0]=='m') {
    fused(); CheckLaunch();
    const auto fm=mixed.Get();
    auto tm=[&] { PrefillHcMixWideKernel<true><<<dim3(kHidden/16,(R+127)/128),256>>>(up.p,w1.p,thi.p,tlo.p,ss.p,H.p,mixed.p,R); };
    mixed.Zero(); tm(); CheckLaunch();
    const bool same=mixed.Get()==fm;
    std::printf("R=%d tallmix bitwise=%d us=%.1f mix_us=%.1f\n",R,(int)same,Time(tm),Time(fmix));
    return same ? 0 : 9;
  }
  if (argc>2) {
    // Tiled dots only: bitwise dots against the fused kernel, squares close.
    fdot(); CheckLaunch();
    const auto fd=dots.Get(),fs=ss.Get();
    for (int waves : {4, 8}) {
      auto tdot=[&] {
        if (waves==4) LaunchHcDotsTile<4>(fn.p,M,H.p,dots.p,ss.p,R);
        else LaunchHcDotsTile<8>(fn.p,M,H.p,dots.p,ss.p,R);
      };
      dots.Zero(); ss.Zero(); tdot(); CheckLaunch();
      const bool same=dots.Get()==fd;
      std::printf("tile%d dots: bitwise=%d err=%.3g ss=%.3g us=%.1f fused_us=%.1f\n",waves,(int)same,
        Compare(dots.Get(),fd).rel_l2,Compare(ss.Get(),fs).rel_l2,Time(tdot),Time(fdot));
      if (!same) return 7;
    }
    return 0;
  }
  dots.Zero(); ss.Zero(); ldot(); CheckLaunch();
  const auto le=Compare(dots.Get(),d0),lse=Compare(ss.Get(),s0);
  std::printf("lds dots: err=%.3g ss=%.3g us=%.1f\n",le.rel_l2,lse.rel_l2,Time(ldot));
  if (le.rel_l2>2e-5 || lse.rel_l2>1e-6) return 6;
  auto sdot=[&] { PrefillHcDotsSplitKernel<<<dim3((R+31)/32,4,2),128>>>(fn.p,M,H.p,dots.p,ss.p,R); };
  dots.Zero(); ss.Zero(); sdot(); CheckLaunch();
  std::printf("split dots: err=%.3g ss=%.3g us=%.1f\n",Compare(dots.Get(),d0).rel_l2,Compare(ss.Get(),s0).rel_l2,Time(sdot));
  for (int u=0;u<2;++u) {
    auto gdot=[&] {
      if (u) PrefillHcDotsGroupKernel<true><<<dim3((R+31)/32,4),256>>>(fn.p,M,H.p,dots.p,ss.p,R);
      else PrefillHcDotsGroupKernel<false><<<dim3((R+31)/32,4),256>>>(fn.p,M,H.p,dots.p,ss.p,R);
    };
    dots.Zero(); gdot(); CheckLaunch();
    std::printf("group dots unroll=%d: err=%.3g us=%.1f\n",u,Compare(dots.Get(),d0).rel_l2,Time(gdot));
  }
  {
    for (int O : {96, 512, 513, 12800}) {
      Buffer<__half> W(Half(Random((size_t)O*kHidden,.02f,331+O)));
      Buffer<float> y1((size_t)R*O), y2((size_t)R*O);
      PrefillPackKernel<<<(R*kHidden+255)/256,256>>>(mixed.p,hi.p,nullptr,R*kHidden);
      auto d1=[&] { PrefillDenseKernel<<<dim3((R+15)/16,(O+63)/64),128>>>(W.p,O,hi.p,y1.p,O,R); };
      auto d2=[&] { PrefillDenseTileKernel<<<dim3((R+63)/64,(O+127)/128),256>>>(W.p,O,hi.p,y2.p,O,R); };
      Buffer<float> y3((size_t)R*O);
      auto d3=[&] { PrefillDenseLdsKernel<<<dim3((R+63)/64,(O+63)/64),128>>>(W.p,O,hi.p,y3.p,O,R); };
      Buffer<float> y4((size_t)R*O);
      auto d4=[&] { PrefillDenseBigKernel<<<dim3((R+127)/128,(O+127)/128),256>>>(W.p,O,hi.p,y4.p,O,R); };
      d1(); d2(); d3(); d4(); CheckLaunch();
      std::printf("dense O=%d err=%.3g lds_err=%.3g big_bitwise_lds=%d old_us=%.1f tile_us=%.1f lds_us=%.1f big_us=%.1f\n",O,Compare(y2.Get(),y1.Get()).rel_l2,
        Compare(y3.Get(),y1.Get()).rel_l2,(int)(y4.Get()==y3.Get()),Time(d1),Time(d2),Time(d3),Time(d4));
    }
  }
  const double flop=2.0*R*((double)M*kWide+(double)kWide*kRank)*2;
  std::printf("hi/lo WMMA work %.2f GFLOP\n",flop/1e9);
  if (de.rel_l2>2e-5 || se.rel_l2>1e-6 || pe.rel_l2>3e-6 || me.rel_l2>4e-4) return 4;
}
