// Independent circular-bit CPU oracle for K4/K6 engine GEMM bodies.
#include "prefill_common.h"
#include "../engine/exl3/prefill_linear.h"
using namespace crucible;
using namespace prefill_bench;
constexpr int I=256,O=128;
float CpuWeight(const std::vector<uint32_t>& tiles,int K,int i,int o) {
  const int r=i&15,c=o&15,lane=4*(c&7)+(r&7)/2;
  const int pos=8*lane+(r&1)+((r&8)?2:0)+((c&8)?4:0),bits=256*K;
  const auto* t=tiles.data()+((size_t)(o/16)*(I/16)+i/16)*8*K;
  uint32_t state=0;
  for(int z=0;z<16;++z) {
    const int bit=((pos+1)*K-16+z+bits)%bits;
    state=(state<<1)|((t[bit/32]>>(31-bit%32))&1);
  }
  const uint32_t p=state*kMul1;
  const int sum=(p&255)+((p>>8)&255)+((p>>16)&255)+(p>>24);
  return __half2float(__float2half((float)((1024.+sum)*0.00676727294921875-10.3828125)));
}
template<int K,int RM,int NB,bool KEEP,bool BASE=false,int PF=1> __global__ void WideCheck(
    Exl3GemmJob j,const Exl3GemmItem* items,int n) {
  __shared__ uint32_t ws[kGemmWeightWords];
  if((int)blockIdx.x>=n)return;
  const auto it=items[blockIdx.x];
  GemmF8 acc[RM*NB];
  if constexpr(BASE) {
    Exl3GemmBody<K,RM,true,false,PF,true,true,true,true>(j,it,ws);
  } else if constexpr(KEEP) {
    PrefillWideBody<K,RM,NB,true,false,false>(j,it,ws,acc);
    const int lane=threadIdx.x&31,wave=threadIdx.x>>5;
#pragma unroll
    for(int c=0;c<NB;++c)
#pragma unroll
      for(int r=0;r<RM;++r)
#pragma unroll
        for(int z=0;z<8;++z) {
          const int row=it.m0+r*16+2*z+lane/16;
          if(row<j.M)j.partial[(size_t)row*j.O+(it.b0+wave*NB+c)*16+(lane&15)]=acc[c*RM+r][z];
        }
  } else PrefillWideBody<K,RM,NB>(j,it,ws);
}
template<int K,int RM,int NB,bool KEEP,bool BASE=false,int PF=1> bool Point(const std::vector<uint32_t>& hw,
    const std::vector<float>& w,int M) {
  const auto hx=Half(Random((size_t)M*I,.1f,19));
  std::vector<__half> packed(hx.size());
  for(int r=0;r<M;++r)for(int i=0;i<I;++i)packed[((size_t)(i/16)*M+r)*16+i%16]=hx[(size_t)r*I+i];
  Buffer<uint32_t> tiles(hw);Buffer<__half> x(packed);
  constexpr int S=BASE?3:1;
  Buffer<float> y((size_t)S*M*O);
  Exl3GemmJob j{tiles.p,x.p,y.p,nullptr,I,O,M,S,K,M};
  std::vector<Exl3GemmItem> items;
  for(int s=0;s<S;++s)for(int m=0;m<M;m+=16*RM)for(int b=0;b<O/16;b+=4*NB)items.push_back({0,s,m,b});
  Buffer<Exl3GemmItem> di(items);
  WideCheck<K,RM,NB,KEEP,BASE,PF><<<items.size(),128>>>(j,di.p,(int)items.size());CheckLaunch();
  const auto raw=y.Get();std::vector<float> got((size_t)M*O);
  for(int s=0;s<S;++s)for(size_t i=0;i<got.size();++i)got[i]+=raw[(size_t)s*M*O+i];
  double worst=0;
  for(int r=0;r<M;++r)for(int o=0;o<O;++o) {
    double ref=0,mag=0;
    for(int i=0;i<I;++i) {const double p=Value(hx[(size_t)r*I+i])*w[(size_t)i*O+o];ref+=p;mag+=std::fabs(p);}
    const double error=std::fabs(got[(size_t)r*O+o]-ref)/(mag+1e-30);
    if(!std::isfinite(error))return false;
    worst=std::max(worst,error);
  }
  std::printf("K=%d M=%d RM=%d NB=%d KEEP=%d BASE=%d PF=%d fp64=%.3e %s\n",K,M,RM,NB,KEEP,BASE,PF,worst,worst<1e-5?"PASS":"FAIL");
  return worst<1e-5;
}
template<int K> int Cases() {
  std::mt19937 rng(31);std::vector<uint32_t> h((size_t)I*O*K/32);
  for(auto& x:h)x=rng();std::vector<float> w((size_t)I*O);
  for(int i=0;i<I;++i)for(int o=0;o<O;++o)w[(size_t)i*O+o]=CpuWeight(h,K,i,o);
  int bad=0;
  for(int M:{1,15,16,17,31,33,63,65,80,96,127,129,256}) {
#define CHECK(R,N,Q) bad+=!Point<K,R,N,Q>(h,w,M)
    CHECK(2,2,false);CHECK(4,2,false);CHECK(6,2,false);CHECK(8,2,false);
    CHECK(4,2,true);CHECK(8,2,true);CHECK(16,1,true);
    bad+=!Point<K,4,1,false,true,1>(h,w,M);
    bad+=!Point<K,8,1,false,true,8>(h,w,M);
#undef CHECK
  }
  return bad;
}
int main(){const int bad=Cases<3>()+Cases<4>()+Cases<5>()+Cases<6>();std::printf("failures=%d\n",bad);return bad?1:0;}
