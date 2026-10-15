// Cold dense shapes. Compare complete partial arrays before timing each variant.
#include "prefill_common.h"
#include <cstring>
#include "kernels/exl3_gemm_stage.h"
using namespace crucible;
using namespace prefill_bench;
__global__ void ColdRead(const uint4* data,unsigned* sum,size_t n) {
  const size_t t=(size_t)blockIdx.x*blockDim.x+threadIdx.x;
  unsigned v=0;
  for(size_t i=t;i<n;i+=(size_t)gridDim.x*blockDim.x) {const auto q=data[i];v^=q.x^q.y^q.z^q.w;}
  sum[t]=v;
}
template<int RM,int STRIDE,int NW,bool STAGE> __global__ void __launch_bounds__(NW*32) DenseStage(
    Exl3GemmJob j,const Exl3GemmItem* items) {
  __shared__ uint32_t ws[NW*kGemmWaveWords+(STAGE?RM*16*STRIDE:0)];
  const auto it=items[blockIdx.x];
  constexpr int NB=8/NW;
#define BODY(K) if constexpr(STAGE) GemmStageBody<K,RM,NB,true,true,true,true,STRIDE,NW>(j,it,ws); else PrefillWideBody<K,RM,NB>(j,it,ws)
  if(j.K==3) {BODY(3);} else if(j.K==4) {BODY(4);} else {BODY(6);}
#undef BODY
}
template<int RM,int STRIDE,int NW,bool STAGE> void Measure(const char* name,Exl3GemmJob j,
    const std::vector<float>& ref,const uint4* cold,unsigned* sum) {
  std::vector<Exl3GemmItem> items;
  for(int m=0;m<j.M;m+=16*RM)for(int b=0;b<j.O/16;b+=8)items.push_back({0,0,m,b});
  Buffer<Exl3GemmItem> di(items);
  auto launch=[&]{DenseStage<RM,STRIDE,NW,STAGE><<<items.size(),NW*32>>>(j,di.p);};
  launch();CheckLaunch();std::vector<float> got(ref.size());
  HIP_OK(hipMemcpy(got.data(),j.partial,got.size()*4,hipMemcpyDeviceToHost));
  const bool same=std::memcmp(got.data(),ref.data(),got.size()*4)==0;
  if(!same) {std::printf("%s bitwise=0\n",name);std::exit(1);}
  bench::EventTimer timer;std::vector<double> times;
  for(int rep=0;rep<4;++rep) {
    ColdRead<<<320,256>>>(cold,sum,16u*1024*1024);
    timer.Begin();launch();HIP_OK(hipGetLastError());const double ms=timer.EndMs();if(rep)times.push_back(ms);
  }
  const double ms=bench::Median(times),tf=2.*j.M*j.I*j.O/ms/1e9;
  std::printf("%s I=%d O=%d K=%d M=%d ms=%.4f tflops=%.3f roof=%.1f%% bitwise=1\n",name,j.I,j.O,j.K,j.M,ms,tf,tf/49*100);
  std::fflush(stdout);
}
void Shape(int I,int O,int K,int M,const uint4* cold,unsigned* sum) {
  std::mt19937 rng(29);std::vector<uint32_t> h((size_t)I*O*K/32);
  for(auto& v:h)v=rng();Buffer<uint32_t> w(h);
  auto hx=Half(Random((size_t)M*I,.1f,31));std::vector<__half> xp(hx.size());
  for(int r=0;r<M;++r)for(int i=0;i<I;++i)xp[((size_t)(i/16)*M+r)*16+i%16]=hx[(size_t)r*I+i];
  Buffer<__half> x(xp);Buffer<float> y((size_t)M*O);
  Exl3GemmJob j{w.p,x.p,y.p,nullptr,I,O,M,1,K,M};
  std::vector<Exl3GemmItem> items;
  for(int m=0;m<M;m+=128)for(int b=0;b<O/16;b+=8)items.push_back({0,0,m,b});
  Buffer<Exl3GemmItem> di(items);
  DenseStage<8,8,4,false><<<items.size(),128>>>(j,di.p);CheckLaunch();const auto ref=y.Get();
  Measure<8,8,4,false>("base8n2",j,ref,cold,sum);
  Measure<8,8,4,true>("stage8n2s8",j,ref,cold,sum);
  Measure<8,12,4,true>("stage8n2s12",j,ref,cold,sum);
  Measure<16,8,8,true>("stage16n1s8",j,ref,cold,sum);
  Measure<16,12,8,true>("stage16n1s12",j,ref,cold,sum);
  Measure<20,8,8,true>("stage20n1s8",j,ref,cold,sum);
  Measure<24,8,8,true>("stage24n1s8",j,ref,cold,sum);
  Measure<24,12,8,true>("stage24n1s12",j,ref,cold,sum);
  Measure<26,8,8,true>("stage26n1s8",j,ref,cold,sum);
}
int main(int argc,char** argv) {
  const int M=argc>1?std::atoi(argv[1]):4096;
  if(M<1||M>4096)return 2;
  Buffer<uint4> cold(16u*1024*1024);Buffer<unsigned> sum(320*256);
  for(const auto [I,O]:std::vector<std::pair<int,int>>{{6144,2560},{2560,16384}})
    for(int K:{3,4,6})Shape(I,O,K,M,cold.p,sum.p);
}
