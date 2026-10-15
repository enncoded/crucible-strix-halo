// Actual pack weights, controlled synthetic H, and a pinned production reference.
#include "common.h"
#include "../engine/exl3/safetensors.h"
#include "isa_hc_ws.h"
#include <charconv>
#include <cmath>
#include <cstring>
using namespace crucible;
constexpr int kRows=4096,kM=kRank+kStreams,kStride=kWide+64;

template<class T> struct Device {
  T* p=nullptr;
  explicit Device(size_t n) { HIP_CHECK(hipMalloc(&p,n*sizeof(T))); }
  ~Device() { (void)hipFree(p); }
  Device(const Device&)=delete;
};
__global__ void FillInput(float* h) {
  const size_t i=size_t(blockIdx.x)*256+threadIdx.x;
  if (i<size_t(kRows)*kWide)
    h[(i/kWide)*kStride+i%kWide]=float(int((i*1664525u+1013904223u)%8191)-4095)*.00037f;
}
float Read16(const uint8_t* data,size_t i,bool bf) {
  uint16_t u; std::memcpy(&u,data+2*i,2);
  if (bf) { const uint32_t bits=uint32_t(u)<<16; float f; std::memcpy(&f,&bits,4); return f; }
  _Float16 h; std::memcpy(&h,&u,2); return float(h);
}
std::vector<__half> Weights(st::Checkpoint& ck,const std::string& p) {
  const auto& norm=ck.Require(p+"hc_norm.weight",{"F16","BF16"},{kWide});
  ck.Require(p+"input_mix_weight_down.weight",{"F16"},{kRank,kWide});
  ck.Require(p+"block_inject_weight.weight",{"F16"},{kStreams,kWide});
  const auto* n=ck.Data(p+"hc_norm.weight");
  const auto* d=ck.Data(p+"input_mix_weight_down.weight");
  const auto* b=ck.Data(p+"block_inject_weight.weight");
  const bool bf=norm.dtype=="BF16";
  std::vector<__half> out(size_t(kM)*kWide);
  for (int m=0;m<kM;++m) for (int c=0;c<kWide;++c) {
    const float x=Read16(m<kRank ? d : b,size_t(m<kRank ? m : m-kRank)*kWide+c,false);
    out[size_t(m)*kWide+c]=__float2half(x*(1.f+Read16(n,c,bf)));
  }
  return out;
}
using Launch=void(*)(const __half*,const float*,float*,float*);
void Reference(const __half* fn,const float* h,float* dots,float* ss) {
  PrefillHcDotsWsKernel<64,0,true,2,2,false,4><<<kStreams*((kRows+63)/64),288>>>(
      fn,kM,h,dots,ss,kRows,kStride);
}
template<int D,int P,int RT=4,int FNSTRIDE=kWide,bool FULL_ROWS=false,int FIXED_M=0>
void Candidate(const __half* fn,const float* h,float* dots,float* ss) {
  static_assert(!FULL_ROWS || kRows%(16*RT)==0);
  static_assert(!FIXED_M || FIXED_M==kM);
  IsaHcWsKernel<D,P,RT,FNSTRIDE,FULL_ROWS,FIXED_M><<<kStreams*((kRows+16*RT-1)/(16*RT)),288>>>(
      fn,kM,h,dots,ss,kRows,kStride);
}
struct Variant { const char* name; Launch launch; int stride=0; };
#if defined(ISA_HC_BOUNDS)
const Variant kVariants[]={
  {"copy_d2_p0",Candidate<2,0>},
  {"full_rows",Candidate<2,0,4,kWide,true>},
  {"fixed_m",Candidate<2,0,4,kWide,false,kM>},
  {"full_rows_fixed_m",Candidate<2,0,4,kWide,true,kM>}
};
#elif defined(ISA_HC_PITCH)
const Variant kVariants[]={
  {"copy_d2_p0",Candidate<2,0>},
  {"fn_pad0",Candidate<2,0>,kWide},
  {"fn_pad32",Candidate<2,0,4,kWide+32>,kWide+32},
  {"fn_pad64",Candidate<2,0,4,kWide+64>,kWide+64}
};
#else
const Variant kVariants[]={
  {"copy_d2_p0",Candidate<2,0>},{"d2_p1",Candidate<2,1>},{"d2_p2",Candidate<2,2>},
  {"d1_p0",Candidate<1,0>},{"d1_p1",Candidate<1,1>},{"d1_p2",Candidate<1,2>},
  {"rt2_d1_p1",Candidate<1,1,2>}
};
#endif
bool Finite(const std::vector<float>& x) {
  return std::all_of(x.begin(),x.end(),[](float v){return std::isfinite(v);});
}
std::vector<float> Get(float* p,size_t n) {
  std::vector<float> x(n);
  HIP_CHECK(hipMemcpy(x.data(),p,n*sizeof(float),hipMemcpyDeviceToHost)); return x;
}
double Time(Launch launch,const __half* fn,const float* h,float* dots,float* ss) {
  bench::EventTimer timer; timer.Begin(); launch(fn,h,dots,ss);
  HIP_CHECK(hipGetLastError()); return timer.EndMs()*1000.;
}
bool Site(st::Checkpoint& ck,int site,Device<float>& h,Device<__half>& fn,
          Device<float>& dots,Device<float>& ss) {
  const std::string path="model.language_model.layers."+std::to_string(site/2)+"."+
      (site%2 ? "mlp_hyper_connection." : "attn_hyper_connection.");
  const auto weights=Weights(ck,path);
  HIP_CHECK(hipMemcpy(fn.p,weights.data(),weights.size()*2,hipMemcpyHostToDevice));
  const size_t nd=size_t(kRows)*kM*kStreams,ns=size_t(kRows)*kStreams;
  HIP_CHECK(hipMemset(dots.p,0xff,nd*4)); HIP_CHECK(hipMemset(ss.p,0xff,ns*4));
  Reference(fn.p,h.p,dots.p,ss.p); HIP_CHECK(hipGetLastError());
  const auto ref_dots=Get(dots.p,nd),ref_ss=Get(ss.p,ns);
  if (!Finite(ref_dots)||!Finite(ref_ss)) { std::fprintf(stderr,"nonfinite reference site=%d\n",site); return false; }
  bool pass=true;
#ifdef ISA_HC_PITCH
  Device<__half> padded(size_t(kM)*(kWide+64));
#endif
  for (size_t index=0;index<std::size(kVariants);++index) {
    const auto& v=kVariants[(site&1) ? std::size(kVariants)-1-index : index];
    const __half* current=fn.p;
#ifdef ISA_HC_PITCH
    if (v.stride) {
      std::vector<__half> packed(size_t(kM)*v.stride,__float2half(0.f));
      for (int row=0;row<kM;++row)
        std::copy_n(weights.data()+size_t(row)*kWide,kWide,packed.data()+size_t(row)*v.stride);
      HIP_CHECK(hipMemcpy(padded.p,packed.data(),packed.size()*sizeof(__half),hipMemcpyHostToDevice));
      current=padded.p;
    }
#endif
    HIP_CHECK(hipMemset(dots.p,0xff,nd*4)); HIP_CHECK(hipMemset(ss.p,0xff,ns*4));
    v.launch(current,h.p,dots.p,ss.p); HIP_CHECK(hipGetLastError());
    const auto got_dots=Get(dots.p,nd),got_ss=Get(ss.p,ns);
    const bool same=Finite(got_dots)&&Finite(got_ss)&&
        !std::memcmp(ref_dots.data(),got_dots.data(),nd*4)&&!std::memcmp(ref_ss.data(),got_ss.data(),ns*4);
    std::vector<double> baseline,candidate;
    for (int sample=0;sample<6;++sample) {
      // Alternate order to reduce systematic thermal and cache bias.
      for (int order=0;order<2;++order) {
        const bool base=((sample+order)&1)==0;
        (base ? baseline : candidate).push_back(Time(base ? Reference : v.launch,base ? fn.p : current,h.p,dots.p,ss.p));
      }
    }
    std::printf("site=%d variant=%s R=%d M=%d ldh=%d bitwise=%d baseline_us=%.3f candidate_us=%.3f\n",
        site,v.name,kRows,kM,kStride,same,bench::Median(baseline),bench::Median(candidate));
    std::fflush(stdout); pass=same&&pass;
  }
  return pass;
}
int main(int argc,char** argv) {
  int sites=96;
  if (argc<2||argc>3) { std::fprintf(stderr,"usage: isa_hc_regs CKPT [SITES:1..96]\n"); return 2; }
  if (argc==3) {
    const auto end=argv[2]+std::strlen(argv[2]); const auto result=std::from_chars(argv[2],end,sites);
    if (result.ec!=std::errc{}||result.ptr!=end||sites<1||sites>96) return 2;
  }
  st::Checkpoint ck(argv[1]);
  std::printf("reference=a38b9f7 pack=%s sites=%d synthetic_H=1 engine_gate=0\n",argv[1],sites);
  Device<float> h(size_t(kRows)*kStride),dots(size_t(kRows)*kM*kStreams),ss(size_t(kRows)*kStreams);
  Device<__half> fn(size_t(kM)*kWide);
  FillInput<<<(size_t(kRows)*kWide+255)/256,256>>>(h.p); HIP_CHECK(hipGetLastError());
  bool pass=true;
  for (int site=0;site<sites;++site) pass=Site(ck,site,h,fn,dots,ss)&&pass;
  return pass ? 0 : 1;
}
