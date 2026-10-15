// Bench-only N8/16/32 family. Use the existing N8 contraction as the byte oracle.
#include <cstring>
#include <random>
#include <stdexcept>
#include "exact_projection_rows.h"
#include "exact_dense_validate.h"
#include "../engine/exl3/plans.h"
#include "../engine/exl3/front_order.h"
#include "common.h"
#include "../engine/exl3/safetensors.h"
using namespace crucible;
namespace {
constexpr int kMaxN = 32;
constexpr double kBandwidth = 245.5e9;
struct Matrix { int I, O, K; uint32_t* tiles; __half *suh, *svh; size_t bytes; const uint8_t* raw; };
struct Plan {
  std::string name;
  std::vector<LinJob> jobs;
  LinJob* dj;
  GemvItem* di;
  XfItem *input, *output;
  size_t full_lds = 0, partial_count = 0, output_count = 0, bytes = 0;
  int ni, nin, nout, rows;
  bool front=false,complete=true;
};
template <typename T> T* Device(const std::vector<T>& values) { return Upload(values); }
template <typename T> T* Empty(size_t count) { return Zeros<T>(count); }

Matrix Load(st::Checkpoint& ck, const std::string& name) {
  const auto& info = ck.Info(name + ".trellis");
  if (info.dtype != "I16" || info.shape.size() != 3 || info.shape[2] % 16 ||
      info.shape[0] < 8 || info.shape[0] > 8192 || info.shape[1] < 8 || info.shape[1] > 32768 ||
      info.shape[2] < 32 || info.shape[2] > 96)
    throw std::runtime_error("bad trellis: " + name);
  const int A = info.shape[0], B = info.shape[1], K = info.shape[2] / 16, T = 8 * K;
  if (K < 2 || K > 6 || A % 8 || B % 8 || size_t(info.end-info.begin) != size_t(A)*B*T*4)
    throw std::runtime_error("unsupported matrix: " + name);
  const auto* src = reinterpret_cast<const uint8_t*>(ck.Data(name + ".trellis"));
  std::vector<uint32_t> tiles((size_t)A * B * T);
  for (int a = 0; a < A; ++a) for (int b = 0; b < B; ++b)
    std::memcpy(tiles.data() + ((size_t)b * A + a) * T, src + ((size_t)a * B + b) * T * 4, T * 4);
  auto scale = [&](const char* suffix, int count) {
    const auto& ti = ck.Info(name + suffix);
    if (ti.dtype != "F16" || size_t(ti.end - ti.begin) != size_t(count) * 2) throw std::runtime_error("bad scale: " + name);
    std::vector<__half> values(count);
    std::memcpy(values.data(), ck.Data(name + suffix), values.size() * 2);
    return Device(values);
  };
  return {A * 16, B * 16, K, Device(tiles), scale(".suh", A * 16), scale(".svh", B * 16), tiles.size() * 4, src};
}

uint32_t* Front(const Matrix& m,int split) {
  const int A=m.I/16,B=m.O/16,T=8*m.K;
  std::vector<uint32_t> host(m.bytes/4);
  for (int b=0;b<B;++b) for (int a=0;a<A;++a)
    std::memcpy(host.data()+FrontTile(b,a,A,B,split)*T,m.raw+((size_t)a*B+b)*T*4,T*4);
  return Device(host);
}
Plan Make(const std::string& name, const std::vector<Matrix>& matrices, int rows, int force_s = 0,bool front=false) {
  Plan p{}; p.name = name; p.rows = rows;
  const bool has6=std::any_of(matrices.begin(),matrices.end(),[](auto m){return m.K==6;});
  const bool lower=std::any_of(matrices.begin(),matrices.end(),[](auto m){return m.K<6;});
  p.front=front && name.find("shared")==std::string::npos && !(has6 && lower);
  int target = 160;
  for (int t : {160,80,40,20}) {
    long waves = 0;
    for (const auto& m : matrices) waves += long(m.O / 16) * SplitFor(m.I / 16, t);
    target = t; if (waves >= 900) break;
  }
  for (const auto& m:matrices) {
    const int split=force_s?force_s:SplitFor(m.I/16,target);
    if ((m.I/16/split)%4) p.front=false;
    p.complete=p.complete && wide_probe::CompleteChunks(m.I/16/split) && wide_probe::FullWaves(m.O);
  }
  std::vector<GemvItem> items; std::vector<XfItem> input, output;
  std::mt19937 rng(17); std::normal_distribution<float> normal(0.f, .125f);
  for (const auto& m : matrices) {
    LinJob j{}; j.I=m.I; j.O=m.O; j.K=m.K; j.R=rows;
    j.S=force_s ? force_s : SplitFor(m.I / 16, target);
    if ((m.I / 16) % j.S) throw std::runtime_error("invalid split");
    j.tiles=p.front?Front(m,j.S):m.tiles; j.suh=m.suh; j.svh=m.svh; j.cache_policy=4; j.lds=m.I; j.ldd=m.O;
    std::vector<float> raw((size_t)rows * m.I);
    for (auto& f : raw) f=normal(rng);
    for (int r=0; r<rows; ++r) { raw[(size_t)r*m.I]=0.f; raw[(size_t)r*m.I+1]=-0.f; }
    j.src=Device(raw); j.xh=Empty<__half>(raw.size());
    j.partial=Empty<float>((size_t)j.S*rows*m.O); j.dst=Empty<float>((size_t)rows*m.O);
    const int index=p.jobs.size(); p.jobs.push_back(j);
    for (int s=0; s<j.S; ++s) for (int b=0; b<m.O/16; b+=8) items.push_back({index,s,b,8});
    for (int r=0; r<rows; ++r) {
      for (int b=0; b<m.I/128; ++b) input.push_back({index,r,b});
      for (int b=0; b<m.O/128; ++b) output.push_back({index,r,b});
    }
    p.full_lds=std::max(p.full_lds, size_t(rows)*(m.I/16/j.S)*32);
    p.partial_count+=size_t(j.S)*rows*m.O; p.output_count+=size_t(rows)*m.O;
    p.bytes+=m.bytes+size_t(m.I+m.O)*2+size_t(rows)*(m.I*8ull+m.O*(8ull*j.S+4));
  }
  p.dj=Device(p.jobs); p.di=Device(items); p.input=Device(input); p.output=Device(output);
  p.ni=items.size(); p.nin=input.size(); p.nout=output.size();
  return p;
}

template <int N,bool Front> void Run(const Plan& p, int mode) {
  if (mode==3 && !p.complete) throw std::runtime_error("incomplete rolling chunk");
  LinInKernel<<<(p.nin+7)/8,256>>>(p.dj,p.input,p.nin);
  const size_t stripe_lds=p.full_lds*8/N;
  if (mode==0) wide_probe::StripeKernel<N,Front><<<dim3(p.ni,N/8),256,stripe_lds>>>(p.dj,p.di);
  else if (mode==1) wide_probe::RollingKernel<N,Front><<<p.ni,256,wide_probe::RollingLdsBytes(N)>>>(p.dj,p.di);
  else if (mode==3) wide_probe::RollingKernel<N,Front,true><<<p.ni,256,wide_probe::RollingLdsBytes(N)>>>(p.dj,p.di);
  else if (mode==4) wide_probe::StripeKernel<N,Front,true><<<p.ni*(N/8),256,stripe_lds>>>(p.dj,p.di);
  else wide_probe::FullKernel<N,Front><<<p.ni,256,p.full_lds>>>(p.dj,p.di);
  LinOutKernel<<<(p.nout+7)/8,256>>>(p.dj,p.output,p.nout);
  HIP_CHECK(hipGetLastError());
}
void Run(const Plan& p,int mode) {
  if (p.front) {
    if (p.rows==8) Run<8,true>(p,mode);
    else if (p.rows==16) Run<16,true>(p,mode);
    else Run<32,true>(p,mode);
  } else {
    if (p.rows==8) Run<8,false>(p,mode);
    else if (p.rows==16) Run<16,false>(p,mode);
    else Run<32,false>(p,mode);
  }
}
std::vector<uint32_t> Capture(const Plan& p) {
  std::vector<uint32_t> data(p.partial_count+p.output_count); size_t at=0;
  for (const auto& j:p.jobs) {
    const size_t count=size_t(j.S)*j.R*j.O;
    HIP_CHECK(hipMemcpy(data.data()+at,j.partial,count*4,hipMemcpyDeviceToHost)); at+=count;
    const size_t out=size_t(j.R)*j.O;
    HIP_CHECK(hipMemcpy(data.data()+at,j.dst,out*4,hipMemcpyDeviceToHost)); at+=out;
  }
  return data;
}
void Poison(const Plan& p,uint32_t value);
void Check(const Plan& p) {
  Poison(p,0x7fc00001);
  Run(p,0); const auto baseline=Capture(p);
  for (int mode=1; mode<=4; ++mode) {
    if ((mode==2 && p.full_lds>65536) || (mode==3 && !p.complete)) continue;
    Poison(p,0x7fc00002);
    Run(p,mode); const auto actual=Capture(p);
    size_t mismatches=0;
    for (size_t i=0; i<actual.size(); ++i)
      mismatches+=actual[i]!=baseline[i] || !wide_probe::FiniteWord(actual[i]) || !wide_probe::FiniteWord(baseline[i]);
    std::printf("check group=%s N=%d mode=%d S=%d words=%zu mismatches=%zu\n",p.name.c_str(),p.rows,mode,p.jobs[0].S,actual.size(),mismatches);
    if (mismatches) throw std::runtime_error("bitwise projection mismatch");
  }
}
void Free(const Plan& p) {
  for (const auto& j:p.jobs) for (void* ptr:{const_cast<float*>(j.src),static_cast<float*>(j.partial),j.dst}) HIP_CHECK(hipFree(ptr));
  for (const auto& j:p.jobs) {
    HIP_CHECK(hipFree(j.xh));
    if (p.front) HIP_CHECK(hipFree(const_cast<uint32_t*>(j.tiles)));
  }
  HIP_CHECK(hipFree(p.dj)); HIP_CHECK(hipFree(p.di)); HIP_CHECK(hipFree(p.input)); HIP_CHECK(hipFree(p.output));
}
using NamedGroup=std::pair<std::string,std::vector<Matrix>>;
std::vector<NamedGroup> Groups(st::Checkpoint& ck) {
  std::vector<NamedGroup> groups;
  for (int l=0; l<48; ++l) {
    const auto prefix="model.language_model.layers."+std::to_string(l);
    if (l%4==3) {
      groups.push_back({"qsa_in_"+std::to_string(l),{Load(ck,prefix+".self_attn.q_proj"),Load(ck,prefix+".self_attn.k_proj"),Load(ck,prefix+".self_attn.v_proj"),Load(ck,prefix+".self_attn.indexer.index_qk_proj")}});
      groups.push_back({"qsa_out_"+std::to_string(l),{Load(ck,prefix+".self_attn.o_proj")}});
    } else {
      groups.push_back({"gdn_in_"+std::to_string(l),{Load(ck,prefix+".linear_attn.in_proj_qkv"),Load(ck,prefix+".linear_attn.in_proj_z")}});
      groups.push_back({"gdn_out_"+std::to_string(l),{Load(ck,prefix+".linear_attn.out_proj")}});
    }
  }
  groups.push_back({"head",{Load(ck,"lm_head")}});
  groups.push_back({"shared_gu",{Load(ck,"model.language_model.layers.0.mlp.shared_expert.gate_proj"),Load(ck,"model.language_model.layers.0.mlp.shared_expert.up_proj")}});
  groups.push_back({"shared_down",{Load(ck,"model.language_model.layers.0.mlp.shared_expert.down_proj")}});
  return groups;
}
__global__ void PoisonWords(uint32_t* data,size_t count,uint32_t value) {
  for (size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<count;i+=size_t(gridDim.x)*blockDim.x) data[i]=value;
}
void Poison(const Plan& p,uint32_t value) {
  for (const auto& job:p.jobs) {
    PoisonWords<<<256,256>>>(reinterpret_cast<uint32_t*>(job.partial),size_t(job.S)*job.R*job.O,value);
    PoisonWords<<<256,256>>>(reinterpret_cast<uint32_t*>(job.dst),size_t(job.R)*job.O,value);
  }
  HIP_CHECK(hipGetLastError());
}
void CheckLayout(const Plan& p,const std::vector<Matrix>& matrices,int force_s=0) {
  if (!p.front) return;
  Plan engine=Make(p.name,matrices,p.rows,force_s,false);
  Poison(engine,0x7fc00001); Poison(p,0x7fc00002);
  Run(engine,0); Run(p,0);
  const auto expected=Capture(engine),actual=Capture(p);
  size_t bad=0;
  for (size_t i=0;i<expected.size();++i)
    bad+=expected[i]!=actual[i] || !wide_probe::FiniteWord(expected[i]) || !wide_probe::FiniteWord(actual[i]);
  Free(engine);
  std::printf("layout group=%s N=%d front=1 mismatches=%zu\n",p.name.c_str(),p.rows,bad);
  if (bad) throw std::runtime_error("front layout mismatch");
}
void Measure(const std::vector<Plan>& plans,int reps) {
  for (int mode:{0,1,3,4,2,2,4,3,1,0}) {
    std::vector<std::vector<double>> times(plans.size());
    for (int rep=0; rep<=reps; ++rep) for (size_t i=0;i<plans.size();++i) {
      if ((mode==2 && plans[i].full_lds>65536) || (mode==3 && !plans[i].complete)) continue;
      bench::EventTimer timer; timer.Begin(); Run(plans[i],mode);
      const double us=timer.EndMs()*1000; if (rep) times[i].push_back(us);
    }
    for (size_t i=0;i<plans.size();++i) if (!times[i].empty()) {
      const auto& p=plans[i]; const double us=bench::Median(times[i]),floor=p.bytes/kBandwidth*1e6;
      std::printf("time group=%s N=%d mode=%d front=%d bytes=%zu floor_us=%.4f median_us=%.4f DRAM_pct=%.3f lds=%zu S=%d\n",p.name.c_str(),p.rows,mode,int(p.front),p.bytes,floor,us,100*floor/us,(mode==1 || mode==3)?size_t(p.rows*kPrefetch*32):(mode==0 || mode==4)?p.full_lds*8/p.rows:p.full_lds,p.jobs[0].S);
    }
  }
}
} // namespace
int main(int argc,char** argv) {
  if (argc<3 || argc>5 || (std::string(argv[1])!="check" && std::string(argv[1])!="bench")) {
    std::fprintf(stderr,"usage: exact_projection_rows check|bench CKPT [repetitions] [front=0|1]\n"); return 2;
  }
  try {
    const int reps=argc>=4?std::stoi(argv[3]):3;
    const int front=argc==5?std::stoi(argv[4]):1;
    if (front<0 || front>1) throw std::runtime_error("front must be zero or one");
    if (reps<1 || reps>20) throw std::runtime_error("repetitions must be in [1,20]");
    st::Checkpoint ck(argv[2]); const auto groups=Groups(ck);
    for (int rows:{8,16,32}) {
      std::vector<Plan> plans;
      for (const auto& [name,matrices]:groups) {
        plans.push_back(Make(name,matrices,rows,0,front));
        Check(plans.back()); CheckLayout(plans.back(),matrices);
      }
      if (std::string(argv[1])=="bench") Measure(plans,reps);
      for (const auto& p:plans) Free(p);
    }
    for (const auto& [name,matrices]:groups) if (name=="gdn_in_0" || name=="shared_down")
      for (int split:{1,2,4,8}) for (int rows:{8,16,32}) {
        auto p=Make(name+"_split"+std::to_string(split),matrices,rows,split,front);
        Check(p); CheckLayout(p,matrices,split); Free(p);
      }
    std::puts("PASS: all partial and output words match independent N8 stripes");
    return 0;
  } catch (const std::exception& error) { std::fprintf(stderr,"FAIL: %s\n",error.what()); return 1; }
}
