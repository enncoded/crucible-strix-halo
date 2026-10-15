// Bench-only N16/32 instantiations of the unchanged FP16 dense row contraction.
#include <cstring>
#include <random>
#include <stdexcept>
#include "../engine/layers/dense_rows.h"
#include "../engine/exl3/plans.h"
#include "common.h"
#include "exact_dense_validate.h"
#include "../engine/exl3/safetensors.h"
using namespace crucible;
namespace {
constexpr double kBandwidth=245.5e9;
struct Dense { std::string name; __half* weights; int outputs; };
Dense Load(st::Checkpoint& ck,const std::string& name,const std::vector<std::string>& parts) {
  std::vector<__half> weights;
  for (const auto& part:parts) {
    const auto& info=ck.Info(part);
    if (info.dtype!="F16" || info.shape.size()!=2 || info.shape[1]!=kHidden ||
        info.shape[0]<1 || info.shape[0]>16384 || size_t(info.end-info.begin)!=size_t(info.shape[0])*kHidden*2)
      throw std::runtime_error("bad dense matrix: "+part);
    const size_t at=weights.size(); weights.resize(at+size_t(info.shape[0])*kHidden);
    std::memcpy(weights.data()+at,ck.Data(part),size_t(info.shape[0])*kHidden*2);
  }
  return {name,Upload(weights),int(weights.size()/kHidden)};
}
std::vector<Dense> Load(st::Checkpoint& ck) {
  std::vector<Dense> matrices;
  for (int layer=0;layer<48;++layer) {
    const auto prefix="model.language_model.layers."+std::to_string(layer);
    matrices.push_back(Load(ck,"router_"+std::to_string(layer),
      {prefix+".mlp.gate.weight",prefix+".mlp.shared_expert_gate.weight"}));
    if (layer%4!=3) matrices.push_back(Load(ck,"ab_"+std::to_string(layer),
      {prefix+".linear_attn.in_proj_a.weight",prefix+".linear_attn.in_proj_b.weight"}));
  }
  const std::string prefix="model.language_model.layers.1.ple.";
  matrices.push_back(Load(ck,"ple_kv",{prefix+"key_proj.weight",prefix+"value_proj.weight"}));
  return matrices;
}
void Run(const Dense& matrix,int rows,int mode,const float* input,int ldx,float* output,int ldy) {
  if (mode==2) {
    for (int row=0;row<rows;++row) DenseGemvKernel<1><<<(matrix.outputs+7)/8,256>>>(
      matrix.weights,matrix.outputs,input+size_t(row)*ldx,ldx,output+size_t(row)*ldy,ldy);
  } else if (mode==0) {
    for (int row=0;row<rows;row+=8) LaunchDenseRows<8,1>(matrix.weights,matrix.outputs,
      input+size_t(row)*ldx,ldx,output+size_t(row)*ldy,ldy);
  } else if (rows==8) LaunchDenseRows<8,1>(matrix.weights,matrix.outputs,input,ldx,output,ldy);
  else if (rows==16) LaunchDenseRows<16,1>(matrix.weights,matrix.outputs,input,ldx,output,ldy);
  else LaunchDenseRows<32,1>(matrix.weights,matrix.outputs,input,ldx,output,ldy);
  HIP_CHECK(hipGetLastError());
}
void Check(const Dense& matrix,int rows,const float* input,int ldx,float* output,int ldy) {
  const size_t count=size_t(rows)*ldy;
  constexpr uint32_t reference_poison=0x7fc00001,candidate_poison=0x7fc00002;
  std::vector<uint32_t> expected(count,reference_poison),actual(count);
  const std::vector<uint32_t> seed(count,candidate_poison);
  HIP_CHECK(hipMemcpy(output,expected.data(),count*4,hipMemcpyHostToDevice));
  Run(matrix,rows,2,input,ldx,output,ldy);
  HIP_CHECK(hipMemcpy(expected.data(),output,count*4,hipMemcpyDeviceToHost));
  for (int mode:{0,1}) {
    HIP_CHECK(hipMemcpy(output,seed.data(),count*4,hipMemcpyHostToDevice));
    Run(matrix,rows,mode,input,ldx,output,ldy);
    HIP_CHECK(hipMemcpy(actual.data(),output,count*4,hipMemcpyDeviceToHost));
    const size_t bad=wide_probe::DenseMismatch(expected,actual,rows,matrix.outputs,ldy,reference_poison,candidate_poison);
    std::printf("check dense=%s N=%d mode=%d O=%d mismatches=%zu finite=checked padding=checked\n",
      matrix.name.c_str(),rows,mode,matrix.outputs,bad);
    if (bad) throw std::runtime_error("dense live row or padding mismatch");
  }
}
void Measure(const std::vector<Dense>& matrices,int rows,const float* input,int ldx,float* output,int ldy) {
  for (int mode:{0,1,1,0}) {
    std::vector<std::vector<double>> times(matrices.size());
    for (int rep=0;rep<4;++rep) for (size_t i=0;i<matrices.size();++i) {
      bench::EventTimer timer; timer.Begin(); Run(matrices[i],rows,mode,input,ldx,output,ldy);
      const double us=timer.EndMs()*1000; if (rep) times[i].push_back(us);
    }
    for (size_t i=0;i<matrices.size();++i) {
      const auto& matrix=matrices[i];
      const size_t bytes=size_t(matrix.outputs)*kHidden*2+size_t(rows)*(kHidden+matrix.outputs)*4;
      const double floor=bytes/kBandwidth*1e6,us=bench::Median(times[i]);
      std::printf("time dense=%s N=%d mode=%d bytes=%zu floor_us=%.4f median_us=%.4f DRAM_pct=%.3f\n",
        matrix.name.c_str(),rows,mode,bytes,floor,us,100*floor/us);
    }
  }
}
} // namespace
int main(int argc,char** argv) {
  if (argc!=3 || (std::string(argv[1])!="check" && std::string(argv[1])!="bench")) {
    std::fprintf(stderr,"usage: exact_dense_rows check|bench CKPT\n"); return 2;
  }
  try {
    st::Checkpoint ck(argv[2]); const auto matrices=Load(ck);
    constexpr int ldx=kHidden+32,ldy=12800+17;
    std::mt19937 rng(17); std::normal_distribution<float> normal(0.f,.125f);
    std::vector<float> raw(32*ldx); for (auto& f:raw) f=normal(rng);
    for (int row=0;row<32;++row) { raw[row*ldx]=0.f;raw[row*ldx+1]=-0.f; }
    float* input=Upload(raw); float* output=Zeros<float>(32*ldy);
    for (int rows:{8,16,32}) {
      for (const auto& matrix:matrices) Check(matrix,rows,input,ldx,output,ldy);
      if (std::string(argv[1])=="bench") Measure(matrices,rows,input,ldx,output,ldy);
    }
    std::puts("PASS: finite dense rows match independent R1; per-run padding stays poisoned"); return 0;
  } catch (const std::exception& error) { std::fprintf(stderr,"FAIL: %s\n",error.what()); return 1; }
}
