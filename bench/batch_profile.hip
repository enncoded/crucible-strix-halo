// Decode profile for independent requests. Event profiling is separate from wall timing.
#define ENGINE_LIBRARY
#include "../engine/engine.hip"
#include <numeric>
#ifdef CRUCIBLE_ROCTX
#include <rocprofiler-sdk-roctx/roctx.h>
#endif

namespace {
using Prompts=std::vector<std::vector<int>>;
Prompts Load(const std::string& path) {
  Prompts prompts;
  std::ifstream paths(path);
  for (std::string name; std::getline(paths,name);) {
    if (name.empty()) continue;
    std::ifstream file(name); std::stringstream text; text<<file.rdbuf();
    const auto ids=ParseIds(text.str());
    if (!file || !ValidTokenIds(ids,kVocab)) std::exit(2);
    prompts.push_back(ids);
  }
  if (!paths.eof() || prompts.size()!=8) std::exit(2);
  prompts[6].push_back(kEos); prompts[7].insert(prompts[7].end(),{kEos,1,kEos});
  return prompts;
}
std::vector<int> Prepare(Engine& e, const Prompts& prompts) {
  std::vector<int> tokens(8);
  for (int s=0; s<8; ++s) {
    e.SelectStream(s); e.Reset();
    const auto& ids=prompts[s];
    for (size_t i=0; i<ids.size(); i+=8) {
      const int r=std::min<size_t>(8,ids.size()-i);
      tokens[s]=e.Forward(ids.data()+i,r,0,r).back();
    }
  }
  return tokens;
}
}
#ifndef BATCH_PROFILE_INPUTS_ONLY
int main(int argc, char** argv) {
  if (argc!=3) return 2;
  setenv("HSA_USERPTR_FOR_PAGED_MEM","0",0);
  const auto prompts=Load(argv[2]);
  Engine e(argv[1],4096);
  e.ConfigureStreams(8); e.WarmPlans();
  std::vector<int> streams(8); std::iota(streams.begin(),streams.end(),0);
  auto tokens=Prepare(e,prompts);
  tokens=e.ForwardBatch(tokens,streams);
  constexpr int steps=32;
#ifdef CRUCIBLE_ROCTX
  roctxRangePush("batch_decode_32");
#endif
  const double start=Now();
  for (int i=0; i<steps; ++i) tokens=e.ForwardBatch(tokens,streams);
  const double seconds=Now()-start;
#ifdef CRUCIBLE_ROCTX
  roctxRangePop();
#endif
  std::printf("batch_profile B=8 steps=%d wall_ms=%.6f aggregate=%.6f per_stream=%.6f\n",
              steps,1000*seconds/steps,8*steps/seconds,steps/seconds);
  tokens=Prepare(e,prompts);
  tokens=e.ForwardBatch(tokens,streams);
  e.CountExperts(true); e.ClearExpertCounts(); e.EnableProfile();
  const double prof_start=Now();
  for (int i=0; i<steps; ++i) tokens=e.ForwardBatch(tokens,streams);
  const double prof_seconds=Now()-prof_start;
  e.Report(steps);
  const auto counts=e.ExpertCounts();
  const auto unions=counts[2*8], calls=counts[2*8+1];
  if (calls!=size_t(steps*kLayers)) return 1;
  std::printf("batch_profile prof_wall_ms=%.6f union_sum=%llu union_calls=%llu mean_union=%.6f\n",
              1000*prof_seconds/steps,unions,calls,double(unions)/calls);
  std::printf("batch_profile temporal_R=1 projection_R=8 routed_R=compact\n");
}
#endif
