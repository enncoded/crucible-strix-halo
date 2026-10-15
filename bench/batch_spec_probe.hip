// Default native probe. Define CRUCIBLE_BATCH_SPEC_BENCH only for the candidate.
#define ENGINE_LIBRARY
#include "../engine/engine.hip"
#include <bit>
#include <filesystem>
#include "exact_dense_validate.h"
#include "batch_spec_provenance.h"
#ifndef CRUCIBLE_BATCH_SPEC_SOURCE_SHA
#error "Build the probe with its complete source digest."
#endif
namespace fs=std::filesystem;
using namespace crucible;
namespace {
struct RequestState {
  int slot,mpos=0,replay=0,step=0;
  long accepted=0,proposed=0;
  size_t base=0;
  float *stacks=nullptr,*carry=nullptr;
  std::vector<int> prompt,emitted,pending,input;
};
void Copy(float* dst,const float* src,size_t count) {
  HIP_OK(hipMemcpyAsync(dst,src,count*sizeof(float),hipMemcpyDeviceToDevice,0));
}
template<class T> void Write(const fs::path& path,const std::vector<T>& data) {
  std::ofstream file(path,std::ios::binary);
  file.write(reinterpret_cast<const char*>(data.data()),data.size()*sizeof(T));
  if (!file) throw std::runtime_error("cannot write probe record: "+path.string());
}
std::vector<int> ReadIds(const fs::path& path) {
  std::ifstream file(path); if (!file) throw std::runtime_error("cannot read probe IDs: "+path.string());
  std::stringstream buffer;buffer<<file.rdbuf();
  auto text=buffer.str(); std::replace(text.begin(),text.end(),',',' ');
  std::stringstream fields(text);std::vector<int> ids;std::string field;
  while (fields>>field) ids.push_back(BatchInteger(field));
  if (ids.empty() || !ValidTokenIds(ids,kVocab)) throw std::invalid_argument("invalid probe prompt");
  return ids;
}
void Init(Engine& engine,RequestState& state) {
  engine.SelectStream(state.slot);engine.Reset();
  state.stacks=Zeros<float>(size_t(kMaxRows+1)*kWide);state.carry=Zeros<float>(kWide);
  const auto ids=RunPrompt(engine,std::span<const int>(state.prompt),8,true,"",state.stacks,state.carry,state.mpos,Copy,
    [&](float* stacks,const int* tokens,int rows,int pos){engine.Mtp(stacks,tokens,rows,pos);},[](int){});
  state.emitted={ids.back()};state.pending=state.emitted;
  Copy(state.stacks,state.carry,kWide);
}
void Draft(Engine& engine,RequestState& state,int spec) {
  engine.SelectStream(state.slot);state.base=engine.Length();
  const int count=state.pending.size();
  const int depth=BoundedDraftDepth(spec,engine.Context(),int(state.base),state.mpos,count);
  state.input={state.emitted.back()};
  if (!depth) return;
  state.proposed+=depth;
  std::vector<int> draft{engine.Mtp(state.stacks,state.pending.data(),count,state.mpos)[count-1]};
  state.mpos+=count;
  for (int row=1;row<depth;++row) {
    Copy(state.stacks+size_t(kMaxRows)*kWide,engine.H()+size_t(row==1?count-1:0)*kWide,kWide);
    draft.push_back(engine.Mtp(state.stacks+size_t(kMaxRows)*kWide,&draft.back(),1,state.mpos+row-1)[0]);
  }
  state.input.insert(state.input.end(),draft.begin(),draft.end());
}
void Capture(Engine& engine,const RequestState& state,int first,const std::vector<int>& output,const fs::path& root) {
  const auto path=root/("request_"+std::to_string(state.slot))/("step_"+std::to_string(state.step));
  fs::create_directories(path);
  const auto logits=engine.CopyLogits(state.input.size(),first);
  for (float value:logits) if (!wide_probe::FiniteWord(std::bit_cast<uint32_t>(value))) throw std::runtime_error("nonfinite probe logit");
  std::vector<float> stacks(state.input.size()*kWide);
  HIP_OK(hipMemcpy(stacks.data(),engine.H()+size_t(first)*kWide,stacks.size()*sizeof(float),hipMemcpyDeviceToHost));
  Write(path/"input.i32",state.input);Write(path/"target.i32",output);Write(path/"logits.f32",logits);Write(path/"stacks.f32",stacks);
  engine.SelectStream(state.slot);
  const auto& sequence=engine.ActiveSequence();
  Write(path/"history.i32",sequence.hist);
  const auto spans=sequence.Spans();
  for (size_t index=0;index<spans.size();++index) {
    std::vector<uint8_t> bytes(spans[index].bytes);
    HIP_OK(hipMemcpy(bytes.data(),spans[index].pointer,bytes.size(),hipMemcpyDeviceToHost));
    Write(path/("state_"+std::to_string(index)+".bin"),bytes);
  }
  std::ofstream schedule(path/"schedule.txt");
  schedule<<"request="<<state.slot<<" generation=1 base="<<state.base<<" rows="<<state.input.size()<<" replay="<<state.replay<<" mpos="<<state.mpos<<"\n";
  if (!schedule) throw std::runtime_error("cannot write probe schedule");
}
void Accept(Engine& engine,RequestState& state,int first,const std::vector<int>& output) {
  int accepted=0;
  while (size_t(accepted+1)<state.input.size() && state.input[accepted+1]==output[accepted]) ++accepted;
  engine.SelectStream(state.slot);engine.Truncate(state.base+accepted+1);
  Copy(state.stacks,engine.H()+size_t(first)*kWide,size_t(accepted+1)*kWide);
  state.pending.assign(output.begin(),output.begin()+accepted+1);
  state.emitted.insert(state.emitted.end(),state.pending.begin(),state.pending.end());
  state.accepted+=accepted;
  state.replay=accepted;++state.step;
}
void Finish(RequestState& state,int count,const fs::path& root) {
  const std::vector<int> published(state.emitted.begin(),state.emitted.begin()+count);
  Write(root/("ids_"+std::to_string(state.slot)+".i32"),published);
  HIP_OK(hipFree(state.stacks));HIP_OK(hipFree(state.carry));
}
double Run(Engine& engine,std::vector<RequestState>& requests,int count,int spec,bool packed,bool capture,const fs::path& root) {
  if (capture) engine.CountExperts(true);
  for (auto& state:requests) Init(engine,state);
  if (capture) engine.ClearExpertCounts();
  engine.Synchronize();const double start=GenerationNow();
  if (!packed) for (auto& state:requests) while (int(state.emitted.size())<count) {
    Draft(engine,state,spec);
    const auto output=engine.Forward(state.input.data(),state.input.size(),state.replay,1);
    if (capture) Capture(engine,state,0,output,root);
    Accept(engine,state,0,output);
  }
  else {
#ifdef CRUCIBLE_BATCH_SPEC_BENCH
    while (true) {
      std::vector<VerifySpan> spans;std::vector<int> active;
      for (auto& state:requests) if (int(state.emitted.size())<count) {
        Draft(engine,state,spec);active.push_back(state.slot);spans.push_back({state.slot,state.input,state.replay,1});
      }
      if (spans.empty()) break;
      const auto output=engine.ForwardVerifySpans(spans);
      int first=0;
      for (int slot:active) {
        auto& state=requests.at(slot);
        const std::vector<int> rows(output.begin()+first,output.begin()+first+state.input.size());
        if (capture) Capture(engine,state,first,rows,root);
        Accept(engine,state,first,rows);first+=state.input.size();
      }
    }
#else
    throw std::invalid_argument("native reference cannot run packed verification");
#endif
  }
  engine.Synchronize();const double elapsed=GenerationNow()-start;
  if (capture) Write(root.parent_path()/(root.filename().string()+"_expert_counts.u64"),engine.ExpertCounts());
  for (auto& state:requests) {
    engine.SelectStream(state.slot);
    auto prefix=engine.ActiveSequence().retained;
    if (const auto error=prefix.Forward(1,state.replay,1); !error.empty()) throw std::runtime_error(error);
  }
  for (auto& state:requests) Finish(state,count,root);
  std::printf("probe mode=%s capture=%d streams=%zu spec=%d seconds=%.6f aggregate_tok_s=%.3f\n",
    packed?"packed":"alone",int(capture),requests.size(),spec,elapsed,requests.size()*double(count-1)/elapsed);
  return elapsed;
}
#include "batch_spec_trials.h"
} // namespace
int main(int argc,char** argv) {
  if (argc<6 || argc>9) {std::fprintf(stderr,"usage: batch_spec_probe CKPT PATHS OUT alone|packed capture|bench [tokens=24] [spec=3] [blocks]\n");return 2;}
  try {
    const fs::path out(argv[3]);if (fs::exists(out)) throw std::invalid_argument("use a new probe output path");
    const std::string mode(argv[4]),kind(argv[5]);
    if ((mode!="alone" && mode!="packed" && mode!="paired") || (kind!="capture" && kind!="bench") || (mode=="paired" && kind!="bench")) throw std::invalid_argument("invalid probe mode");
    const int count=argc>6?BatchInteger(argv[6]):24,spec=argc>7?BatchInteger(argv[7]):3;
    if (count<2 || count>64 || spec<1 || spec>3) throw std::invalid_argument("require tokens 2..64 and spec 1..3");
    std::ifstream paths(argv[2]);if (!paths) throw std::runtime_error("cannot open probe path list");
    std::vector<RequestState> requests;
    std::string path;
    while (requests.size()<2 && std::getline(paths,path)) if (!path.empty()) {
      RequestState request;request.slot=requests.size();request.prompt=ReadIds(path);
      if (!RequestFitsContext(request.prompt.size(),count,1024)) throw std::invalid_argument("probe context exceeds 1024");
      requests.push_back(std::move(request));
    }
    if (requests.size()!=2) throw std::invalid_argument("probe needs at least two prompt paths");
    std::vector<int> blocks;
    if (argc==9) {std::ifstream file(argv[8]);if (!file) throw std::runtime_error("cannot open probe draft blocks");std::stringstream text;text<<file.rdbuf();blocks=ParseDraftSelection(text.str());}
    if (!blocks.empty() && !ValidDraftSelection(blocks,kVocab/128)) throw std::invalid_argument("invalid probe draft selection");
    fs::create_directories(out);
    spec_probe::Provenance(out,mode,CRUCIBLE_BATCH_SPEC_SOURCE_SHA,argc,argv);
    Engine engine(argv[1],1024);engine.ConfigureStreams(2);engine.WarmPlans();engine.DraftHead(blocks,true,false,false);
    std::printf("draft_columns=%d policy=fixed temperature=0 lookup=0 overlap=0 head_auto=0\n",blocks.empty()?kVocab:int(blocks.size()*128));
    if (mode=="paired") PairedTrials(engine,requests,count,spec,blocks,out);
    else Run(engine,requests,count,spec,mode=="packed",kind=="capture",out);
    return 0;
  } catch(const std::exception& error) {std::fprintf(stderr,"batch spec probe: %s\n",error.what());return 1;}
}
