// Separate stream and external-stack checks. No cache or serving hooks run here.
#define ENGINE_LIBRARY
#include "../engine/engine.hip"
#include <filesystem>
#include <cassert>

namespace {
struct EngineStateIO {
  static SequenceState& State(Engine& e, int stream) { return *e.sequences_.at(stream); }
};
struct Observation { std::vector<int> ids; std::vector<float> logits; };
Observation Continue(Engine& e, const std::vector<int>& ids, int rows, int replay, bool bulk,
                     const std::string& dump) {
  Observation out;
  for (size_t i=0; i<ids.size();) {
    const int r=std::min<size_t>(rows,ids.size()-i);
    const auto y=bulk ? e.Prefill(ids.data()+i,r,dump) : e.Forward(ids.data()+i,r,replay,r,dump);
    replay=0;
    out.ids.insert(out.ids.end(),y.begin(),y.end());
    const int count=bulk ? 1 : r;
    const size_t start=out.logits.size();
    out.logits.resize(start+(size_t)count*kVocab);
    HIP_OK(hipMemcpy(out.logits.data()+start,e.logits(),(size_t)count*kVocab*sizeof(float),hipMemcpyDeviceToHost));
    i+=r;
  }
  return out;
}
bool DumpsEqual(const std::string& a, const std::string& b) {
  for (int l=0; l<=kLayers; ++l) {
    char name[32]; std::snprintf(name,sizeof(name),"/h_%02d.f32",l);
    std::ifstream fa(a+name,std::ios::binary), fb(b+name,std::ios::binary);
    const std::string av(std::istreambuf_iterator<char>(fa),{}), bv(std::istreambuf_iterator<char>(fb),{});
    if (av.empty() || av!=bv) return false;
  }
  return true;
}
}
int main(int argc, char** argv) {
  if (argc!=4 || std::filesystem::exists(argv[3])) return 2;
  setenv("HSA_USERPTR_FOR_PAGED_MEM","0",0);
  std::ifstream file(argv[2]); std::stringstream text; text<<file.rdbuf();
  const auto source=ParseIds(text.str());
  if (source.size()<24) return 2;
  Engine e(argv[1],128,64);
  e.ConfigureStreams(2);
  float* stk=Zeros<float>((kMaxRows+1)*kWide);
  float* carry=Zeros<float>(kWide);
  float* peer_stk=Zeros<float>((kMaxRows+1)*kWide);
  float* peer_carry=Zeros<float>(kWide);
  for (int kind=0; kind<3; ++kind) {
    e.SelectStream(0); e.Reset();
    const int length=kind==2 ? 64 : 14;
    std::vector<int> prefix(length);
    for (int i=0; i<length; ++i) prefix[i]=source[i%source.size()];
    Continue(e,prefix,kind==2 ? 64 : 8,0,kind==2,"");
    int replay=0;
    if (kind==1) { e.Forward(source.data(),3,0,1); replay=2; }
    e.Mtp(stk,source.data(),3,0,false);
    ConversationResume resume{e.Length(),3,replay,0,false,{},{}};
    const auto saved=ConversationStateIO::Save(e,EngineStateIO::State(e,0),resume,stk,carry);
    const std::vector<int> suffix(source.begin()+14,source.begin()+24);
    const std::string root=std::string(argv[3])+"/"+std::to_string(kind);
    std::filesystem::create_directories(root+"/a");
    std::filesystem::create_directories(root+"/b");
    const auto a=Continue(e,suffix,kind==2 ? 10 : 4,replay,kind==2,root+"/a");
    e.Forward(source.data()+9,3,0,1);
    e.SelectStream(1); e.Reset();
    Continue(e,suffix,4,0,false,"");
    ConversationResume peer_resume{e.Length(),0,0,0,false,{},{}};
    const auto peer_before=ConversationStateIO::Save(e,EngineStateIO::State(e,1),peer_resume,peer_stk,peer_carry);
    const auto restored=ConversationStateIO::Restore(e,EngineStateIO::State(e,0),*saved,stk,carry);
    if (restored.mpos!=3 || restored.replay!=replay || e.Length()!=peer_resume.length) return 1;
    const auto peer_after=ConversationStateIO::Save(e,EngineStateIO::State(e,1),peer_resume,peer_stk,peer_carry);
    if (peer_before->hist!=peer_after->hist || peer_before->bytes!=peer_after->bytes) return 1;
    const auto restored_state=ConversationStateIO::Save(e,EngineStateIO::State(e,0),restored,stk,carry);
    if (restored_state->bytes!=saved->bytes || restored_state->hist!=saved->hist) return 1;
    e.SelectStream(0);
    const auto b=Continue(e,suffix,kind==2 ? 10 : 4,replay,kind==2,root+"/b");
    if (a.ids!=b.ids || a.logits.size()!=b.logits.size() ||
        std::memcmp(a.logits.data(),b.logits.data(),a.logits.size()*sizeof(float)) ||
        !DumpsEqual(root+"/a",root+"/b")) return 1;
    std::printf("PASS explicit snapshot kind=%d: 49 dumps, %zu logits, %zu IDs, mpos=3, peer state unchanged\n",
                kind,a.logits.size(),a.ids.size());
  }
  for (void* p : {stk,carry,peer_stk,peer_carry}) HIP_OK(hipFree(p));
}
