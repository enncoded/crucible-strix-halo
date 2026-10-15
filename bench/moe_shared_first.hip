// Compare static shared-first dispatch without host route capture.
#define ENGINE_LIBRARY
#include "../engine/engine.hip"
#include <array>
#include <cmath>

namespace {
struct Saved {void* target;uint8_t* copy;size_t bytes;};
struct Snapshot {
  std::vector<Saved> buffers;
  std::vector<std::vector<int>> hist;
  std::vector<RetainedPrefix> retained;
};
struct ItemChoice {Group* group;GemvItem* original;GemvItem* shared;int rows;};
std::vector<ItemChoice> item_choices;
std::vector<void*> extra_allocations;
struct EngineStateIO {
  static Snapshot Save(Engine& e) {
    Snapshot out;
    for(const auto& s:e.sequences_){out.hist.push_back(s->hist);out.retained.push_back(s->retained);}
    for(const auto& b:e.buffers_)if(b.pointer&&b.bytes){
      uint8_t* copy;HIP_OK(hipMalloc(&copy,b.bytes));HIP_OK(hipMemcpy(copy,b.pointer,b.bytes,hipMemcpyDeviceToDevice));
      out.buffers.push_back({const_cast<void*>(b.pointer),copy,b.bytes});}
    return out;
  }
  static void Restore(Engine& e,const Snapshot& s) {
    for(const auto& b:s.buffers)HIP_OK(hipMemcpyAsync(b.target,b.copy,b.bytes,hipMemcpyDeviceToDevice));
    for(size_t i=0;i<e.sequences_.size();++i){e.sequences_[i]->hist=s.hist[i];e.sequences_[i]->retained=s.retained[i];}
    e.sequence_=e.sequences_[0].get();e.batch_streams_.clear();HIP_OK(hipDeviceSynchronize());
  }
  static void Items(Engine& e) {
    for(auto& [r,p]:e.plans_)for(auto& l:p->L)for(int down=0;down<2;++down){
      Group& g=down?l.dn:l.gu;std::vector<GemvItem> items;
      for(size_t j=0;j<g.jobs.size();++j)for(int s=0;s<g.jobs[j].S;++s)
        for(int b=0;b<g.jobs[j].O/16;b+=kMoeItemBlocks<kMoeCols>)
          items.push_back({int(j),s,b,std::min(kMoeItemBlocks<kMoeCols>,g.jobs[j].O/16-b)});
      if(items.size()!=size_t(g.nitems)){fprintf(stderr,"static item count changes\n");std::exit(2);}
      const auto shared=MoeSharedFirstItems(items,int(g.jobs.size())-(down?1:2));
      const auto& expected=r>1?shared:items;std::vector<GemvItem> production(g.nitems);
      HIP_OK(hipMemcpy(production.data(),g.d_items,production.size()*sizeof(GemvItem),hipMemcpyDeviceToHost));
      static_assert(sizeof(GemvItem)==4*sizeof(int));
      if(std::memcmp(production.data(),expected.data(),production.size()*sizeof(GemvItem))!=0){
        fprintf(stderr,"production item order differs from the resident probe at R=%d\n",r);std::exit(2);}
      auto* old=crucible::Upload(items);auto* first=crucible::Upload(shared);
      extra_allocations.push_back(old);extra_allocations.push_back(first);
      item_choices.push_back({&g,old,first,r});
    }
  }
  static void Mode(Engine& e,int mode) {
    if(mode!=0&&mode!=1){fprintf(stderr,"invalid static order mode\n");std::exit(2);}
    for(const auto& c:item_choices)c.group->d_items=mode&&c.rows>1?c.shared:c.original;
  }
  static std::vector<uint8_t> Output(Engine& e,int r) {
    const size_t h=r*kWide*sizeof(float),logits=r*kVocab*sizeof(float);std::vector<uint8_t> out(h+logits);
    HIP_OK(hipMemcpy(out.data(),e.H_,h,hipMemcpyDeviceToHost));
    HIP_OK(hipMemcpy(out.data()+h,e.logits_,logits,hipMemcpyDeviceToHost));
    for(size_t i=0;i<out.size();i+=sizeof(float)){
      float value;std::memcpy(&value,out.data()+i,sizeof(value));
      if(!std::isfinite(value)){fprintf(stderr,"nonfinite final H or logits\n");std::exit(1);}}
    return out;
  }
};
double Median(std::vector<double> v){std::sort(v.begin(),v.end());const int n=v.size();return n%2?v[n/2]:(v[n/2-1]+v[n/2])/2;}
std::vector<int> Forward(Engine& e,int workload) {
  const std::array<int,8> tok{71093,12305,198,727,73111,1393,25,514};
  if(workload==2)return e.ForwardBatch(std::vector<int>(tok.begin(),tok.end()),{0,1,2,3,4,5,6,7});
  const int r=workload?8:1;return e.Forward(tok.data(),r,0,r);
}
void Check(Engine& e,const Snapshot& prefix,int work) {
  const char* label=work==2?"B8":work?"R8":"R1";const int rows=work?8:1;
  EngineStateIO::Mode(e,0);EngineStateIO::Restore(e,prefix);const auto ids=Forward(e,work);
  const auto baseline=EngineStateIO::Output(e,rows);
  for(int mode:{1}){
    EngineStateIO::Mode(e,mode);EngineStateIO::Restore(e,prefix);
    if(Forward(e,work)!=ids||EngineStateIO::Output(e,rows)!=baseline){
      fprintf(stderr,"mode %d changes %s H, logits, or IDs\n",mode,label);std::exit(1);}
    printf("EQUALITY %s mode=%d H=1 logits=1 ids=1\n",label,mode);
  }
}
double Pairs(Engine& e,const Snapshot& prefix,int work,int reps) {
  const char* label=work==2?"B8":work?"R8":"R1";
  std::vector<double> base,cut,delta;
  for(int rep=0;rep<reps+4;++rep){
    double ms[2]{};
    for(int order=0;order<2;++order){
      const int candidate=(order+(rep&1))&1;EngineStateIO::Mode(e,candidate);
      EngineStateIO::Restore(e,prefix);const double start=Now();Forward(e,work);ms[candidate]=(Now()-start)*1000;}
    if(rep>=4){base.push_back(ms[0]);cut.push_back(ms[1]);delta.push_back(ms[0]-ms[1]);}
  }
  printf("SHARED %s base_ms=%.6f candidate_ms=%.6f paired_saved_ms=%.6f\n",label,Median(base),Median(cut),Median(delta));
  printf("SHARED_SAMPLES %s",label);
  for(int i=0;i<reps;++i)printf(" %.6f/%.6f",base[i],cut[i]);printf("\n");fflush(stdout);
  return Median(delta);
}
void LiveSpec(Engine& e,const Snapshot& prefix,const std::vector<int>& prompt) {
  auto options=CliGenOptions();options.ids=prompt;options.ctx=4096;options.n=112;options.prefill_rows=0;
  std::vector<int> reference;
  for(int round=0;round<7;++round)for(int order=0;order<2;++order){
    const int mode=(order+(round&1))&1;
    EngineStateIO::Mode(e,mode);EngineStateIO::Restore(e,prefix);const auto result=RunGen(e,options,{});
    if(result.gen.size()!=size_t(options.n)){fprintf(stderr,"live spec-7 ID count changes\n");std::exit(1);}
    if(reference.empty())reference=result.gen;
    else if(reference!=result.gen){fprintf(stderr,"live spec-7 IDs change\n");std::exit(1);}
    if(round)printf("SHARED_SPEC mode=%d round=%d tok_s=%.6f steps=%ld decode_ms=%.6f ids_equal=1\n",mode,round,(options.n-1)/result.decode_s,result.forwards,1000*result.decode_s);
  }
}
void LiveBatch(Engine& e,const Snapshot& prefix,const std::vector<int>& prompt) {
  BatchOptions options;options.streams=8;options.n=33;options.ctx=4096;options.chunk=8;options.prefill_rows=0;
  options.prompts=std::vector<std::vector<int>>(8,prompt);
  std::vector<std::vector<int>> reference;
  for(int round=0;round<3;++round)for(int order=0;order<2;++order){
    const int mode=(order+(round&1))&1;
    EngineStateIO::Mode(e,mode);EngineStateIO::Restore(e,prefix);std::vector<int> tokens(8);
    PrefillBatch(e,options,tokens);std::vector<std::vector<int>> generated(8);
    for(int s=0;s<8;++s)generated[s].push_back(tokens[s]);
    const double start=Now();
    for(int i=1;i<options.n;++i){tokens=e.ForwardBatch(tokens,{0,1,2,3,4,5,6,7});
      for(int s=0;s<8;++s)generated[s].push_back(tokens[s]);}
    const double seconds=Now()-start;
    if(reference.empty())reference=generated;
    else if(reference!=generated){fprintf(stderr,"live B8 IDs change\n");std::exit(1);}
    if(round)printf("SHARED_BATCH mode=%d round=%d aggregate_tok_s=%.6f wall_ms=%.6f ids_equal=1\n",mode,round,8.*32/seconds,1000*seconds/32);
  }
}
}
int main(int argc,char** argv) {
  if(argc!=3)return 2;
  std::ifstream file(argv[2]);std::stringstream text;text<<file.rdbuf();const auto prompt=ParseIds(text.str());
  if(!file||prompt.empty()||prompt.size()>480||!ValidTokenIds(prompt,kVocab))return 2;
  printf("SHARED_FIRST C=2 PF=4 cache=default mode0=original mode1=shared R1=unchanged\n");
  Engine e(argv[1],4096);e.ConfigureStreams(8);e.WarmPlans();EngineStateIO::Items(e);EngineStateIO::Mode(e,0);
  for(int stream=0;stream<8;++stream){e.SelectStream(stream);for(size_t i=0;i<prompt.size();i+=8){
    const int r=std::min<size_t>(8,prompt.size()-i);e.Forward(prompt.data()+i,r,0,r);}}
  e.SelectStream(0);const auto prefix=EngineStateIO::Save(e);
  Check(e,prefix,0);Check(e,prefix,1);const double r8=Pairs(e,prefix,1,32);
  const bool stop=r8<0.5;
  if(stop)printf("STOP R8 shared-first saved %.6f ms is below 0.5 ms\n",r8);
  else {
    Check(e,prefix,2);Pairs(e,prefix,2,32);
    LiveSpec(e,prefix,prompt);LiveBatch(e,prefix,prompt);
  }
  for(const auto& b:prefix.buffers)HIP_OK(hipFree(b.copy));
  for(void* p:extra_allocations)HIP_OK(hipFree(p));
  return 0;
}
