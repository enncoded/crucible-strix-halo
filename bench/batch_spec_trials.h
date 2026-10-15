// Included after the private probe runtime. These trials reuse one engine load.
struct PairStats {
  double seconds=0;
  long steps=0;
  std::array<long,2> accepted{},proposed{},request_steps{};
};
std::vector<RequestState> FreshRequests(const std::vector<RequestState>& source) {
  std::vector<RequestState> fresh;
  for (const auto& old:source) {RequestState state;state.slot=old.slot;state.prompt=old.prompt;fresh.push_back(std::move(state));}
  return fresh;
}
PairStats BatchedSpecCase(Engine& engine,std::vector<RequestState>& requests,int count,int spec,
                          const std::vector<int>& blocks,const fs::path& root) {
  engine.DraftHead(blocks,true,false,false);
  PairStats stats;stats.seconds=Run(engine,requests,count,spec,true,false,root);
  for (const auto& state:requests) {
    stats.accepted[state.slot]=state.accepted;stats.proposed[state.slot]=state.proposed;
    stats.request_steps[state.slot]=state.step;stats.steps=std::max(stats.steps,long(state.step));
  }
  return stats;
}
PairStats SoloDefaultCase(Engine& engine,const std::vector<RequestState>& requests,int count,const fs::path& root) {
  PairStats stats;
  for (const auto& state:requests) {
    engine.SelectStream(state.slot);
    GenOptions options=CliGenOptions();options.ids=state.prompt;options.n=count;options.ctx=engine.Context();
    options.chunk=8;options.prefill_rows=0;
    const auto result=RunGen(engine,options,{});
    Write(root/("ids_"+std::to_string(state.slot)+".i32"),result.gen);
    stats.seconds+=result.decode_s;stats.steps+=result.forwards;
    stats.accepted[state.slot]=result.accepted;stats.proposed[state.slot]=result.proposed;
    stats.request_steps[state.slot]=result.forwards;
  }
  return stats;
}
PairStats BatchZeroCase(Engine& engine,const std::vector<RequestState>& requests,int count,const fs::path& root) {
  BatchOptions options;options.streams=2;options.n=count;options.ctx=engine.Context();options.chunk=8;
  for (const auto& state:requests) options.prompts.push_back(state.prompt);
  std::vector<int> tokens(2);PrefillBatch(engine,options,tokens);
  std::array<std::vector<int>,2> generated{std::vector<int>{tokens[0]},std::vector<int>{tokens[1]}};
  engine.Synchronize();const double start=GenerationNow();
  for (int step=1;step<count;++step) {
    tokens=engine.ForwardBatch(tokens,{0,1});
    for (int slot=0;slot<2;++slot) generated[slot].push_back(tokens[slot]);
  }
  engine.Synchronize();PairStats stats;stats.seconds=GenerationNow()-start;stats.steps=count-1;
  for (int slot=0;slot<2;++slot) {Write(root/("ids_"+std::to_string(slot)+".i32"),generated[slot]);stats.request_steps[slot]=count-1;}
  return stats;
}
void PrintPair(char mode,int count,const PairStats& stats) {
  const char* lookahead=mode=='c'?"default_batch_off":"native_on";
  std::printf("PAIR {\"case\":\"%c\",\"tokens\":%d,\"seconds\":%.9f,\"steps\":%ld,\"tokens_per_step\":%.6f,\"lookahead\":\"%s\",\"requests\":[",
    mode,2*(count-1),stats.seconds,stats.steps,2.*(count-1)/stats.steps,lookahead);
  for (int slot=0;slot<2;++slot) std::printf("%s{\"slot\":%d,\"accepted\":%ld,\"proposed\":%ld,\"steps\":%ld,\"tokens_per_step\":%.6f}",
    slot?",":"",slot,stats.accepted[slot],stats.proposed[slot],stats.request_steps[slot],double(count-1)/stats.request_steps[slot]);
  std::printf("]}\n");std::fflush(stdout);
}
void PairedTrials(Engine& engine,const std::vector<RequestState>& source,int count,int spec,
                  const std::vector<int>& blocks,const fs::path& root) {
#ifndef CRUCIBLE_BATCH_SPEC_BENCH
  throw std::invalid_argument("paired trials require the candidate probe");
#else
  if (std::getenv("CRUCIBLE_PROFILE")) throw std::invalid_argument("paired trials require no profiler");
  engine.CountExperts(false);
  std::printf("paired solo_profile=CliGenOptions spec=7 draft_q=0.3 head=64k adaptive=1 mtp_prefix=256 lookup=0 overlap=0 head_auto=0\n");
  constexpr std::array<char,6> order{'a','b','c','c','b','a'};
  for (size_t index=0;index<order.size();++index) {
    auto requests=FreshRequests(source);const char mode=order[index];
    const auto path=root/(std::to_string(index)+"_"+mode);fs::create_directories(path);
    const auto stats=mode=='a'?BatchedSpecCase(engine,requests,count,spec,blocks,path):
                     mode=='b'?SoloDefaultCase(engine,requests,count,path):BatchZeroCase(engine,requests,count,path);
    PrintPair(mode,count,stats);
  }
#endif
}
