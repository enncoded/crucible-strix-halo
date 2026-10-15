// Bounded full-state checkpoints for long-context speculative decode.
#define CRUCIBLE_QSA_NO_MAIN
#include "qsa_engine_long.hip"

namespace {
void SpecPrefill(Engine& engine, char** argv, int ctx, float* carry, float* stacks) {
  const auto ids = ReadIds(argv[3]);
  if (std::string(argv[5]) != "-") {
    EngineStateIO::Transfer(engine,argv[5],false);
    EngineStateIO::Mtp(engine,argv[5],false,carry);
  }
  const int start = engine.Length(), limit = std::stoi(argv[6]);
  Check(limit > 0 && limit <= ctx,"prefill limit");
  const int end = std::min((int)ids.size(),start+limit);
  Check(end > start && end <= ctx,"prefill range");
  int mpos = std::max(0,start-1), next = 0;
  bool have_carry = start > 0;
  const double begin = Now();
  for (int i = start; i < end; i += 8) {
    const int rows = std::min(8,end-i);
    next = engine.Forward(ids.data()+i,rows,0,rows).back();
    std::vector<int> tokens;
    int m = 0;
    if (have_carry) {
      HIP_OK(hipMemcpyAsync(stacks,carry,kWide*sizeof(float),hipMemcpyDeviceToDevice));
      tokens.push_back(ids[i]); m = 1;
    }
    CopyRows(engine,stacks,m,0,rows-1);
    for (int r = 0; r+1 < rows; ++r) tokens.push_back(ids[i+r+1]);
    CopyRows(engine,carry,0,rows-1,1); have_carry = true;
    if (!tokens.empty()) {
      engine.Mtp(stacks,tokens.data(),tokens.size(),mpos); mpos += tokens.size();
    }
  }
  const double elapsed = Now()-begin;
  Check(mpos == end-1,"MTP frontier");
  EngineStateIO::Transfer(engine,argv[4],true,next);
  EngineStateIO::Mtp(engine,argv[4],true,carry);
  std::printf("spec_prefill %d..%d in %.6f s, %.3f tok/s, next %d\n",start,end,elapsed,(end-start)/elapsed,next);
}

std::vector<int> Generate(Engine& engine, int token, int n, int depth, float* carry, float* stacks) {
  std::vector<int> generated{token};
  if (!depth) {
    while ((int)generated.size() < n)
      generated.push_back(engine.Forward(&generated.back(),1,0,1).back());
    return generated;
  }
  HIP_OK(hipMemcpyAsync(stacks,carry,kWide*sizeof(float),hipMemcpyDeviceToDevice));
  std::vector<int> catchup{token};
  int mpos = engine.Length()-1, replay = 0, forwards = 0, accepted = 0;
  while ((int)generated.size() < n) {
    const int rc = catchup.size();
    std::vector<int> drafts{engine.Mtp(stacks,catchup.data(),rc,mpos).back()};
    mpos += rc;
    for (int k = 1; k < depth; ++k) {
      CopyRows(engine,stacks,kMaxRows,k == 1 ? rc-1 : 0,1);
      drafts.push_back(engine.Mtp(stacks+(size_t)kMaxRows*kWide,&drafts.back(),1,mpos+k-1).back());
    }
    std::vector<int> rows{generated.back()}; rows.insert(rows.end(),drafts.begin(),drafts.end());
    const size_t base = engine.Length();
    const auto ys = engine.Forward(rows.data(),depth+1,replay,1);
    int a = 0;
    while (a < depth && drafts[a] == ys[a]) ++a;
    engine.Truncate(base+a+1);
    CopyRows(engine,stacks,0,0,a+1);
    catchup.assign(ys.begin(),ys.begin()+a+1);
    generated.insert(generated.end(),ys.begin(),ys.begin()+a+1);
    replay = a; accepted += a; ++forwards;
  }
  generated.resize(n);
  std::printf("acceptance %d/%d = %.4f\n",accepted,forwards*depth,double(accepted)/(forwards*depth));
  return generated;
}

void SpecDecode(Engine& engine, char** argv, int ctx, float* carry, float* stacks) {
  const int count = std::stoi(argv[4]), depth = std::stoi(argv[6]);
  Check(count >= 2 && depth >= 0 && depth <= 2,"decode arguments");
  for (int trial = 0; trial < 3; ++trial) {
    const int token = EngineStateIO::Transfer(engine,argv[3],false);
    EngineStateIO::Mtp(engine,argv[3],false,carry);
    const int start = engine.Length();
    Check(start+count+depth <= ctx,"decode range");
    const double begin = Now();
    const auto ids = Generate(engine,token,count,depth,carry,stacks);
    const double elapsed = Now()-begin;
    std::printf("spec_decode trial=%d context=%d spec=%d tokens=%d seconds=%.6f tok_s=%.6f\n",
                trial,start,depth,count-1,elapsed,(count-1)/elapsed);
    std::printf("ids:"); for (int id : ids) std::printf(" %d",id); std::printf("\n");
  }
}
}  // namespace

// prefill CKPT IDS OUTPUT INPUT|- LIMIT CTX
// decode CKPT STATE COUNT CTX DEPTH
int main(int argc, char** argv) {
  Check(argc >= 7,"prefill CKPT IDS OUT IN|- LIMIT CTX or decode CKPT STATE COUNT CTX DEPTH");
  const std::string mode = argv[1];
  Check(mode == "prefill" || mode == "decode","mode");
  const bool prefill = mode == "prefill";
  Check(argc == (prefill ? 8 : 7),"argument count");
  const int ctx = std::stoi(argv[prefill ? 7 : 5]);
  Check(ctx > 0 && ctx <= 262144,"context");
  Engine engine(argv[2],ctx); engine.Reset();
  float* carry = Zeros<float>(kWide);
  float* stacks = Zeros<float>((kMaxRows+1)*kWide);
  if (prefill) SpecPrefill(engine,argv,ctx,carry,stacks);
  else SpecDecode(engine,argv,ctx,carry,stacks);
  HIP_OK(hipFree(stacks)); HIP_OK(hipFree(carry));
}
