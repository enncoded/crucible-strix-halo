// Keep each full-model prefill run below the shared 300-second GPU limit.
// Checkpoints contain serial inference state, not weights or MTP state.
// The baseline build reads the same state but restores the old pool layout.
#ifndef CRUCIBLE_ENGINE_SOURCE
#define CRUCIBLE_ENGINE_SOURCE "../engine/engine.hip"
#endif
// The rename is kept for builds against engine sources older than train 20 (an unguarded main in gen_cli.h).
#define ENGINE_LIBRARY
#define main CrucibleCliMain
#include CRUCIBLE_ENGINE_SOURCE
#undef main
#include "../engine/runtime/bounds.h"

namespace {

void Check(bool ok, const char* text) {
  if (!ok) { std::fprintf(stderr,"state error: %s\n",text); std::exit(1); }
}

struct EngineStateIO {
  static constexpr uint64_t magic = 0x4352554351534131ull;

  template<class E> static void RestorePrefix(E& engine) {
    if constexpr (requires { engine.retained_.Reset(); }) {
      engine.retained_.Reset();
      const int n = engine.hist_.size();
      if (n) {
        const auto error = engine.retained_.Forward(n,0,n);
        Check(error.empty(),error.c_str());
      }
    }
  }

  static void Bytes(std::fstream& file, void* ptr, size_t bytes, bool save) {
    if (save) file.write(static_cast<const char*>(ptr),bytes);
    else file.read(static_cast<char*>(ptr),bytes);
    Check(bool(file),"checkpoint I/O failed");
  }

  template<class T>
  static void Device(std::fstream& file, T* ptr, size_t count, bool save) {
    constexpr size_t chunk = 32u<<20;
    std::vector<char> host(std::min(chunk,count*sizeof(T)));
    for (size_t off = 0; off < count*sizeof(T); off += chunk) {
      const size_t bytes = std::min(chunk,count*sizeof(T)-off);
      char* device = reinterpret_cast<char*>(ptr)+off;
      if (save) HIP_OK(hipMemcpy(host.data(),device,bytes,hipMemcpyDeviceToHost));
      Bytes(file,host.data(),bytes,save);
      if (!save) HIP_OK(hipMemcpy(device,host.data(),bytes,hipMemcpyHostToDevice));
    }
  }

  static void Pool(std::fstream& file, float* device, int n, bool save) {
    const int blocks = n/4;
    std::vector<float> canonical((size_t)blocks*128);
#ifdef CRUCIBLE_BASELINE_STATE
    Device(file,device,canonical.size(),save);
#else
    std::vector<float> tiled((size_t)((n+127)/128)*4096);
    if (save) {
      HIP_OK(hipMemcpy(tiled.data(),device,tiled.size()*4,hipMemcpyDeviceToHost));
      for (int b = 0; b < blocks; ++b) for (int d = 0; d < 128; ++d)
        canonical[b*128+d] = tiled[(size_t)(b/32)*4096+d*32+b%32];
    }
    Bytes(file,canonical.data(),canonical.size()*4,save);
    if (!save) {
      for (int b = 0; b < blocks; ++b) for (int d = 0; d < 128; ++d)
        tiled[(size_t)(b/32)*4096+d*32+b%32] = canonical[b*128+d];
      HIP_OK(hipMemcpy(device,tiled.data(),tiled.size()*4,hipMemcpyHostToDevice));
    }
#endif
  }

  static void Mtp(Engine& engine, const std::string& path, bool save, float* carry) {
#ifndef CRUCIBLE_BASELINE_STATE
    Check(!engine.compact_state_,"legacy MTP state format does not support compact storage; use live warmup");
#endif
    const std::string target = path+(save ? ".mtp.tmp" : ".mtp");
    std::fstream file(target,std::ios::binary | (save ? std::ios::out|std::ios::trunc : std::ios::in));
    Check(bool(file),"cannot open MTP state");
    int n = (int)engine.Length()-1;
    const int expected = n;
    Bytes(file,&n,sizeof(n),save);
    Check(n == expected && n >= 0,"MTP state length");
#ifdef CRUCIBLE_BASELINE_STATE
    const auto& cache = engine.qc_.at(kLayers);
#else
    const auto& cache = engine.sequence_->qc.at(kLayers);
#endif
    Device(file,cache.k,(size_t)n*512,save);
    Device(file,cache.v,(size_t)n*512,save);
    Device(file,cache.ik_raw,(size_t)n*128,save);
    Pool(file,cache.ik_pool,n,save);
    Device(file,carry,kWide,save);
    file.close(); Check(bool(file),"MTP state close failed");
    if (save) Check(std::rename(target.c_str(),(path+".mtp").c_str()) == 0,"MTP rename failed");
  }

  static int Transfer(Engine& engine, const std::string& path, bool save, int next = 0) {
#ifndef CRUCIBLE_BASELINE_STATE
    Check(!engine.compact_state_,"legacy state format does not support compact storage; use a live prompt or conversation snapshot");
#endif
    auto mode = std::ios::binary | (save ? std::ios::out|std::ios::trunc : std::ios::in);
    const std::string target = save ? path+".tmp" : path;
    std::fstream file(target,mode);
    Check(bool(file),"cannot open checkpoint");
    uint64_t tag = magic;
#ifdef CRUCIBLE_BASELINE_STATE
    auto& hist = engine.hist_;
    auto& qc = engine.qc_;
    auto& gstate = engine.gstate_;
    auto& ghist = engine.ghist_;
    float* ple_ring = engine.ple_ring_;
#else
    SequenceState& state = *engine.sequence_;
    auto& hist = state.hist;
    auto& qc = state.qc;
    auto& gstate = state.gstate;
    auto& ghist = state.ghist;
    float* ple_ring = state.ple_ring;
#endif
    int n = (int)hist.size();
    Bytes(file,&tag,sizeof(tag),save);
    Bytes(file,&n,sizeof(n),save);
    Bytes(file,&next,sizeof(next),save);
    Check(tag == magic && n >= 1 && n <= engine.ctx_,"bad checkpoint header");
    if (!save) hist.resize(n);
    Bytes(file,hist.data(),(size_t)n*sizeof(int),save);
    Check(next >= 0 && next < kVocab,"bad next token");
    Check(std::all_of(hist.begin(),hist.end(),[](int t) {
      return t >= 0 && t < kVocab;
    }),"bad history token");
    if (!save) RestorePrefix(engine);
    for (int layer = 0; layer < kLayers; ++layer) {
      if (layer%4 == 3) {
        const auto& cache = qc.at(layer);
        Device(file,cache.k,(size_t)n*512,save);
        Device(file,cache.v,(size_t)n*512,save);
        Device(file,cache.ik_raw,(size_t)n*128,save);
        Pool(file,cache.ik_pool,n,save);
      } else {
        Device(file,gstate.at(layer),48*128*128,save);
        Device(file,ghist.at(layer),16*10240,save);
      }
    }
    Device(file,ple_ring,16*kWide,save);
    file.close();
    Check(bool(file),"checkpoint close failed");
#ifndef CRUCIBLE_BASELINE_STATE
    if (!save) {
      // Imported serial state has no pending rollback window or draft rows.
      state.retained.Reset();
      Check(state.retained.Forward(n,0,n).empty(),"cannot restore retained prefix");
    }
#endif
    if (save) Check(std::rename(target.c_str(),path.c_str()) == 0,"checkpoint rename failed");
    return next;
  }
};

std::vector<int> ReadIds(const std::string& path) {
  std::ifstream file(path);
  Check(bool(file),"cannot open IDs");
  std::stringstream text; text << file.rdbuf();
  const auto ids = ParseIds(text.str());
  Check(std::all_of(ids.begin(),ids.end(),[](int t) { return t >= 0 && t < kVocab; }),"bad input token");
  return ids;
}

void Prefill(Engine& engine, char** argv, int ctx) {
  const auto ids = ReadIds(argv[3]);
  if (std::string(argv[5]) != "-") EngineStateIO::Transfer(engine,argv[5],false);
  const int start = (int)engine.Length(), limit = std::stoi(argv[6]);
  const int end = std::min((int)ids.size(),start+limit);
  Check(limit > 0 && end > start && end <= ctx,"prefill range");
  int next = 0;
  const double t0 = Now();
  for (int i = start; i < end; i += 8) {
    const int rows = std::min(8,end-i);
    next = engine.Forward(ids.data()+i,rows,0,rows).back();
  }
  const double elapsed = Now()-t0;
  EngineStateIO::Transfer(engine,argv[4],true,next);
  std::printf("prefill %d..%d in %.3f s, %.3f tok/s, next %d\n",start,end,elapsed,(end-start)/elapsed,next);
}

void Decode(Engine& engine, const std::string& state, int steps, int ctx, int trial) {
  int token = EngineStateIO::Transfer(engine,state,false);
  const int start = (int)engine.Length();
  Check(steps > 0 && start+steps <= ctx,"decode range");
  const double t0 = Now();
  std::vector<int> result;
  std::vector<double> times;
  for (int i = 0; i < steps; ++i) {
    const double begin = Now();
    token = engine.Forward(&token,1,0,1).back(); result.push_back(token);
    times.push_back(Now()-begin);
  }
  const double elapsed = Now()-t0;
  const int warmup = std::min(8,steps-1);
  double steady = 0;
  for (int i = warmup; i < steps; ++i) steady += times[i];
  std::vector<double> sorted(times.begin()+warmup,times.end());
  std::sort(sorted.begin(),sorted.end());
  const double median = (sorted[(sorted.size()-1)/2]+sorted[sorted.size()/2])/2;
  std::printf("decode trial %d context %d..%d steps %d time %.6f s, %.4f ms/token, %.3f tok/s\n",
              trial,start,start+steps,steps,elapsed,elapsed*1000/steps,steps/elapsed);
  std::printf("steady skip=%d steps=%d %.4f ms/token %.3f tok/s median=%.4f ms\n",
              warmup,steps-warmup,steady*1000/(steps-warmup),(steps-warmup)/steady,median*1000);
  std::printf("step_ms:"); for (double t : times) std::printf(" %.4f",t*1000); std::printf("\n");
  std::printf("ids:"); for (int id : result) std::printf(" %d",id); std::printf("\n");
}

std::vector<int> SerialRestored(Engine& engine, const std::string& state, int count) {
  engine.Reset();
  std::vector<int> ids{EngineStateIO::Transfer(engine,state,false)};
  for (int i = 1; i < count; ++i) ids.push_back(engine.Forward(&ids.back(),1,0,1).back());
  return ids;
}

std::vector<int> ColdDrafts(Engine& engine, float* stacks, const std::vector<int>& catchup,
                            int& pos, int depth) {
  const int rows = (int)catchup.size();
  std::vector<int> drafts{engine.Mtp(stacks,catchup.data(),rows,pos).back()};
  pos += rows;
  for (int k = 1; k < depth; ++k) {
    CopyRows(engine,stacks,kMaxRows,k == 1 ? rows-1 : 0,1);
    drafts.push_back(engine.Mtp(stacks+(size_t)kMaxRows*kWide,&drafts.back(),1,pos+k-1).back());
  }
  return drafts;
}

std::vector<int> SpecRestored(Engine& engine, const std::string& state, int count, int ctx) {
  engine.Reset();
  std::vector<int> ids{EngineStateIO::Transfer(engine,state,false)}, catchup{ids.back()};
  float* stacks = Zeros<float>((size_t)(kMaxRows+1)*kWide);
  // Snapshots omit MTP state. Start at its own position zero, not the trunk position.
  int mpos = 0, replay = 0;
  long attempted = 0, accepted = 0;
  while ((int)ids.size() < count) {
    const size_t base = engine.Length();
    const int depth = BoundedDraftDepth(2,ctx,(int)base,mpos,(int)catchup.size());
    if (depth == 0) {
      ids.push_back(engine.Forward(&ids.back(),1,replay,1).back());
      replay = 0;
      continue;
    }
    const auto drafts = ColdDrafts(engine,stacks,catchup,mpos,depth);
    std::vector<int> rows{ids.back()};
    rows.insert(rows.end(),drafts.begin(),drafts.end());
    const auto verified = engine.Forward(rows.data(),depth+1,replay,1);
    int keep = 0;
    while (keep < depth && drafts[keep] == verified[keep]) ++keep;
    engine.Truncate(base+keep+1);
    CopyRows(engine,stacks,0,0,keep+1);
    catchup.assign(verified.begin(),verified.begin()+keep+1);
    ids.insert(ids.end(),verified.begin(),verified.begin()+keep+1);
    replay = keep;
    accepted += keep; attempted += depth;
  }
  ids.resize(count);
  HIP_OK(hipFree(stacks));
  std::printf("restored-spec2 drafts %ld accepted %ld (%.1f%%), cold MTP position starts at 0\n",
              attempted,accepted,attempted ? 100.0*accepted/attempted : 0.0);
  return ids;
}

void VerifyRestored(Engine& engine, const std::string& state, int count, int ctx) {
  Check(count > 1,"verify needs at least two output tokens");
  const auto serial = SerialRestored(engine,state,count);
  const auto spec = SpecRestored(engine,state,count,ctx);
  Check(serial == spec,"restored serial/spec2 token mismatch");
  std::printf("PASS restored serial/spec2: %zu identical token IDs\n",serial.size());
}

}  // namespace

#ifndef CRUCIBLE_QSA_NO_MAIN
// prefill CKPT IDS OUTPUT_STATE INPUT_STATE|- LIMIT CTX
// decode CKPT INPUT_STATE STEPS CTX [REPEATS=3]
// verify CKPT INPUT_STATE OUTPUT_TOKENS CTX (cold MTP, exact verification)
int main(int argc, char** argv) {
  Check(argc >= 6,"use prefill, decode, or verify with a checkpoint and context size");
  const std::string mode = argv[1];
  Check(mode == "prefill" || mode == "decode" || mode == "verify","unknown mode");
  const bool prefill = mode == "prefill", verify = mode == "verify";
  Check(prefill ? argc == 8 : verify ? argc == 6 : argc == 6 || argc == 7,"argument count");
  const int ctx = std::stoi(argv[prefill ? 7 : 5]);
  Check(ctx > 0 && ctx <= 262144,"context size");
  Engine engine(argv[2],ctx);
  engine.Reset();
  if (prefill) Prefill(engine,argv,ctx);
  else if (verify) VerifyRestored(engine,argv[3],std::stoi(argv[4]),ctx);
  else {
    const int repeats = argc == 7 ? std::stoi(argv[6]) : 3;
    Check(repeats >= 1 && repeats <= 10,"repeat count");
    for (int trial = 0; trial < repeats; ++trial)
      Decode(engine,argv[3],std::stoi(argv[4]),ctx,trial);
  }
}
#endif
