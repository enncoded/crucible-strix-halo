// Separate host enqueue delays from per-layer GPU intervals.
#include <cstddef>
namespace crucible { void TraceBuffer(const void* pointer, size_t bytes); }
namespace { void QsaTrace(int layer, int phase); void StageTrace(const char* name); }
#define CRUCIBLE_QSA_TRACE
#define CRUCIBLE_QSA_NO_MAIN
#include "qsa_engine_long.hip"
#include <sys/resource.h>
#include "qsa_settle.h"

namespace {
struct Trace {
  hipEvent_t events[12][4];
  double cpu[12][4]{};
  bool active = false, stages = false;
  hipEvent_t stage_events[1024];
  const char* names[1024]{};
  double stage_cpu[1024]{};
  int layer[1024]{}, count = 0, current_layer = 0;
  FILE* stage_output = nullptr;
  Trace() {
    for (auto& row : events) for (auto& e : row) HIP_OK(hipEventCreate(&e));
    for (auto& e : stage_events) HIP_OK(hipEventCreate(&e));
  }
  ~Trace() {
    for (auto& row : events) for (auto& e : row) HIP_OK(hipEventDestroy(e));
    for (auto& e : stage_events) HIP_OK(hipEventDestroy(e));
  }
};
Trace* trace = nullptr;
void StageTrace(const char* name) {
  if (!trace || !trace->active || !trace->stages) return;
  if (std::strcmp(name,"start") == 0) { trace->count = 0; trace->current_layer = 0; }
  const int i = trace->count++;
  Check(i < 1024,"stage event capacity");
  trace->names[i] = name; trace->layer[i] = trace->current_layer;
  trace->stage_cpu[i] = Now();
  HIP_OK(hipEventRecord(trace->stage_events[i]));
  if (std::strcmp(name,"moe_combine") == 0) ++trace->current_layer;
}
void QsaTrace(int layer, int phase) {
  if (!trace || !trace->active || layer == kLayers) return;
  const int l = layer/4;
  trace->cpu[l][phase] = Now();
  HIP_OK(hipEventRecord(trace->events[l][phase]));
}

void Trial(Engine& engine, const char* state, int trial, int steps, FILE* output) {
  int token = EngineStateIO::Transfer(engine,state,false);
  if (trial == 0 && std::getenv("CRUCIBLE_QSA_MIGRATION_SETTLE")) {
    engine.Forward(&token,1,0,1);
    token = EngineStateIO::Transfer(engine,state,false);
    TouchAndSettle();
  }
  if (trial == 0) {
    if (const char* path = std::getenv("CRUCIBLE_QSA_READY_FILE")) {
      std::ofstream ready(path); ready << "ready\n";
      Check(bool(ready),"ready marker");
    }
  }
  const int start = engine.Length();
  Check(start+steps <= 262144,"trace context");
  std::vector<int> ids;
  trace->active = true;
  for (int step = 0; step < steps; ++step) {
    rusage before{},after{}; getrusage(RUSAGE_SELF,&before);
    const double begin = Now();
    token = engine.Forward(&token,1,0,1).back();
    const double wall = (Now()-begin)*1e6;
    getrusage(RUSAGE_SELF,&after); ids.push_back(token);
    for (int i = 1; i < trace->count; ++i) {
      float ms;
      HIP_OK(hipEventElapsedTime(&ms,trace->stage_events[i-1],trace->stage_events[i]));
      std::fprintf(trace->stage_output,"%d,%d,%d,%s,%.6f,%.3f,%.3f,%.3f\n",trial,step,
                   trace->layer[i],trace->names[i],begin,wall,ms*1000,
                   (trace->stage_cpu[i]-trace->stage_cpu[i-1])*1e6);
    }
    for (int l = 0; l < 12; ++l) {
      float gpu[3];
      for (int p = 0; p < 3; ++p)
        HIP_OK(hipEventElapsedTime(&gpu[p],trace->events[l][p],trace->events[l][p+1]));
      std::fprintf(output,"%d,%d,%d,%d,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%ld,%ld,%ld,%ld\n",
                   trial,step,start+step,l*4+3,wall,gpu[0]*1000,gpu[1]*1000,gpu[2]*1000,
                   (trace->cpu[l][1]-trace->cpu[l][0])*1e6,(trace->cpu[l][2]-trace->cpu[l][1])*1e6,
                   (trace->cpu[l][3]-trace->cpu[l][2])*1e6,after.ru_minflt-before.ru_minflt,
                   after.ru_majflt-before.ru_majflt,after.ru_nvcsw-before.ru_nvcsw,after.ru_nivcsw-before.ru_nivcsw);
    }
  }
  trace->active = false;
  std::printf("ids:"); for (int id : ids) std::printf(" %d",id); std::printf("\n");
}
}  // namespace

int main(int argc, char** argv) {
  Check(argc == 6 || argc == 7,"CKPT STATE STEPS TRIALS OUTPUT_CSV [--stages]");
  const int steps = std::stoi(argv[3]), trials = std::stoi(argv[4]);
  Check(steps > 0 && steps <= 4096 && trials > 0 && trials <= 10,"trace range");
  Engine engine(argv[1],262144); engine.Reset();
  hipDeviceProp_t properties{}; HIP_OK(hipGetDeviceProperties(&properties,0));
  const char* xnack = std::getenv("HSA_XNACK");
  std::fprintf(stderr,"architecture=%s HSA_XNACK=%s\n",properties.gcnArchName,xnack ? xnack : "unset");
  for (const char* file : {"maps","smaps"}) {
    std::ifstream source(std::string("/proc/self/")+file);
    std::ofstream target(std::string(argv[5])+"."+file); target << source.rdbuf();
    Check(bool(source) && bool(target),"process mapping snapshot");
  }
  Trace storage; trace = &storage;
  if (argc == 7) {
    Check(std::string(argv[6]) == "--stages","trace option");
    storage.stages = true;
    storage.stage_output = std::fopen((std::string(argv[5])+".stages").c_str(),"w");
    Check(storage.stage_output,"stage output");
    std::fprintf(storage.stage_output,"trial,step,layer,stage,begin_s,wall_us,gpu_us,host_us\n");
  }
  FILE* output = std::fopen(argv[5],"w"); Check(output,"trace output");
  std::fprintf(output,"trial,step,position,layer,wall_us,prep_gpu_us,select_gpu_us,attn_gpu_us,prep_host_us,select_host_us,attn_host_us,minflt,majflt,nvcsw,nivcsw\n");
  for (int trial = 0; trial < trials; ++trial) Trial(engine,argv[2],trial,steps,output);
  Check(std::fclose(output) == 0,"trace close");
  if (storage.stage_output) Check(std::fclose(storage.stage_output) == 0,"stage close");
}
