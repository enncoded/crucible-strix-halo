// One model load serves a bounded set of production and clipped captures.
// The rename is kept for builds against engine sources older than train 20 (an unguarded main in gen_cli.h).
#define ENGINE_LIBRARY
#define main CrucibleCliMain
#ifndef CRUCIBLE_ENGINE_HEADER
#define CRUCIBLE_ENGINE_HEADER "../engine/engine.hip"
#endif
#include CRUCIBLE_ENGINE_HEADER
#undef main
#include "long_probe_runtime.h"
#include "resident_probe_runtime.h"
#include <bit>
#include <filesystem>
#include <fcntl.h>
#include <mutex>
#include <unistd.h>

namespace rp = crucible::residentprobe;
namespace lp = crucible::longprobe;
constexpr size_t kResidentMaxJobs = 7;
constexpr size_t kResidentMaxPromptTokens = 32768;
constexpr int kResidentMaxOutputTokens = 160;
constexpr uintmax_t kResidentMaxIdsBytes = kResidentMaxPromptTokens * 8;
constexpr uintmax_t kResidentMaxManifestBytes = 131072;
struct CaptureJob {
  std::vector<int> input, points;
  std::filesystem::path root;
  int tokens;
};

int ResidentNumber(const std::string& value) {
  return GenNumber<int>(value);
}

std::vector<int> ResidentIds(const std::string& path) {
  if (std::filesystem::file_size(path)>kResidentMaxIdsBytes)
    throw std::runtime_error("resident IDs exceed their bounded input size");
  std::ifstream input(path);
  if (!input) throw std::runtime_error("cannot open resident IDs");
  std::stringstream text;
  text << input.rdbuf();
  return ParseIds(text.str());
}

std::vector<std::string> ResidentFields(const std::string& line) {
  std::vector<std::string> values;
  size_t start = 0;
  for (;;) {
    const size_t end = line.find('\t',start);
    values.push_back(line.substr(start,end == std::string::npos ? end : end-start));
    if (end == std::string::npos) break;
    start=end+1;
  }
  if (values.size()!=4) throw std::runtime_error("resident manifest needs four fields per request");
  return values;
}

std::vector<CaptureJob> ResidentJobs(const char* manifest,int context,int rows) {
  if (std::filesystem::file_size(manifest)>kResidentMaxManifestBytes)
    throw std::runtime_error("resident manifest exceeds its bounded input size");
  std::ifstream input(manifest);
  if (!input) throw std::runtime_error("cannot open resident manifest");
  std::vector<CaptureJob> jobs;
  size_t total = 0;
  for (std::string line;std::getline(input,line);) {
    const auto fields=ResidentFields(line);
    CaptureJob job{ResidentIds(fields[0]),{},fields[2],ResidentNumber(fields[3])};
    if (job.tokens<1 || job.tokens>kResidentMaxOutputTokens ||
        !ValidTokenIds(job.input,kVocab) || !RequestFitsContext(job.input.size(),job.tokens,context))
      throw std::runtime_error("invalid resident dimensions or IDs");
    if (fields[1]!="-") {
      job.points=ResidentIds(fields[1]);
      lp::Validate(job.input,job.points,rows ? rows : 8,kVocab);
    }
    total+=job.input.size();
    if (total>kResidentMaxPromptTokens || jobs.size()>=kResidentMaxJobs)
      throw std::runtime_error("resident role exceeds its bounded capture budget");
    jobs.push_back(std::move(job));
  }
  if (!input.eof() || jobs.empty()) throw std::runtime_error("empty or unreadable resident manifest");
  return jobs;
}

class ResidentStdout {
 public:
  explicit ResidentStdout(const std::filesystem::path& path) {
    std::fflush(stdout);
    saved_=dup(STDOUT_FILENO);
    const int output=open(path.c_str(),O_WRONLY|O_CREAT|O_EXCL,0600);
    if (saved_<0 || output<0) {
      if (saved_>=0) close(saved_);
      if (output>=0) close(output);
      throw std::runtime_error("cannot create private resident stdout");
    }
    const int result=dup2(output,STDOUT_FILENO);
    close(output);
    if (result<0) { close(saved_); throw std::runtime_error("cannot redirect resident stdout"); }
  }
  ~ResidentStdout() {
    std::fflush(stdout);
    if (dup2(saved_,STDOUT_FILENO)<0) std::terminate();
    close(saved_);
  }
  ResidentStdout(const ResidentStdout&)=delete;
  ResidentStdout& operator=(const ResidentStdout&)=delete;
 private:
  int saved_=-1;
};

void ResidentNative(Engine& engine,const CaptureJob& job,int context,int rows) {
  if (!std::filesystem::create_directory(job.root)) throw std::runtime_error("resident request root must be new");
  const auto out=job.root/"native";
  if (!std::filesystem::create_directory(out)) throw std::runtime_error("resident native output must be new");
  ResidentStdout redirect(out/"stdout.log");
  GenOptions options;
  options.ids=job.input;
  options.n=job.tokens;
  options.ctx=context;
  options.prefill_rows=rows;
  options.spec=0;
  options.chunk=8;
  options.draft_head="full";
  options.head_auto=0;
  options.mtp_prefix=0;
  options.draft_q=0;
  options.draft_margin=0;
  options.adapt=0;
  options.adapt_set=true;
  options.lookup=0;
  options.overlap=0;
  options.last_head=1;
  options.draft_dense=false;
  options.logits_dump=(out/"logits.f32").string();
  std::vector<int> blocks;
  if (CheckGenOptions(options,blocks)) throw std::runtime_error("invalid resident generation options");
  const auto result=RunGen(engine,options,blocks);
  PrintGen(options,result);
}

void ResidentSchedule(const std::vector<lp::Block>& blocks) {
  std::printf("{\"event\":\"schedule\",\"kind\":\"clipped schedule\",\"blocks\":[");
  for (size_t i=0;i<blocks.size();++i)
    std::printf("%s[%zu,%d]",i ? "," : "",blocks[i].start,blocks[i].rows);
  std::printf("]}\n");
  std::fflush(stdout);
}

void ResidentClipped(Engine& engine,const CaptureJob& job,int rows) {
  if (job.points.empty()) return;
  const auto out=job.root/"clipped";
  if (!std::filesystem::create_directory(out)) throw std::runtime_error("resident clipped output must be new");
  ResidentStdout redirect(out/"stdout.log");
  std::ofstream binary(out/"logits.f32",std::ios::binary);
  if (!binary) throw std::runtime_error("cannot create resident clipped logits");
  const auto copy=[](const float* source,size_t count) {
    std::vector<float> values(count);
    HIP_OK(hipMemcpy(values.data(),source,count*sizeof(float),hipMemcpyDeviceToHost));
    return values;
  };
  const auto emit=[&](int position,int id,const std::vector<float>& values) {
    binary.write(reinterpret_cast<const char*>(values.data()),values.size()*sizeof(float));
    if (!binary) throw std::runtime_error("cannot write resident logits");
    std::printf("{\"event\":\"point\",\"position\":%d,\"argmax\":%d,\"copied_argmax\":true}\n",position,id);
  };
  const auto blocks=lp::Capture(engine,job.input,job.points,rows ? rows : 8,rows!=0,kVocab,copy,emit);
  binary.close();
  if (!binary) throw std::runtime_error("cannot finish resident logits");
  ResidentSchedule(blocks);
}

int main(int argc,char** argv) {
  try {
    if (argc!=5) return 2;
    static_assert(sizeof(float)==4 && std::endian::native==std::endian::little);
    const int context=ResidentNumber(argv[2]),rows=ResidentNumber(argv[3]);
    if (context<1 || context>262144 || rows<0 || rows>kMaxPrefillRows)
      throw std::runtime_error("invalid resident role dimensions");
    const auto jobs=ResidentJobs(argv[4],context,rows);
    Engine engine(argv[1],context,rows);
    std::printf("{\"event\":\"resident\",\"requests\":%zu,\"bulk_rows\":%d}\n",jobs.size(),rows);
    std::fflush(stdout);
    std::once_flag warmed;
    rp::Run(engine,jobs,
      [&](Engine& e,const CaptureJob& job) { ResidentNative(e,job,context,rows); },
      [&](Engine& e,const CaptureJob& job) {
        if (job.points.empty()) return;
        std::call_once(warmed,[&] { e.WarmPlans(); });
        ResidentClipped(e,job,rows);
      });
  } catch (const std::exception& error) {
    std::fprintf(stderr,"resident probe: %s\n",error.what());
    return 1;
  }
  return 0;
}
