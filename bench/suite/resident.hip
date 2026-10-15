// Reuse RunGen so suite timing follows the engine's own generation path.
#ifndef CRUCIBLE_ENGINE_HEADER
#define CRUCIBLE_ENGINE_HEADER "../../engine/engine.hip"
#endif
// The rename is kept for builds against engine sources older than train 20 (an unguarded main in gen_cli.h).
#define ENGINE_LIBRARY
#define main CrucibleCliMain
#include CRUCIBLE_ENGINE_HEADER
#undef main
#include "full_head_policy.h"
#include <iomanip>
#include <stdexcept>

namespace {
struct Request {
  std::string name;
  int repeat;
  GenOptions options;
};

int Integer(const std::string& text) {
  size_t used = 0;
  const int value = std::stoi(text, &used);
  if (used != text.size()) throw std::runtime_error("invalid integer");
  return value;
}

std::vector<Request> ReadRequests(const char* file, int ctx, int prefill_rows) {
  std::ifstream input(file);
  if (!input) throw std::runtime_error("cannot open request file");
  std::vector<Request> requests;
  for (std::string line; std::getline(input, line);) {
    std::istringstream stream(line);
    Request r;
    std::string path, extra;
    auto& o = r.options;
    SetFullHeadPolicy(o);
    if (!(stream >> r.name >> std::quoted(path) >> o.n >> o.chunk >> o.spec >> r.repeat) ||
        stream >> extra || r.name.find_first_not_of("abcdefghijklmnopqrstuvwxyz0123456789_") != std::string::npos)
      throw std::runtime_error("invalid request line");
    std::ifstream ids(path);
    if (!ids) throw std::runtime_error("cannot open token IDs");
    std::stringstream text;
    text << ids.rdbuf();
    o.ids = ParseIds(text.str());
    o.ctx = ctx;
#ifdef CRUCIBLE_SUITE_PREFILL
    o.prefill_rows = prefill_rows;
#endif
    std::vector<int> blocks;
    if (CheckGenOptions(o, blocks) != 0 || r.repeat < -1)
      throw std::runtime_error("invalid generation options");
    requests.push_back(r);
  }
  if (requests.empty()) throw std::runtime_error("empty request file");
  return requests;
}

void PrintResult(const Request& request, const GenResult& result) {
  std::printf("{\"event\":\"result\",\"name\":\"%s\",\"repeat\":%d,\"spec\":%d,"
              "\"prompt_tokens\":%zu,\"prefill_seconds\":%.9f,\"ttft_seconds\":%.9f,"
              "\"decode_seconds\":%.9f,\"forwards\":%ld,\"accepted\":%ld,\"proposed\":%ld,\"output_ids\":[",
              request.name.c_str(), request.repeat, request.options.spec, request.options.ids.size(),
              result.prefill_s, result.prefill_s, result.decode_s, result.forwards,
              result.accepted, result.proposed);
  for (size_t i = 0; i < result.gen.size(); ++i)
    std::printf("%s%d", i ? "," : "", result.gen[i]);
  std::printf("]}\n");
  std::fflush(stdout);
}
}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 4 && argc != 5) throw std::runtime_error("usage: suite-resident CKPT REQUESTS CONTEXT [PREFILL_ROWS]");
    const int ctx = Integer(argv[3]), prefill_rows = argc == 5 ? Integer(argv[4]) : 0;
    if (ctx < 1 || ctx > 262144) throw std::runtime_error("context must be in [1,262144]");
#ifndef CRUCIBLE_SUITE_PREFILL
    if (prefill_rows) throw std::runtime_error("this build has no bulk prefill adapter");
#endif
    const auto requests = ReadRequests(argv[2], ctx, prefill_rows);
    // This entry point bypasses the engine CLI's process policy.
    if (!SetRuntimeMemoryDefaults())
      throw std::runtime_error("cannot set the HSA memory policy");
    const double start = Now();
#ifdef CRUCIBLE_SUITE_PREFILL
    Engine engine(argv[1], ctx, prefill_rows);
#else
    Engine engine(argv[1], ctx);
#endif
    engine.WarmPlans();
    HIP_OK(hipDeviceSynchronize());
    std::printf("{\"event\":\"ready\",\"load_seconds\":%.9f,\"max_rows\":%d,"
                "\"draft_head\":\"full\",\"draft_q\":0,\"lookup\":0}\n", Now() - start, kMaxRows);
    std::fflush(stdout);
    for (const auto& request : requests) {
      // RunGen resets state and creates no persistent prefix between requests.
      const auto result = RunGen(engine, request.options, {});
      if (result.gen.size() != static_cast<size_t>(request.options.n))
        throw std::runtime_error("generation returned the wrong output length");
      PrintResult(request, result);
    }
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "suite: %s\n", error.what());
    return 2;
  }
}
