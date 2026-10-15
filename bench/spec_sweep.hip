// Resident speculative-decode sweep: loads the checkpoint once, then runs
// every request line of a config file through the engine's own RunGen.
//
//   spec_sweep CKPT CONFIG
//
// CONFIG lines: NAME [crucible gen options] [--count 1] [--profile 1]
// '#' starts a comment. --steps 1 is implied. A --spec 0 line is the serial
// reference for its ids file and --n; later lines with the same request are
// checked against it token for token (Gate 3).
// A predefined CRUCIBLE_ENGINE_HEADER is the CPU exit-status test, which has
// no HIP runtime for the settle and counter instrumentation.
#ifndef CRUCIBLE_ENGINE_HEADER
#define CRUCIBLE_ENGINE_HEADER "../engine/engine.hip"
#define CRUCIBLE_SWEEP_SETTLE 1
#include "quality_settle.h"

template <class T>
hipError_t SweepAlloc(T** pointer, size_t bytes) {
  const hipError_t result = hipMalloc(pointer, bytes);
  if (result == hipSuccess) crucible::TraceBuffer(*pointer, bytes);
  return result;
}

#define hipMalloc SweepAlloc
#endif
// The rename is kept for builds against engine sources older than train 20 (an unguarded main in gen_cli.h).
#define ENGINE_LIBRARY
#define main CrucibleCliMain
#include CRUCIBLE_ENGINE_HEADER
#undef main
#undef hipMalloc

#ifndef CRUCIBLE_SWEEP_SETTLE
inline void CounterSnapshot(const char*) {}
inline void TouchAndSettle() {}
#endif

namespace {

uint64_t Fnv(const std::vector<int>& v) {
  uint64_t h = 1469598103934665603ull;
  for (int x : v) h = (h ^ (uint32_t)x) * 1099511628211ull;
  return h;
}

std::string Option(const std::vector<std::string>& args, const std::string& key) {
  for (size_t i = 1; i + 1 < args.size(); i += 2)
    if (args[i] == key) return args[i + 1];
  return "";
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 3) {
    std::fprintf(stderr, "usage: spec_sweep CKPT CONFIG\n");
    return 2;
  }
  if (setenv("HSA_USERPTR_FOR_PAGED_MEM", "0", 0) != 0) {
    std::perror("cannot set HIP host allocation policy");
    return 1;
  }
  std::ifstream cfg(argv[2]);
  if (!cfg) {
    std::fprintf(stderr, "cannot open %s\n", argv[2]);
    return 2;
  }
  std::vector<std::vector<std::string>> lines;
  for (std::string line; std::getline(cfg, line);) {
    if (const size_t h = line.find('#'); h != std::string::npos) line.resize(h);
    std::stringstream ss(line);
    std::vector<std::string> args;
    for (std::string t; ss >> t;) args.push_back(t);
    if (!args.empty()) lines.push_back(args);
  }
  int ctx = 32768, prefill_rows = 0;
  for (auto& a : lines) {
    if (const std::string c = Option(a, "--ctx"); !c.empty()) ctx = std::max(ctx, std::stoi(c));
    if (const std::string p = Option(a, "--prefill"); !p.empty())
      prefill_rows = std::max(prefill_rows, std::stoi(p));
  }
  Engine eng(argv[1], ctx, prefill_rows);
  eng.WarmPlans();
  if (std::getenv("CRUCIBLE_SETTLE")) {
    CounterSnapshot("loaded");
    TouchAndSettle();
    CounterSnapshot("ready");
  }
  std::map<std::string, std::vector<int>> serial;
  bool mismatch = false;
  for (auto& args : lines) {
    std::vector<char*> av{(char*)"x"};
    av.push_back(args[0].data());
    for (size_t i = 1; i + 1 < args.size(); i += 2) {
      if (args[i] == "--count" || args[i] == "--profile") continue;
      av.push_back(args[i].data()); av.push_back(args[i + 1].data());
    }
    GenOptions o;
    std::vector<int> blocks;
    if (!ParseGenOptions((int)av.size(), av.data(), 2, o) || CheckGenOptions(o, blocks) != 0) {
      std::fprintf(stderr, "%s: bad options\n", args[0].c_str());
      return 2;
    }
    if (o.ctx > ctx) return 2;
    o.steps = true;
    const bool count = Option(args, "--count") == "1", profile = Option(args, "--profile") == "1";
    eng.CountExperts(count);
    if (profile) setenv("CRUCIBLE_PROFILE", "1", 1);
    else unsetenv("CRUCIBLE_PROFILE");
    eng.EnableProfile(false);
    const double request_start = Now();
    const GenResult r = RunGen(eng, o, blocks);
    const double request_ms = 1000 * (Now() - request_start);
    const std::string key = Option(args, "--ids-file") + "#" + std::to_string(o.n);
    const char* verdict = "noref";
    if (o.sampling.temperature > 0) {
      // Serial and speculative sampling have the same law, not the same RNG path.
      verdict = "sampled";
      if (o.sampling.top_k == 1)
        if (auto it = serial.find(key); it != serial.end()) verdict = it->second == r.gen ? "exact-top1" : "MISMATCH";
    } else if (o.spec == 0 && o.lookup == 0) {
      serial[key] = r.gen;
      verdict = "ref";
    } else if (auto it = serial.find(key); it != serial.end()) {
      const bool exact = it->second == r.gen;
      verdict = exact ? "exact" : "MISMATCH";
      mismatch = mismatch || !exact;
    }
    const double tps = (o.n - 1) / r.decode_s;
    std::printf("== %s: %.2f tok/s, %ld steps, %.3f tok/step, accept %.1f%%, hash %016llx, %s\n",
                args[0].c_str(), tps, r.forwards, r.forwards ? (double)(o.n - 1) / r.forwards : 0.,
                r.proposed ? 100.0 * r.accepted / r.proposed : 0., (unsigned long long)Fnv(r.gen),
                verdict);
    if (o.lookup > 0) std::printf("  lookup steps %ld, accepted %ld\n", r.lookup_steps, r.lookup_accepted);
    if (std::getenv("CRUCIBLE_SWEEP_TIMINGS"))
      std::printf("timings %s: prefill %.3f ms, decode %.3f ms, request %.3f ms\n",
                  args[0].c_str(), r.prefill_s * 1000, r.decode_s * 1000, request_ms);
    if (std::getenv("CRUCIBLE_SWEEP_IDS")) {
      std::printf("ids %s:", args[0].c_str());
      for (int id : r.gen) std::printf(" %d", id);
      std::printf("\n");
    }
    std::fflush(stdout);
    r.steps.Print(stdout);
    if (o.head_auto)
      std::printf("  head policy: restricted %ld steps, full %ld steps, final %s\n",
                  r.head_policy.Steps(false), r.head_policy.Steps(true),
                  r.head_policy.IncumbentFull() ? "full" : "restricted");
    if (count) {
      const auto c = eng.ExpertCounts();
      for (int s = 0; s < 2; ++s) {
        const unsigned long long* v = c.data() + s * 2 * (kMaxRows + 1);
        std::printf("  experts %s:", s ? "mtp" : "trunk");
        for (int R = 1; R <= kMaxRows; ++R)
          if (v[2 * R + 1])
            std::printf(" R=%d %.2f/call (%llu calls)", R, (double)v[2 * R] / v[2 * R + 1], v[2 * R + 1]);
        std::printf("\n");
      }
    }
    if (profile) {
      std::fflush(stdout);
      eng.Report((int)r.forwards);
      eng.EnableProfile(false);
    }
    std::fflush(stdout);
    if (std::getenv("CRUCIBLE_SWEEP_CASE_COUNTERS")) CounterSnapshot(args[0].c_str());
  }
  if (std::getenv("CRUCIBLE_SETTLE")) CounterSnapshot("done");
  return mismatch ? 1 : 0;
}
