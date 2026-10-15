// Compare generic dispatch variants in one allocation, with identical token IDs.
// Usage: quality_dispatch CKPT IDS_FILE
#define ENGINE_LIBRARY
#include "../engine/engine.hip"

namespace {
struct EngineStateIO {
  static void WideDispatch(Engine& engine, bool wide) {
    auto set = [wide](Group& group) {
      group.has_k6 = wide || std::any_of(group.jobs.begin(), group.jobs.end(),
                                       [](const LinJob& job) { return job.K == 6; });
    };
    for (auto& [rows, plan] : engine.plans_) {
      for (auto& layer : plan->L) {
        set(layer.a_in); set(layer.a_out); set(layer.gu); set(layer.dn);
      }
      set(plan->head); set(plan->mtp_in);
    }
  }
};
}

int main(int argc, char** argv) {
  if (argc != 3) {
    std::fprintf(stderr, "usage: quality_dispatch CKPT IDS_FILE\n");
    return 2;
  }
  std::ifstream file(argv[2]);
  if (!file) return 2;
  std::stringstream text;
  text << file.rdbuf();
  GenOptions options;
  options.ids = ParseIds(text.str());
  options.n = 160;
  std::vector<int> blocks;
  if (CheckGenOptions(options, blocks)) return 2;
  Engine engine(argv[1], options.ctx);
  engine.WarmPlans();
  std::vector<int> reference;
  for (int trial = 0; trial < 10; ++trial) {
    const bool wide = trial % 2 == 0;
    EngineStateIO::WideDispatch(engine, wide);
    const auto result = RunGen(engine, options, blocks);
    if (trial == 0) reference = result.gen;
    if (reference != result.gen) {
      std::fprintf(stderr, "dispatch outputs differ at trial %d\n", trial);
      return 1;
    }
    std::printf("trial %d %s %s: %.3f tok/s, exact\n", trial,
                trial < 2 ? "warmup" : "measured", wide ? "wide" : "split",
                (options.n - 1) / result.decode_s);
    std::fflush(stdout);
  }
}
