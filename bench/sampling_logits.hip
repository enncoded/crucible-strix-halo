// A greedy-ID gate cannot prove equality of the sampling probabilities.
// Trace sampled target logits, then force its exact continuation through serial.
#define ENGINE_LIBRARY
#include "../engine/engine.hip"
#include <bit>

static std::vector<int> ReadAuditIds(const std::string& path) {
  std::ifstream file(path);
  if (!file) throw std::runtime_error("cannot open audit prompt: " + path);
  std::stringstream text;
  text << file.rdbuf();
  return ParseIds(text.str());
}

struct LogitAudit {
  long rows = 0, different_rows = 0, different_values = 0;
  double maximum_absolute = 0;
  void Compare(std::span<const float> sampled, std::span<const float> serial) {
    if (sampled.size() != serial.size()) throw std::runtime_error("audit row size differs");
    bool differs = false;
    for (size_t i = 0; i < serial.size(); ++i) {
      if (std::bit_cast<uint32_t>(sampled[i]) != std::bit_cast<uint32_t>(serial[i])) {
        differs = true;
        ++different_values;
        maximum_absolute = std::max(maximum_absolute, double(std::abs(sampled[i] - serial[i])));
      }
    }
    ++rows;
    different_rows += differs;
  }
};

static bool Audit(Engine& engine, GenOptions options, const std::string& name) {
  std::vector<int> blocks;
  if (CheckGenOptions(options, blocks)) return false;
  std::vector<std::vector<float>> trace;
  options.sampling_trace = [&](std::span<const float> logits) { trace.emplace_back(logits.begin(), logits.end()); };
  engine.EnableProfile(false);
  const GenResult sampled = RunGen(engine, options, blocks);
  if (trace.size() != sampled.gen.size()) throw std::runtime_error("audit token and logit counts differ");
  engine.Reset();
  int last_rows = 0;
  for (size_t i = 0; i < options.ids.size(); i += options.chunk) {
    last_rows = int(std::min<size_t>(options.chunk, options.ids.size() - i));
    engine.Forward(options.ids.data() + i, last_rows, 0, last_rows);
  }
  LogitAudit audit;
  audit.Compare(trace[0], engine.CopyLogits(1, last_rows - 1));
  for (size_t i = 1; i < sampled.gen.size(); ++i) {
    engine.Forward(&sampled.gen[i - 1], 1, 0, 1);
    audit.Compare(trace[i], engine.CopyLogits(1));
  }
  std::printf("audit %s: %ld rows, %ld different rows, %ld different floats, max abs %.9g; accepted %ld/%ld\n",
              name.c_str(), audit.rows, audit.different_rows, audit.different_values, audit.maximum_absolute,
              sampled.accepted, sampled.proposed);
  std::fflush(stdout);
  return audit.different_rows == 0;
}

int main(int argc, char** argv) {
  if (argc != 2) {
    std::fprintf(stderr, "usage: sampling_logits CKPT\n");
    return 2;
  }
  Engine engine(argv[1], 32768);
  engine.WarmPlans();
  GenOptions options;
  options.ids = ReadAuditIds("tests/data/draft/code_fibonacci.ids");
  options.n = 96;
  options.sampling.temperature = 0.7;
  options.sampling.seed = 87326;
  bool passed = true;
  for (int depth = 1; depth < kMaxRows; ++depth) {
    options.spec = depth;
    passed &= Audit(engine, options, "R" + std::to_string(depth + 1));
  }
  options.spec = 7;
  options.lookup = 7;
  options.lookup_min = 3;
  options.ids = ReadAuditIds("tests/data/draft/code_copy.ids");
  passed &= Audit(engine, options, "lookup");
  options.lookup = 0;
  options.draft_blocks = "tests/data/draft/blocks32768.txt";
  options.adapt = 1;
  passed &= Audit(engine, options, "adaptive");
  const auto long_ids = ReadAuditIds("tests/data/draft-long/code_copy_long.ids");
  options.ids.clear();
  for (int repeat = 0; repeat < 4; ++repeat) options.ids.insert(options.ids.end(), long_ids.begin(), long_ids.end());
  options.draft_blocks.clear();
  options.adapt = 0;
  options.n = 24;
  passed &= Audit(engine, options, "sparse-QSA");
  std::printf("sampling target-logit audit: %s\n", passed ? "PASS" : "FAIL");
  return passed ? 0 : 1;
}
