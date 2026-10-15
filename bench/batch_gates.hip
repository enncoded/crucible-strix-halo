// Resident Gates 1(a), 2, 3, and independent-stream exactness for B=1..8.
#define ENGINE_LIBRARY
#include "../engine/engine.hip"

#include <filesystem>
#include <numeric>

namespace {
using Prompts = std::vector<std::vector<int>>;

Prompts LoadPrompts(const std::string& list) {
  Prompts prompts;
  std::ifstream paths(list);
  for (std::string path; std::getline(paths, path);) {
    if (path.empty()) continue;
    std::ifstream file(path);
    std::stringstream text;
    text << file.rdbuf();
    auto ids = ParseIds(text.str());
    if (!file || ids.empty() || !ValidTokenIds(ids, kVocab)) std::exit(2);
    prompts.push_back(ids);
  }
  if (!paths.eof() || prompts.size() != 8) std::exit(2);
  // Unequal absolute positions and both previous/current EOS exercise PLE isolation.
  prompts[6].push_back(kEos);
  prompts[7].insert(prompts[7].end(), {kEos, 1, kEos});
  return prompts;
}

int Prefill(Engine& eng, int stream, const std::vector<int>& prompt, const std::string& dump = "") {
  eng.SelectStream(stream);
  eng.Reset();
  int token = 0;
  for (size_t i = 0; i < prompt.size(); i += 8) {
    const int R = (int)std::min<size_t>(8, prompt.size() - i);
    token = eng.Forward(prompt.data() + i, R, 0, R, dump).back();
  }
  return token;
}

bool SameFile(const std::filesystem::path& a, const std::filesystem::path& b) {
  std::ifstream fa(a, std::ios::binary), fb(b, std::ios::binary);
  if (!fa || !fb) return false;
  return std::vector<char>(std::istreambuf_iterator<char>(fa), {}) ==
         std::vector<char>(std::istreambuf_iterator<char>(fb), {});
}

void SaveLogits(Engine& eng, const std::filesystem::path& path, int row = 0) {
  std::vector<float> logits(kVocab);
  HIP_OK(hipMemcpy(logits.data(), eng.logits() + (size_t)row * kVocab,
                    sizeof(float) * kVocab, hipMemcpyDeviceToHost));
  std::ofstream out(path, std::ios::binary);
  out.write(reinterpret_cast<const char*>(logits.data()), sizeof(float) * kVocab);
  if (!out) std::exit(1);
}

void SaveIds(const std::filesystem::path& path, const std::vector<int>& ids) {
  std::ofstream out(path);
  for (size_t i = 0; i < ids.size(); ++i) out << (i ? "," : "") << ids[i];
  out << '\n';
  if (!out) std::exit(1);
}

bool DumpsEqual(const std::filesystem::path& a, const std::filesystem::path& b) {
  for (int l = 0; l <= kLayers; ++l) {
    char name[40];
    std::snprintf(name, sizeof(name), "h_%02d.f32", l);
    if (!SameFile(a / name, b / name)) {
      std::fprintf(stderr, "dump mismatch: %s / %s\n", a.c_str(), name);
      return false;
    }
  }
  return true;
}
}  // namespace

int main(int argc, char** argv) {
#if defined(BATCHING_ENGINE_SHA) && defined(BATCHING_SOURCE_SHA)
  if (argc == 2 && std::string(argv[1]) == "--build-info") {
    std::printf("{\"candidate_sha256\":\"%s\",\"source_sha256\":\"%s\"}\n",
                BATCHING_ENGINE_SHA, BATCHING_SOURCE_SHA);
    return 0;
  }
#endif
  if (argc != 5) {
    std::fprintf(stderr, "usage: batch_gates CKPT PROMPT_PATHS OUT BASELINE\n");
    return 2;
  }
#ifdef BATCHING_GATED_SHA
  std::printf("build_commit=%s\n", BATCHING_GATED_SHA);
  std::fflush(stdout);
#endif
  const auto prompts = LoadPrompts(argv[2]);
  const std::filesystem::path root(argv[3]);
  if (std::filesystem::exists(root)) {
    std::fprintf(stderr, "gate output already exists; choose a new directory\n");
    return 2;
  }
  std::filesystem::create_directories(root / "chunk1");
  std::filesystem::create_directories(root / "chunk8");
  Engine eng(argv[1], 4096);
  eng.WarmPlans();
  GenOptions o;
  o.ids = prompts[0];
  o.n = 1;
  o.chunk = 1;
  o.dump = (root / "chunk1").string();
  const auto gate1 = RunGen(eng, o, {});
  o.chunk = 8;
  o.dump = (root / "chunk8").string();
  const auto gate2 = RunGen(eng, o, {});
  if (!DumpsEqual(root / "chunk1", argv[4]) || !DumpsEqual(root / "chunk1", root / "chunk8")) return 1;
  std::printf("gate1a=49/49 gate2=49/49\n");
  o.n = 64;
  o.dump.clear();
  o.spec = 0;
  const auto serial = RunGen(eng, o, {});
  o.spec = 2;
  const auto speculative = RunGen(eng, o, {});
  if (serial.gen != speculative.gen) return 1;
  std::printf("gate3=exact serial=%.2f tok/s spec2=%.2f tok/s\n", 63 / serial.decode_s, 63 / speculative.decode_s);

  constexpr int kSteps = 8, kPerfSteps = 32;
  Prompts reference(8);
  for (int s = 0; s < 8; ++s) {
    const auto dir = root / "single" / ("stream_" + std::to_string(s));
    std::filesystem::create_directories(dir);
    int token = Prefill(eng, 0, prompts[s]);
    for (int step = 0; step <= kPerfSteps; ++step) {
      token = eng.Forward(&token, 1, 0, 1, step < kSteps ? dir.string() : "")[0];
      reference[s].push_back(token);
      if (step < kSteps) SaveLogits(eng, dir / ("logits_" + std::to_string(step) + ".f32"));
    }
    SaveIds(dir / "ids.txt", reference[s]);
  }
  eng.ConfigureStreams(8);
  Engine::ReportMemory("eight-streams");
  for (int B = 1; B <= 8; ++B) {
    const auto dir = root / ("batch" + std::to_string(B));
    std::vector<int> ids(B), tokens(B);
    std::iota(ids.begin(), ids.end(), 0);
    for (int s = 0; s < B; ++s) {
      std::filesystem::create_directories(dir / ("stream_" + std::to_string(s)));
      tokens[s] = Prefill(eng, s, prompts[s]);
    }
    Prompts dump_ids(B), perf_ids(B);
    double seconds = 0;
    for (int step = 0; step < kSteps; ++step) {
      // Rotate row IDs without changing stream identity. State must follow the ID.
      std::rotate(ids.begin(), ids.begin() + 1 % B, ids.end());
      std::vector<int> input;
      for (int s : ids) input.push_back(tokens[s]);
      const double start = Now();
      const auto output = eng.ForwardBatch(input, ids, dir.string());
      seconds += Now() - start;
      for (int r = 0; r < B; ++r) {
        const int s = ids[r];
        tokens[s] = output[r];
        dump_ids[s].push_back(output[r]);
        if (output[r] != reference[s][step]) {
          std::fprintf(stderr, "token mismatch B=%d stream=%d step=%d\n", B, s, step);
          return 1;
        }
        const auto file = dir / ("stream_" + std::to_string(s)) / ("logits_" + std::to_string(step) + ".f32");
        SaveLogits(eng, file, r);
        if (!SameFile(file, root / "single" / ("stream_" + std::to_string(s)) / file.filename())) {
          std::fprintf(stderr, "logit mismatch B=%d stream=%d step=%d\n", B, s, step);
          return 1;
        }
      }
    }
    for (int s = 0; s < B; ++s)
      if (!DumpsEqual(dir / ("stream_" + std::to_string(s)), root / "single" / ("stream_" + std::to_string(s)))) return 1;
    std::printf("batch_gate B=%d: IDs, full logits, 49 dumps exact; dump-inclusive %.2f aggregate tok/s\n",
                B, B * kSteps / seconds);
    std::fflush(stdout);

    // Measure without dumps or logit copies. Warm this row count before timing.
    for (int s = 0; s < B; ++s) tokens[s] = Prefill(eng, s, prompts[s]);
    std::iota(ids.begin(), ids.end(), 0);
    tokens = eng.ForwardBatch(tokens, ids);
    for (int s = 0; s < B; ++s) perf_ids[s].push_back(tokens[s]);
    eng.CountExperts(true);
    const double start = Now();
    for (int step = 0; step < kPerfSteps; ++step) {
      tokens = eng.ForwardBatch(tokens, ids);
      for (int s = 0; s < B; ++s) {
        perf_ids[s].push_back(tokens[s]);
        if (tokens[s] != reference[s][step + 1]) {
          std::fprintf(stderr, "performance-run token mismatch B=%d stream=%d step=%d\n", B, s, step);
          return 1;
        }
      }
    }
    const double elapsed = Now() - start;
    for (int s = 0; s < B; ++s) {
      const auto stream_dir = dir / ("stream_" + std::to_string(s));
      SaveIds(stream_dir / "ids-dump.txt", dump_ids[s]);
      SaveIds(stream_dir / "ids-perf.txt", perf_ids[s]);
    }
    const auto counts = eng.ExpertCounts();
    const double union_mean = (double)counts[2 * B] / counts[2 * B + 1];
    // Tier-specific traffic accounting belongs in the report, not this checkpoint-neutral probe.
    const double rate = B * kPerfSteps / elapsed;
    std::printf("batch_perf B=%d: %.2f aggregate tok/s, %.2f per-stream, %.3f ms/step, "
                "union %.3f\n", B, rate, rate / B, elapsed * 1000 / kPerfSteps, union_mean);
    eng.CountExperts(false);
    std::fflush(stdout);
  }
  return 0;
}
