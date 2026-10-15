#define ENGINE_LIBRARY
#include "../engine/engine.hip"
// G15 quality probe ("Session replay"). For each prompt P+S: generate G tokens after P with
// the serve defaults and save the generated tail; then compare the next-token logits after P+gen+S on three paths:
//   replay: restore the tail (decode-path state for gen) and prefill only S;
//   bulk:   the default fresh-request path (bulk prefill of P+gen+S on the fixed grid);
//   shift:  bulk prefill with the grid shifted by one row, a floor for schedule-only differences.
// Usage: session_replay_probe CKPT GEN SUFFIX IDS_FILE...   (comma-separated IDs; one JSON line per prompt)
//        session_replay_probe CKPT dump DIR IDS_FILE GEN    (Gate 1(b) dumps, see DumpMode)
#include <cmath>
#include <filesystem>

namespace {
std::vector<float> ReadLogits(const std::string& path) {
  std::vector<float> logits(kVocab);
  std::ifstream file(path, std::ios::binary);
  if (!file.read(reinterpret_cast<char*>(logits.data()), logits.size() * sizeof(float)))
    throw std::runtime_error("cannot read logits " + path);
  return logits;
}
std::vector<double> LogSoftmax(const std::vector<float>& x) {
  const double peak = *std::max_element(x.begin(), x.end());
  double sum = 0;
  for (const float v : x) sum += std::exp(double(v) - peak);
  std::vector<double> out(x.size());
  for (size_t i = 0; i < x.size(); ++i) out[i] = double(x[i]) - peak - std::log(sum);
  return out;
}
// KL(test || base), relative L2 of the raw logits, and both argmax IDs.
std::string Compare(const std::vector<float>& test, const std::vector<float>& base) {
  const auto lt = LogSoftmax(test), lb = LogSoftmax(base);
  double kl = 0, diff = 0, norm = 0;
  for (size_t i = 0; i < lt.size(); ++i) {
    kl += std::exp(lt[i]) * (lt[i] - lb[i]);
    diff += (double(test[i]) - base[i]) * (double(test[i]) - base[i]);
    norm += double(base[i]) * base[i];
  }
  const auto top = [](const std::vector<float>& x) { return int(std::max_element(x.begin(), x.end()) - x.begin()); };
  char text[160];
  std::snprintf(text, sizeof text, "{\"kl\":%.6e,\"rel_l2\":%.6e,\"top1\":%d,\"top1_base\":%d}",
                kl, std::sqrt(diff / norm), top(test), top(base));
  return text;
}
}  // namespace

// Gate 1(b) input: residual dumps of the same tokens on the decode path (what a replayed tail holds; spec 0 so
// every dumped row is a kept position) and on the bulk path. ref/oracle.py then scores each against fp64.
int DumpMode(char** argv) {
  const std::string root = argv[3];
  const int gen = GenNumber<int>(argv[5]);
  std::ifstream file(argv[4]); std::stringstream text; text << file.rdbuf();
  const auto prompt = ParseIds(text.str());
  Engine e(argv[1], 4096, kPrefillAuto);
  std::filesystem::create_directories(root + "/decode");
  std::filesystem::create_directories(root + "/bulk");
  GenOptions o = CliGenOptions();
  o.ids = prompt; o.n = gen; o.ctx = 4096; o.spec = 0; o.dump = root + "/decode";
  const auto decoded = RunGen(e, o, {});
  std::vector<int> ids = prompt;
  ids.insert(ids.end(), decoded.gen.begin(), decoded.gen.end() - 1);
  GenOptions b = CliGenOptions();
  b.ids = ids; b.n = 1; b.ctx = 4096; b.spec = 0; b.dump = root + "/bulk";
  RunGen(e, b, {});
  std::ofstream out(root + "/ids.txt");
  for (size_t i = 0; i < ids.size(); ++i) out << (i ? "," : "") << ids[i];
  std::printf("dumped %zu positions (prompt %zu, decode rows %zu)\n", ids.size(), prompt.size(), ids.size() - prompt.size());
  return 0;
}

int main(int argc, char** argv) {
  setenv("HSA_USERPTR_FOR_PAGED_MEM", "0", 0);
  if (argc == 6 && std::string(argv[2]) == "dump") return DumpMode(argv);
  if (argc < 5) {
    std::fprintf(stderr, "usage: session_replay_probe CKPT GEN SUFFIX IDS_FILE... | CKPT dump DIR IDS_FILE GEN\n");
    return 2;
  }
  const int gen = GenNumber<int>(argv[2]), suffix = GenNumber<int>(argv[3]);
  constexpr int kCtx = 32768;
  Engine e(argv[1], kCtx, kPrefillAuto);
  const std::string scratch = "build/session-replay-probe";
  std::filesystem::create_directories(scratch);
  for (int index = 4; index < argc; ++index) {
    std::ifstream file(argv[index]); std::stringstream text; text << file.rdbuf();
    const auto source = ParseIds(text.str());
    if (source.size() <= size_t(suffix) + 8) throw std::runtime_error("prompt shorter than its suffix");
    const std::vector<int> prompt(source.begin(), source.end() - suffix), user(source.end() - suffix, source.end());
    GenOptions o = CliGenOptions();
    o.ids = prompt; o.n = gen; o.ctx = kCtx;
    ConversationHooks hooks;
    std::shared_ptr<const ConversationSnapshot> tail;
    hooks.save = [&](const ConversationResume& state, float* s, float* c) {
      if (!state.prompt_only) tail = ConversationStateIO::Save(e, e.ActiveSequence(), state, s, c);
    };
    hooks.skip_prompt_interior_save = true;  // only the generated tail matters here
    o.conversation = &hooks;
    const auto first = RunGen(e, o, {});
    std::vector<int> next = prompt;
    next.insert(next.end(), first.gen.begin(), first.gen.end());
    next.insert(next.end(), user.begin(), user.end());
    if (!tail || tail->hist.size() >= next.size() || !std::equal(tail->hist.begin(), tail->hist.end(), next.begin())) {
      std::printf("{\"file\":\"%s\",\"tail\":false}\n", argv[index]);
      continue;
    }
    const auto run = [&](const std::string& name, ConversationHooks* conversation) {
      GenOptions r = CliGenOptions();
      r.ids = next; r.n = 1; r.ctx = kCtx; r.conversation = conversation;
      r.logits_dump = scratch + "/" + name + ".f32";
      RunGen(e, r, {});
      return ReadLogits(r.logits_dump);
    };
    ConversationHooks replay;
    replay.continue_tail = true;
    replay.restore = [&](float* s, float* c) { return ConversationStateIO::Restore(e, e.ActiveSequence(), *tail, s, c); };
    const double t0 = GenerationNow();
    const auto replayed = run("replay", &replay);
    const double t1 = GenerationNow();
    const auto bulk = run("bulk", nullptr);
    const double t2 = GenerationNow();
    // The shifted floor: one plain row first moves every later bulk chunk boundary by one position.
    e.Reset();
    e.Forward(next.data(), 1, 0, 1);
    const int rows = e.PrefillRows();
    std::vector<int> last;
    for (size_t i = 1; i < next.size(); i += rows) {
      const int r = int(std::min<size_t>(rows, next.size() - i));
      last = e.Prefill(next.data() + i, r, "", i + r == next.size());
    }
    const auto shifted = e.CopyLogits(1, 0);
    std::printf("{\"file\":\"%s\",\"tail\":true,\"prompt\":%zu,\"generated\":%zu,\"tail_tokens\":%zu,\"tail_replay\":%d,"
                "\"next\":%zu,\"replay_s\":%.4f,\"bulk_s\":%.4f,\"replay_vs_bulk\":%s,\"shift_vs_bulk\":%s,"
                "\"replay_vs_shift\":%s}\n",
                argv[index], prompt.size(), first.gen.size(), tail->hist.size(), tail->resume.replay, next.size(),
                t1 - t0, t2 - t1, Compare(replayed, bulk).c_str(), Compare(shifted, bulk).c_str(),
                Compare(replayed, shifted).c_str());
    std::fflush(stdout);
  }
  return 0;
}
