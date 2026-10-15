// Capture real top-10 routes without modifying the gated engine or its kernels.
#define ENGINE_LIBRARY
#include "../engine/engine.hip"

#include <array>

namespace {
struct EngineStateIO {
  static int Step(Engine& e, int token, std::array<int, kLayers * kTopK>& routes) {
    SequenceState& state = *e.sequence_;
    const int pos = (int)state.hist.size();
    if (!ContextRowsFit(pos, 1, e.ctx_, kMaxRows)) std::exit(2);
    if (!state.retained.Forward(1, 0, 1).empty()) std::exit(2);
    state.hist.push_back(token);
    const auto& plan = e.PlansFor(1);
    const auto rows = e.PleRows(pos, 1);
    HIP_OK(hipMemcpy(e.tok_, &token, sizeof(token), hipMemcpyHostToDevice));
    HIP_OK(hipMemcpy(e.rows_, rows.data(), rows.size() * sizeof(rows[0]), hipMemcpyHostToDevice));
    EmbedKernel<<<dim3(kHidden / 256, 1), 256>>>(e.m_.embed, e.tok_, e.H_);
    for (int l = 0; l < kLayers; ++l) {
      if (l == 1) e.Ple(1, pos);
      e.Block(l, e.m_.L[l], plan.L[l], 1, pos, 0, 1);
      HIP_OK(hipMemcpy(routes.data() + l * kTopK, e.sel_moe_, kTopK * sizeof(int), hipMemcpyDeviceToHost));
    }
    return e.Head(e.m_.final_hc, plan, 1)[0];
  }
};
}  // namespace

int main(int argc, char** argv) {
  if (argc != 4) {
    std::fprintf(stderr, "usage: batch_routing CKPT PROMPT_PATHS OUTPUT.bin\n");
    return 2;
  }
  std::ifstream paths(argv[2]);
  std::vector<std::vector<int>> prompts;
  for (std::string path; std::getline(paths, path);) {
    if (path.empty()) continue;
    std::ifstream file(path);
    std::stringstream text;
    text << file.rdbuf();
    auto ids = ParseIds(text.str());
    if (!file || ids.empty() || !ValidTokenIds(ids, kVocab)) return 2;
    prompts.push_back(ids);
  }
  if (!paths.eof() || prompts.size() != 8) return 2;
  FILE* out = std::fopen(argv[3], "wb");
  if (!out) return 2;
  Engine eng(argv[1], 4096);
  const int header[]{8, 16, kLayers, kTopK};
  if (std::fwrite(header, sizeof(int), 4, out) != 4) return 1;
  for (size_t p = 0; p < prompts.size(); ++p) {
    eng.Reset();
    std::vector<int> last;
    for (size_t i = 0; i < prompts[p].size(); i += 8) {
      const int R = (int)std::min<size_t>(8, prompts[p].size() - i);
      last = eng.Forward(prompts[p].data() + i, R, 0, R);
    }
    int token = last.back();
    const double start = Now();
    for (int step = 0; step < 16; ++step) {
      std::array<int, kLayers * kTopK> routes;
      token = EngineStateIO::Step(eng, token, routes);
      if (std::fwrite(routes.data(), sizeof(int), routes.size(), out) != routes.size()) return 1;
    }
    std::printf("prompt %zu: %zu prefill tokens, 16 routing rows, %.3f s\n", p, prompts[p].size(), Now() - start);
  }
  if (std::fclose(out) != 0) return 1;
  return 0;
}
