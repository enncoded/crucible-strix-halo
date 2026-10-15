// Sparse QSA and changing active-stream sets at unequal positions above 2048.
#define ENGINE_LIBRARY
#include "../engine/engine.hip"

#include <filesystem>

namespace {
int PrefillSparse(Engine& eng, int stream, const std::vector<int>& prompt) {
  eng.SelectStream(stream);
  eng.Reset();
  int next = 0;
  for (size_t i = 0; i < prompt.size(); i += 8) {
    const int R = (int)std::min<size_t>(8, prompt.size() - i);
    next = eng.Forward(prompt.data() + i, R, 0, R).back();
  }
  return next;
}

std::vector<int> Pattern(const std::string& path, size_t size) {
  std::ifstream file(path);
  std::stringstream text;
  text << file.rdbuf();
  const auto pattern = ParseIds(text.str());
  if (!file || pattern.empty() || !ValidTokenIds(pattern, kVocab)) std::exit(2);
  std::vector<int> prompt(size);
  for (size_t i = 0; i < size; ++i) prompt[i] = pattern[i % pattern.size()];
  return prompt;
}

std::vector<float> HostLogits(Engine& eng, int row) {
  std::vector<float> result(kVocab);
  HIP_OK(hipMemcpy(result.data(), eng.logits() + (size_t)row * kVocab,
                    kVocab * sizeof(float), hipMemcpyDeviceToHost));
  return result;
}
}  // namespace

int main(int argc, char** argv) {
  if (argc != 4) {
    std::fprintf(stderr, "usage: batch_sparse_gate CKPT PROMPT1.ids PROMPT2.ids\n");
    return 2;
  }
  const std::vector<std::vector<int>> prompts{Pattern(argv[2], 2055), {}, Pattern(argv[3], 2061)};
  Engine eng(argv[1], 4096);
  eng.WarmPlans();
  std::vector<std::vector<int>> reference(3);
  std::vector<std::vector<std::vector<float>>> logits(3);
  for (int s : {0, 2}) {
    int token = PrefillSparse(eng, 0, prompts[s]);
    for (int step = 0; step < 4; ++step) {
      token = eng.Forward(&token, 1, 0, 1)[0];
      reference[s].push_back(token);
      logits[s].push_back(HostLogits(eng, 0));
    }
  }
  eng.ConfigureStreams(3);
  std::vector<int> tokens(3), positions(3);
  for (int s : {0, 2}) tokens[s] = PrefillSparse(eng, s, prompts[s]);
  for (const std::vector<int>& active : std::vector<std::vector<int>>{{2, 0}, {0}, {2}, {0, 2}, {2, 0}}) {
    std::vector<int> input;
    for (int s : active) input.push_back(tokens[s]);
    const auto output = eng.ForwardBatch(input, active);
    for (size_t r = 0; r < active.size(); ++r) {
      const int s = active[r], step = positions[s]++;
      const auto got = HostLogits(eng, (int)r);
      if (output[r] != reference[s][step] ||
          std::memcmp(got.data(), logits[s][step].data(), kVocab * sizeof(float)) != 0) {
        std::fprintf(stderr, "sparse batch mismatch: stream %d local step %d\n", s, step);
        return 1;
      }
      tokens[s] = output[r];
    }
  }
  std::printf("sparse_batch_gate: IDs and full logits exact; unequal contexts 2055/2061, active sets 2/1/1/2/2\n");
  return 0;
}
