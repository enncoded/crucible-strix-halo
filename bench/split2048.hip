// This probe uses the production prompt code with one explicit row-schedule substitution.
#define ENGINE_LIBRARY
#include "../engine/engine.hip"
#include "split2048_prompt.h"
#include <filesystem>

namespace {
void Clear(Engine& e) {
  e.Reset();
  for (const auto& span : e.ActiveSequence().Spans())
    HIP_OK(hipMemset(const_cast<void*>(span.pointer), 0, span.bytes));
}
void WriteBytes(const std::filesystem::path& path, const void* data, size_t bytes) {
  std::ofstream output(path, std::ios::binary);
  if (!output || !output.write(static_cast<const char*>(data), bytes))
    throw std::runtime_error("cannot write probe capture");
}
std::vector<int> ReadIds(const char* path) {
  std::ifstream input(path);
  if (!input) throw std::runtime_error("cannot read probe IDs");
  std::stringstream text; text << input.rdbuf();
  return ParseIds(text.str());
}
std::vector<int> Schedule(int length, int cut) {
  std::vector<int> result;
  bool found = false;
  for (int at = 0; at < length; at += 4096) {
    const int end = std::min(at + 4096, length);
    if (at < cut && cut < end) {
      if ((cut - at) % kPrefillCoreRows || std::min(cut - at, end - cut) <= kMaxRows)
        throw std::runtime_error("invalid core cut");
      found = true;
      result.push_back(cut - at); result.push_back(end - cut);
    } else result.push_back(end - at);
  }
  if (cut && !found) throw std::runtime_error("cut is not an interior core boundary");
  return result;
}
void Capture(Engine& e, const std::vector<int>& ids, int cut, const std::filesystem::path& root,
             float* stk, float* carry, bool restore_cut = false) {
  std::filesystem::create_directory(root);
  Clear(e);
  HIP_OK(hipMemset(stk, 0, (kMaxRows + 1) * kWide * sizeof(float)));
  HIP_OK(hipMemset(carry, 0, kWide * sizeof(float)));
  ConversationHooks hooks;
  hooks.admit_prompt = [&](size_t n) { return n == ids.size() || (restore_cut && n == size_t(cut)); };
  hooks.save_generated_tail = false;
  hooks.save = [&](const ConversationResume& resume, float* s, float* c) {
    const auto snapshot = ConversationStateIO::Save(e, e.ActiveSequence(), resume, s, c);
    if (restore_cut && resume.length == size_t(cut)) {
      Clear(e);
      HIP_OK(hipMemset(s, 0, (kMaxRows + 1) * kWide * sizeof(float)));
      HIP_OK(hipMemset(c, 0, kWide * sizeof(float)));
      ConversationStateIO::Restore(e, e.ActiveSequence(), *snapshot, s, c);
      std::ofstream marker(root / "restore.json");
      if (!marker) throw std::runtime_error("cannot write restore marker");
      marker << "{\"length\":" << resume.length << ",\"bytes\":" << snapshot->bytes.size() << "}\n";
      return;
    }
    const auto state_bytes = snapshot->bytes.Copy();
    WriteBytes(root / "state.bin", state_bytes.data(), state_bytes.size());
    WriteBytes(root / "logits.f32", resume.logits.data(), resume.logits.size() * sizeof(float));
    std::ofstream meta(root / "state.json");
    meta << "{\"length\":" << resume.length << ",\"mpos\":" << resume.mpos
         << ",\"have_carry\":" << (resume.have_carry ? "true" : "false")
         << ",\"next_id\":" << resume.next_id << ",\"bytes\":" << snapshot->bytes.size() << "}\n";
  };
  const auto copy = [](float* dst, const float* src, size_t count) {
    HIP_OK(hipMemcpyAsync(dst, src, count * sizeof(float), hipMemcpyDeviceToDevice, 0));
  };
  const auto mtp = [&](float* stacks, const int* tokens, int rows, int pos) { e.Mtp(stacks, tokens, rows, pos, false); };
  int mpos = 0;
  const double start = Now();
  const auto last = cut ? RunSplitPrompt(e, std::span<const int>(ids), 8, true, "", stk, carry, mpos,
      copy, mtp, [](int) {}, 256, {}, &hooks, Schedule(int(ids.size()), cut))
      : RunPrompt(e, std::span<const int>(ids), 8, true, "", stk, carry, mpos,
      copy, mtp, [](int) {}, 256, {}, &hooks);
  HIP_OK(hipDeviceSynchronize());
  const double seconds = Now() - start;
  std::vector<int> generated{last.back()};
  for (int i = 1; i < 8; ++i) generated.push_back(e.Forward(&generated.back(), 1, 0, 1).at(0));
  WriteBytes(root / "ids.i32", generated.data(), generated.size() * sizeof(int));
  std::printf("RESULT %s rows=%zu cut=%d seconds=%.6f mpos=%d\n", root.c_str(), ids.size(), cut, seconds, mpos);
  std::fflush(stdout);
}
}

int main(int argc, char** argv) {
  try {
    if (argc != 9) return 2;
    const std::filesystem::path root(argv[2]);
    if (!std::filesystem::create_directory(root)) return 2;
    Engine e(argv[1], 24576, 4096);
    e.DraftHead({}, true);
    float* stk = Zeros<float>((kMaxRows + 1) * kWide);
    float* carry = Zeros<float>(kWide);
    for (int i = 3; i < 9; i += 2) {
      const auto ids = ReadIds(argv[i]);
      const int cut = std::stoi(argv[i + 1]);
      if (ids.empty() || ids.size() > 20480 || !ValidTokenIds(ids, kVocab)) return 2;
      const auto case_root = root / ("case" + std::to_string((i - 3) / 2));
      std::filesystem::create_directory(case_root);
      Capture(e, ids, 0, case_root / "whole", stk, carry);
      Capture(e, ids, cut, case_root / "split", stk, carry);
      Capture(e, ids, cut, case_root / "restored", stk, carry, true);
    }
    HIP_OK(hipFree(stk)); HIP_OK(hipFree(carry));
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "split2048 probe: %s\n", error.what()); return 1;
  }
}
