// API checks run without an Engine instance or a visible GPU.
#define ENGINE_LIBRARY
#include "../engine/engine.hip"
#include <cassert>

GenOptions Options(std::initializer_list<const char*> args) {
  std::vector<char*> av;
  for (const char* arg : args) av.push_back(const_cast<char*>(arg));
  GenOptions o;
  o.ids = {1};
  assert(ParseGenOptions((int)av.size(), av.data(), 0, o));
  return o;
}

int main() {
  std::vector<int> blocks;
  const auto defaults = Options({});
  assert(defaults.spec == 0 && defaults.draft_q == 0 && defaults.head_auto == 0);
  assert(defaults.draft_head == "full" && !AdaptiveDraft(defaults));
  assert(CheckGenOptions(defaults, blocks) == 0 && blocks.empty());
  assert(ResolveDraftBlocks(defaults, {}).empty());
  const auto full = Options({"--spec", "2", "--draft-head", "full"});
  assert(CheckGenOptions(full, blocks) == 0 && blocks.empty());
  const auto builtin = Options({"--spec", "7", "--draft-head", "64k"});
  assert(CheckGenOptions(builtin, blocks) == 0 && blocks.size() == 512);
  assert(ResolveDraftBlocks(builtin, {}) == blocks && AdaptiveDraft(builtin));
  const auto fixed = Options({"--spec", "7", "--draft-head", "64k", "--draft-adapt", "0"});
  assert(CheckGenOptions(fixed, blocks) == 0 && !AdaptiveDraft(fixed));
  const auto online = Options({"--spec", "2", "--draft-head", "64k", "--draft-head-auto", "1"});
  assert(CheckGenOptions(online, blocks) == 0 && online.head_auto == 1);
  const auto serial = Options({"--draft-head", "64k"});
  assert(CheckGenOptions(serial, blocks) == 0 && blocks.empty());
  for (const auto& invalid : {
      Options({"--draft-head", "bad"}),
      Options({"--draft-head-auto", "1"}),
      Options({"--spec", "2", "--draft-head", "full", "--draft-head-auto", "1"}),
      Options({"--draft-head", "full", "--draft-adapt", "1"}),
      Options({"--draft-head", "64k", "--draft-blocks", "unused"})})
    assert(CheckGenOptions(invalid, blocks) == 2);
  // Prefill dispatch: auto by default, --chunk alone keeps the small-row path.
  assert(defaults.prefill_rows == kPrefillAuto);
  assert(Options({"--chunk", "8"}).prefill_rows == 0);
  assert(Options({"--chunk", "8", "--prefill", "24"}).prefill_rows == 24);
  assert(Options({"--prefill", "0"}).prefill_rows == 0);
  assert(CheckGenOptions(Options({"--prefill", "-2"}), blocks) == 2);
}
