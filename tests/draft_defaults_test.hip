// CPU-only checks. No Engine instance or HIP allocation is needed.
#define ENGINE_LIBRARY
#include "../engine/engine.hip"
#include <cassert>

GenOptions Options(std::initializer_list<const char*> args) {
  std::vector<char*> av;
  for (const char* arg : args) av.push_back(const_cast<char*>(arg));
  GenOptions o = CliGenOptions();
  o.ids = {1};
  assert(ParseGenOptions((int)av.size(), av.data(), 0, o));
  return o;
}

int main() {
  std::vector<int> blocks;
  const auto defaults = Options({});
  assert(defaults.spec == 7 && defaults.draft_q == 0.3f && defaults.mtp_prefix == 256);
  assert(defaults.draft_head == "64k" && defaults.lookup == 0 && defaults.head_auto == 0);
  // Programmatic users keep serial full-head defaults.
  const GenOptions plain;
  assert(plain.spec == 0 && plain.draft_q == 0 && plain.draft_head == "full" && plain.mtp_prefix == 0);
  assert(CheckGenOptions(defaults, blocks) == 0);
  assert(blocks.size() == 512 && blocks.front() == 0 && blocks.back() == 1761);
  assert(BuiltinDraft(defaults) && AdaptiveDraft(defaults));
  assert(ValidDraftSelection(blocks, kVocab / 128));
  assert(ResolveDraftBlocks(defaults, {}) == blocks);

  const auto serial = Options({"--spec", "0"});
  assert(CheckGenOptions(serial, blocks) == 0 && blocks.empty());
  assert(ResolveDraftBlocks(serial, {1, 2}).empty());
  const auto single = Options({"--n", "1"});
  assert(CheckGenOptions(single, blocks) == 0 && blocks.empty());
  assert(ResolveDraftBlocks(single, {1, 2}).empty());
  const auto capped = Options({"--mtp-prefix", "8"});
  assert(CheckGenOptions(capped, blocks) == 0 && capped.mtp_prefix == 8);
  const auto uncapped = Options({"--mtp-prefix", "0"});
  assert(CheckGenOptions(uncapped, blocks) == 0 && uncapped.mtp_prefix == 0);
  const auto full = Options({"--draft-head", "full"});
  assert(CheckGenOptions(full, blocks) == 0 && blocks.empty() && full.spec == 7);
  assert(!BuiltinDraft(full) && !AdaptiveDraft(full));
  const auto fixed = Options({"--draft-head", "64k", "--draft-adapt", "0"});
  assert(CheckGenOptions(fixed, blocks) == 0 && blocks.size() == 512);
  assert(!AdaptiveDraft(fixed));
  const auto automatic = Options({"--draft-head-auto", "1"});
  assert(CheckGenOptions(automatic, blocks) == 0 && blocks.size() == 512);
  const auto depth = Options({"--draft-head", "full", "--spec", "2", "--draft-q", "0"});
  assert(CheckGenOptions(depth, blocks) == 0 && blocks.empty());
  assert(depth.spec == 2 && depth.draft_q == 0);
  auto custom = Options({});
  custom.draft_blocks = "custom-selection";
  assert(!BuiltinDraft(custom) && !AdaptiveDraft(custom));
  custom.adapt = 1;
  assert(AdaptiveDraft(custom));
  assert(ResolveDraftBlocks(custom, {0, 1, 2}) == std::vector<int>({0, 1, 2}));

  for (const auto& invalid : {
      Options({"--draft-head", "bad"}),
      Options({"--mtp-prefix", "-1"}),
      Options({"--mtp-prefix", "262145"}),
      Options({"--draft-head", "full", "--draft-adapt", "1"}),
      Options({"--draft-head", "full", "--draft-head-auto", "1"}),
      Options({"--draft-head", "64k", "--draft-blocks", "unused"})})
    assert(CheckGenOptions(invalid, blocks) == 2);
}
