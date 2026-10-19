#include <cassert>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <unistd.h>
#include <vector>
#include <random>
#include "../engine/runtime/draft_policy.h"

using crucible::LookupDraft;
std::vector<int> Brute(const std::vector<int>& h, int depth) {
  for (int len = std::min(32, (int)h.size() - 1); len >= 3; --len)
    for (int end = (int)h.size() - 1; end >= len; --end)
      if (std::equal(h.begin() + end - len, h.begin() + end, h.end() - len))
        return {h.begin() + end, h.begin() + std::min(end + depth, (int)h.size())};
  return {};
}
int main() {
  using crucible::ParseDraftSelection;
  using crucible::ValidDraftSelection;
  assert(ParseDraftSelection(" 0, 42,1939\n") == std::vector<int>({0, 42, 1939}));
  for (const auto* invalid : {"", "1,", "1,,2", "1junk", "1.5", "2147483648", "x"}) {
    bool caught = false;
    try { ParseDraftSelection(invalid); } catch (const std::invalid_argument&) { caught = true; }
    assert(caught);
  }
  assert(ValidDraftSelection({0, 1, 1939}, 1940));
  assert(!ValidDraftSelection({}, 1940));
  assert(!ValidDraftSelection({-1}, 1940));
  assert(!ValidDraftSelection({1940}, 1940));
  assert(!ValidDraftSelection({1, 1}, 1940));
  assert(!ValidDraftSelection({2, 1}, 1940));
  // serve --draft-blocks: a block file for the served head; any fault is one fixed text.
  assert(crucible::ServeDraftBlocks("1,5,1937\n", 1940) == std::vector<int>({1, 5, 1937}));
  for (const char* invalid : {"", "1,x", "5,1", "1,1", "1940", "-1"}) {
    bool caught = false;
    try { crucible::ServeDraftBlocks(invalid, 1940); } catch (const std::invalid_argument&) { caught = true; }
    assert(caught);
  }
  // An empty or blank file gets its own text.
  for (const char* blank : {"", " \n\t"}) {
    std::string what;
    try { crucible::ServeDraftBlocks(blank, 1940); } catch (const std::invalid_argument& e) { what = e.what(); }
    assert(what == "draft-blocks file is empty");
  }
  // The file read of serve --draft-blocks: a regular file of at most 64 KiB.
  {
    namespace fs = std::filesystem;
    const fs::path dir = fs::temp_directory_path() / ("draft_policy_test." + std::to_string(::getpid()));
    fs::remove_all(dir);
    fs::create_directories(dir);
    auto write = [&](const char* name, const std::string& text) {
      std::ofstream(dir / name, std::ios::binary) << text;
      return (dir / name).string();
    };
    auto fault = [](const std::string& path) {
      try { crucible::ReadServeDraftBlocks(path, 1940); } catch (const std::exception& e) { return std::string(e.what()); }
      return std::string();
    };
    assert(crucible::ReadServeDraftBlocks(write("head.txt", "1,5,1937\n"), 1940) == std::vector<int>({1, 5, 1937}));
    std::string full;
    for (int block = 0; block < 1940; ++block) full += (block ? "," : "") + std::to_string(block);
    assert(crucible::ReadServeDraftBlocks(write("full.txt", full + "\n"), 1940).size() == 1940);
    assert(fault((dir / "missing.txt").string()) == "draft-blocks file could not be read");
    assert(fault(dir.string()) == "draft-blocks file must be a regular file");
    assert(fault(write("empty.txt", "")) == "draft-blocks file is empty");
    assert(fault(write("big.txt", "1" + std::string(64 * 1024, ' '))) == "draft-blocks file is larger than 64 KiB");
    assert(crucible::ReadServeDraftBlocks(write("edge.txt", "1" + std::string(64 * 1024 - 1, ' ')), 1940) ==
           std::vector<int>({1}));
    fs::remove_all(dir);
  }
  // The startup line gives the head size and the built-in head size.
  assert(crucible::ServeDraftHeadLine(1940, 512, 2560 * 5 / 8 + 2) ==
         "serve draft head: 1940 blocks from --draft-blocks, 397.81 MB (the built-in head: 512 blocks, 104.99 MB)");
  assert(LookupDraft({}, 4).empty());
  assert(LookupDraft({1, 2, 3}, 4).empty());
  assert(LookupDraft({1, 2, 3, 4, 5, 1, 2, 3}, 4) == std::vector<int>({4, 5, 1, 2}));
  assert(LookupDraft({1, 2, 3, 4, 5, 1, 2, 3}, 1) == std::vector<int>({4}));
  assert(LookupDraft({1, 2, 7, 1, 2}, 4).empty());
  assert(LookupDraft({1, 2, 3, 4, 1, 2, 3, 9, 1, 2, 3}, 2) == std::vector<int>({9, 1}));
  assert(LookupDraft({0, 1, 2, 3, 8, 1, 2, 3, 9, 0, 1, 2, 3}, 2) == std::vector<int>({8, 1}));
  assert(LookupDraft({1, 2, 3, 4, 1, 2, 3}, 0).empty());
  assert(LookupDraft({1, 2, 3, 4, 1, 2, 3}, 4, 4).empty());
  std::mt19937 rng(42);
  for (int trial = 0; trial < 10000; ++trial) {
    std::vector<int> history(rng() % 500);
    for (auto& token : history) token = rng() % 8;
    const int depth = 1 + rng() % 7;
    assert(LookupDraft(history, depth) == Brute(history, depth));
  }
}
