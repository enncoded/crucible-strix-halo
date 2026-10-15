#include "../bench/compact_transition.h"
#include <cassert>
#include <stdexcept>
#include <string>
#include <vector>

struct MockEngine {
  int stream = -1, count = 0;
  std::vector<std::string> trace;
  void ConfigureServeStreams(int n) { count = n; trace.push_back("configure"); }
  void SelectStream(int n) { assert(n >= 0 && n < count); stream = n; trace.push_back("select" + std::to_string(n)); }
};

struct MockSnapshot {
  std::vector<int> hist;
  std::vector<std::pair<int, int>> qsa_layout;
  struct { size_t length; int mpos; } resume;
  std::vector<unsigned char> bytes;
};

template<class Fn>
bool Rejects(Fn fn) {
  try { fn(); } catch (const std::runtime_error&) { return true; }
  return false;
}

int main() {
  // The compared stream-one output must come from its first restore, which performs
  // the lazy MTP promotion. A later restore into a present cache would mask it.
  MockEngine engine;
  bool promoted = false;
  int restores[2] = {0, 0};
  const auto result = crucible::CompactAcrossStreams(engine,
    [&] { assert(engine.stream == 0); engine.trace.push_back("save"); return 42; },
    [&](int saved) {
      assert(saved == 42);
      if (engine.stream == 1) promoted = true;
      engine.trace.push_back("continue" + std::to_string(engine.stream));
      return ++restores[engine.stream];
    },
    [&] { assert(engine.stream == 1); engine.trace.push_back("serial"); },
    [&] { assert(engine.stream == 1); return promoted; });
  assert(result.first == 1 && result.second == 1);
  assert((engine.trace == std::vector<std::string>{"configure", "select0", "save", "continue0",
                                                  "select1", "serial", "continue1"}));
  assert(Rejects([] {
    MockEngine present;
    crucible::CompactAcrossStreams(present, [] { return 1; }, [](int n) { return n; },
      [] {}, [] { return true; });
  }));
  assert(Rejects([] {
    MockEngine absent;
    crucible::CompactAcrossStreams(absent, [] { return 1; }, [](int n) { return n; },
      [] {}, [] { return false; });
  }));

  // The audit compares snapshot content, not only its size.
  const MockSnapshot saved{{1, 2, 3}, {{3, 16}, {48, 16}}, {3, 2}, {7, 8, 9, 10}};
  crucible::RequireSameSnapshot(saved, saved);
  auto content = saved; content.bytes[2] ^= 1;
  auto hist = saved; hist.hist[0] = 5;
  auto layout = saved; layout.qsa_layout[1].second = 0;
  auto length = saved; length.resume.length = 2;
  auto mpos = saved; mpos.resume.mpos = 1;
  for (const auto* changed : {&content, &hist, &layout, &length, &mpos})
    assert(Rejects([&] { crucible::RequireSameSnapshot(saved, *changed); }));
}
