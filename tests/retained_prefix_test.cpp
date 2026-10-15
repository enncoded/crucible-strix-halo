#include "../engine/runtime/retained_prefix.h"
#include <cassert>
#include <string>
using crucible::RetainedPrefix;
int main() {
  RetainedPrefix s;
  assert(s.Forward(8, 0, 0).find("at least one") != std::string::npos);
  assert(s.Forward(8, 0, 9).find("commit") != std::string::npos);
  assert(s.Forward(8, 1, 8).find("replay") != std::string::npos);
  assert(s.Forward(8, 0, 8).empty());
  assert(s.Truncate(0).find("zero-prefix") != std::string::npos);
  assert(s.Truncate(7).find("committed") != std::string::npos);
  assert(s.Truncate(9).find("grow") != std::string::npos);
  assert(s.Forward(8, 0, 1).empty());
  assert(s.Truncate(8).find("zero-prefix") != std::string::npos);
  assert(s.Truncate(12).empty());
  assert(s.Forward(1, 2, 1).find("replay") != std::string::npos);
  assert(s.Forward(1, 4, 1).find("replay") != std::string::npos);
  assert(s.Forward(1, 3, 1).empty());
  assert(s.Truncate(12).find("zero-prefix") != std::string::npos);

  // Every CLI verification retains its first row and replays accepted drafts.
  for (int depth = 1; depth < 8; ++depth) {
    for (int accepted = 0; accepted <= depth; ++accepted) {
      RetainedPrefix cli;
      assert(cli.Forward(8, 0, 8).empty());
      assert(cli.Forward(8, 0, 8).empty());
      assert(cli.Forward(depth + 1, 0, 1).empty());
      assert(cli.Truncate(16 + accepted + 1).empty());
      assert(cli.Forward(3, accepted, 1).empty());
      assert(cli.Truncate(16 + accepted + 2).empty());
      assert(cli.Forward(1, 0, 1).empty());  // Context-end serial fallback.
    }
  }
  s.Reset();
  assert(s.Forward(1, 0, 1).empty());
}
