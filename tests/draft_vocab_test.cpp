#include <cassert>
#include <fstream>
#include <sstream>
#include "../engine/runtime/draft_policy.h"
#include "../engine/runtime/draft_vocab.h"

int main(int argc, char** argv) {
  assert(argc == 2);
  std::ifstream file(argv[1]);
  assert(file);
  std::stringstream text;
  text << file.rdbuf();
  const auto expected = crucible::ParseDraftSelection(text.str());
  const std::vector<int> actual(crucible::kDefaultDraftBlocks.begin(),
                                crucible::kDefaultDraftBlocks.end());
  assert(actual == expected);
  assert(actual.size() == 512);
  assert(crucible::ValidDraftSelection(actual, 1940));
}
