#include "../engine/model/flashnext/weight_validation.h"
int main(int argc, char** argv) {
  if (argc == 2) { const std::string s = argv[1]; st::Scanner sc(s); sc.Str(); return 0; }
  st::Checkpoint ck(argv[1]);
  ck.Info("x");
  if (argc > 3) crucible::ValidateLinear(ck, "lin", 128, 128);
}
