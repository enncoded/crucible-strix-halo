// the --inject spec parser of the debug build (layers/nonfinite_probe.h, mask bit 4).
#define CRUCIBLE_DEBUG_NONFINITE 4
#include "../engine/layers/nonfinite_probe.h"

#include <cassert>
#include <cstdio>
#include <stdexcept>

using namespace crucible;

static bool Rejects(const char* spec) {
  try { ParseInjectPlan(spec); } catch (const std::invalid_argument&) { return true; }
  return false;
}

int main() {
  assert(!ParseInjectPlan(""));
  const auto kv = ParseInjectPlan("kv,11,26,1");
  assert(kv && kv->kind == "kv" && kv->layer == 11 && kv->pos == 26 && kv->stream == 1 && !kv->done);
  const auto hidden = ParseInjectPlan("hidden,10,30");
  assert(hidden && hidden->stream == 0);
  assert(ParseInjectPlan("gdn,0,0") && ParseInjectPlan("index,47,1027") && ParseInjectPlan("hidden,47,5"));
  // QSA layers are 3, 7, ..., 47: kv and index need one, gdn needs a GDN layer.
  for (const char* bad : {"kv,10,26", "index,10,1027", "gdn,11,0", "index,11,1026", "nan,10,1", "hidden,48,1",
                          "hidden,-1,1", "hidden,10,-1", "hidden,10,1,-1", "hidden,10", "hidden,10,1,0,9", "hidden,x,1",
                          "hidden,99999999999,1", "hidden,10,99999999999", "kv,11,26,99999999999"})
    assert(Rejects(bad));
  std::puts("nonfinite_inject_test: ok");
}
