#include <cassert>
#include "../engine/runtime/batch_spec_contract.h"
using namespace crucible;
int main() {
  SpecPrefix first{10,64,{}};
  assert(first.retained.Forward(8,0,8).empty());
  assert(first.retained.Forward(2,0,2).empty());
  SpecPrefix pending=first;
  assert(pending.retained.Forward(4,0,1).empty());
  assert(pending.retained.Truncate(13).empty());
  pending.length=13;
  const std::vector<SpecPrefix> prefixes{first,pending};
  const std::vector<VerifySpan> good{{0,{1,2,3,4},0,1},{1,{5,6,7,8},2,1}};
  assert(SpecSpansError(good,prefixes,8,8,100).empty());
  assert(!SpecSpansError(good,prefixes,7,8,100).empty());
  assert(!SpecSpansError({},prefixes,8,8,100).empty());
  assert(!SpecSpansError({{0,{1},0,1},{0,{2},0,1}},prefixes,8,8,100).empty());
  assert(!SpecSpansError({{-1,{1},0,1}},prefixes,8,8,100).empty());
  assert(!SpecSpansError({{2,{1},0,1}},prefixes,8,8,100).empty());
  assert(!SpecSpansError({{0,{},0,1}},prefixes,8,8,100).empty());
  assert(!SpecSpansError({{0,{100},0,1}},prefixes,8,8,100).empty());
  assert(!SpecSpansError({{0,{-1},0,1}},prefixes,8,8,100).empty());
  assert(!SpecSpansError({{1,{1},1,1}},prefixes,8,8,100).empty());
  assert(!SpecSpansError({{0,{1},0,0}},prefixes,8,8,100).empty());
  assert(!SpecSpansError({{0,{1},0,2}},prefixes,8,8,100).empty());
  assert(!SpecSpansError({{0,std::vector<int>(9,1),0,1}},prefixes,16,8,100).empty());
  assert(SpecSpansError(good,{SpecPrefix{63,64,first.retained},pending},8,8,100).size()>0);
  // Validation must not mutate the committed or pending prefix.
  auto unchanged=first.retained;
  assert(unchanged.Forward(1,0,1).empty());
  auto same_pending=pending.retained;
  assert(same_pending.Forward(1,2,1).empty());
}
