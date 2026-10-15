#include "../bench/moe_partial_check.h"
#include <cassert>
#include <vector>

using namespace crucible_bench;
float Word(uint32_t value) { return std::bit_cast<float>(value); }
int main() {
  const std::vector<int> rows{1,0,2};
  const std::vector<float> finite{1,2,3,4,5,6};
  assert(PartialCovered<false>(finite,rows,2,PartialPoison<false,false>()));
  assert(PartialMatches<false>(finite,finite,rows,2));
  auto missing=finite;missing[2]=Word(PartialPoison<false,false>());
  auto other_missing=finite;other_missing[2]=Word(PartialPoison<false,true>());
  assert(!PartialMatches<false>(missing,other_missing,rows,2));
  assert(!PartialCovered<false>(finite,rows,0,0));
  assert(!PartialCovered<false>(finite,std::vector<int>{-1,4},2,0));
  assert(!PartialCovered<false>(finite,std::vector<int>{4},2,0));
  assert(!PartialCovered<false>(finite,std::vector<int>{2},2,0));
  assert(!PartialCovered<false>({},std::vector<int>{0},2,0));
  const auto rp=PartialPoison<true,false>(),cp=PartialPoison<true,true>();
  const std::vector<float> ref{Word(0x40003c00u),Word(rp),Word(0x40003c00u),
                             Word(0x44004200u),Word(rp),Word(rp)};
  auto got=ref;got[1]=Word(cp);got[4]=Word(cp);got[5]=Word(cp);
  const auto before=got;
  assert(PartialMatches<true>(ref,got,rows,2));
  assert(std::memcmp(got.data(),before.data(),got.size()*4)==0);
  auto bad=got;bad[1]=0;
  assert(!PartialMatches<true>(ref,bad,rows,2));
  bad=got;bad[2]=Word(cp);
  assert(!PartialMatches<true>(ref,bad,rows,2));
  bad=got;bad[2]=Word(0x40003c01u);
  assert(!PartialMatches<true>(ref,bad,rows,2));
  auto both=ref;both[2]=Word(rp);
  assert(!PartialMatches<true>(both,bad,rows,2));
}
