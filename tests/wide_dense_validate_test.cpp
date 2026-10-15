#include <array>
#include <cassert>
#include <stdexcept>
#include "../bench/exact_dense_validate.h"
using namespace crucible::wide_probe;
int main() {
  constexpr uint32_t a=0x7fc00001,b=0x7fc00002,v=0x3f800000;
  const std::array<uint32_t,4> expected{v,a,v,a},actual{v,b,v,b};
  assert(DenseMismatch(expected,actual,2,1,2,a,b)==0);
  assert(DenseMismatch(std::array<uint32_t,4>{a,a,v,a},std::array<uint32_t,4>{a,b,v,b},2,1,2,a,b)>0);
  assert(DenseMismatch(std::array<uint32_t,4>{0x7f800000,a,v,a},std::array<uint32_t,4>{0x7f800000,b,v,b},2,1,2,a,b)>0);
  assert(DenseMismatch(std::array<uint32_t,4>{a,a,v,a},actual,2,1,2,a,b)>0);
  assert(DenseMismatch(expected,std::array<uint32_t,4>{b,b,v,b},2,1,2,a,b)>0);
  assert(DenseMismatch(std::array<uint32_t,4>{a,a,v,a},std::array<uint32_t,4>{b,b,v,b},2,1,2,a,b)>0);
  assert(DenseMismatch(expected,std::array<uint32_t,4>{0x7f800000,b,v,b},2,1,2,a,b)>0);
  assert(DenseMismatch(expected,std::array<uint32_t,4>{v,a,v,b},2,1,2,a,b)>0);
  assert(DenseMismatch(expected,std::array<uint32_t,4>{0x40000000,b,v,b},2,1,2,a,b)>0);
  bool rejected=false;
  try { DenseMismatch(expected,actual,2,3,2,a,b); } catch(const std::invalid_argument&) { rejected=true; }
  assert(rejected);
}
