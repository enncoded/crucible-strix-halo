// A shared missed live cell must not pass a byte comparison.
#pragma once
#include <cstdint>
#include <span>
#include <stdexcept>
namespace crucible::wide_probe {
// Do not let finite-math compiler flags remove the missed-write check.
constexpr bool FiniteWord(uint32_t bits) { return (bits&0x7f800000u)!=0x7f800000u; }
inline size_t DenseMismatch(std::span<const uint32_t> expected,std::span<const uint32_t> actual,
                            int rows,int columns,int stride,uint32_t reference_poison,uint32_t candidate_poison) {
  if (rows<=0 || columns<=0 || stride<columns || expected.size()!=size_t(rows)*stride || actual.size()!=expected.size())
    throw std::invalid_argument("invalid dense validation shape");
  size_t bad=0;
  for (int row=0;row<rows;++row) for (int column=0;column<stride;++column) {
    const size_t at=size_t(row)*stride+column;
    if (column<columns) bad+=!FiniteWord(expected[at]) || !FiniteWord(actual[at]) || expected[at]!=actual[at];
    else bad+=expected[at]!=reference_poison || actual[at]!=candidate_poison;
  }
  return bad;
}
} // namespace crucible::wide_probe
