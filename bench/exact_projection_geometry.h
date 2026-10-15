// Geometry stays independent of HIP so the row and split mappings have a CPU proof.
#pragma once
#include <cstddef>
#include <initializer_list>
#ifdef __HIPCC__
#define WIDE_HD __host__ __device__
#else
#define WIDE_HD
#endif
namespace crucible::wide_probe {
WIDE_HD constexpr size_t StripePartialBase(int split,int rows,int stripe,int columns) {
  return (size_t(split)*(rows-8)+stripe)*columns;
}
WIDE_HD constexpr size_t RollingSlot(int row,int tile,int word) {
  return (size_t(row)*8+tile)*8+word;
}
WIDE_HD constexpr int InterleavedItem(int block,int rows) { return block/(rows/8); }
WIDE_HD constexpr int InterleavedRow(int block,int rows) { return block%(rows/8)*8; }
WIDE_HD constexpr bool FullWaves(int columns) { return columns>0 && columns%128==0; }
WIDE_HD constexpr bool CompleteChunks(int span) { return span>0 && span%8==0; }
WIDE_HD constexpr size_t RollingLdsBytes(int rows) { return size_t(rows)*256; }
} // namespace crucible::wide_probe
#undef WIDE_HD
