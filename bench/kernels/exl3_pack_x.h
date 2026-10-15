// Reorder fp16 values without rounding them. Contiguous row fragments avoid
// the large row strides in row-major WMMA operand loads.
#pragma once
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

namespace crucible {
// Launch (I/16, ceil(rows/32)) blocks of 256 threads. No row padding is needed.
__global__ void PackX16Kernel(const __half* src, __half* dst, int rows, int I) {
  const int row = blockIdx.y * 32 + (threadIdx.x >> 3);
  if (row >= rows) return;
  const int a = blockIdx.x, q = threadIdx.x & 7;
  const size_t from = (size_t)row * (I / 2) + a * 8 + q;
  const size_t to = ((size_t)a * rows + row) * 8 + q;
  reinterpret_cast<uint32_t*>(dst)[to] = reinterpret_cast<const uint32_t*>(src)[from];
}
struct PackXJob {
  const __half* src;
  __half* dst;
  const int* row_ids;
  int I, rows;
};
struct PackXItem { int job, a, row0; };
// Compact each route list before packing. Each expert then has a small input
// working set rather than sparse references into the entire prompt pool.
__global__ void PackCompactX16Kernel(const PackXJob* jobs, const PackXItem* items, int n) {
  if (blockIdx.x >= n) return;
  const PackXItem it = items[blockIdx.x];
  const PackXJob j = jobs[it.job];
  const int row = it.row0 + (threadIdx.x >> 3), q = threadIdx.x & 7;
  if (row >= j.rows) return;
  const int source = j.row_ids ? j.row_ids[row] : row;
  const size_t from = (size_t)source * (j.I / 2) + it.a * 8 + q;
  const size_t to = ((size_t)it.a * j.rows + row) * 8 + q;
  reinterpret_cast<uint32_t*>(j.dst)[to] = reinterpret_cast<const uint32_t*>(j.src)[from];
}
}  // namespace crucible
