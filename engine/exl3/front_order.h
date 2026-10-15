// Front tile order for dense decode matrices. Chunk (q, s * O/128 + g) holds
// column blocks 8g..8g+7 and row tiles 4q..4q+3 of split s, column-major
// inside the chunk. Concurrent decode blocks read chunk q of every item side
// by side, where the engine order gives eight short runs at a column stride.
// Every column keeps its tile order, so the matvec results stay bitwise equal.
#pragma once
#include <cstddef>
#include <cstdint>
#include <hip/hip_runtime.h>

namespace crucible {

constexpr int kFrontCols = 8, kFrontTiles = 4;

// Tile index of column block b, row tile a, for A row tiles, B column blocks, split S.
__host__ __device__ inline size_t FrontTile(int b, int a, int A, int B, int S) {
  const int span = A / S, items = S * (B / kFrontCols), s = a / span, a2 = a % span;
  return ((size_t)(a2 / kFrontTiles) * items + (size_t)s * (B / kFrontCols) + b / kFrontCols) *
             (kFrontCols * kFrontTiles) +
         (b % kFrontCols) * kFrontTiles + a2 % kFrontTiles;
}

// A front order applies when every split span and the column count divide.
inline bool FrontFitsShape(int A, int B, int S) {
  return S > 0 && A % S == 0 && (A / S) % kFrontTiles == 0 && B % kFrontCols == 0;
}

// Engine-order copy of the 128 columns of block g (column blocks 8g..8g+7).
__global__ void FrontColumnsKernel(const uint32_t* __restrict__ src, uint32_t* __restrict__ dst,
                                   int A, int B, int T, int S, int g) {
  const size_t words = (size_t)kFrontCols * A * T;
  for (size_t w = (size_t)blockIdx.x * blockDim.x + threadIdx.x; w < words;
       w += (size_t)gridDim.x * blockDim.x) {
    const size_t tile = w / T;
    const int c = (int)(tile / A), a = (int)(tile % A);
    dst[w] = src[FrontTile(g * kFrontCols + c, a, A, B, S) * T + w % T];
  }
}

}  // namespace crucible
