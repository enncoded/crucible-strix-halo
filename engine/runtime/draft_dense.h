// Draft-only fp16 weights. Fold the output Hadamard into selected weight rows;
// keep the original activation-side input transform and scales.
#pragma once

#include <vector>

#include "../exl3/exl3.h"
#include "../exl3/front_order.h"

namespace crucible {

__global__ void __launch_bounds__(256) DraftDensePackKernel(
    const uint32_t* tiles, const __half* svh, const int* token_map, const int* blocks,
    int I, __half* dense, int O = 0, int front_s = 0) {
  __shared__ __half tile[16][128];
  const int lane = threadIdx.x & 31, wave = threadIdx.x >> 5;
  const int a = blockIdx.x, group = blocks[blockIdx.y], b = group * 8 + wave;
  const LaneWindow<5> win(lane);
  const uint32_t* src =
      tiles + (front_s ? FrontTile(b, a, I / 16, O / 16, front_s) : (size_t)b * (I / 16) + a) * 40;
  __half2 p[4];
  DecodeLane<5, kMulMode>(src[win.d0], src[(win.d0 + 1) % 40], src[(win.d0 + 2) % 40], win.shift, p);
  const int r = 2 * (lane & 3), c = wave * 16 + lane / 4;
  tile[r][c] = __low2half(p[0]);
  tile[r + 1][c] = __high2half(p[0]);
  tile[r + 8][c] = __low2half(p[1]);
  tile[r + 9][c] = __high2half(p[1]);
  tile[r][c + 8] = __low2half(p[2]);
  tile[r + 1][c + 8] = __high2half(p[2]);
  tile[r + 8][c + 8] = __low2half(p[3]);
  tile[r + 9][c + 8] = __high2half(p[3]);
  __syncthreads();
#pragma unroll
  for (int half = 0; half < 2; ++half) {
    const int row = wave + half * 8;
    float v[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) v[j] = __half2float(tile[row][lane * 4 + j]);
    Wht128(v);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int token = group * 128 + lane * 4 + j;
      const int output = token_map[token];
      if (output >= 0)
        dense[(size_t)output * I + a * 16 + row] = __float2half(v[j] * kHadNorm * __half2float(svh[token]));
    }
  }
}

__global__ void __launch_bounds__(256) DraftDenseGemvKernel(
    const __half* weights, const __half* x, float* logits, int V) {
  const int o = blockIdx.x * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
  if (o >= V) return;
  uint4 raw[10];
#pragma unroll
  for (int k = 0; k < 10; ++k)
    raw[k] = LoadNT(weights + (size_t)o * kHidden + k * 256 + lane * 8);
  float sum = 0.f;
#pragma unroll
  for (int k = 0; k < 10; ++k) {
    const __half2* w = reinterpret_cast<const __half2*>(&raw[k]);
    const __half2* xx = reinterpret_cast<const __half2*>(x + k * 256 + lane * 8);
#pragma unroll
    for (int j = 0; j < 4; ++j) sum = Dot2(w[j], xx[j], sum);
  }
  sum = WaveSum(sum);
  if (lane == 0) logits[o] = sum;
}

// Called once at load time. Only proposals use this rounded weight matrix.
template <typename Linear>
inline __half* PackDraftDense(const Linear& head, const std::vector<int>& tokens) {
  if (head.I != kHidden || head.K != 5) {
    std::fprintf(stderr, "fp16 draft packing requires a 2560-input, 5-bit EXL3 head\n");
    std::exit(2);
  }
  std::vector<int> mapping(kVocab, -1), blocks;
  for (int i = 0; i < (int)tokens.size(); ++i) {
    mapping[tokens[i]] = i;
    const int block = tokens[i] / 128;
    if (blocks.empty() || blocks.back() != block) blocks.push_back(block);
  }
  int *dm, *db;
  __half* weights;
  HIP_OK(hipMalloc(&dm, mapping.size() * sizeof(int)));
  HIP_OK(hipMalloc(&db, blocks.size() * sizeof(int)));
  HIP_OK(hipMalloc(&weights, tokens.size() * kHidden * sizeof(__half)));
  HIP_OK(hipMemcpy(dm, mapping.data(), mapping.size() * sizeof(int), hipMemcpyHostToDevice));
  HIP_OK(hipMemcpy(db, blocks.data(), blocks.size() * sizeof(int), hipMemcpyHostToDevice));
  DraftDensePackKernel<<<dim3(kHidden / 16, blocks.size()), 256>>>(head.tiles, head.svh, dm, db, kHidden, weights,
                                                                   head.O, head.front_s);
  HIP_OK(hipStreamSynchronize(0));
  HIP_OK(hipFree(dm));
  HIP_OK(hipFree(db));
  return weights;
}

}  // namespace crucible
