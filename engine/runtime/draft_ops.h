// Draft-only sampling. The trunk always uses its original full-vocabulary head.
#pragma once

#include "../gfx1151/common.h"

namespace crucible {

struct DraftChoice {
  int token;
  float margin;
};

__global__ void __launch_bounds__(1024) DraftChoiceKernel(const float* logits, int V,
                                                          const int* blocks,
                                                          DraftChoice* out, bool individual = false) {
  __shared__ float best[32], second[32];
  __shared__ int ids[32];
  const int t = threadIdx.x, lane = t & 31, wave = t >> 5;
  float b = -INFINITY, s = -INFINITY;
  int id = 0;
  for (int i = t; i < V; i += 1024) {
    const float v = logits[i];
    const int token = blocks ? (individual ? blocks[i] : blocks[i / 128] * 128 + i % 128) : i;
    if (v > b || (v == b && token < id)) {
      s = b;
      b = v;
      id = token;
    } else {
      s = fmaxf(s, v);
    }
  }
#pragma unroll
  for (int m = 1; m < 32; m <<= 1) {
    const float ob = __shfl_xor(b, m), os = __shfl_xor(s, m);
    const int oi = __shfl_xor(id, m);
    if (ob > b || (ob == b && oi < id)) {
      s = fmaxf(b, os);
      b = ob;
      id = oi;
    } else {
      s = fmaxf(s, ob);
    }
  }
  if (lane == 0) {
    best[wave] = b;
    second[wave] = s;
    ids[wave] = id;
  }
  __syncthreads();
  if (t == 0) {
    b = best[0];
    s = second[0];
    id = ids[0];
    for (int w = 1; w < 32; ++w) {
      if (best[w] > b || (best[w] == b && ids[w] < id)) {
        s = fmaxf(b, second[w]);
        b = best[w];
        id = ids[w];
      } else {
        s = fmaxf(s, best[w]);
      }
    }
    *out = {id, b - s};
  }
}

}  // namespace crucible
