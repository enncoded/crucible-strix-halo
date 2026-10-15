// WMMA issue model on gfx1151: f16 16x16x16 rate by independent accumulators
// per wave, waves per SIMD, and operands from registers or from LDS.
// Usage: wmma_rate (prints one line per configuration).
#include <hip/hip_runtime.h>
#include <cstdio>

typedef _Float16 H16 __attribute__((ext_vector_type(16)));
typedef float F8 __attribute__((ext_vector_type(8)));

// MODE 0: operands in registers. MODE 1: per k step a wave reads TA A and TB
// B fragments from LDS (two 16-byte reads per lane each) and runs TA x TB
// WMMAs, the shape of a TA*16 x TB*16 register tile.
template <int TA, int TB, int MODE>
__global__ void __launch_bounds__(256) Kern(float* out, int iters) {
  __shared__ __attribute__((aligned(16))) _Float16 lds[2][8][16][24];
  const int lane = threadIdx.x & 31, wave = threadIdx.x >> 5;
  for (int i = threadIdx.x; i < 2 * 8 * 16 * 24; i += blockDim.x)
    (&lds[0][0][0][0])[i] = (_Float16)((i % 7) * 0.01f);
  __syncthreads();
  H16 a[TA], b[TB];
  for (int i = 0; i < TA; ++i)
    for (int k = 0; k < 16; ++k) a[i][k] = (_Float16)((lane + i + k) * 0.001f);
  for (int j = 0; j < TB; ++j)
    for (int k = 0; k < 16; ++k) b[j][k] = (_Float16)((lane * 3 + j + k) * 0.001f);
  F8 acc[TA][TB] = {};
  for (int it = 0; it < iters; ++it) {
    if constexpr (MODE == 1) {
#pragma unroll
      for (int i = 0; i < TA; ++i) a[i] = *reinterpret_cast<const H16*>(&lds[0][(wave + i) & 7][lane & 15][(it & 1) * 8 & 0]);
#pragma unroll
      for (int j = 0; j < TB; ++j) b[j] = *reinterpret_cast<const H16*>(&lds[1][(j + it) & 7][lane & 15][0]);
    }
#pragma unroll
    for (int i = 0; i < TA; ++i)
#pragma unroll
      for (int j = 0; j < TB; ++j) acc[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a[i], b[j], acc[i][j]);
  }
  float s = 0;
  for (int i = 0; i < TA; ++i)
    for (int j = 0; j < TB; ++j)
      for (int z = 0; z < 8; ++z) s += acc[i][j][z];
  if (s == 12345.f) out[blockIdx.x] = s;
}

template <int TA, int TB, int MODE>
void Run(float* out, int waves_per_block, int blocks) {
  const int iters = 4096;
  hipEvent_t e0, e1;
  hipEventCreate(&e0); hipEventCreate(&e1);
  Kern<TA, TB, MODE><<<blocks, 32 * waves_per_block>>>(out, iters);
  hipEventRecord(e0);
  for (int r = 0; r < 5; ++r) Kern<TA, TB, MODE><<<blocks, 32 * waves_per_block>>>(out, iters);
  hipEventRecord(e1);
  hipEventSynchronize(e1);
  float ms; hipEventElapsedTime(&ms, e0, e1);
  const double flop = 5.0 * blocks * waves_per_block * (double)iters * TA * TB * 16 * 16 * 16 * 2;
  std::printf("mode %d tile %dx%d waves/block %d blocks %d: %.1f TFLOPS\n", MODE, TA, TB, waves_per_block,
              blocks, flop / (ms * 1e-3) / 1e12);
}

int main() {
  float* out;
  (void)hipMalloc(&out, 1 << 20);
  // 40 CUs x 2 SIMD32: (waves per block, blocks) = (2,40) (4,40) (8,40) (8,80)
  // (8,160) give 1, 2, 4, 8, 16 waves per SIMD when the registers allow.
  const int cfg[5][2] = {{2, 40}, {4, 40}, {8, 40}, {8, 80}, {8, 160}};
  for (const auto& c : cfg) {
    Run<1, 1, 0>(out, c[0], c[1]);
    Run<1, 2, 0>(out, c[0], c[1]);
    Run<2, 2, 0>(out, c[0], c[1]);
    Run<2, 4, 0>(out, c[0], c[1]);
    Run<4, 4, 0>(out, c[0], c[1]);
    Run<2, 4, 1>(out, c[0], c[1]);
    Run<4, 4, 1>(out, c[0], c[1]);
  }
  return 0;
}
