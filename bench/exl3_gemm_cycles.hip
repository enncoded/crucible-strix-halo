// Isolated hot-stage probes. These exclude activation loads and cold DRAM.
#include "common.h"
#include "../engine/exl3/kernels/exl3_gemm.h"
using namespace crucible;

template <int K, bool LDS>
__global__ void DecodeStage(int iters, uint32_t* out) {
  __shared__ uint32_t ws[LDS ? kGemmWeightWords : 1];
  const int lane = threadIdx.x & 31;
  const LaneWindow<K> win(lane);
  uint32_t x = lane * 0x9e3779b9u;
  for (int i = 0; i < iters; ++i) {
    x += 0x9e3779b9u;
    __half2 p[4];
    DecodeLane<K, kMulTrue16>(x, x ^ 0xa387e58au, x + 0x816ec9cbu, win.shift, p);
    GemmH16 b = GemmBFragment<LDS>(p, LDS ? ws + (threadIdx.x >> 5) * kGemmWaveWords : ws, lane);
    auto* v = reinterpret_cast<uint32_t*>(&b);
    // An empty volatile asm keeps every fragment live without adding a hash
    // or stores to the measured loop.
    asm volatile("" : : "v"(v[0]), "v"(v[1]), "v"(v[2]), "v"(v[3]),
                         "v"(v[4]), "v"(v[5]), "v"(v[6]), "v"(v[7]));
  }
  out[blockIdx.x * 128 + threadIdx.x] = x;
}

template <int RM>
__global__ void WmmaStage(int iters, float* out) {
  GemmH16 a, b;
  GemmF8 c[RM];
  const int lane = threadIdx.x & 15;
#pragma unroll
  for (int k = 0; k < 16; ++k) {
    b[k] = (_Float16)((lane + k) % 7) * (_Float16)0.01;
    a[k] = (_Float16)((lane + k) % 11) * (_Float16)0.01;
  }
#pragma unroll
  for (int r = 0; r < RM; ++r)
#pragma unroll
    for (int z = 0; z < 8; ++z) c[r][z] = (float)(r + 1) * 0.001f;
  for (int i = 0; i < iters; ++i)
#pragma unroll
    for (int r = 0; r < RM; ++r)
      c[r] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c[r]);
  float sum = 0;
#pragma unroll
  for (int r = 0; r < RM; ++r) sum += c[r][0];
  out[blockIdx.x * 128 + threadIdx.x] = sum;
}

template <typename Fn, typename T>
static void Point(const char* label, Fn fn, T* out) {
  constexpr int blocks = 160, probe_iters = 32768;
  bench::Sampler smp(2);
  bench::EventTimer timer;
  timer.Begin();
  hipLaunchKernelGGL(fn, dim3(blocks), dim3(128), 0, nullptr, probe_iters, out);
  const double probe_ms = timer.EndMs();
  const int iters = (int)std::min(1e9, probe_iters * 1000.0 / std::max(probe_ms, 0.01));
  // hwmon averages lag short kernels. Warm for 0.5 s and time about 1 s so
  // sampled sclk is usable while keeping the entire command below 20 s.
  hipLaunchKernelGGL(fn, dim3(blocks), dim3(128), 0, nullptr, iters / 2, out);
  HIP_CHECK(hipDeviceSynchronize());
  smp.Start(); timer.Begin();
  hipLaunchKernelGGL(fn, dim3(blocks), dim3(128), 0, nullptr, iters, out);
  const double ms = timer.EndMs(); smp.Stop();
  const double wave_steps = (double)blocks * 4 * iters;
  const double cycles = ms * smp.MeanMhz() * 1000 * 80 / wave_steps;
  std::printf("%-20s %9.4f ms %8.2f ns/SIMD-step %8.1f cycles/SIMD-step [%s]\n",
              label, ms, ms * 1e6 * 80 / wave_steps, cycles, smp.Summary().c_str());
}

int main() {
  uint32_t* out;
  HIP_CHECK(hipMalloc(&out, 160 * 128 * 4));
  std::puts("# 80 SIMD32s; cycles estimated from sampled sclk, not a hardware performance counter");
  Point("WMMA warmup", WmmaStage<8>, reinterpret_cast<float*>(out));
  Point("decode K3 registers", DecodeStage<3, false>, out);
  Point("decode K3 LDS", DecodeStage<3, true>, out);
  Point("decode K5 registers", DecodeStage<5, false>, out);
  Point("decode K5 LDS", DecodeStage<5, true>, out);
  Point("WMMA RM1", WmmaStage<1>, reinterpret_cast<float*>(out));
  Point("WMMA RM4", WmmaStage<4>, reinterpret_cast<float*>(out));
  Point("WMMA RM8", WmmaStage<8>, reinterpret_cast<float*>(out));
  Point("WMMA RM16", WmmaStage<16>, reinterpret_cast<float*>(out));
  HIP_CHECK(hipFree(out));
  return 0;
}
