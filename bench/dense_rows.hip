// DenseGemvKernel vs DenseRowsKernel on cold fp16 weights, R=1..8.
//
//   dense_rows [O] [mats]
//
// Checks every row of every variant bit for bit against DenseGemvKernel<1>
// on that row alone (the serial path), then times back-to-back launches that
// cycle through `mats` weight matrices so each launch reads cold weights.
#include <chrono>
#include <cstring>
#include <vector>

#include "../engine/layers/dense_rows.h"
#include "../engine/exl3/plans.h"

using namespace crucible;

namespace {

__global__ void Fill(__half* p, size_t n, uint32_t seed) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    uint32_t z = (uint32_t)(i * 2654435761u) ^ seed;
    z ^= z >> 13;
    z *= 0x5bd1e995u;
    z ^= z >> 15;
    p[i] = __float2half(((z & 0xffff) / 65536.f - 0.5f) * 0.1f);
  }
}

__global__ void FillF(float* p, size_t n, uint32_t seed) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    uint32_t z = (uint32_t)(i * 2246822519u) ^ seed;
    z ^= z >> 15;
    z *= 0x2c1b3c6du;
    z ^= z >> 12;
    p[i] = ((z & 0xffffff) / 16777216.f - 0.5f) * 4.f;
  }
}

using Launch = void (*)(const __half*, int, const float*, int, float*, int);

template <int R>
void Old(const __half* W, int O, const float* x, int ldx, float* y, int ldy) {
  DenseGemvKernel<R><<<(O + 7) / 8, 256>>>(W, O, x, ldx, y, ldy);
}

template <int R, int E>
void New(const __half* W, int O, const float* x, int ldx, float* y, int ldy) {
  LaunchDenseRows<R, E>(W, O, x, ldx, y, ldy);
}

template <int R>
std::vector<Launch> Variants() {
  return {Old<R>, New<R, 1>, New<R, 2>, New<R, 4>};
}

double Now() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

}  // namespace

int main(int argc, char** argv) {
  const int O = argc > 1 ? std::atoi(argv[1]) : 513;
  const int mats = argc > 2 ? std::atoi(argv[2]) : 48;
  const int ldx = kHidden, ldy = O + 7;
  __half* W = Zeros<__half>((size_t)mats * O * kHidden);
  Fill<<<1024, 256>>>(W, (size_t)mats * O * kHidden, 17);
  float* x = Zeros<float>(kMaxRows * ldx);
  FillF<<<64, 256>>>(x, kMaxRows * ldx, 23);
  float* y = Zeros<float>(kMaxRows * ldy);
  HIP_OK(hipDeviceSynchronize());
  // Serial reference: each row alone through DenseGemvKernel<1>.
  std::vector<float> ref((size_t)kMaxRows * O), got((size_t)kMaxRows * ldy);
  for (int r = 0; r < kMaxRows; ++r) {
    Old<1>(W, O, x + (size_t)r * ldx, ldx, y, ldy);
    HIP_OK(hipMemcpy(ref.data() + (size_t)r * O, y, O * 4, hipMemcpyDeviceToHost));
  }
  const double mb = (double)O * kHidden * 2 / 1e6;
  std::printf("O=%d, %.2f MB per matrix, %d matrices, floor %.1f us\n", O, mb, mats, mb / 245e3 * 1e6);
  const char* names[] = {"DenseGemv", "Rows<1>", "Rows<2>", "Rows<4>"};
  for (int R = 1; R <= kMaxRows; ++R) {
    std::vector<Launch> vs;
    switch (R) {
      case 1: vs = Variants<1>(); break;
      case 2: vs = Variants<2>(); break;
      case 3: vs = Variants<3>(); break;
      case 4: vs = Variants<4>(); break;
      case 5: vs = Variants<5>(); break;
      case 6: vs = Variants<6>(); break;
      case 7: vs = Variants<7>(); break;
      default: vs = Variants<8>(); break;
    }
    std::printf("R=%d:", R);
    for (size_t v = 0; v < vs.size(); ++v) {
      HIP_OK(hipMemset(y, 0xff, kMaxRows * ldy * 4));
      vs[v](W, O, x, ldx, y, ldy);
      HIP_OK(hipMemcpy(got.data(), y, got.size() * 4, hipMemcpyDeviceToHost));
      bool same = true;
      for (int r = 0; r < R; ++r)
        same = same && std::memcmp(&got[(size_t)r * ldy], &ref[(size_t)r * O], O * 4) == 0;
      for (int i = 0; i < mats; ++i) vs[v](W + (size_t)i * O * kHidden, O, x, ldx, y, ldy);
      HIP_OK(hipDeviceSynchronize());
      const int n = 4 * mats;
      const double t0 = Now();
      for (int i = 0; i < n; ++i) vs[v](W + (size_t)(i % mats) * O * kHidden, O, x, ldx, y, ldy);
      HIP_OK(hipDeviceSynchronize());
      const double us = (Now() - t0) / n * 1e6;
      std::printf("  %s %.1f us%s", names[v], us, same ? "" : " (DIFFERENT)");
    }
    std::printf("\n");
  }
  return 0;
}
