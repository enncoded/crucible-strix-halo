// Probe: which lanes of the A and B operands does gfx1151 WMMA read, and is a
// consistent k-half swap in lanes 16-31 bitwise neutral?
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstring>
#include <random>
typedef _Float16 H16 __attribute__((ext_vector_type(16)));
typedef float F8 __attribute__((ext_vector_type(8)));
__global__ void K(const _Float16* A, const _Float16* B, float* out, int mode) {
  const int lane = threadIdx.x, r = lane & 15;
  H16 a, b;
  for (int k = 0; k < 16; ++k) { a[k] = A[r * 16 + k]; b[k] = B[r * 16 + k]; }
  if (lane >= 16) {
    if (mode == 1) a = H16{};
    if (mode == 2) b = H16{};
    if (mode == 3) for (int k = 0; k < 8; ++k) { _Float16 t = a[k]; a[k] = a[k + 8]; a[k + 8] = t; t = b[k]; b[k] = b[k + 8]; b[k + 8] = t; }
  }
  if (lane < 16 && mode == 4) a = H16{};
  if (lane < 16 && mode == 5) b = H16{};
  F8 c = {};
  for (int i = 0; i < 8; ++i) c[i] = out[1024 + lane * 8 + i];
  c = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
  for (int i = 0; i < 8; ++i) out[lane * 8 + i] = c[i];
}
int main() {
  std::mt19937 g(5); std::normal_distribution<float> n(0, 1);
  _Float16 hA[256], hB[256]; float hc[2048];
  for (int i = 0; i < 256; ++i) { hA[i] = (_Float16)(n(g) * (i % 7 ? 1.f : 300.f)); hB[i] = (_Float16)(n(g) * (i % 5 ? 1.f : 0.001f)); }
  for (int i = 0; i < 2048; ++i) hc[i] = n(g) * 50;
  _Float16 *A, *B; float* o;
  hipMalloc(&A, 512); hipMalloc(&B, 512); hipMalloc(&o, 8192);
  hipMemcpy(A, hA, 512, hipMemcpyHostToDevice); hipMemcpy(B, hB, 512, hipMemcpyHostToDevice);
  float ref[256];
  for (int mode = 0; mode < 6; ++mode) {
    hipMemcpy(o, hc, 8192, hipMemcpyHostToDevice);
    K<<<1, 32>>>(A, B, o, mode);
    float r[256]; hipMemcpy(r, o, 1024, hipMemcpyDeviceToHost);
    if (mode == 0) std::memcpy(ref, r, 1024);
    int lo = 0, hi = 0;
    for (int l = 0; l < 32; ++l) for (int i = 0; i < 8; ++i) if (std::memcmp(&r[l * 8 + i], &ref[l * 8 + i], 4)) (l < 16 ? lo : hi)++;
    std::printf("mode %d: differing outputs lanes0-15 %d lanes16-31 %d\n", mode, lo, hi);
  }
}
