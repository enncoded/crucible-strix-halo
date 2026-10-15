// Probe: mul1 pair decode with v_sad_hi_u8 instead of two v_sad_u8 and a
// v_perm_b32. Checks bitwise equality over all 65536 states (each paired
// with 64 partner states) and compares the issue rate of v_sad_hi_u8 with
// v_sad_u8.
#include <hip/hip_fp16.h>

#include "../common.h"

namespace {

__device__ __forceinline__ uint32_t Mul1(uint32_t s) { return (s & 0xffffu) * 0x83DCD12Du; }

__device__ __forceinline__ uint32_t PairOld(uint32_t p0, uint32_t p1) {
  uint32_t s0 = __builtin_amdgcn_sad_u8(p0, 0u, 0x6400u);
  uint32_t s1 = __builtin_amdgcn_sad_u8(p1, 0u, 0x6400u);
  uint32_t packed = __builtin_amdgcn_perm(s1, s0, 0x05040100u);
  __half2 h = *reinterpret_cast<__half2*>(&packed);
  const __half2 inv = __halves2half2(__ushort_as_half(0x1eee), __ushort_as_half(0x1eee));
  const __half2 bias = __halves2half2(__ushort_as_half(0xc931), __ushort_as_half(0xc931));
  h = __hfma2(h, inv, bias);
  return *reinterpret_cast<uint32_t*>(&h);
}

__device__ __forceinline__ uint32_t PairNew(uint32_t p0, uint32_t p1) {
  uint32_t s0 = __builtin_amdgcn_sad_u8(p0, 0u, 0x64006400u);
  uint32_t packed = __builtin_amdgcn_sad_hi_u8(p1, 0u, s0);
  __half2 h = *reinterpret_cast<__half2*>(&packed);
  const __half2 inv = __halves2half2(__ushort_as_half(0x1eee), __ushort_as_half(0x1eee));
  const __half2 bias = __halves2half2(__ushort_as_half(0xc931), __ushort_as_half(0xc931));
  h = __hfma2(h, inv, bias);
  return *reinterpret_cast<uint32_t*>(&h);
}

__global__ void Check(uint32_t* bad) {
  const uint32_t s0 = blockIdx.x * blockDim.x + threadIdx.x;  // 0..65535
  uint32_t n = 0;
  for (uint32_t i = 0; i < 64; ++i) {
    const uint32_t s1 = (s0 * 40503u + i * 977u) & 0xffffu;
    n += PairOld(Mul1(s0), Mul1(s1)) != PairNew(Mul1(s0), Mul1(s1));
  }
  if (n) atomicAdd(bad, n);
}

template <int HI>
__global__ void Rate(uint32_t seed, int iters, uint32_t* out) {
  uint32_t a[8];
#pragma unroll
  for (int j = 0; j < 8; ++j) a[j] = seed + threadIdx.x * 8 + j;
  for (int i = 0; i < iters; ++i) {
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      if constexpr (HI) a[j] = __builtin_amdgcn_sad_hi_u8(a[j], 0x01020304u, a[j]);
      else a[j] = __builtin_amdgcn_sad_u8(a[j], 0x01020304u, a[j]);
    }
  }
  uint32_t r = 0;
#pragma unroll
  for (int j = 0; j < 8; ++j) r ^= a[j];
  if (r == 0x9e3779b9u) out[0] = r;
}

}  // namespace

int main() {
  uint32_t* d;
  HIP_CHECK(hipMalloc(&d, 64));
  HIP_CHECK(hipMemset(d, 0, 64));
  hipLaunchKernelGGL(Check, 256, 256, 0, nullptr, d);
  uint32_t bad = 0;
  HIP_CHECK(hipMemcpy(&bad, d, 4, hipMemcpyDeviceToHost));
  std::printf("pairs checked %u, mismatches %u\n", 65536u * 64u, bad);
  const int iters = 1 << 16, blocks = 40 * 16;
  for (int rep = 0; rep < 2; ++rep)
    for (int hi = 0; hi < 2; ++hi) {
      bench::EventTimer t;
      t.Begin();
      if (hi) hipLaunchKernelGGL(Rate<1>, blocks, 256, 0, nullptr, 7u, iters, d);
      else hipLaunchKernelGGL(Rate<0>, blocks, 256, 0, nullptr, 7u, iters, d);
      const float ms = t.EndMs();
      const double ops = double(blocks) * 256 / 32 * iters * 8;  // wave instructions
      std::printf("%s: %.3f ms, %.2f Gwave-instr/s (%.3f per SIMD-ns)\n", hi ? "v_sad_hi_u8" : "v_sad_u8   ",
                  ms, ops / ms / 1e6, ops / ms / 1e6 / 80);
    }
  return 0;
}
