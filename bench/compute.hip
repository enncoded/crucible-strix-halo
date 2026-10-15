// Sustained instruction throughput on gfx1151, in instructions per clock per
// SIMD (1.0 = full rate for one wave32 VALU op), plus WMMA TFLOPS/TOPS.
// Each op runs 8 independent chains per lane so latency is hidden; the
// emitted ISA is checked in build/isa/compute.s.
//
//   compute valu    VALU / DPP / dot instructions used by trellis decode
//   compute wmma    every gfx11 WMMA variant
#include "common.h"

typedef _Float16 h16 __attribute__((ext_vector_type(16)));
typedef float f8 __attribute__((ext_vector_type(8)));
typedef short s16 __attribute__((ext_vector_type(16)));
typedef int i8v __attribute__((ext_vector_type(8)));
typedef int i4v __attribute__((ext_vector_type(4)));
typedef int i2v __attribute__((ext_vector_type(2)));

constexpr int kChains = 8;
constexpr uint32_t kSentinel = 0x9e3779b9u;

#define OP_LIST(X)                                                        \
  X(0, "v_and_b32", "v_and_b32 %0, %1, %0")                              \
  X(1, "v_xor_b32", "v_xor_b32 %0, %1, %0")                              \
  X(2, "v_mul_lo_u32", "v_mul_lo_u32 %0, %1, %0")                        \
  X(3, "v_mul_u32_u24", "v_mul_u32_u24 %0, %1, %0")                      \
  X(4, "v_mad_u32_u24", "v_mad_u32_u24 %0, %1, %0, %2")                  \
  X(5, "v_lshl_add_u32", "v_lshl_add_u32 %0, %0, 3, %1")                 \
  X(6, "v_dot4_u32_u8", "v_dot4_u32_u8 %0, %1, %2, %0")                  \
  X(7, "v_sad_u8", "v_sad_u8 %0, %1, %2, %0")                            \
  X(8, "v_alignbit_b32", "v_alignbit_b32 %0, %1, %0, %2")                \
  X(9, "v_bfe_u32", "v_bfe_u32 %0, %0, %2, 16")                          \
  X(10, "v_perm_b32", "v_perm_b32 %0, %1, %0, %2")                       \
  X(11, "v_fmac_f32", "v_fmac_f32 %0, %1, %2")                           \
  X(12, "v_pk_fma_f16", "v_pk_fma_f16 %0, %1, %2, %0")                   \
  X(13, "v_dot2_f32_f16", "v_dot2_f32_f16 %0, %1, %2, %0")               \
  X(14, "v_pk_add_f16", "v_pk_add_f16 %0, %1, %0")                       \
  X(15, "v_permlanex16_b32", "v_permlanex16_b32 %0, %1, %3, %4")         \
  X(16, "v_mov_b32_dpp_row_shr", "v_mov_b32_dpp %0, %1 row_shr:1 row_mask:0xf bank_mask:0xf") \
  X(17, "v_cvt_f32_f16", "v_cvt_f32_f16 %0, %1")                        \
  X(18, "v_mad_u32_u16", "v_mad_u32_u16 %0, %1.l, %0.l, %2")              \
  X(19, "v_pk_mul_lo_u16", "v_pk_mul_lo_u16 %0, %1, %0")                 \
  X(20, "v_lshl_or_b32", "v_lshl_or_b32 %0, %0, 3, %1")                  \
  X(21, "v_bfi_b32", "v_bfi_b32 %0, %1, %0, %2")                         \
  X(22, "v_lshrrev_b32", "v_lshrrev_b32 %0, 3, %0")                      \
  X(23, "v_fma_f16", "v_fma_f16 %0.l, %1.l, %2.l, %0.l")                 \
  X(24, "v_and_or_b32", "v_and_or_b32 %0, %0, %1, %2")

template <int OP>
__global__ void ValuKernel(int iters, uint32_t seed, uint32_t* out) {
  uint32_t x[kChains];
  uint32_t a = seed * 2654435761u + threadIdx.x;
  uint32_t b = a ^ 0x5bd1e995u;
  uint32_t s0 = 0x76543210u, s1 = 0xfedcba98u;
#pragma unroll
  for (int c = 0; c < kChains; ++c) x[c] = a + c * 0x01000193u;
  for (int it = 0; it < iters; ++it) {
#pragma unroll
    for (int c = 0; c < kChains; ++c) {
#define EMIT(ID, NAME, ASM)                                                   \
  if constexpr (OP == ID)                                                     \
    asm volatile(ASM : "+v"(x[c]) : "v"(a), "v"(b), "s"(s0), "s"(s1));
      OP_LIST(EMIT)
#undef EMIT
    }
  }
  uint32_t acc = 0;
#pragma unroll
  for (int c = 0; c < kChains; ++c) acc ^= x[c];
  if (acc == kSentinel) out[0] = acc;
}

template <int OP>
static double RunValu(int cus, int waves_per_simd, double* mhz) {
  int threads = 32 * waves_per_simd * 4;  // 4 SIMDs per WGP
  bench::EventTimer t;
  uint32_t* out;
  HIP_CHECK(hipMalloc(&out, 64));
  int iters = 20000;
  t.Begin();
  hipLaunchKernelGGL(ValuKernel<OP>, cus * 4, threads, 0, nullptr, iters, 1u, out);
  double probe_ms = t.EndMs();
  iters = (int)std::min(2.0e9, iters * 1000.0 / std::max(probe_ms, 0.01));
  hipLaunchKernelGGL(ValuKernel<OP>, cus * 4, threads, 0, nullptr, iters / 3, 1u, out);
  bench::Sampler smp(10);
  smp.Start();
  t.Begin();
  hipLaunchKernelGGL(ValuKernel<OP>, cus * 4, threads, 0, nullptr, iters, 1u, out);
  double ms = t.EndMs();
  smp.Stop();
  HIP_CHECK(hipFree(out));
  *mhz = smp.MeanMhz();
  double instrs = (double)cus * 4 * (threads / 32) * iters * kChains;
  double simds = cus * 4.0;  // 20 WGPs x 4 SIMD32
  return instrs / (ms * 1e-3) / (*mhz * 1e6) / simds;
}

static void Valu(int cus) {
  std::printf("# VALU throughput, instructions per clock per SIMD (1.0 = full rate)\n");
  std::printf("%-24s %8s %8s %8s %10s\n", "op", "4w/simd", "8w/simd", "MHz", "Gop/s/SIMD");
#define RUN(ID, NAME, ASM)                                     \
  {                                                            \
    double m4, m8;                                             \
    double r4 = RunValu<ID>(cus, 4, &m4);                      \
    double r8 = RunValu<ID>(cus, 8, &m8);                      \
    std::printf("%-24s %8.3f %8.3f %8.0f %10.2f\n", NAME, r4, r8, m8, r8 * m8 / 1e3); \
  }
  OP_LIST(RUN)
#undef RUN
}

// ---- WMMA ---------------------------------------------------------------

template <int KIND>
__global__ void WmmaKernel(int iters, uint32_t* out) {
  uint32_t lane = threadIdx.x;
  // Two operand sets per chain keep the compiler from merging chains.
  if constexpr (KIND == 0 || KIND == 2) {  // f16 inputs
    h16 a, b;
    for (int i = 0; i < 16; ++i) {
      a[i] = (_Float16)((lane + i) % 7) * (_Float16)0.01f;
      b[i] = (_Float16)((lane * 3 + i) % 5) * (_Float16)0.01f;
    }
    if constexpr (KIND == 0) {
      f8 c[kChains] = {};
      for (int it = 0; it < iters; ++it)
#pragma unroll
        for (int j = 0; j < kChains; ++j)
          c[j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c[j]);
      float s = 0;
      for (int j = 0; j < kChains; ++j) s += c[j][0] + c[j][7];
      if (__float_as_uint(s) == kSentinel) out[0] = 1;
    } else {
      h16 c[kChains] = {};
      for (int it = 0; it < iters; ++it)
#pragma unroll
        for (int j = 0; j < kChains; ++j)
          c[j] = __builtin_amdgcn_wmma_f16_16x16x16_f16_w32(a, b, c[j], false);
      float s = 0;
      for (int j = 0; j < kChains; ++j) s += (float)c[j][0];
      if (__float_as_uint(s) == kSentinel) out[0] = 1;
    }
  } else if constexpr (KIND == 1) {  // bf16 -> f32
    s16 a, b;
    for (int i = 0; i < 16; ++i) {
      a[i] = (short)(0x3c00 + (lane + i) % 7);
      b[i] = (short)(0x3c00 + (lane * 3 + i) % 5);
    }
    f8 c[kChains] = {};
    for (int it = 0; it < iters; ++it)
#pragma unroll
      for (int j = 0; j < kChains; ++j)
        c[j] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, c[j]);
    float s = 0;
    for (int j = 0; j < kChains; ++j) s += c[j][0];
    if (__float_as_uint(s) == kSentinel) out[0] = 1;
  } else if constexpr (KIND == 3) {  // iu8 -> i32
    i4v a, b;
    for (int i = 0; i < 4; ++i) {
      a[i] = (int)(lane * 0x01010101u + i);
      b[i] = (int)(lane * 0x02030405u + i);
    }
    i8v c[kChains] = {};
    for (int it = 0; it < iters; ++it)
#pragma unroll
      for (int j = 0; j < kChains; ++j)
        c[j] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a, true, b, c[j], false);
    int s = 0;
    for (int j = 0; j < kChains; ++j) s ^= c[j][0];
    if ((uint32_t)s == kSentinel) out[0] = 1;
  } else {  // iu4 -> i32
    i2v a, b;
    for (int i = 0; i < 2; ++i) {
      a[i] = (int)(lane * 0x11111111u + i);
      b[i] = (int)(lane * 0x12345671u + i);
    }
    i8v c[kChains] = {};
    for (int it = 0; it < iters; ++it)
#pragma unroll
      for (int j = 0; j < kChains; ++j)
        c[j] = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(true, a, true, b, c[j], false);
    int s = 0;
    for (int j = 0; j < kChains; ++j) s ^= c[j][0];
    if ((uint32_t)s == kSentinel) out[0] = 1;
  }
}

template <int KIND>
static void RunWmma(const char* name, int cus) {
  uint32_t* out;
  HIP_CHECK(hipMalloc(&out, 64));
  for (int wps : {1, 2, 4}) {
    int threads = 32 * 4 * wps;
    bench::EventTimer t;
    int iters = 2000;
    t.Begin();
    hipLaunchKernelGGL(WmmaKernel<KIND>, cus * 8, threads, 0, nullptr, iters, out);
    double probe_ms = t.EndMs();
    iters = (int)(iters * 1000.0 / std::max(probe_ms, 0.01));
    hipLaunchKernelGGL(WmmaKernel<KIND>, cus * 8, threads, 0, nullptr, iters / 3, out);
    bench::Sampler smp(10);
    smp.Start();
    t.Begin();
    hipLaunchKernelGGL(WmmaKernel<KIND>, cus * 8, threads, 0, nullptr, iters, out);
    double ms = t.EndMs();
    smp.Stop();
    double ops = (double)cus * 8 * (threads / 32) * iters * kChains * 16 * 16 * 16 * 2;
    double tops = ops / (ms * 1e-3) / 1e12;
    double per_clk_cu = ops / (ms * 1e-3) / (smp.MeanMhz() * 1e6) / (cus * 2);
    std::printf("%-22s %d waves/SIMD: %6.1f T(FL)OPS, %5.0f ops/clk/CU  [%s]\n", name,
                wps, tops, per_clk_cu, smp.Summary().c_str());
  }
  HIP_CHECK(hipFree(out));
}

int main(int argc, char** argv) {
  std::string mode = argc > 1 ? argv[1] : "all";
  int cus = bench::NumCUs();  // HIP reports WGPs (20) on gfx11
  if (mode == "valu" || mode == "all") Valu(cus);
  if (mode == "wmma" || mode == "all") {
    RunWmma<0>("wmma f16->f32", cus);
    RunWmma<1>("wmma bf16->f32", cus);
    RunWmma<2>("wmma f16->f16", cus);
    RunWmma<3>("wmma iu8->i32", cus);
    RunWmma<4>("wmma iu4->i32", cus);
  }
  return 0;
}
