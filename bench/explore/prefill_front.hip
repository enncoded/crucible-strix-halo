// Round 4 Q2, prefill side: does the prefill wide GEMM body (main d3468ba,
// engine/exl3/prefill_linear.h PrefillWideBody) lose rate when it reads weight
// tiles in mega2's front order (chunk 4, decode split S) instead of the
// engine's column-major order? FrontWideBody is PrefillWideBody with one
// change: the tile address. It is wave-uniform, so it lives in the scalar
// offset of the raw-buffer loads and adds no VALU work. Every output tile
// keeps its WMMA sequence, so the partials must be bitwise equal.
//
// Build on the test machine against a main snapshot (git archive d3468ba engine kernels
// bench/common.h engine/exl3/safetensors.h into build/main-d3468ba):
//   hipcc ... -I build/main-d3468ba -o build/bench/prefill_front bench/explore/prefill_front.hip
// Usage: prefill_front CKPT [M=4096] [S=1] [reps=10]
#include <random>

#include "bench/common.h"
#include "engine/exl3/safetensors.h"
#include "engine/exl3/prefill_linear.h"

using namespace crucible;

namespace {
struct Matrix {
  int I, O, K;
  uint32_t* tiles;  // engine order: tile (b, a) at (b * A + a) * T words
  size_t words;
};

Matrix Load(st::Checkpoint& ck, const std::string& name) {
  const auto& ti = ck.Info(name + ".trellis");
  if (ti.dtype != "I16" || ti.shape.size() != 3 || ti.shape[2] % 16 != 0) std::abort();
  const int A = ti.shape[0], B = ti.shape[1], K = ti.shape[2] / 16, T = 8 * K;
  std::vector<uint32_t> host((size_t)A * B * T);
  const auto* src = reinterpret_cast<const uint32_t*>(ck.Data(name + ".trellis"));
  for (int a = 0; a < A; ++a)
    for (int b = 0; b < B; ++b)
      std::memcpy(host.data() + ((size_t)b * A + a) * T, src + ((size_t)a * B + b) * T, T * 4);
  Matrix m{A * 16, B * 16, K, nullptr, host.size()};
  HIP_CHECK(hipMalloc(&m.tiles, m.words * 4));
  HIP_CHECK(hipMemcpy(m.tiles, host.data(), m.words * 4, hipMemcpyHostToDevice));
  return m;
}

// mega2's front order (bench/mega_ablate2.hip Repack, layout 2, chunk 4).
uint32_t* RepackFront(const Matrix& m, int S) {
  const int A = m.I / 16, B = m.O / 16, T = 8 * m.K, span = A / S, count = span / 4, ncg = B / 8;
  std::vector<uint32_t> src(m.words), dst(m.words);
  HIP_CHECK(hipMemcpy(src.data(), m.tiles, m.words * 4, hipMemcpyDeviceToHost));
  for (int s = 0; s < S; ++s)
    for (int cg = 0; cg < ncg; ++cg)
      for (int q = 0; q < count; ++q)
        for (int c = 0; c < 8; ++c)
          for (int f = 0; f < 4; ++f) {
            const size_t ci = (size_t)q * S * ncg + s * ncg + cg;
            std::memcpy(dst.data() + (ci * 32 + c * 4 + f) * T,
                        src.data() + ((size_t)(cg * 8 + c) * A + s * span + q * 4 + f) * T, T * 4);
          }
  uint32_t* out;
  HIP_CHECK(hipMalloc(&out, m.words * 4));
  HIP_CHECK(hipMemcpy(out, dst.data(), m.words * 4, hipMemcpyHostToDevice));
  return out;
}

// PrefillWideBody (main d3468ba) with front-order tile addresses. base(a) is
// the tile index of column 0 of a's chunk row; column b adds cg * 32 + c * 4.
template <int K, int RM, int NB, bool EARLY_A = true, bool SKIP = true, bool ORDER = SKIP>
__device__ __forceinline__ void FrontWideBody(const Exl3GemmJob& j, const Exl3GemmItem& it, uint32_t* ws, int S) {
  constexpr int T = 8 * K;
  const int lane = threadIdx.x & 31, wave = __builtin_amdgcn_readfirstlane(threadIdx.x >> 5);
  const int b = it.b0 + wave * NB;
  if (b * 16 >= j.O) return;
  const int A = j.I / 16, B = j.O / 16, ncg = B / 8, chunks = A / S / 4;
  const LaneWindow<K> win(lane);
  const auto rs = StreamRsrc(j.tiles, (uint32_t)((size_t)A * B * T * 4));
  int col[NB];
#pragma unroll
  for (int n = 0; n < NB; ++n) col[n] = ((b + n) >> 3) * 32 + ((b + n) & 7) * 4;
  // Scalar walk over (s, q, f): f steps inside a chunk, q jumps S * ncg chunk
  // rows, s restarts q at the next split.
  int s = 0, q = 0, f = 0, base = 0;
  auto advance = [&]() {
    if (++f < 4) { ++base; return; }
    f = 0;
    if (++q < chunks) { base += S * ncg * 32 - 3; return; }
    q = 0; ++s; base = s * ncg * 32;
  };
  GemmF8 acc[NB][RM] = {};
  const uint32_t xbytes = (uint32_t)j.Xrows * j.I * 2;
  const auto xrs = StreamRsrc(reinterpret_cast<const uint32_t*>(j.x), xbytes);
  const int astep = j.Xrows * 32;
  uint32_t xo[RM];
#pragma unroll
  for (int r = 0; r < RM; ++r) {
    const int row = it.m0 + r * 16 + (lane & 15);
    xo[r] = row < j.M ? (uint32_t)row * 32 : xbytes;
  }
  uint32_t* w = ws + wave * kGemmWaveWords;
  const int o0 = win.d0 * 4, o1 = ((win.d0 + 1) % T) * 4, o2 = ((win.d0 + 2) % T) * 4;
  uint32_t raw[NB][3];
#pragma unroll
  for (int n = 0; n < NB; ++n) {
    const int so = (base + col[n]) * T * 4;
    raw[n][0] = __builtin_amdgcn_raw_buffer_load_b32(rs, o0, so, 0);
    raw[n][1] = __builtin_amdgcn_raw_buffer_load_b32(rs, o1, so, 0);
    raw[n][2] = K == 5 ? __builtin_amdgcn_raw_buffer_load_b32(rs, o2, so, 0) : 0;
  }
  for (int a = 0; a < A; ++a) {
    GemmH16 afr[RM];
    auto load_a = [&] {
#pragma unroll
      for (int r = 0; r < RM; ++r) {
        if (SKIP && it.m0 + r * 16 >= j.M) break;
        afr[r] = GemmAFragment<false>(xrs, xo[r], a, nullptr, 0, astep);
      }
      if constexpr (ORDER) asm volatile("" ::: "memory");
    };
    if constexpr (EARLY_A) load_a();
    advance();
    // Past the last tile the walk points at valid data; the loads are unused.
    GemmH16 bfs[EARLY_A ? 1 : NB];
    if constexpr (!EARLY_A) {
#pragma unroll
      for (int n = 0; n < NB; ++n) {
        __half2 p[4];
        DecodeLane<K, kMulTrue16>(raw[n][0], raw[n][1], raw[n][2], win.shift, p);
        bfs[n] = GemmBFragment<true>(p, w, lane);
      }
      load_a();
    }
#pragma unroll
    for (int n = 0; n < NB; ++n) {
      const uint32_t w0 = raw[n][0], w1 = raw[n][1], w2 = raw[n][2];
      const int so = a + 1 < A ? (base + col[n]) * T * 4 : 0;
      raw[n][0] = __builtin_amdgcn_raw_buffer_load_b32(rs, o0, so, 0);
      raw[n][1] = __builtin_amdgcn_raw_buffer_load_b32(rs, o1, so, 0);
      if constexpr (K == 5) raw[n][2] = __builtin_amdgcn_raw_buffer_load_b32(rs, o2, so, 0);
      GemmH16 bf;
      if constexpr (EARLY_A) {
        __half2 p[4];
        DecodeLane<K, kMulTrue16>(w0, w1, w2, win.shift, p);
        bf = GemmBFragment<true>(p, w, lane);
      } else {
        bf = bfs[n];
      }
#pragma unroll
      for (int r = 0; r < RM; ++r) {
        if (SKIP && it.m0 + r * 16 >= j.M) break;
        acc[n][r] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(afr[r], bf, acc[n][r]);
      }
    }
  }
  const auto ps = StreamRsrc(reinterpret_cast<const uint32_t*>(j.partial), (uint32_t)j.M * j.O * 4);
#pragma unroll
  for (int n = 0; n < NB; ++n) {
    const uint32_t po = ((uint32_t)(it.m0 + lane / 16) * j.O + (b + n) * 16 + (lane & 15)) * 4;
#pragma unroll
    for (int r = 0; r < RM; ++r) {
      if (SKIP && it.m0 + r * 16 >= j.M) break;
#pragma unroll
      for (int z = 0; z < 8; ++z)
        __builtin_amdgcn_raw_buffer_store_b32(__float_as_uint(acc[n][r][z]), ps,
                                              po + (uint32_t)(r * 16 + 2 * z) * j.O * 4, 0, 0);
    }
  }
}

// One block = 4 waves x NB=2 = 8 column tiles; blockIdx.y = row group;
// blockIdx.z = job (one matrix per job).
template <int K, int RM, bool FRONT, bool DENSE>
__global__ void __launch_bounds__(128) WideKernel(const Exl3GemmJob* jobs, int S) {
  __shared__ uint32_t ws[kGemmWeightWords];
  const Exl3GemmJob j = jobs[blockIdx.z];
  const Exl3GemmItem it{0, 0, (int)blockIdx.y * RM * 16, (int)blockIdx.x * 8};
  if (it.m0 >= j.M || it.b0 * 16 >= j.O) return;
  // Dense: no skip checks and no order clobber.
  if constexpr (FRONT) FrontWideBody<K, RM, 2, true, !DENSE, !DENSE>(j, it, ws, S);
  else PrefillWideBody<K, RM, 2, true, !DENSE, !DENSE>(j, it, ws);
}

struct Case {
  std::string name;
  std::vector<Matrix> mats;
  int M, RM;
  bool dense;
};

template <int RM, bool DENSE>
void LaunchK(int K, bool front, dim3 grid, const Exl3GemmJob* jobs, int S) {
  if (K == 3) {
    if (front) WideKernel<3, RM, true, DENSE><<<grid, 128>>>(jobs, S);
    else WideKernel<3, RM, false, DENSE><<<grid, 128>>>(jobs, S);
  } else if (K == 4) {
    if (front) WideKernel<4, RM, true, DENSE><<<grid, 128>>>(jobs, S);
    else WideKernel<4, RM, false, DENSE><<<grid, 128>>>(jobs, S);
  } else {
    if (front) WideKernel<5, RM, true, DENSE><<<grid, 128>>>(jobs, S);
    else WideKernel<5, RM, false, DENSE><<<grid, 128>>>(jobs, S);
  }
}

void Launch(const Case& c, bool front, dim3 grid, const Exl3GemmJob* jobs, int S) {
  if (c.dense) LaunchK<8, true>(c.mats[0].K, front, grid, jobs, S);
  else LaunchK<6, false>(c.mats[0].K, front, grid, jobs, S);
  HIP_CHECK(hipGetLastError());
}

int RunCase(const Case& c, int S, int reps) {
  const int nj = (int)c.mats.size(), I = c.mats[0].I, O = c.mats[0].O;
  for (const auto& m : c.mats)
    if (m.I != I || m.O != O || m.K != c.mats[0].K || (m.I / 16 / S) % 4 || (m.O / 16) % 8) {
      std::printf("%s: skipped (shape or split S=%d not aligned)\n", c.name.c_str(), S);
      return 0;
    }
  std::mt19937 rng(5); std::normal_distribution<float> norm;
  std::vector<__half> hx((size_t)c.M * I);
  for (auto& h : hx) h = __float2half(norm(rng));
  __half* x; float* part[2];
  HIP_CHECK(hipMalloc(&x, hx.size() * 2));
  HIP_CHECK(hipMemcpy(x, hx.data(), hx.size() * 2, hipMemcpyHostToDevice));
  const size_t pw = (size_t)nj * c.M * O;
  for (auto& p : part) HIP_CHECK(hipMalloc(&p, pw * 4));
  std::vector<Exl3GemmJob> hj[2];
  size_t bytes = 0;
  for (int i = 0; i < nj; ++i) {
    const Matrix& m = c.mats[i];
    bytes += m.words * 4;
    uint32_t* front = RepackFront(m, S);
    for (int l = 0; l < 2; ++l)
      hj[l].push_back({l ? front : m.tiles, x, part[l] + (size_t)i * c.M * O, nullptr, I, O, c.M, 1, m.K, c.M});
  }
  Exl3GemmJob* dj[2];
  for (int l = 0; l < 2; ++l) {
    HIP_CHECK(hipMalloc(&dj[l], nj * sizeof(Exl3GemmJob)));
    HIP_CHECK(hipMemcpy(dj[l], hj[l].data(), nj * sizeof(Exl3GemmJob), hipMemcpyHostToDevice));
  }
  const dim3 grid(O / 16 / 8, (c.M + c.RM * 16 - 1) / (c.RM * 16), nj);
  for (int l = 0; l < 2; ++l) {
    HIP_CHECK(hipMemset(part[l], 0xff, pw * 4));
    Launch(c, l == 1, grid, dj[l], S);
  }
  HIP_CHECK(hipDeviceSynchronize());
  std::vector<float> h0(pw), h1(pw);
  HIP_CHECK(hipMemcpy(h0.data(), part[0], pw * 4, hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(h1.data(), part[1], pw * 4, hipMemcpyDeviceToHost));
  const bool exact = std::memcmp(h0.data(), h1.data(), pw * 4) == 0;
  double med[2];
  for (int l : {0, 1, 0, 1}) {
    std::vector<double> t;
    for (int r = 0; r < reps + 1; ++r) {
      bench::EventTimer timer; timer.Begin();
      Launch(c, l == 1, grid, dj[l], S);
      const float ms = timer.EndMs();
      if (r) t.push_back(ms);
    }
    med[l] = bench::Median(t);
  }
  const double flop = 2.0 * nj * c.M * (double)I * O;
  std::printf("%-26s M=%5d jobs=%2d K=%d S=%d  engine %8.3f ms  front %8.3f ms  front/engine %.4f  %5.1f TFLOPS  weights %.1f MB  %s\n",
              c.name.c_str(), c.M, nj, c.mats[0].K, S, med[0], med[1], med[1] / med[0], flop / med[0] / 1e9,
              bytes / 1e6, exact ? "bitwise equal" : "MISMATCH");
  std::fflush(stdout);
  for (auto& p : part) HIP_CHECK(hipFree(p));
  HIP_CHECK(hipFree(x));
  for (auto* d : dj) HIP_CHECK(hipFree(d));
  for (int i = 0; i < nj; ++i) HIP_CHECK(hipFree(const_cast<uint32_t*>(hj[1][i].tiles)));
  return exact ? 0 : 1;
}
}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) { std::fprintf(stderr, "usage: prefill_front CKPT [M=4096] [S=1] [reps=10]\n"); return 2; }
  const int M = argc > 2 ? std::atoi(argv[2]) : 4096;
  const int S = argc > 3 ? std::atoi(argv[3]) : 1;
  const int reps = argc > 4 ? std::atoi(argv[4]) : 10;
  st::Checkpoint ck(argv[1]);
  const std::string L0 = "model.language_model.layers.0.", L3 = "model.language_model.layers.3.";
  std::vector<Case> cases;
  cases.push_back({"dense gdn in_proj_qkv", {Load(ck, L0 + "linear_attn.in_proj_qkv")}, M, 8, true});
  cases.push_back({"dense gdn out_proj", {Load(ck, L0 + "linear_attn.out_proj")}, M, 8, true});
  cases.push_back({"dense qsa q_proj", {Load(ck, L3 + "self_attn.q_proj")}, M, 8, true});
  Case gate{"moe gate x64 (96 rows)", {}, 96, 6, false}, down{"moe down x64 (96 rows)", {}, 96, 6, false};
  for (int e = 0; e < 64; ++e) {
    const std::string E = L0 + "mlp.experts." + std::to_string(e) + ".";
    gate.mats.push_back(Load(ck, E + "gate_proj"));
    down.mats.push_back(Load(ck, E + "down_proj"));
  }
  cases.push_back(gate);
  cases.push_back(down);
  int fails = 0;
  for (const auto& c : cases) fails += RunCase(c, S, reps);
  return fails ? 1 : 0;
}
