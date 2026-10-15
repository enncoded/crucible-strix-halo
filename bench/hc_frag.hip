// Fragment-order HC weights for prefill (explore item 5): does storing fn, up
// and t as 16x16 tiles (one 512-byte run per wave load, lanes 0..15 own one
// row each) speed up the prefill HC dots and mix? Every variant keeps each
// output's WMMA chain and epilogue, so it must be bitwise equal to the engine
// kernel it replaces. Usage: hc_frag R [ldh]
#include "prefill_common.h"
#include "../engine/layers/prefill_hc.h"
using namespace crucible;
using namespace prefill_bench;

namespace crucible {
constexpr int kKt = kRank / 16;          // 20 k tiles in the mix
constexpr int kDotKt = kHidden / 16;     // 160 k tiles per stream in the dots
constexpr int kFnTiles = (kRank + 4 + 15) / 16;  // 21 column tiles of fn

// Tile helpers: tile id * 256 halves, lane row (lane & 15) at 16 halves.
__device__ __forceinline__ size_t LaneOff() { return (size_t)(threadIdx.x & 15) * 16; }

// up [4*kHidden][kRank] -> [row/16][kt][16][16].
__global__ void UpFragKernel(const __half* up, __half* uf) {
  const size_t i = (size_t)blockIdx.x * 256 + threadIdx.x;
  if (i >= (size_t)kWide * kRank) return;
  const int row = i / kRank, k = i % kRank;
  uf[((size_t)(row >> 4) * kKt + k / 16) * 256 + (row & 15) * 16 + (k & 15)] = up[i];
}
// fn [M][kWide] -> [nt][s][kt][16][16], rows past M are zero.
__global__ void FnFragKernel(const __half* fn, int M, __half* ff) {
  const size_t i = (size_t)blockIdx.x * 256 + threadIdx.x;
  if (i >= (size_t)kFnTiles * 16 * kWide) return;
  const int row = i / kWide, c = i % kWide, s = c / kHidden, k = c % kHidden;
  ff[(((size_t)(row >> 4) * 4 + s) * kDotKt + k / 16) * 256 + (row & 15) * 16 + (k & 15)] =
      row < M ? fn[(size_t)row * kWide + c] : __half(0.f);
}
// t [R][kRank] -> [R/16][kt][16][16] (R padded to 16 rows).
__global__ void TFragKernel(const __half* t, __half* tf, int R) {
  const size_t i = (size_t)blockIdx.x * 256 + threadIdx.x;
  if (i >= (size_t)R * kRank) return;
  const int r = i / kRank, k = i % kRank;
  tf[((size_t)(r >> 4) * kKt + k / 16) * 256 + (r & 15) * 16 + (k & 15)] = t[i];
}

// PrefillHcMixWideKernel<true> with optional fragment-order t (FA) and up
// (FB) and RT row tiles per wave. DIAG 2 replaces the WMMAs with one VALU op.
template <bool FA, bool FB, int RT = 1, int DIAG = 0>
__global__ void __launch_bounds__(256) MixFragKernel(const __half* up, const float* w1,
    const __half* thi, const __half* tlo, const float* ss, const float* H, float* mixed, int R,
    __half* xh, int ldh) {
  const int lane = threadIdx.x & 31, wave = threadIdx.x >> 5;
  const int r0 = blockIdx.y * 128 * RT + wave * 16 * RT;
  const int ct = blockIdx.x, d = ct * 16 + (lane & 15);
  if (r0 >= R) return;
  prefill::F8 hi[RT][4] = {}, lo[RT][4] = {};
#pragma unroll 2
  for (int k = 0; k < kRank; k += 16) {
    const int kt = k / 16;
    prefill::H16 ah[RT], al[RT];
#pragma unroll
    for (int i = 0; i < RT; ++i) {
      const int ra = r0 + 16 * i + (lane & 15);
      if (FA) {
        const size_t o = ((size_t)((r0 >> 4) + i) * kKt + kt) * 256 + LaneOff();
        ah[i] = ra < R ? PrefillLoad16(thi + o) : prefill::H16{};
        al[i] = ra < R ? PrefillLoad16(tlo + o) : prefill::H16{};
      } else {
        ah[i] = ra < R ? PrefillLoad16(thi + (size_t)ra * kRank + k) : prefill::H16{};
        al[i] = ra < R ? PrefillLoad16(tlo + (size_t)ra * kRank + k) : prefill::H16{};
      }
    }
#pragma unroll
    for (int s = 0; s < 4; ++s) {
      const prefill::H16 b = FB ? PrefillLoad16(up + ((size_t)(s * (kHidden / 16) + ct) * kKt + kt) * 256 + LaneOff())
                                : PrefillLoad16(up + ((size_t)s * kHidden + d) * kRank + k);
#pragma unroll
      for (int i = 0; i < RT; ++i) {
        if constexpr (DIAG & 2) {
          hi[i][s][0] += (float)(ah[i][0] * b[1]);
          lo[i][s][0] += (float)(al[i][0] * b[2]);
        } else {
          hi[i][s] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(ah[i], b, hi[i][s]);
          lo[i][s] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(al[i], b, lo[i][s]);
        }
      }
    }
  }
  float w[4];
#pragma unroll
  for (int s = 0; s < 4; ++s) w[s] = w1[s * kHidden + d];
#pragma unroll
  for (int i = 0; i < RT; ++i)
#pragma unroll
    for (int z = 0; z < 8; ++z) {
      const int r = r0 + 16 * i + prefill::FragmentRow(z);
      if (r >= R) continue;
      float v[4];
#pragma unroll
      for (int s = 0; s < 4; ++s) {
        const float g = hi[i][s][z] + lo[i][s][z] * (1.f / prefill::kResidualScale);
        const float inv = rsqrtf(ss[r * 4 + s] / kHidden + kEps);
        v[s] = Sigmoid(g) * H[(size_t)r * ldh + s * kHidden + d] * inv * w[s];
      }
      const float x = RoundH(((v[0] + v[1]) + (v[2] + v[3])) * .25f);
      mixed[(size_t)r * kHidden + d] = x;
      if (xh) xh[(size_t)r * kHidden + d] = __float2half(x);
    }
}

// PrefillHcDotsWsKernel<64,0,true,2,2,false,4> (the engine's R >= 512 launch)
// with fn read either row-major (FB false, the engine) or in fragment order,
// and the B fragments PF k blocks ahead (the engine uses 1).
template <bool FB, int PF>
__global__ void __launch_bounds__((kHcWsCompute + 2) * 32) DotsFragKernel(const __half* fn, int M,
    const float* H, float* dots, float* ss, int R, int ldh) {
  typedef _Float16 H4 __attribute__((ext_vector_type(4)));
  constexpr int STEP = 64, LW = 2, D = 2, RT = 4;
  constexpr int kPad = STEP + 8, kI = STEP / 32, kSteps = kHidden / STEP, G = 8 / LW, TPR = 8 / G,
                ROWS = 16 * RT, NP = RT / 2, KB = STEP / 16;
  static_assert(kSteps % D == 0 && KB % PF == 0);
  __shared__ __attribute__((aligned(16))) _Float16 ahi[2][ROWS][kPad];
  const int t = threadIdx.x, lane = t & 31, wave = t >> 5;
  const int r0 = (blockIdx.x >> 2) * ROWS, s = blockIdx.x & 3;
  if (wave >= kHcWsCompute) {
    const int u = t - kHcWsCompute * 32, sr = u / TPR, p = u % TPR;
    bool live[NP];
    const float* src[NP];
#pragma unroll
    for (int pp = 0; pp < NP; ++pp) {
      live[pp] = r0 + sr + 32 * pp < R;
      src[pp] = H + (size_t)(r0 + sr + 32 * pp) * ldh + s * kHidden + 4 * G * p;
    }
    float sq[NP][G] = {};
    float4 v[D][NP][kI][G];
    auto load = [&](float4 (&w)[NP][kI][G], int step) {
#pragma unroll
      for (int pp = 0; pp < NP; ++pp)
#pragma unroll
        for (int i = 0; i < kI; ++i)
#pragma unroll
          for (int h = 0; h < G; ++h)
            w[pp][i][h] = live[pp] ? *reinterpret_cast<const float4*>(src[pp] + step * STEP + i * 32 + 4 * h) : float4{};
    };
    auto store = [&](const float4 (&w)[NP][kI][G], int buf) {
#pragma unroll
      for (int pp = 0; pp < NP; ++pp)
#pragma unroll
        for (int i = 0; i < kI; ++i)
#pragma unroll
          for (int h = 0; h < G; ++h) {
            const float x[4] = {w[pp][i][h].x, w[pp][i][h].y, w[pp][i][h].z, w[pp][i][h].w};
            H4 hi4;
#pragma unroll
            for (int e = 0; e < 4; ++e) {
              sq[pp][h] += x[e] * x[e];
              hi4[e] = (_Float16)x[e];
            }
            *reinterpret_cast<H4*>(&ahi[buf][sr + 32 * pp][4 * G * p + 32 * i + 4 * h]) = hi4;
          }
    };
#pragma unroll
    for (int d = 0; d < D; ++d) load(v[d], d);
    store(v[0], 0);
    if (D < kSteps) load(v[0], D);
    PrefillLdsSync();
#pragma unroll 1
    for (int s0 = 0; s0 < kSteps; s0 += D)
#pragma unroll
      for (int d = 0; d < D; ++d) {
        const int step = s0 + d;
        if (step + 1 < kSteps) {
          store(v[(d + 1) % D], (step + 1) & 1);
          if (step + 1 + D < kSteps) load(v[(d + 1) % D], step + 1 + D);
        }
        PrefillLdsSync();
      }
#pragma unroll
    for (int pp = 0; pp < NP; ++pp) {
      float q[G];
#pragma unroll
      for (int h = 0; h < G; ++h) q[h] = sq[pp][h];
#pragma unroll
      for (int w = 1; w < G; w <<= 1)
#pragma unroll
        for (int h = 0; h < G; h += 2 * w) q[h] = q[h] + q[h + w];
      float a = q[0];
#pragma unroll
      for (int w = 1; w < TPR; w <<= 1) a += __shfl_xor(a, w);
      if (p == 0 && live[pp]) ss[(r0 + sr + 32 * pp) * kStreams + s] = a;
    }
    return;
  }
  // B source for k block g (16 halves) of column group j.
  auto bload = [&](int j, int g) -> prefill::H16 {
    const int nt = wave + kHcWsCompute * j;
    if (FB) return PrefillLoad16(fn + (((size_t)nt * 4 + s) * kDotKt + g) * 256 + LaneOff());
    const int n = min(nt * 16 + (lane & 15), M - 1);
    return PrefillLoad16(fn + (size_t)n * kWide + s * kHidden + g * 16);
  };
  prefill::F8 hi[RT][3] = {};
  prefill::H16 ring[PF][3];
#pragma unroll
  for (int f = 0; f < PF; ++f)
#pragma unroll
    for (int j = 0; j < 3; ++j) ring[f][j] = bload(j, f);
  PrefillLdsSync();
#pragma unroll 1
  for (int step = 0; step < kSteps; ++step) {
    const int buf = step & 1;
#pragma unroll
    for (int kb = 0; kb < KB; ++kb) {
      const int g = step * KB + kb, slot = kb % PF, gn = g + PF < kDotKt ? g + PF : g;
      prefill::H16 b[3];
#pragma unroll
      for (int j = 0; j < 3; ++j) b[j] = ring[slot][j];
#pragma unroll
      for (int j = 0; j < 3; ++j) ring[slot][j] = bload(j, gn);
      prefill::H16 ah[RT];
#pragma unroll
      for (int i = 0; i < RT; ++i)
        ah[i] = *reinterpret_cast<const prefill::H16*>(&ahi[buf][i * 16 + (lane & 15)][kb * 16]);
#pragma unroll
      for (int j = 0; j < 3; ++j)
#pragma unroll
        for (int i = 0; i < RT; ++i) hi[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(ah[i], b[j], hi[i][j]);
    }
    PrefillLdsSync();
  }
#pragma unroll
  for (int j = 0; j < 3; ++j) {
    const int n = (wave + kHcWsCompute * j) * 16 + (lane & 15);
    if (n >= M) continue;
#pragma unroll
    for (int i = 0; i < RT; ++i)
#pragma unroll
      for (int z = 0; z < 8; ++z) {
        const int r = r0 + i * 16 + prefill::FragmentRow(z);
        if (r < R) dots[((size_t)r * M + n) * 4 + s] = hi[i][j][z];
      }
  }
}
}  // namespace crucible

int main(int argc, char** argv) {
  const int R = argc > 1 ? std::atoi(argv[1]) : 4096, M = kRank + 4;
  const int ldh = argc > 2 ? std::atoi(argv[2]) : kWide;
  if (R < 16 || R > 8192 || R % 16 || ldh < kWide) return 2;
  const int Rp = (R + 255) / 256 * 256;
  Buffer<float> H(Random((size_t)R * ldh, .4f, 311));
  Buffer<__half> fn(Half(Random((size_t)M * kWide, .02f, 313)));
  Buffer<__half> up(Half(Random((size_t)kWide * kRank, .02f, 317)));
  auto ws = Random(kWide, .05f, 319);
  for (auto& v : ws) v += 1.f;
  Buffer<float> w1(ws), dots((size_t)R * M * 4), ss(R * 4), post(R * 4), mixed((size_t)R * kHidden);
  Buffer<__half> thi((size_t)Rp * kRank), tlo((size_t)Rp * kRank), thf((size_t)Rp * kRank), tlf((size_t)Rp * kRank);
  Buffer<__half> uf((size_t)kWide * kRank), ff((size_t)kFnTiles * 16 * kWide), xh((size_t)R * kHidden);
  UpFragKernel<<<(kWide * kRank + 255) / 256, 256>>>(up.p, uf.p);
  FnFragKernel<<<(kFnTiles * 16 * kWide + 255) / 256, 256>>>(fn.p, M, ff.p);
  CheckLaunch();
  bool all = true;

  // Dots: engine kernel, its copy, and fragment-order fn.
  const int nb4 = 4 * ((R + 63) / 64);
  auto eng = [&] {
    PrefillHcDotsWsKernel<64, 0, true, 2, 2, false, 4><<<nb4, (kHcWsCompute + 2) * 32>>>(fn.p, M, H.p, dots.p, ss.p, R
#ifdef HC_FRAG_LDH
      , ldh
#endif
    );
  };
#ifndef HC_FRAG_LDH
  if (ldh != kWide) { std::printf("engine kernel has no ldh here; build with -DHC_FRAG_LDH\n"); return 2; }
#endif
  eng(); CheckLaunch();
  const auto d0 = dots.Get(), s0 = ss.Get();
  auto runDots = [&](const char* name, auto launch) {
    dots.Zero(); ss.Zero(); launch(); CheckLaunch();
    const bool same = dots.Get() == d0 && ss.Get() == s0;
    all = all && same;
    std::printf("R=%d ldh=%d dots %-16s bitwise=%d us=%.1f\n", R, ldh, name, (int)same, Time(launch));
  };
  runDots("engine", eng);
#define DF(FB, PF) runDots("copy fb" #FB " pf" #PF, [&] { \
    DotsFragKernel<FB, PF><<<nb4, (kHcWsCompute + 2) * 32>>>(FB ? ff.p : fn.p, M, H.p, dots.p, ss.p, R, ldh); })
  DF(false, 1); DF(false, 2); DF(false, 4); DF(true, 1); DF(true, 2); DF(true, 4);
#undef DF

  // Mid (engine) gives t; the fragment copies come from it.
  PrefillHcMidPackKernel<<<R, 256>>>(dots.p, M, ss.p, thi.p, tlo.p, post.p, R);
  TFragKernel<<<(R * kRank + 255) / 256, 256>>>(thi.p, thf.p, R);
  TFragKernel<<<(R * kRank + 255) / 256, 256>>>(tlo.p, tlf.p, R);
  CheckLaunch();
  auto mixEng = [&] {
    PrefillHcMixWideKernel<true><<<dim3(kHidden / 16, (R + 127) / 128), 256>>>(up.p, w1.p, thi.p, tlo.p, ss.p, H.p,
        mixed.p, R, xh.p
#ifdef HC_FRAG_LDH
        , ldh
#endif
    );
  };
  mixEng(); CheckLaunch();
  const auto m0 = mixed.Get();
  const auto x0 = xh.Get();
  auto runMix = [&](const char* name, auto launch, bool check) {
    mixed.Zero(); xh.Zero(); launch(); CheckLaunch();
    const bool same = mixed.Get() == m0 && xh.Get() == x0;
    if (check) all = all && same;
    std::printf("R=%d ldh=%d mix %-18s bitwise=%d us=%.1f\n", R, ldh, name, (int)same, Time(launch));
  };
  runMix("engine", mixEng, true);
#define MF(FA, FB, RT, DIAG) runMix("fa" #FA " fb" #FB " rt" #RT " diag" #DIAG, [&] { \
    MixFragKernel<FA, FB, RT, DIAG><<<dim3(kHidden / 16, (R + 128 * RT - 1) / (128 * RT)), 256>>>( \
        FB ? uf.p : up.p, w1.p, FA ? thf.p : thi.p, FA ? tlf.p : tlo.p, ss.p, H.p, mixed.p, R, xh.p, ldh); }, DIAG == 0)
  MF(false, false, 1, 0); MF(false, true, 1, 0); MF(true, false, 1, 0); MF(true, true, 1, 0);
  MF(false, false, 2, 0); MF(true, true, 2, 0);
  MF(false, false, 1, 2); MF(true, true, 1, 2); MF(true, true, 2, 2);
#undef MF
  std::printf("R=%d all_bitwise=%d\n", R, (int)all);
  return all ? 0 : 9;
}
