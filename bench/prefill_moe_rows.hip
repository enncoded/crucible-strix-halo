// MoE expert GEMM cost per 4096 prompt rows against the launch row count R
// (prefill-moe lesson 1). The engine's packed kernels (LaunchPrefillMoeGemm)
// run gate/up (K3 routed, K5 shared, fp32 out) and down (fp16 out) over
// random weights for top-10 routing of R rows; the shared expert takes all R
// rows. The weights (about 1.9 GB) do not fit the MALL, so every launch reads
// them from memory, as a layer does in the engine.
// Usage: prefill_moe_rows [skew] [rm...]   skew 0 = uniform routing.
#include "prefill_common.h"
#include "../engine/layers/prefill_moe.h"
using namespace crucible;
using namespace prefill_bench;

__global__ void FillKernel(uint32_t* p, size_t n, uint32_t seed, uint32_t mask) {
  for (size_t i = blockIdx.x * 256ull + threadIdx.x; i < n; i += 256ull * gridDim.x) {
    uint32_t x = (uint32_t)i * 2654435761u ^ seed;
    x ^= x >> 15; x *= 2246822519u; x ^= x >> 13; x *= 3266489917u; x ^= x >> 16;
    p[i] = x & mask;
  }
}
static void Fill(void* p, size_t bytes, uint32_t seed, uint32_t mask = ~0u) {
  FillKernel<<<4096, 256>>>(static_cast<uint32_t*>(p), bytes / 4, seed, mask);
  CheckLaunch();
}

std::vector<int> Counts(int R, double skew, unsigned seed) {
  std::mt19937 rng(seed);
  std::vector<double> w(kExperts);
  std::vector<int> perm(kExperts);
  std::iota(perm.begin(), perm.end(), 0);
  std::shuffle(perm.begin(), perm.end(), rng);
  for (int e = 0; e < kExperts; ++e) w[perm[e]] = std::pow(e + 1.0, -skew);
  std::vector<int> count(kExperts + 1, 0);
  for (int r = 0; r < R; ++r) {
    std::vector<double> ww = w;
    for (int k = 0; k < kTopK; ++k) {
      std::discrete_distribution<int> d(ww.begin(), ww.end());
      const int e = d(rng);
      ++count[e]; ww[e] = 0;
    }
  }
  count[kExperts] = R;
  return count;
}

int main(int argc, char** argv) {
  const double skew = argc > 1 ? std::atof(argv[1]) : 0.0;
  std::vector<int> rms;
  for (int a = 2; a < argc; ++a) rms.push_back(std::atoi(argv[a]));
  if (rms.empty()) rms = {6};
  constexpr int kMaxR = 32768;
  const size_t maxrows = (size_t)kMaxR * kSlots;
  // Routed K3, shared K5; gate, up and down per expert.
  const size_t w3 = (size_t)kHidden * kMoeInter * 3 / 32, w5 = (size_t)kHidden * kMoeInter * 5 / 32;
  const size_t wwords = w3 * 3 * kExperts + w5 * 3;
  uint32_t* tiles; __half *xg, *xu, *xd; float *pg, *pu, *pd;
  HIP_OK(hipMalloc(&tiles, wwords * 4));
  HIP_OK(hipMalloc(&xg, maxrows * kHidden * 2)); HIP_OK(hipMalloc(&xu, maxrows * kHidden * 2));
  HIP_OK(hipMalloc(&xd, maxrows * kMoeInter * 2));
  HIP_OK(hipMalloc(&pg, maxrows * kMoeInter * 4)); HIP_OK(hipMalloc(&pu, maxrows * kMoeInter * 4));
  HIP_OK(hipMalloc(&pd, maxrows * kHidden * 4));
  Fill(tiles, wwords * 4, 1);
  // Small fp16 values (exponent below 2^-1), so no partial overflows.
  const uint32_t hm = 0xb7ffb7ffu;
  Fill(xg, maxrows * kHidden * 2, 2, hm); Fill(xu, maxrows * kHidden * 2, 3, hm); Fill(xd, maxrows * kMoeInter * 2, 4, hm);
  auto wptr = [&](int e, int m) {
    return e < kExperts ? tiles + ((size_t)e * 3 + m) * w3 : tiles + (size_t)kExperts * 3 * w3 + m * w5;
  };
  std::printf("skew %.2f\n", skew);
  for (int rm : rms)
    for (int R : {4096, 8192, 16384, 32768}) {
      const auto count = Counts(R, skew, 7 + R);
      const int T = 16 * rm;
      std::vector<Exl3GemmJob> gu, dn;
      std::vector<Exl3GemmItem> igu, idn;
      size_t off = 0;
      int maxm = 0;
      for (int e = 0; e <= kExperts; ++e) {
        const int M = count[e], K = e < kExperts ? 3 : 5;
        maxm = std::max(maxm, e < kExperts ? M : 0);
        gu.push_back({wptr(e, 0), xg + off * kHidden, pg + off * kMoeInter, nullptr, kHidden, kMoeInter, M, 1, K, M});
        gu.push_back({wptr(e, 1), xu + off * kHidden, pu + off * kMoeInter, nullptr, kHidden, kMoeInter, M, 1, K, M});
        dn.push_back({wptr(e, 2), xd + off * kMoeInter, pd + off * kHidden, nullptr,
                      kMoeInter, kHidden, M, 1, K, M});
        for (int m = 0; m < M; m += T) {
          for (int b = 0; b < 10 / kMoeNb; ++b) {
            igu.push_back({e * 2, 0, m, b * 4 * kMoeNb});
            igu.push_back({e * 2 + 1, 0, m, b * 4 * kMoeNb});
          }
          for (int b = 0; b < 40 / kMoeNb; ++b) idn.push_back({e, 0, m, b * 4 * kMoeNb});
        }
        off += M;
      }
      Buffer<Exl3GemmJob> dgu(gu), ddn(dn);
      Buffer<Exl3GemmItem> digu(igu), didn(idn);
      Buffer<int> ngu(std::vector<int>{(int)igu.size()}), ndn(std::vector<int>{(int)idn.size()});
      const PrefillGemmShape g{rm, true};
      const double tg = Time([&] { LaunchPrefillMoeGemm(g, dgu.p, digu.p, ngu.p); }, 7);
      const double td = Time([&] { LaunchPrefillMoeGemm(g, ddn.p, didn.p, ndn.p, false, PrefillMoeHalfOut(g)); }, 7);
      const double f = 4096.0 / R;
      std::printf("rm %d R %5d max_expert_rows %5d items %6zu/%6zu  gateup %8.1f us  down %8.1f us  "
                  "per4096: gateup %7.1f down %7.1f sum %7.1f us\n",
                  rm, R, maxm, igu.size(), idn.size(), tg, td, tg * f, td * f, (tg + td) * f);
      std::fflush(stdout);
    }
  return 0;
}
