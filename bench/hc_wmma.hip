// HC read on WMMA (engine/layers/hc_wmma.h) against the VALU kernels, R = 1..16.
//
//   hc_wmma [sites] [calls]
//
// 1. Exactness across R: each WMMA variant runs R = 1..16 on the same H, and
//    every row of x and post must be bitwise equal to the R = 16 run.
// 2. Accuracy at R = 4 against an fp64 host reference, for the VALU path and
//    each WMMA variant (x before rounding to fp16 is not available, so the
//    error includes the final fp16 rounding).
// 3. Time per site call, cycling `sites` cold weight sets (13.3 MB each).
#include <chrono>
#include <cmath>
#include <cstring>
#include <vector>

#include "../engine/layers/hc_wmma.h"
#include "../engine/exl3/plans.h"
#include "../engine/layers/kernels/hc_multirow.h"

using namespace crucible;

namespace {

constexpr int kM = kRank + kStreams;  // inject site

__global__ void Fill(__half* p, size_t n, uint32_t seed, float scale) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    uint32_t z = (uint32_t)(i * 2654435761u) ^ seed;
    z ^= z >> 13;
    z *= 0x5bd1e995u;
    z ^= z >> 15;
    p[i] = __float2half(((z & 0xffff) / 65536.f - 0.5f) * scale);
  }
}

__global__ void FillF(float* p, size_t n, uint32_t seed, float scale, float bias) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    uint32_t z = (uint32_t)(i * 2246822519u) ^ seed;
    z ^= z >> 15;
    z *= 0x2c1b3c6du;
    z ^= z >> 12;
    p[i] = ((z & 0xffffff) / 16777216.f - 0.5f) * scale + bias;
  }
}

struct Site {
  __half *fn, *up;
  float* w1;
};

struct Bufs {
  float *H, *x, *post, *dots, *ss;
  HcWScratch sc;
};

unsigned g_epoch = 0;

template <int R>
void ValuR(const Site& w, const Bufs& b) {
  if constexpr (R >= 4) {
    HcRowsDotsShared<R, false, 2><<<dim3((kM + 15) / 16 + 1, kStreams), 256>>>(w.fn, kM, b.H, b.dots, b.ss);
    HcRowsMix<R, HcMixMode::Ordered><<<kHidden / 16, 256>>>(w.up, w.w1, b.dots, kM, b.ss, b.H, b.x, b.post);
  } else {
    HcDotsKernel<R><<<((kM + 1) * kStreams + 7) / 8, 256>>>(w.fn, kM, b.H, b.dots, b.ss);
    HcMixKernel<R><<<kHidden / 16, 256>>>(w.up, w.w1, b.dots, kM, b.ss, b.H, b.x, b.post);
  }
}

// VALU phases alone at R = 1 (dots, then mix on the same dots).
void ValuDots(int R, const Site& w, const Bufs& b) {
  HcDotsKernel<1><<<((kM + 1) * kStreams + 7) / 8, 256>>>(w.fn, kM, b.H, b.dots, b.ss);
}

void ValuMix(int R, const Site& w, const Bufs& b) {
  HcMixKernel<1><<<kHidden / 16, 256>>>(w.up, w.w1, b.dots, kM, b.ss, b.H, b.x, b.post);
}

void Valu(int R, const Site& w, const Bufs& b) {
  switch (R) {
    case 1: ValuR<1>(w, b); break;
    case 2: ValuR<2>(w, b); break;
    case 3: ValuR<3>(w, b); break;
    case 4: ValuR<4>(w, b); break;
    case 5: ValuR<5>(w, b); break;
    case 6: ValuR<6>(w, b); break;
    case 7: ValuR<7>(w, b); break;
    default: ValuR<8>(w, b); break;
  }
}

template <int KS, int PF>
void Wmma(int R, const Site& w, const Bufs& b) {
  HcWmmaKernel<KS, PF><<<HcWDotsBlocks<KS>() + kHcWMixBlocks, 256, HcWLds<KS>(R)>>>(w.fn, kM, w.up, w.w1, b.H, R, b.x, b.post,
                                                                     b.sc, ++g_epoch);
}

// Phases alone: dots and mid (no mix blocks), and the mix after the flag.
template <int KS, int PF>
void WmmaDots(int R, const Site& w, const Bufs& b) {
  HcWmmaKernel<KS, PF><<<HcWDotsBlocks<KS>(), 256, HcWLds<KS>(R)>>>(w.fn, kM, w.up, w.w1, b.H, R, b.x, b.post, b.sc, ++g_epoch);
}

template <int KS, int PF>
void WmmaMix(int R, const Site& w, const Bufs& b) {
  HcWmmaKernel<KS, PF><<<kHcWMixBlocks, 256, HcWLds<KS>(R)>>>(w.fn, kM, w.up, w.w1, b.H, R, b.x, b.post, b.sc, g_epoch,
                                              HcWDotsBlocks<KS>());
}

using Fn = void (*)(int, const Site&, const Bufs&);

struct Variant {
  const char* name;
  Fn fn;
  int rmax;
};

double Now() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

std::vector<float> Get(const float* d, size_t n) {
  std::vector<float> h(n);
  HIP_OK(hipMemcpy(h.data(), d, n * 4, hipMemcpyDeviceToHost));
  return h;
}

std::vector<__half> GetH(const __half* d, size_t n) {
  std::vector<__half> h(n);
  HIP_OK(hipMemcpy(h.data(), d, n * 2, hipMemcpyDeviceToHost));
  return h;
}

// fp64 reference of x (before the final fp16 rounding) and post.
void Reference(const Site& w, const std::vector<float>& H, int R, std::vector<double>& x, std::vector<double>& post) {
  const auto fn = GetH(w.fn, (size_t)kM * kWide);
  const auto up = GetH(w.up, (size_t)kWide * kRank);
  const auto w1 = Get(w.w1, kWide);
  x.assign((size_t)R * kHidden, 0);
  post.assign((size_t)R * kStreams, 0);
  for (int r = 0; r < R; ++r) {
    double inv[kStreams];
    for (int s = 0; s < kStreams; ++s) {
      double a = 0;
      for (int d = 0; d < kHidden; ++d) a += (double)H[(size_t)r * kWide + s * kHidden + d] * H[(size_t)r * kWide + s * kHidden + d];
      inv[s] = 1.0 / std::sqrt(a / kHidden + kEps);
    }
    std::vector<double> t(kRank);
    for (int m = 0; m < kM; ++m) {
      double a = 0;
      for (int s = 0; s < kStreams; ++s) {
        double d = 0;
        for (int k = 0; k < kHidden; ++k)
          d += (double)__half2float(fn[(size_t)m * kWide + s * kHidden + k]) * H[(size_t)r * kWide + s * kHidden + k];
        a += inv[s] * d;
      }
      a *= 0.25;
      if (m < kRank) t[m] = a / (1.0 + std::exp(-a));
      else post[r * kStreams + m - kRank] = 2.0 / (1.0 + std::exp(-a));
    }
    for (int d = 0; d < kHidden; ++d) {
      double v = 0;
      for (int s = 0; s < kStreams; ++s) {
        double g = 0;
        for (int m = 0; m < kRank; ++m) g += (double)__half2float(up[((size_t)s * kHidden + d) * kRank + m]) * t[m];
        v += 1.0 / (1.0 + std::exp(-g)) * H[(size_t)r * kWide + s * kHidden + d] * inv[s] * w1[s * kHidden + d];
      }
      x[(size_t)r * kHidden + d] = v * 0.25;
    }
  }
}

}  // namespace

int main(int argc, char** argv) {
  const int nsites = argc > 1 ? std::atoi(argv[1]) : 24;
  const int calls = argc > 2 ? std::atoi(argv[2]) : 96;
  std::vector<Site> sites(nsites);
  for (int i = 0; i < nsites; ++i) {
    HIP_OK(hipMalloc(&sites[i].fn, sizeof(__half) * kM * kWide));
    HIP_OK(hipMalloc(&sites[i].up, sizeof(__half) * kWide * kRank));
    HIP_OK(hipMalloc(&sites[i].w1, sizeof(float) * kWide));
    Fill<<<1024, 256>>>(sites[i].fn, (size_t)kM * kWide, 17 + i, 0.04f);
    Fill<<<1024, 256>>>(sites[i].up, (size_t)kWide * kRank, 91 + i, 0.25f);
    FillF<<<64, 256>>>(sites[i].w1, kWide, 7 + i, 0.4f, 1.f);
  }
  Bufs b{};
  HIP_OK(hipMalloc(&b.H, sizeof(float) * kHcWRows * kWide));
  HIP_OK(hipMalloc(&b.x, sizeof(float) * kHcWRows * kHidden));
  HIP_OK(hipMalloc(&b.post, sizeof(float) * kHcWRows * kStreams));
  HIP_OK(hipMalloc(&b.dots, sizeof(float) * kHcWRows * kM * kStreams));
  HIP_OK(hipMalloc(&b.ss, sizeof(float) * kHcWRows * kStreams));
  HIP_OK(hipMalloc(&b.sc.part, sizeof(float) * HcWPartFloats<8>()));
  HIP_OK(hipMalloc(&b.sc.pss, sizeof(float) * 8 * kStreams * kHcWRows));
  HIP_OK(hipMalloc(&b.sc.inv, sizeof(float) * kHcWRows * kStreams));
  HIP_OK(hipMalloc(&b.sc.thi, sizeof(__half) * kHcWRows * kRank));
  HIP_OK(hipMalloc(&b.sc.tlo, sizeof(__half) * kHcWRows * kRank));
  HIP_OK(hipMalloc(&b.sc.count, sizeof(unsigned) * 2));
  HIP_OK(hipMemset(b.sc.count, 0, sizeof(unsigned) * 2));
  b.sc.flag = b.sc.count + 1;
  // Residual streams: a few large channels, like the model's.
  FillF<<<256, 256>>>(b.H, (size_t)kHcWRows * kWide, 5, 2.f, 0.f);
  HIP_OK(hipDeviceSynchronize());
  const std::vector<float> H = Get(b.H, (size_t)kHcWRows * kWide);

  const std::vector<Variant> vars = {
      {"valu", Valu, 8},
      {"valu dots", ValuDots, 99},
      {"valu mix", ValuMix, 99},
      {"wmma ks4p20", Wmma<4, 20>, 16},
      {"wmma ks8p20", Wmma<8, 20>, 16},
      {"ks4p20 dots", WmmaDots<4, 20>, 99},
      {"ks8p20 dots", WmmaDots<8, 20>, 99},
      {"ks4 mix", WmmaMix<4, 8>, 99},
  };

  // 1. Exactness across R.
  for (const Variant& v : vars) {
    if (v.rmax != 16) continue;
    v.fn(16, sites[0], b);
    HIP_OK(hipDeviceSynchronize());
    const auto x16 = Get(b.x, (size_t)16 * kHidden);
    const auto p16 = Get(b.post, 16 * kStreams);
    int bad = 0;
    for (int R = 1; R < 16; ++R) {
      HIP_OK(hipMemset(b.x, 0xff, sizeof(float) * kHcWRows * kHidden));
      v.fn(R, sites[0], b);
      HIP_OK(hipDeviceSynchronize());
      const auto x = Get(b.x, (size_t)R * kHidden);
      const auto p = Get(b.post, R * kStreams);
      if (std::memcmp(x.data(), x16.data(), x.size() * 4) || std::memcmp(p.data(), p16.data(), p.size() * 4)) ++bad;
    }
    // FNV-1a over the R=16 x and post bytes: compares kernel versions.
    uint64_t h = 1469598103934665603ull;
    auto mix = [&](const void* p, size_t n) {
      for (size_t i = 0; i < n; ++i) h = (h ^ static_cast<const unsigned char*>(p)[i]) * 1099511628211ull;
    };
    mix(x16.data(), x16.size() * 4);
    mix(p16.data(), p16.size() * 4);
    std::printf("%-9s rows R=1..15 vs R=16: %s, R=16 hash %016llx\n", v.name, bad ? "DIFFERENT" : "bitwise equal",
                (unsigned long long)h);
  }

  // 2. Accuracy at R=4.
  {
    const int R = 4;
    std::vector<double> xr, pr;
    Reference(sites[0], H, R, xr, pr);
    double xmax = 0;
    for (double v : xr) xmax = std::max(xmax, std::fabs(v));
    for (const Variant& v : vars) {
      if (v.rmax == 99) continue;
      v.fn(R, sites[0], b);
      HIP_OK(hipDeviceSynchronize());
      const auto x = Get(b.x, (size_t)R * kHidden);
      const auto p = Get(b.post, R * kStreams);
      double emax = 0, esum = 0, pmax = 0;
      for (size_t i = 0; i < x.size(); ++i) {
        const double e = std::fabs(x[i] - xr[i]);
        emax = std::max(emax, e);
        esum += e;
      }
      for (size_t i = 0; i < p.size(); ++i) pmax = std::max(pmax, std::fabs(p[i] - pr[i]));
      std::printf("%-9s R=4 vs fp64: x max err %.3e (|x| max %.3f), mean %.3e; post max err %.3e\n", v.name, emax,
                  xmax, esum / x.size(), pmax);
    }
  }

  // 3. Time per site call.
  const double mb = (double)sizeof(__half) * (kM * kWide + kWide * kRank) / 1e6;
  std::printf("site %.2f MB, floor %.1f us at 245.5 GB/s; %d sites, %d calls\n", mb, mb / 245.5e3 * 1e6, nsites, calls);
  for (int R : {1, 4, 8, 16}) {
    std::printf("R=%2d", R);
    for (const Variant& v : vars) {
      if (R > v.rmax) continue;
      for (int i = 0; i < nsites; ++i) v.fn(R, sites[i], b);
      HIP_OK(hipDeviceSynchronize());
      const double t0 = Now();
      for (int i = 0; i < calls; ++i) v.fn(R, sites[i % nsites], b);
      HIP_OK(hipDeviceSynchronize());
      const double us = (Now() - t0) / calls * 1e6;
      std::printf("  %s %.1f us (%.0f%%)", v.name, us, mb / 245.5e3 * 1e6 / us * 100);
    }
    std::printf("\n");
    std::fflush(stdout);
  }
  return 0;
}
