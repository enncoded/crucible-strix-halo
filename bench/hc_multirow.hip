// Compare every result with independent R=1 calls, then time cold weight sets.
#include <cstring>
#include <random>

#include "kernels/hc_multirow.h"
#include "common.h"

using namespace crucible;

namespace {
constexpr int kSets = 16;
constexpr int kM = kRank + kStreams;

template <typename T> T* Put(const std::vector<T>& v) {
  T* p;
  HIP_OK(hipMalloc(&p, v.size() * sizeof(T)));
  HIP_OK(hipMemcpy(p, v.data(), v.size() * sizeof(T), hipMemcpyHostToDevice));
  return p;
}

template <typename T> std::vector<T> Get(T* p, int n) {
  std::vector<T> v(n);
  HIP_OK(hipMemcpy(v.data(), p, n * sizeof(T), hipMemcpyDeviceToHost));
  return v;
}

struct Buffers {
  __half *fn, *up;
  float *H, *w1, *dots, *ss, *mixed, *post;
};

template <int R> void Dots(Buffers b, int M, int mode) {
  if (mode == 0)
    HcDotsKernel<R><<<((M + 1) * kStreams + 7) / 8, 256>>>(b.fn, M, b.H, b.dots, b.ss);
  else if (mode == 1)
    HcRowsDotsRegister<R><<<((M + 1) * kStreams + 7) / 8, 256>>>(b.fn, M, b.H, b.dots, b.ss);
  else if (mode == 2)
    HcRowsDotsShared<R><<<dim3((M + 7) / 8 + 1, kStreams), 256>>>(b.fn, M, b.H, b.dots, b.ss);
  else if (mode == 3)
    HcRowsDotsShared<R, false><<<dim3((M + 7) / 8 + 1, kStreams), 256>>>(b.fn, M, b.H, b.dots, b.ss);
  else if (mode == 4)
    HcRowsDotsShared<R, false, 2><<<dim3((M + 15) / 16 + 1, kStreams), 256>>>(b.fn, M, b.H, b.dots, b.ss);
  else if (mode == 5)
    HcRowsDotsShared<R, true, 2><<<dim3((M + 15) / 16 + 1, kStreams), 256>>>(b.fn, M, b.H, b.dots, b.ss);
  else
    HcRowsDotsShared<R, false, 4><<<dim3((M + 31) / 32 + 1, kStreams), 256>>>(b.fn, M, b.H, b.dots, b.ss);
}

template <int R> void Mix(Buffers b, int M, int mode) {
  if (mode == 0)
    HcMixKernel<R><<<kHidden / 16, 256>>>(b.up, b.w1, b.dots, M, b.ss, b.H, b.mixed, M > kRank ? b.post : nullptr);
  else if (mode == 1)
    HcRowsMix<R><<<kHidden / 16, 256>>>(b.up, b.w1, b.dots, M, b.ss, b.H, b.mixed, M > kRank ? b.post : nullptr);
  else if (mode == 2)
    HcRowsMix<R, HcMixMode::Parallel><<<kHidden / 16, 256>>>(b.up, b.w1, b.dots, M, b.ss, b.H, b.mixed, M > kRank ? b.post : nullptr);
  else if (mode == 3)
    HcRowsMix<R, HcMixMode::Pair><<<kHidden / 16, 256>>>(b.up, b.w1, b.dots, M, b.ss, b.H, b.mixed, M > kRank ? b.post : nullptr);
  else if (mode == 4)
    HcRowsMix<R, HcMixMode::Quad><<<kHidden / 16, 256>>>(b.up, b.w1, b.dots, M, b.ss, b.H, b.mixed, M > kRank ? b.post : nullptr);
  else
    HcRowsMix<R, HcMixMode::Ordered><<<kHidden / 16, 256>>>(b.up, b.w1, b.dots, M, b.ss, b.H, b.mixed, M > kRank ? b.post : nullptr);
}

bool Same(const char* name, const std::vector<float>& ref, float* got) {
  const auto values = Get(got, ref.size());
  if (std::memcmp(ref.data(), values.data(), ref.size() * sizeof(float)) == 0) return true;
  int count = 0, first = -1;
  for (int i = 0; i < (int)ref.size(); ++i)
    if (std::memcmp(&ref[i], &values[i], 4)) { if (first < 0) first = i; ++count; }
  std::printf("FAIL %s: %d/%zu values, first %d %.9g != %.9g\n",
              name, count, ref.size(), first, ref[first], values[first]);
  return false;
}

template <int R> bool Exact(Buffers b, int M) {
  for (int r = 0; r < R; ++r) {
    Buffers row = b;
    row.H += r * kWide;
    row.dots += r * M * kStreams;
    row.ss += r * kStreams;
    row.mixed += r * kHidden;
    row.post += r * kStreams;
    Dots<1>(row, M, 0);
    Mix<1>(row, M, 0);
  }
  const auto dots = Get(b.dots, R * M * kStreams), ss = Get(b.ss, R * kStreams);
  const auto mixed = Get(b.mixed, R * kHidden), post = Get(b.post, R * kStreams);
  bool ok = true;
  for (int mode = 0; mode < 7; ++mode) {
    Dots<R>(b, M, mode);
    ok &= Same("dots", dots, b.dots);
    ok &= Same("ss", ss, b.ss);
    Mix<R>(b, M, std::min(mode, 5));
    ok &= Same("mixed", mixed, b.mixed);
    ok &= Same("post", post, b.post);
    std::printf("exact R=%d M=%d mode=%d %s\n", R, M, mode, ok ? "PASS" : "FAIL");
  }
  return ok;
}

template <int R> void Timing(Buffers b) {
  for (int op = 0; op < 2; ++op) {
    for (int mode = 0; mode < (op == 0 ? 7 : 6); ++mode) {
      std::vector<double> samples;
      for (int repeat = 0; repeat < 10; ++repeat) {
        bench::EventTimer timer;
        timer.Begin();
        for (int set = 0; set < kSets; ++set) {
          Buffers weights = b;
          weights.fn += (size_t)set * kM * kWide;
          weights.up += (size_t)set * kWide * kRank;
          if (op == 0) Dots<R>(weights, kM, mode);
          else Mix<R>(weights, kM, mode);
        }
        const double us = timer.EndMs() * 1000 / kSets;
        if (repeat) samples.push_back(us);
      }
      std::printf("time R=%d %s mode=%d %.3f us\n", R, op ? "mix" : "dots", mode, bench::Median(samples));
    }
  }
}

template <int R> bool Run(Buffers b) {
  bool ok = true;
  for (int set = 0; set < 4; ++set) {
    Buffers weights = b;
    weights.fn += (size_t)set * kM * kWide;
    weights.up += (size_t)set * kWide * kRank;
    std::printf("dataset R=%d set=%d\n", R, set);
    ok &= Exact<R>(weights, kRank);
    ok &= Exact<R>(weights, kM);
  }
  if (ok) Timing<R>(b);
  return ok;
}
}  // namespace

int main() {
  std::mt19937 rng(123);
  std::normal_distribution<float> weight(0.f, .02f), hidden(0.f, 2.f);
  std::vector<__half> fn((size_t)kSets * kM * kWide), up((size_t)kSets * kWide * kRank);
  for (auto& x : fn) x = __float2half(weight(rng));
  for (auto& x : up) x = __float2half(weight(rng));
  std::vector<float> H(kMaxRows * kWide), w1(kWide);
  for (auto& x : H) x = hidden(rng);
  for (auto& x : w1) x = 1.f + weight(rng);
  Buffers b{Put(fn), Put(up), Put(H), Put(w1), Put(std::vector<float>(kMaxRows * kM * kStreams)),
            Put(std::vector<float>(kMaxRows * kStreams)), Put(std::vector<float>(kMaxRows * kHidden)),
            Put(std::vector<float>(kMaxRows * kStreams))};
  bench::Sampler sampler;
  sampler.Start();
  bool ok = Run<1>(b);
  ok &= Run<2>(b); ok &= Run<3>(b); ok &= Run<4>(b);
  ok &= Run<5>(b); ok &= Run<6>(b); ok &= Run<7>(b); ok &= Run<8>(b);
  HIP_OK(hipDeviceSynchronize());
  sampler.Stop();
  std::puts(sampler.Summary().c_str());
  std::printf("all exact to R1: %s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
