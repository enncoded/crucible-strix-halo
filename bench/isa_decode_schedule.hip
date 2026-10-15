// Compare existing decode bodies without changing engine dispatch or arithmetic.
#include "common.h"
#include "kernels/moe_rows.h"
#include <cmath>
#include <cstring>
#include <random>
#include <utility>

using namespace crucible;

// By-value descriptors keep dispatch overhead identical between the two bodies.
template<int K, int R, bool Scheduled>
__global__ void __launch_bounds__(256) IsaDecodeBody(LinJob job) {
  extern __shared__ uint32_t xs[];
  const GemvItem item{0, 0, int(blockIdx.x) * 8, 8};
  if constexpr (Scheduled) MoeGemvBody<K, R, 8>(job, item, xs);
  else LinGemvBody<K, R>(job, item, xs);
}

template<class T> struct Device {
  T* ptr = nullptr;
  explicit Device(size_t count) { HIP_CHECK(hipMalloc(&ptr, count * sizeof(T))); }
  ~Device() { (void)hipFree(ptr); }
  Device(const Device&) = delete;
  Device& operator=(const Device&) = delete;
};

template<int K, int R>
std::pair<double, double> Measure(LinJob job, size_t lds) {
  std::vector<double> base_us, scheduled_us;
  bench::EventTimer timer;
  constexpr int repeats = 20;
  for (int sample = 0; sample < 8; ++sample) {
    for (int order = 0; order < 2; ++order) {
      const bool scheduled = ((sample + order) & 1) != 0;
      timer.Begin();
      for (int repeat = 0; repeat < repeats; ++repeat) {
        if (scheduled) IsaDecodeBody<K, R, true><<<job.O / 128, 256, lds>>>(job);
        else IsaDecodeBody<K, R, false><<<job.O / 128, 256, lds>>>(job);
      }
      HIP_CHECK(hipGetLastError());
      (scheduled ? scheduled_us : base_us).push_back(timer.EndMs() * 1000. / repeats);
    }
  }
  return {bench::Median(base_us), bench::Median(scheduled_us)};
}

template<int K, int R>
bool Check(int span, int O = 4096) {
  const int I = span * 16;
  const size_t nw = size_t(span) * (O / 16) * (8 * K);
  std::mt19937 random(20261003 + K * 8 + R);
  std::vector<uint32_t> weights(nw);
  std::vector<__half> input(size_t(R) * I);
  for (auto& w : weights) w = random();
  for (auto& x : input) x = __float2half(float(int(random() % 2049) - 1024) / 1024.f);
  Device<uint32_t> dw(nw);
  Device<__half> dx(input.size());
  Device<float> dy(size_t(R) * O);
  HIP_CHECK(hipMemcpy(dw.ptr, weights.data(), nw * sizeof(uint32_t), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(dx.ptr, input.data(), input.size() * sizeof(__half), hipMemcpyHostToDevice));
  LinJob job{};
  job.tiles = dw.ptr; job.xh = dx.ptr; job.partial = dy.ptr;
  job.I = I; job.O = O; job.K = K; job.S = 1; job.R = R;
  const size_t lds = size_t(R) * span * 32;
  std::vector<float> baseline(size_t(R) * O), scheduled(baseline.size());
  HIP_CHECK(hipMemset(dy.ptr, 0xff, baseline.size() * sizeof(float)));
  IsaDecodeBody<K, R, false><<<O / 128, 256, lds>>>(job);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipMemcpy(baseline.data(), dy.ptr, baseline.size() * sizeof(float), hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemset(dy.ptr, 0xff, scheduled.size() * sizeof(float)));
  IsaDecodeBody<K, R, true><<<O / 128, 256, lds>>>(job);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipMemcpy(scheduled.data(), dy.ptr, scheduled.size() * sizeof(float), hipMemcpyDeviceToHost));
  const bool same = std::memcmp(baseline.data(), scheduled.data(), baseline.size() * sizeof(float)) == 0;
  bool finite = true;
  for (const float value : baseline) finite = finite && std::isfinite(value);
  const auto [base_us, scheduled_us] = Measure<K, R>(job, lds);
  std::printf("K=%d R=%d span=%d O=%d bitwise=%d finite=%d baseline_us=%.3f scheduled_us=%.3f\n",
              K, R, span, O, same, finite, base_us, scheduled_us);
  return same && finite;
}

template<int K> bool Cases() {
  bool pass = true;
  for (int span : {20, 40, 160}) {
    pass = Check<K, 1>(span) && pass;
    pass = Check<K, 8>(span) && pass;
  }
  return pass;
}

template<int R=1> bool K5Sweep() {
  bool pass = true;
  for (int span : {8, 12, 16, 20, 24, 28, 32, 40, 48, 64, 80, 96, 128, 160})
    pass = Check<5, R>(span) && pass;
  if constexpr (R < 8) pass = K5Sweep<R + 1>() && pass;
  return pass;
}

template<int R=1> bool K5Shapes() {
  bool pass = true;
  for (int O : {640, 2560, 10240})
    for (int span : {20, 40, 160}) pass = Check<5, R>(span, O) && pass;
  if constexpr (R < 8) pass = K5Shapes<R + 1>() && pass;
  return pass;
}

int main(int argc, char** argv) {
  if (argc > 2 || (argc == 2 && std::strcmp(argv[1], "k5-sweep") &&
                  std::strcmp(argv[1], "k5-shapes"))) return 2;
  std::puts("Synthetic cache-local schedule check. This is not engine gate evidence or cold bandwidth.");
  std::puts("Timing: eight paired samples, alternating order, twenty launches per sample.");
  if (argc == 2) return (std::strcmp(argv[1], "k5-sweep") == 0 ? K5Sweep() : K5Shapes()) ? 0 : 1;
  bool pass = Cases<3>();
  pass = Cases<4>() && pass;
  pass = Cases<5>() && pass;
  pass = Cases<6>() && pass;
  return pass ? 0 : 1;
}
