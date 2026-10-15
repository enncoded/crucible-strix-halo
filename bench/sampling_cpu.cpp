// CPU probability work can be measured without loading a model or taking the GPU.
#ifndef SAMPLING_HEADER
#define SAMPLING_HEADER "../engine/runtime/sampling.h"
#endif
#include SAMPLING_HEADER
#include "../engine/model/flashnext/geometry.h"
#include <chrono>
#include <cstdio>

using namespace crucible;
using namespace crucible::sampling;

static double Seconds() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

int main() {
  std::vector<float> logits(kVocab);
  std::mt19937_64 values(1234);
  for (float& value : logits) value = -14.f + float(values() % 1000000) * 1e-5f;
  logits[kEos] = 10;
  Rng rng(42);
  for (int mode = 0; mode < 4; ++mode) {
    Options options;
    options.temperature = 0.7;
    if (mode == 1) options.top_k = 50;
    if (mode == 2) options.top_p = 0.95;
    if (mode == 3) { options.top_k = 50; options.top_p = 0.95; options.min_p = 0.01; }
    const char* names[] = {"unfiltered", "top_k50", "top_p95", "all_filters"};
    std::vector<double> milliseconds;
    long checksum = 0;
    for (int repeat = 0; repeat < 41; ++repeat) {
      const double start = Seconds();
      const auto distribution = FromLogits(logits, options);
      checksum += distribution.Sample(rng);
      const double elapsed = (Seconds() - start) * 1000;
      if (repeat) milliseconds.push_back(elapsed);
    }
    std::sort(milliseconds.begin(), milliseconds.end());
    std::printf("%s V=%d median %.6f ms, checksum %ld\n", names[mode], kVocab,
                (milliseconds[19] + milliseconds[20]) * 0.5, checksum);
  }
}
