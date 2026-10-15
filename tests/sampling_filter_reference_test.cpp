// Keep the simple full-sort algorithm as an oracle for faster nucleus filtering.
#include "../engine/runtime/sampling.h"
#include <cassert>
#include <cstdio>
#include <numeric>

using namespace crucible::sampling;

static Distribution Reference(std::span<const float> logits, const Options& options, std::span<const int> mapping) {
  std::vector<Candidate> candidates;
  Candidate best{-1, -INFINITY};
  for (size_t i = 0; i < logits.size(); ++i) {
    if (!std::isfinite(logits[i])) continue;
    Candidate item{mapping.empty() ? int(i) : mapping[i], logits[i]};
    if (best.token < 0 || Higher(item, best)) best = item;
    candidates.push_back(item);
  }
  if (options.temperature == 0) return Distribution::Point(best.token);
  if (options.top_k > 0 && size_t(options.top_k) < candidates.size()) {
    std::nth_element(candidates.begin(), candidates.begin() + options.top_k, candidates.end(), Higher);
    candidates.resize(options.top_k);
  }
  double sum = 0;
  for (Candidate& item : candidates) { item.value = std::exp((item.value - best.value) / options.temperature); sum += item.value; }
  if (options.top_p < 1) {
    std::sort(candidates.begin(), candidates.end(), Higher);
    size_t keep = 0;
    double prefix = 0;
    do { prefix += candidates[keep++].value; }
    while (keep < candidates.size() && prefix < options.top_p * sum);
    candidates.resize(keep);
  }
  std::vector<Candidate> kept;
  for (const Candidate& item : candidates) if (item.value > 0 && item.value >= options.min_p) kept.push_back(item);
  std::sort(kept.begin(), kept.end(), [](const Candidate& a, const Candidate& b) { return a.token < b.token; });
  std::vector<int> ids;
  std::vector<double> weights;
  for (const Candidate& item : kept) { ids.push_back(item.token); weights.push_back(item.value); }
  return Distribution(std::move(ids), std::move(weights));
}

int main() {
  std::mt19937_64 random(7654321);
  size_t cases = 0;
  for (int size : {7, 64, 65, 129, 1024, 10000}) {
    std::vector<float> logits(size);
    std::vector<int> mapping(size);
    std::iota(mapping.begin(), mapping.end(), 0);
    std::shuffle(mapping.begin(), mapping.end(), random);
    for (float& value : logits) value = float(int(random() % 101) - 50) / 10;
    for (double temperature : {0., 0.0001, 0.7, 2.})
      for (int top_k : {0, 1, 50})
        for (double top_p : {0., 0.1, 0.5, 0.95, 0.999999, 1.})
          for (double min_p : {0., 0.05, 1.}) {
            Options options;
            options.temperature = temperature; options.top_k = top_k; options.top_p = top_p; options.min_p = min_p;
            const auto expected = Reference(logits, options, mapping);
            const auto actual = FromLogits(logits, options, mapping);
            assert(expected.Ids() == actual.Ids());
            assert(expected.Probabilities() == actual.Probabilities());
            ++cases;
          }
  }
  std::printf("PASS: %zu filter cases match the full-sort reference bit for bit\n", cases);
}
