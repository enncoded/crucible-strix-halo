#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace crucible_bench {
using PairSample=std::array<double,4>;  // Baseline, candidate, candidate, baseline.
struct PairResult {
  double baseline,candidate,delta,minimum_delta,maximum_delta;
};
inline double PairMedian(std::vector<double> values) {
  if(values.empty())throw std::invalid_argument("empty pair samples");
  std::sort(values.begin(),values.end());
  const size_t n=values.size();
  return n%2?values[n/2]:(values[n/2-1]+values[n/2])*.5;
}
inline PairResult PairStats(const std::vector<PairSample>& samples) {
  if(samples.empty())throw std::invalid_argument("empty pair samples");
  std::vector<double> baseline,candidate,delta;
  for(const auto& sample:samples) {
    for(double ms:sample)
      if(!std::isfinite(ms)||ms<=0)throw std::invalid_argument("invalid pair duration");
    baseline.insert(baseline.end(),{sample[0],sample[3]});
    candidate.insert(candidate.end(),{sample[1],sample[2]});
    // ABBA cancels linear drift within each round. Reduce the paired deltas.
    delta.push_back((sample[0]+sample[3]-sample[1]-sample[2])*.5);
  }
  return {PairMedian(baseline),PairMedian(candidate),PairMedian(delta),
          *std::min_element(delta.begin(),delta.end()),*std::max_element(delta.begin(),delta.end())};
}
}  // namespace crucible_bench
