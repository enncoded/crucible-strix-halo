#include <random>
#include <map>
#include "../engine/runtime/sampling.h"
#include <array>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <limits>
#include <string>
#include <vector>

using namespace crucible::sampling;

static bool Close(double a, double b) { return std::abs(a - b) < 1e-12; }

static void Filters() {
  Options options;
  options.temperature = 1;
  const float logits[] = {0, float(std::log(2.)), float(std::log(4.)), float(std::log(8.))};
  auto p = FromLogits(logits, options);
  assert(std::abs(p.Probability(3) - 8. / 15.) < 1e-7);
  options.top_k = 2;
  p = FromLogits(logits, options);
  assert(p.Probability(0) == 0 && p.Probability(1) == 0);
  assert(std::abs(p.Probability(3) - 2. / 3.) < 1e-7);
  options.top_p = 0.6;
  p = FromLogits(logits, options);
  assert(Close(p.Probability(3), 1));
  options.top_k = 0;
  options.top_p = 1;
  options.min_p = 0.49;
  p = FromLogits(logits, options);
  assert(p.Probability(1) == 0 && p.Probability(2) > 0);
  const float threshold_logits[] = {-1, 0};
  options.min_p = std::exp(-1.);
  assert(FromLogits(threshold_logits, options).Probability(0) > 0);
  options.min_p = 1;
  p = FromLogits(logits, options);
  assert(Close(p.Probability(3), 1));
  options = Options{};
  const float ties[] = {2, 2, 1};
  const int remap[] = {9, 3, 7};
  p = FromLogits(ties, options, remap);
  assert(Close(p.Probability(3), 1));
  options.temperature = 0.5;
  options.top_k = 1;
  p = FromLogits(ties, options, remap);
  assert(Close(p.Probability(3), 1));
  options.top_k = 0;
  options.top_p = 0;
  p = FromLogits(ties, options, remap);
  assert(Close(p.Probability(3), 1));
  const float masked[] = {-std::numeric_limits<float>::infinity(), 2};
  options.top_p = 1;
  assert(Close(FromLogits(masked, options).Probability(1), 1));
}

static void Invalid() {
  Options options;
  for (double bad : {-1., std::numeric_limits<double>::infinity(), std::nan("")}) {
    options.temperature = bad;
    assert(!options.Error(4).empty());
  }
  options = Options{};
  options.top_k = 5;
  assert(!options.Error(4).empty());
  options.top_k = 0;
  options.top_p = 1.1;
  assert(!options.Error(4).empty());
  options.top_p = 1;
  options.min_p = -0.1;
  assert(!options.Error(4).empty());
  bool threw = false;
  try { FromLogits(std::span<const float>{}, Options{}); }
  catch (const std::invalid_argument&) { threw = true; }
  assert(threw);
}

// a greedy head ID below zero (a NaN or +Inf logit row) must give the sampled path's error.
static std::string Message(auto&& call) {
  try { call(); } catch (const std::invalid_argument& error) { return error.what(); }
  return "";
}

static void NonfiniteHeadIds() {
  const float nan_row[] = {0, std::nanf(""), 1};
  const std::string sampled = Message([&] { FromLogits(nan_row, Options{}); });
  assert(sampled == "nonfinite model logits");
  for (const std::vector<int>& bad : {std::vector<int>{-1}, {5, -1, 3}, {7, 9, -1}})
    assert(Message([&] { RequireFiniteHeadIds(bad); }) == sampled);
  const std::vector<int> good{0, 1, 248319};
  assert(Message([&] { RequireFiniteHeadIds(good); }).empty());
  assert(Message([&] { RequireFiniteHeadIds(std::span<const int>{}); }).empty());
  std::vector<int> draft{-1, 7, -1, 0};
  ReplaceNonfiniteDraftIds(draft);
  assert((draft == std::vector<int>{0, 7, 0, 0}));
}

static void Determinism() {
  Rng first(123), second(123), other(456);
  bool differs = false;
  for (int i = 0; i < 100; ++i) {
    const double x = first.Unit();
    assert(x >= 0 && x < 1 && x == second.Unit());
    differs |= x != other.Unit();
  }
  assert(differs);
  const Distribution p({0, 1, 2}, {0.1, 0.3, 0.6});
  Rng a(42), b(42);
  for (int i = 0; i < 100; ++i) assert(p.Sample(a) == p.Sample(b));
}

static void Statistics(const Distribution& p, const Distribution& q, const char* name) {
  constexpr int count = 250000;
  std::array<int, 5> serial{}, speculative{};
  Rng reference(1234567), proposals(2345678), verification(3456789);
  for (int i = 0; i < count; ++i) {
    ++serial[p.Sample(reference)];
    const int draft = q.Sample(proposals);
    const auto result = Verify(p, q, draft, verification);
    ++speculative[result.token];
    if (result.accepted) assert(result.token == draft);
  }
  double serial_chi = 0, spec_chi = 0;
  for (int i = 0; i < 5; ++i) {
    const double expected = count * p.Probability(i);
    if (expected == 0) { assert(serial[i] == 0 && speculative[i] == 0); continue; }
    serial_chi += std::pow(serial[i] - expected, 2) / expected;
    spec_chi += std::pow(speculative[i] - expected, 2) / expected;
  }
  // For four degrees of freedom, 30 has a tail probability below 5e-6.
  assert(serial_chi < 30 && spec_chi < 30);
  std::printf("%s: serial chi2 %.4f, speculative chi2 %.4f\n", name, serial_chi, spec_chi);
}

static void Sequences(bool lookup) {
  constexpr int count = 150000;
  const std::array<Distribution, 3> p = {
      Distribution({0, 1, 2}, {0.7, 0.2, 0.1}), Distribution({0, 1, 2}, {0.1, 0.2, 0.7}),
      Distribution({0, 1, 2}, {0.3, 0.4, 0.3})};
  const std::array<Distribution, 3> q = lookup ? std::array<Distribution, 3>{
      Distribution::Point(1), Distribution::Point(2), Distribution::Point(0)} : std::array<Distribution, 3>{
      Distribution({0, 1, 2}, {0.1, 0.4, 0.5}), Distribution({0, 1, 2}, {0.6, 0.2, 0.2}),
      Distribution({0, 1, 2}, {0.2, 0.7, 0.1})};
  std::array<int, 9> observed{};
  Rng proposals(87123), verification(99327);
  for (int i = 0; i < count; ++i) {
    const int first = q[0].Sample(proposals);
    const int second = q[first].Sample(proposals);
    const auto a = Verify(p[0], q[0], first, verification);
    const int next = a.accepted ? Verify(p[first], q[first], second, verification).token : p[a.token].Sample(verification);
    ++observed[a.token * 3 + next];
  }
  double chi = 0;
  for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) {
    const double expected = count * p[0].Probability(i) * p[i].Probability(j);
    chi += std::pow(observed[i * 3 + j] - expected, 2) / expected;
  }
  assert(chi < 50);
  std::printf("%s sequence chi2 %.4f\n", lookup ? "lookup" : "MTP", chi);
}

// Keyed Gumbel-max draws follow the distribution: across positions with one seed, and across seeds at one position.
void KeyedLaw() {
  const Distribution p({3, 7, 11, 19, 40}, {0.05, 0.1, 0.2, 0.3, 0.35});
  constexpr int kDraws = 200000;
  const auto chi_square = [&](auto draw) {
    std::map<int, int> counts;
    for (int i = 0; i < kDraws; ++i) ++counts[draw(i)];
    double total = 0;
    for (size_t k = 0; k < p.Ids().size(); ++k) {
      const double expected = kDraws * p.Probabilities()[k];
      const double diff = counts[p.Ids()[k]] - expected;
      total += diff * diff / expected;
    }
    assert(counts.size() == p.Ids().size());
    return total;
  };
  // Four degrees of freedom: 30 is far beyond the 0.9999 quantile (23.5).
  assert(chi_square([&](int i) { return p.KeyedSample(1234, uint64_t(i)); }) < 30);
  assert(chi_square([&](int i) { return p.KeyedSample(uint64_t(i) * 7919 + 5, 42); }) < 30);
  // A filtered distribution from logits keeps the filter's support and weights.
  Options options; options.temperature = 0.7; options.top_k = 3; options.seed = 9;
  const std::vector<float> logits{1.f, 3.f, 2.f, -1.f, 2.5f, 0.f};
  const Distribution filtered = FromLogits(logits, options);
  std::map<int, int> counts;
  for (int i = 0; i < kDraws; ++i) ++counts[filtered.KeyedSample(options.seed, uint64_t(i))];
  assert(counts.size() == 3 && counts.count(1) && counts.count(2) && counts.count(4));
  for (size_t k = 0; k < filtered.Ids().size(); ++k)
    assert(std::fabs(double(counts[filtered.Ids()[k]]) / kDraws - filtered.Probabilities()[k]) < 0.005);
  // The same key gives the same draw; a point distribution always returns its token.
  assert(p.KeyedSample(5, 6) == p.KeyedSample(5, 6));
  assert(Distribution::Point(17).KeyedSample(1, 2) == 17);
  // A draft with the same key matches the target exactly when both choose the same token.
  const Distribution q({3, 7, 11, 19, 40}, {0.3, 0.25, 0.2, 0.15, 0.1});
  int matches = 0;
  for (int i = 0; i < kDraws; ++i) matches += q.KeyedSample(77, uint64_t(i)) == p.KeyedSample(77, uint64_t(i));
  double overlap = 0;
  for (size_t k = 0; k < p.Ids().size(); ++k) overlap += std::min(p.Probabilities()[k], q.Probabilities()[k]);
  // Shared-noise matching accepts at least (1 - TV) / (1 + TV) and at most 1 - TV, where 1 - TV = overlap.
  const double rate = double(matches) / kDraws, tv = 1 - overlap;
  assert(rate <= overlap + 0.005 && rate >= (1 - tv) / (1 + tv) - 0.005);
  std::printf("keyed match acceptance %.4f, rejection sampling %.4f\n", rate, overlap);
  // The pruned scan equals a full Gumbel-max scan on a large, heavy-tailed support.
  std::vector<int> ids(60000);
  std::vector<double> weights(ids.size());
  for (size_t i = 0; i < ids.size(); ++i) { ids[i] = int(i) * 3 + 1; weights[i] = 1.0 / std::pow(double(i % 997 + 1), 1.3); }
  const Distribution wide(ids, weights);
  const auto full = [&](uint64_t seed, uint64_t position) {
    int best = wide.Ids().front();
    double score = -std::numeric_limits<double>::infinity();
    for (size_t i = 0; i < wide.Ids().size(); ++i) {
      const double value = wide.Probabilities()[i] / -std::log(KeyedUniform(seed, position, wide.Ids()[i]));
      if (value > score) { score = value; best = wide.Ids()[i]; }
    }
    return best;
  };
  for (uint64_t position = 0; position < 300; ++position) assert(wide.KeyedSample(11, position) == full(11, position));
  assert(Distribution::kMinExponential > 0 && Distribution::kMinExponential < 1e-15);
  assert(KeyedUniform(0, 0, 0) > 0 && KeyedUniform(0, 0, 0) < 1);
}

// The one-pass top-k equals the full-list selection it replaces: ties, -inf, mappings, and every filter.
void FastTopK() {
  const auto reference = [](std::span<const float> logits, const Options& o, std::span<const int> mapping) {
    auto candidates = AllCandidates(logits, mapping);
    const Candidate best = *std::min_element(candidates.begin(), candidates.end(), Higher);
    if (size_t(o.top_k) < candidates.size()) {
      std::nth_element(candidates.begin(), candidates.begin() + o.top_k, candidates.end(), Higher);
      candidates.resize(o.top_k);
    }
    // Canonical order, as the one-pass selection returns it.
    std::sort(candidates.begin(), candidates.end(), Higher);
    double sum = 0;
    for (Candidate& item : candidates) { item.value = std::exp((item.value - best.value) / o.temperature); sum += item.value; }
    if (o.top_p < 1) Nucleus(candidates, sum, o.top_p);
    std::vector<Candidate> kept;
    for (const Candidate& item : candidates) if (item.value > 0 && item.value >= o.min_p) kept.push_back(item);
    std::sort(kept.begin(), kept.end(), [](const Candidate& a, const Candidate& b) { return a.token < b.token; });
    std::vector<int> ids; std::vector<double> weights;
    for (const Candidate& item : kept) { ids.push_back(item.token); weights.push_back(item.value); }
    return Distribution(ids, weights);
  };
  std::mt19937 rng(5);
  for (int trial = 0; trial < 400; ++trial) {
    const size_t vocab = 50 + rng() % 3000;
    std::vector<float> logits(vocab);
    // Coarse values force many ties; some rows carry -inf.
    for (float& value : logits) value = float(int(rng() % 40) - 20) * (trial % 2 ? 0.25f : 0.37f);
    for (int i = 0; i < 5; ++i) logits[rng() % vocab] = -std::numeric_limits<float>::infinity();
    std::vector<int> mapping;
    if (trial % 3 == 0) for (size_t i = 0; i < vocab; ++i) mapping.push_back(int(vocab - i) * 2);
    Options o; o.temperature = 0.3 + (rng() % 10) * 0.1; o.top_k = int(1 + rng() % 80);
    o.top_p = trial % 4 ? 0.5 + (rng() % 50) * 0.01 : 1.0; o.min_p = trial % 5 ? 0 : 0.01;
    const Distribution fast = FromLogits(logits, o, mapping), slow = reference(logits, o, mapping);
    assert(fast.Ids() == slow.Ids() && fast.Probabilities() == slow.Probabilities());
  }
  // Nonfinite values still fail, wherever they sit.
  std::vector<float> bad(1000, 0.f);
  Options o; o.temperature = 0.7; o.top_k = 20;
  for (const float value : {std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::infinity()}) {
    bad[777] = value;
    try { (void)FromLogits(bad, o); assert(false); } catch (const std::invalid_argument&) {}
  }
  std::vector<float> none(100, -std::numeric_limits<float>::infinity());
  try { (void)FromLogits(none, o); assert(false); } catch (const std::invalid_argument&) {}
  std::vector<int> negative(1000, 1); negative[3] = -1;
  bad[777] = 0;
  try { (void)FromLogits(bad, o, negative); assert(false); } catch (const std::invalid_argument&) {}
}

int main() {
  FastTopK();
  KeyedLaw();
  Filters();
  Invalid();
  NonfiniteHeadIds();
  Determinism();
  const Distribution p({0, 1, 2, 3, 4}, {0.05, 0.1, 0.2, 0.3, 0.35});
  Statistics(p, Distribution({0, 1, 2, 3, 4}, {0.3, 0.25, 0.2, 0.15, 0.1}), "MTP");
  Statistics(p, Distribution({0, 2, 4}, {0.2, 0.3, 0.5}), "sparse MTP");
  Statistics(p, Distribution::Point(4), "lookup");
  Statistics(p, p, "identical");
  Statistics(Distribution({0, 1}, {0.2, 0.8}), Distribution({2, 3}, {0.5, 0.5}), "disjoint");
  Options filtered;
  filtered.temperature = 0.7; filtered.top_k = 4; filtered.top_p = 0.9; filtered.min_p = 0.05;
  const float target[] = {1, 1.5, 2, 2.2, 1.1}, draft[] = {2, 1.8, 1.6, 1.3, 2.2};
  Statistics(FromLogits(target, filtered), FromLogits(draft, filtered), "filtered MTP");
  Sequences(false);
  Sequences(true);
  std::puts("PASS: filters, seed determinism, and exact speculative rejection");
}
