// Value-aware draft length (exact: it chooses drafts only; verification commits the target's tokens).
// A chain grows while the next draft's expected accepted tokens pay for its extra time:
//   E(d) = 1 + Q_1 + ... + Q_d expected tokens of a step with d drafts, Q_k = p_1 * ... * p_k,
//   T(d) = verify time of d + 1 rows plus d MTP chain calls.
// Adding draft d + 1 raises tokens per ms when Q_{d+1} * T(d) > E(d) * (T(d + 1) - T(d)).
// p comes from the MTP margin table, calibrated by the request's own recent acceptance.
#pragma once

#include <algorithm>
#include <array>

namespace crucible {

inline constexpr int kDraftValueRows = 8;
inline constexpr int kDraftBuckets = 8;

// Margin buckets of DraftAcceptProb and SampledDepthConfidence (same edges and values).
inline int DraftMarginBucket(float margin) {
  static constexpr float edges[] = {1, 2, 3, 4, 6, 8, 12};
  int b = 0;
  while (b < 7 && margin >= edges[b]) ++b;
  return b;
}
inline constexpr std::array<float, kDraftBuckets> kDraftBucketPrior{0.35f, 0.43f, 0.51f, 0.75f, 0.87f, 0.94f,
                                                                   0.99f, 0.99f};

// Verify time by row count (ms) and the cost of one MTP chain call. Only ratios matter for the rule.
struct DraftCosts {
  std::array<float, kDraftValueRows + 1> verify_ms{};  // [R], R = 1..8
  float mtp_ms = 1.0f;
  // Short-context single-stream defaults (3.05, forward_rows on the Strix Halo test host); serve may replace them.
  static DraftCosts Default() {
    DraftCosts c;
    // forward_rows medians, Oct 6. R=7 runs an untuned row policy today.
    c.verify_ms = {0.f, 25.9f, 31.3f, 32.8f, 36.8f, 39.1f, 43.1f, 51.7f, 49.7f};
    c.mtp_ms = 1.3f;
    return c;
  }
  float StepMs(int drafts) const {
    return verify_ms[size_t(std::clamp(drafts + 1, 1, kDraftValueRows))] + mtp_ms * float(std::max(0, drafts));
  }
};

// Per-request acceptance by margin bucket: a prior from the static table plus decayed observations.
class DraftCalibration {
 public:
  explicit DraftCalibration(const std::array<float, kDraftBuckets>& prior = kDraftBucketPrior, float weight = 8.f, float decay = 0.98f)
      : prior_(prior), weight_(weight), decay_(decay) {}
  float P(int bucket) const {
    const size_t b = size_t(std::clamp(bucket, 0, kDraftBuckets - 1));
    return std::clamp((hits_[b] + weight_ * prior_[b]) / (trials_[b] + weight_), 0.02f, 0.995f);
  }
  // Acceptance of a draft whose margin is not known yet: the request's recent mean.
  float Mean() const { return std::clamp((all_hits_ + weight_ * 0.75f) / (all_trials_ + weight_), 0.02f, 0.995f); }
  // One verified chain: drafts [0, accepted) matched; draft `accepted` (if proposed) did not. Later drafts were not
  // observed, because acceptance stops at the first mismatch.
  template <class Buckets> void Observe(const Buckets& buckets, int proposed, int accepted) {
    for (int k = 0; k < proposed && k <= accepted; ++k) {
      const size_t b = size_t(std::clamp(int(buckets[size_t(k)]), 0, kDraftBuckets - 1));
      const float hit = k < accepted ? 1.f : 0.f;
      hits_[b] = hits_[b] * decay_ + hit;
      trials_[b] = trials_[b] * decay_ + 1.f;
      all_hits_ = all_hits_ * decay_ + hit;
      all_trials_ = all_trials_ * decay_ + 1.f;
    }
  }

 private:
  std::array<float, kDraftBuckets> prior_, hits_{}, trials_{};
  float weight_, decay_, all_hits_ = 0.f, all_trials_ = 0.f;
};

// The growing chain of one step.
struct DraftValue {
  float q = 1.f;   // Q of the last kept draft (1 before any draft)
  float e = 1.f;   // E(d)
  int drafts = 0;  // d
  // Before an MTP call: is draft d + 1 worth computing, with acceptance p_next expected?
  bool Try(const DraftCosts& c, float p_next) const {
    return q * p_next * c.StepMs(drafts) > e * (c.StepMs(drafts + 1) - c.StepMs(drafts));
  }
  // After the call (its MTP cost is spent): keep the draft with acceptance p?
  bool Keep(const DraftCosts& c, float p) const {
    const float extra = c.verify_ms[size_t(std::min(drafts + 2, kDraftValueRows))] -
                        c.verify_ms[size_t(std::min(drafts + 1, kDraftValueRows))];
    return q * p * c.StepMs(drafts) > e * extra;
  }
  void Add(float p) {
    q *= p;
    e += q;
    ++drafts;
  }
};

}  // namespace crucible
