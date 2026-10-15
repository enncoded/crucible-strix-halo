// CPU checks of the value-aware draft length rule (engine/runtime/draft_value.h).
#include "../engine/runtime/draft_accept.h"
#include "../engine/runtime/draft_value.h"
#include <cassert>
#include <cstdio>
#include <vector>

using namespace crucible;

int main() {
  const auto c = DraftCosts::Default();
  assert(c.StepMs(0) == c.verify_ms[1]);
  assert(c.StepMs(7) == c.verify_ms[8] + 7 * c.mtp_ms);
  assert(DraftMarginBucket(0.5f) == 0 && DraftMarginBucket(4.f) == 4 && DraftMarginBucket(100.f) == 7);
  // DraftAcceptProb (runtime/draft_accept.h) reads the same buckets: the edges and values must not drift.
  for (float m = -1.f; m < 15.f; m += 0.25f)
    assert(DraftAcceptProb(m) == kDraftBucketPrior[size_t(DraftMarginBucket(m))]);
  assert(kDraftNextPrior == 0.75f);
  // A confident chain keeps growing; a weak draft stops it.
  DraftValue v;
  assert(v.Keep(c, 0.9f));
  v.Add(0.9f);
  assert(v.Try(c, 0.9f) && v.Keep(c, 0.9f));
  v.Add(0.9f);
  assert(!v.Keep(c, 0.05f));
  // The bar rises with E: after many drafts a mid-confidence draft no longer pays.
  DraftValue deep;
  for (int k = 0; k < 6; ++k) deep.Add(0.95f);
  DraftValue shallow;
  shallow.Add(0.95f);
  assert(shallow.Keep(c, 0.5f) || !deep.Keep(c, 0.5f));
  // Calibration: misses lower the bucket's estimate, hits raise it, and the prior holds with no data.
  DraftCalibration cal;
  assert(cal.P(4) == kDraftBucketPrior[4]);
  const std::vector<int> buckets(8, 4);
  for (int i = 0; i < 40; ++i) cal.Observe(buckets, 3, 0);  // draft 0 rejected each time
  assert(cal.P(4) < 0.3f && cal.Mean() < 0.3f);
  DraftCalibration up;
  for (int i = 0; i < 40; ++i) up.Observe(buckets, 3, 3);  // all accepted
  assert(up.P(4) > 0.95f);
  // Only observed drafts count: proposed 5, accepted 1 records draft 0 (hit) and draft 1 (miss), not drafts 2-4.
  DraftCalibration one;
  one.Observe(std::vector<int>{0, 1, 2, 3, 4}, 5, 1);
  assert(one.P(2) == kDraftBucketPrior[2] && one.P(0) > kDraftBucketPrior[0] && one.P(1) < kDraftBucketPrior[1]);
  std::puts("draft_value_test PASS");
}
