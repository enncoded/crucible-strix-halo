// Pruned prefill router versus MoeRouteKernel: bitwise selections and weights,
// including rows with many tied logits.
#include "prefill_common.h"
#include "kernels/prefill_moe_route.h"
using namespace crucible;
using namespace prefill_bench;

int main() {
  const int R = 4096;
  auto logits = Random((size_t)R * (kExperts + 1), 2.f, 71);
  // Coarse values in some rows force ties at and around the threshold.
  for (int r = 0; r < R; r += 3)
    for (int e = 0; e <= kExperts; ++e) logits[(size_t)r * (kExperts + 1) + e] = std::round(logits[(size_t)r * (kExperts + 1) + e] * (r % 2 ? 1.f : 4.f));
  for (int e = 0; e <= kExperts; ++e) logits[(size_t)7 * (kExperts + 1) + e] = 1.f;  // all equal
  for (int e = 0; e < kExperts; e += 2) logits[(size_t)11 * (kExperts + 1) + e] = -INFINITY;  // half -inf
  for (int e = 0; e < kExperts; ++e) logits[(size_t)13 * (kExperts + 1) + e] = e < 500 ? -INFINITY : (float)(e % 3);
  Buffer<float> lg(logits), w0((size_t)R * kSlots), w1((size_t)R * kSlots);
  Buffer<int> s0(R * kTopK), s1(R * kTopK);
  MoeRouteKernel<<<R, 512>>>(lg.p, s0.p, w0.p);
  PrefillMoeRouteKernel<<<R, 512>>>(lg.p, s1.p, w1.p);
  CheckLaunch();
  const bool same = s0.Get() == s1.Get() && w0.Get() == w1.Get();
  Buffer<float> w2((size_t)R * kSlots);
  Buffer<int> s2(R * kTopK);
  PrefillMoeRouteWaveKernel<<<(R + 7) / 8, 256>>>(lg.p, s2.p, w2.p, R);
  CheckLaunch();
  const bool wave = s0.Get() == s2.Get() && w0.Get() == w2.Get();
  std::printf("route R=%d bitwise=%d wave_bitwise=%d old_us=%.1f new_us=%.1f wave_us=%.1f\n", R, (int)same,
    (int)wave, Time([&] { MoeRouteKernel<<<R, 512>>>(lg.p, s0.p, w0.p); }),
    Time([&] { PrefillMoeRouteKernel<<<R, 512>>>(lg.p, s1.p, w1.p); }),
    Time([&] { PrefillMoeRouteWaveKernel<<<(R + 7) / 8, 256>>>(lg.p, s2.p, w2.p, R); }));
  return same && wave ? 0 : 1;
}
