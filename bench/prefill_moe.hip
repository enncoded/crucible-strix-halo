// Standalone deterministic GPU dispatch correctness and latency.
#include "prefill_common.h"
#include "../engine/layers/kernels/prefill_moe.h"
using namespace crucible;
using namespace crucible::prefill;
using namespace prefill_bench;

int main() {
  // 5000 and 32768 rows have more than 256 count tiles (merged launches, prefill_launch.h).
  for (int rows : {256, 512, 1024, 2048, 4096, 5000, 32768}) {
    std::vector<int> input(rows * kTopK);
    std::mt19937 rng(43);
    for (int r = 0; r < rows; ++r) {
      std::vector<int> experts(kExperts);
      std::iota(experts.begin(), experts.end(), 0);
      std::shuffle(experts.begin(), experts.end(), rng);
      for (int k = 0; k < kTopK; ++k) input[r * kTopK + k] = experts[k];
    }
    Buffer<int> sel(input), tiles(kExperts * ((rows * kTopK + 255) / 256));
    Buffer<int> counts(kExperts), offsets(kExperts + 1), pairs(input.size()), row_ids(input.size());
    MoeDispatch a{sel.p, tiles.p, counts.p, offsets.p, pairs.p, row_ids.p, rows};
    LaunchMoeDispatch(a);
    CheckLaunch();
    const auto count = counts.Get(), off = offsets.Get(), pair = pairs.Get(), ids = row_ids.Get();
    std::vector<int> ref;
    for (int e = 0; e < kExperts; ++e) {
      if (off[e] != (int)ref.size()) return 2;
      int n = 0;
      for (int i = 0; i < (int)input.size(); ++i)
        if (input[i] == e) { ref.push_back(i); ++n; }
      if (count[e] != n) return 3;
    }
    if (pair != ref || off.back() != rows * kTopK) return 4;
    for (size_t i = 0; i < pair.size(); ++i) if (ids[i] != pair[i] / kTopK) return 5;
    const double us = Time([&] { LaunchMoeDispatch(a); }, 20);
    if (pairs.Get() != ref) return 6;
    std::printf("moe rows=%d us=%.3f exact=1 routes=%d\n", rows, us, rows * kTopK);
  }
}
