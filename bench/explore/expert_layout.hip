// Round 4 Q2: does the front tile order help the routed-expert stream as
// much as the dense stream? Reuses mega2's layout ablation (bench/
// mega_ablate2.hip on the mega branch, be5e2eb) on real expert matrices:
// per layer one gate/up phase (U routed experts + the shared expert) and one
// down phase. U = 10 is R=1; U = 51 is the measured distinct count at R=8.
// The R=1 consumer is used for every U, so the R=8 case counts bytes, not
// the extra v_dot2 rows (about 0.6 points in mega2's layout-2 run).
//
// Build on the test machine (mega_ablate2's main is renamed into a library copy):
//   sed 's/^int main(/int Ablate2Main(/' <mega worktree>/bench/mega_ablate2.hip \
//     > build/bench/mega_ablate2_lib.hip
//   hipcc ... -I build/bench -I <mega worktree>/bench -o build/bench/expert_layout \
//     bench/explore/expert_layout.hip
// Usage: expert_layout CKPT [U=10] [grid=80] [seed=1]
#include "mega_ablate2_lib.hip"

namespace {
// Split counts that keep span = I/16/S a multiple of the chunk (8 for chunk 8).
constexpr int kSplitGateUp = 4, kSplitDown = 1;

std::vector<std::vector<Matrix>> LoadExpertGroups(st::Checkpoint& ck, int U, unsigned seed) {
  std::vector<std::vector<Matrix>> groups;
  std::mt19937 rng(seed);
  for (int l = 0; l < 48; ++l) {
    const std::string L = "model.language_model.layers." + std::to_string(l) + ".mlp.";
    std::vector<int> ex;
    while ((int)ex.size() < U) {
      const int e = (int)(rng() % 512);
      if (std::find(ex.begin(), ex.end(), e) == ex.end()) ex.push_back(e);
    }
    std::vector<Matrix> gu, dn;
    for (int e : ex) {
      const std::string E = L + "experts." + std::to_string(e) + ".";
      gu.push_back(Load(ck, E + "gate_proj"));
      gu.push_back(Load(ck, E + "up_proj"));
      dn.push_back(Load(ck, E + "down_proj"));
    }
    gu.push_back(Load(ck, L + "shared_expert.gate_proj"));
    gu.push_back(Load(ck, L + "shared_expert.up_proj"));
    dn.push_back(Load(ck, L + "shared_expert.down_proj"));
    groups.push_back(std::move(gu));
    groups.push_back(std::move(dn));
  }
  return groups;
}
}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) { std::fprintf(stderr, "usage: expert_layout CKPT [U=10] [grid=80] [seed=1]\n"); return 2; }
  const int U = argc > 2 ? std::atoi(argv[2]) : 10;
  const int grid = argc > 3 ? std::atoi(argv[3]) : 80;
  const unsigned seed = argc > 4 ? (unsigned)std::atoi(argv[4]) : 1u;
  if (U < 1 || U > 512 || grid < 1 || grid > 1024) return 2;
  st::Checkpoint ck(argv[1]);
  const auto groups = LoadExpertGroups(ck, U, seed);
  std::mt19937 rng(17); std::normal_distribution<float> norm;
  std::vector<__half> hx(10240);
  for (auto& h : hx) h = __float2half(norm(rng));
  __half* x; float* partial; uint32_t* sinks; uint64_t* clocks;
  HIP_CHECK(hipMalloc(&x, hx.size() * 2));
  HIP_CHECK(hipMemcpy(x, hx.data(), hx.size() * 2, hipMemcpyHostToDevice));
  HIP_CHECK(hipMalloc(&partial, 64 << 20));
  HIP_CHECK(hipMalloc(&sinks, (size_t)grid * 256 * 4));
  HIP_CHECK(hipMalloc(&clocks, (size_t)grid * 16));
  std::vector<Plan> plans;
  size_t bytes = 0, lds = 0, items = 0;
  for (size_t i = 0; i < groups.size(); ++i) {
    plans.push_back(MakePlan(groups[i], x, partial, i % 2 ? kSplitDown : kSplitGateUp));
    bytes += plans.back().bytes; lds = std::max(lds, plans.back().lds);
    items += plans.back().items.size();
  }
  for (const auto& p : plans) for (const auto& j : p.jobs)
    if ((j.I / 16 / j.S) % 4 || (j.O / 16) % 8) { std::fprintf(stderr, "I %d O %d S %d not aligned\n", j.I, j.O, j.S); return 1; }
  // Chunk 4 only: mega2 measured chunk 8 slower in every full row.
  constexpr int kChunk = 4;
  std::vector<ab2::Phase> host[3][2];
  for (int layout = 0; layout < 3; ++layout) {
    int start = 0;
    for (const auto& p : plans) {
      std::vector<LinJob> jobs = p.jobs;
      if (layout) for (auto& j : jobs) j.tiles = ab2::Repack(j, layout, kChunk);
      LinJob* dj;
      HIP_CHECK(hipMalloc(&dj, jobs.size() * sizeof(LinJob)));
      HIP_CHECK(hipMemcpy(dj, jobs.data(), jobs.size() * sizeof(LinJob), hipMemcpyHostToDevice));
      for (int bal = 0; bal < 2; ++bal) host[layout][bal].push_back({dj, p.di, (int)p.items.size(), bal ? start : 0});
      start += (int)p.items.size();
    }
  }
  ab2::Phase* dph[3][2];
  for (int a = 0; a < 3; ++a) for (int b = 0; b < 2; ++b) {
    HIP_CHECK(hipMalloc(&dph[a][b], plans.size() * sizeof(ab2::Phase)));
    HIP_CHECK(hipMemcpy(dph[a][b], host[a][b].data(), plans.size() * sizeof(ab2::Phase), hipMemcpyHostToDevice));
  }
  std::printf("# experts U=%d seed=%u phases=%zu items=%zu bytes=%.6fGB grid=%d L2 C8 chunk4 roof=245.5GB/s\n",
              U, seed, plans.size(), items, bytes / 1e9, grid);
  // Exactness: each layout against LinGemvKernel<1> on gate/up and down phases.
  for (int layout = 0; layout < 3; ++layout)
    for (int i : {0, 1, 46, 47, 94, 95}) {
      const Plan& p = plans[i];
      std::vector<float> reference(p.partials), actual(p.partials);
      Launch(p, false, grid);
      HIP_CHECK(hipMemcpy(reference.data(), partial, p.partials * 4, hipMemcpyDeviceToHost));
      HIP_CHECK(hipMemset(partial, 0xff, p.partials * 4));
      hipLaunchKernelGGL(ab2::Pick(2, layout, kChunk), grid, 320, lds, nullptr, dph[layout][0] + i, 1, sinks, clocks);
      HIP_CHECK(hipMemcpy(actual.data(), partial, p.partials * 4, hipMemcpyDeviceToHost));
      if (std::memcmp(reference.data(), actual.data(), p.partials * 4)) {
        std::fprintf(stderr, "full mismatch layout%d phase%d\n", layout, i);
        return 1;
      }
    }
  std::printf("# full partials exact for 3 layouts on six phases\n");
  // Per-op reference: today's decode GEMV kernel, one launch per phase.
  {
    std::vector<double> times;
    for (int rep = 0; rep < 7; ++rep) {
      bench::EventTimer timer; timer.Begin();
      for (const auto& p : plans) Launch(p, false, grid);
      const float ms = timer.EndMs();
      if (rep) times.push_back(ms);
    }
    const double ms = bench::Median(times), bw = bytes / ms / 1e6;
    std::printf("gemv  engine  per-phase  %8.3fms %7.2fGB/s %6.2f%%DRAM\n", ms, bw, bw / 245.5 * 100);
  }
  for (int mode : {0, 2})
    for (int layout : {0, 1, 2})
      for (int kind = 0; kind < 3; ++kind) {  // 0 persistent zero, 1 persistent balanced, 2 per-phase
        const auto fn = ab2::Pick(mode, layout, kChunk);
        const ab2::Phase* ph = dph[layout][kind == 1];
        const int g = std::min(grid, ab2::ResidentBlocks(fn, lds));
        std::vector<double> times;
        for (int rep = 0; rep < 7; ++rep) {
          bench::EventTimer timer; timer.Begin();
          if (kind < 2) hipLaunchKernelGGL(fn, g, 320, lds, nullptr, ph, (int)plans.size(), sinks, clocks);
          else for (size_t i = 0; i < plans.size(); ++i) hipLaunchKernelGGL(fn, g, 320, lds, nullptr, ph + i, 1, sinks, clocks);
          const float ms = timer.EndMs(); HIP_CHECK(hipGetLastError());
          if (rep) times.push_back(ms);
        }
        const double ms = bench::Median(times), bw = bytes / ms / 1e6;
        std::printf("%-5s layout%d %-21s %8.3fms %7.2fGB/s %6.2f%%DRAM min=%.3f max=%.3f grid=%d\n",
                    mode ? "full" : "bytes", layout,
                    kind == 0 ? "persistent zero" : kind == 1 ? "persistent balanced" : "per-phase", ms, bw,
                    bw / 245.5 * 100, *std::min_element(times.begin(), times.end()),
                    *std::max_element(times.begin(), times.end()), g);
        std::fflush(stdout);
      }
  return 0;
}
