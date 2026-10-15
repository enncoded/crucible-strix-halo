// Exact partial sums and cold dense stream timing for the R=1 LDS ring.
// Usage: mega_dense check|bench CKPT [grid=80]
#include <cmath>
#include <random>
#include "kernels/exl3_ring.h"
#include "../engine/exl3/plans.h"
#include "common.h"
#include "../engine/exl3/safetensors.h"

using namespace crucible;

namespace {
constexpr double kRoofGBps = 245.5;
struct Matrix {
  int I, O, K;
  uint32_t* tiles;
  size_t bytes;
};
struct Plan {
  std::vector<LinJob> jobs;
  std::vector<GemvItem> items;
  LinJob* dj;
  GemvItem* di;
  size_t lds, partials, bytes;
};

Matrix Load(st::Checkpoint& ck, const std::string& name) {
  const auto& ti = ck.Info(name + ".trellis");
  if (ti.dtype != "I16" || ti.shape.size() != 3 || ti.shape[2] % 16 != 0) std::abort();
  const int A = ti.shape[0], B = ti.shape[1], K = ti.shape[2] / 16, T = 8 * K;
  std::vector<uint32_t> host((size_t)A * B * T);
  const auto* src = reinterpret_cast<const uint32_t*>(ck.Data(name + ".trellis"));
  for (int a = 0; a < A; ++a)
    for (int b = 0; b < B; ++b)
      std::memcpy(host.data() + ((size_t)b * A + a) * T,
                  src + ((size_t)a * B + b) * T, T * 4);
  Matrix m{A * 16, B * 16, K, nullptr, host.size() * 4};
  HIP_CHECK(hipMalloc(&m.tiles, m.bytes));
  HIP_CHECK(hipMemcpy(m.tiles, host.data(), m.bytes, hipMemcpyHostToDevice));
  return m;
}

Plan MakePlan(const std::vector<Matrix>& matrices, __half* x, float* partial, int splits = 0) {
  Plan p{};
  int target = 160;
  if (!splits) {
    for (int t : {160, 80, 40, 20}) {
      long waves = 0;
      for (const auto& m : matrices) waves += (long)(m.O / 16) * SplitFor(m.I / 16, t);
      target = t;
      if (waves >= 900) break;
    }
  }
  for (const auto& m : matrices) {
    LinJob j{};
    j.tiles = m.tiles; j.xh = x; j.partial = partial + p.partials;
    j.I = m.I; j.O = m.O; j.K = m.K; j.R = 1;
    j.S = splits ? splits : SplitFor(m.I / 16, target);
    if ((m.I / 16) % j.S) std::abort();
    const int ji = p.jobs.size();
    p.jobs.push_back(j);
    for (int s = 0; s < j.S; ++s)
      for (int b = 0; b < m.O / 16; b += 8)
        p.items.push_back({ji, s, b, std::min(8, m.O / 16 - b)});
    p.partials += (size_t)j.S * j.O;
    p.lds = std::max(p.lds, (size_t)(j.I / 16 / j.S) * 32);
    p.bytes += m.bytes;
  }
  HIP_CHECK(hipMalloc(&p.dj, p.jobs.size() * sizeof(LinJob)));
  HIP_CHECK(hipMalloc(&p.di, p.items.size() * sizeof(GemvItem)));
  HIP_CHECK(hipMemcpy(p.dj, p.jobs.data(), p.jobs.size() * sizeof(LinJob), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(p.di, p.items.data(), p.items.size() * sizeof(GemvItem), hipMemcpyHostToDevice));
  return p;
}

void Launch(const Plan& p, bool ring, int grid) {
  if (ring) LinRingKernel<><<<std::min<int>(grid, p.items.size()), 320, p.lds>>>(p.dj, p.di, p.items.size());
  else LinGemvKernel<1><<<p.items.size(), 256, p.lds>>>(p.dj, p.di, p.items.size());
  HIP_CHECK(hipGetLastError());
}
void Free(const Plan& p) {
  HIP_CHECK(hipFree(p.dj)); HIP_CHECK(hipFree(p.di));
}

std::vector<std::vector<Matrix>> LoadGroups(st::Checkpoint& ck, bool all) {
  std::vector<std::vector<Matrix>> groups;
  for (int l = 0; l < (all ? 48 : 4); ++l) {
    if (!all && l != 0 && l != 3) continue;
    const std::string L = "model.language_model.layers." + std::to_string(l);
    if (l % 4 == 3) {
      groups.push_back({Load(ck, L + ".self_attn.q_proj"), Load(ck, L + ".self_attn.k_proj"),
                        Load(ck, L + ".self_attn.v_proj"), Load(ck, L + ".self_attn.indexer.index_qk_proj")});
      groups.push_back({Load(ck, L + ".self_attn.o_proj")});
    } else {
      groups.push_back({Load(ck, L + ".linear_attn.in_proj_qkv"), Load(ck, L + ".linear_attn.in_proj_z")});
      groups.push_back({Load(ck, L + ".linear_attn.out_proj")});
    }
  }
  if (all) groups.push_back({Load(ck, "lm_head")});
  else groups.push_back({Load(ck, "model.language_model.layers.0.mlp.shared_expert.down_proj")});
  return groups;
}
}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) { std::fprintf(stderr, "usage: mega_dense check|bench CKPT [grid]\n"); return 2; }
  const std::string mode = argv[1];
  if (mode != "check" && mode != "bench") return 2;
  const int grid = argc > 3 ? std::atoi(argv[3]) : 80;
  if (grid <= 0 || grid > 1024) return 2;
  st::Checkpoint ck(argv[2]);
  auto groups = LoadGroups(ck, mode == "bench");
  std::mt19937 rng(17); std::normal_distribution<float> norm;
  std::vector<__half> hx(10240);
  for (auto& h : hx) h = __float2half(norm(rng));
  __half* x; float* partial;
  HIP_CHECK(hipMalloc(&x, hx.size() * 2));
  HIP_CHECK(hipMemcpy(x, hx.data(), hx.size() * 2, hipMemcpyHostToDevice));
  HIP_CHECK(hipMalloc(&partial, 64 << 20));
  if (mode == "check") {
    int fails = 0;
    for (const auto& group : groups) for (int splits : {1, 2, 4, 8}) {
      if ((group[0].I / 16) % splits) continue;
      Plan p = MakePlan(group, x, partial, splits);
      std::vector<float> expected(p.partials), actual(p.partials);
      Launch(p, false, grid);
      HIP_CHECK(hipMemcpy(expected.data(), partial, p.partials * 4, hipMemcpyDeviceToHost));
      HIP_CHECK(hipMemset(partial, 0xff, p.partials * 4));
      Launch(p, true, grid);
      HIP_CHECK(hipMemcpy(actual.data(), partial, p.partials * 4, hipMemcpyDeviceToHost));
      size_t bad = 0;
      for (size_t i = 0; i < p.partials; ++i) bad += std::memcmp(&actual[i], &expected[i], 4) != 0;
      std::printf("check I=%d O=%d K=%d jobs=%zu S=%d partials=%zu mismatch=%zu\n",
                  group[0].I, group[0].O, group[0].K, group.size(), splits, p.partials, bad);
      fails += bad != 0;
      Free(p);
    }
    return fails ? 1 : 0;
  }
  std::vector<Plan> plans;
  size_t bytes = 0;
  for (const auto& group : groups) { plans.push_back(MakePlan(group, x, partial)); bytes += plans.back().bytes; }
  std::printf("# dense phases=%zu trellis=%.6f GB grid=%d DRAM=%.1f GB/s\n", plans.size(), bytes / 1e9, grid, kRoofGBps);
  for (bool ring : {false, true, false, true}) {
    std::vector<double> times;
    for (int rep = 0; rep < 8; ++rep) {
      bench::EventTimer timer;
      timer.Begin();
      for (const auto& p : plans) Launch(p, ring, grid);
      const float ms = timer.EndMs();
      if (rep) times.push_back(ms);
    }
    const double ms = bench::Median(times), gbps = bytes / ms / 1e6;
    std::printf("%-8s %.3f ms %.2f GB/s %.2f%% DRAM min=%.3f max=%.3f\n",
                ring ? "ring" : "baseline", ms, gbps, gbps / kRoofGBps * 100,
                *std::min_element(times.begin(), times.end()), *std::max_element(times.begin(), times.end()));
  }
  return 0;
}
