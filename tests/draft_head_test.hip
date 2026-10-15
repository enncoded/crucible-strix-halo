// Tests draft-only sampling and exact retention of EXL3 output blocks.
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "../engine/model/flashnext/model.h"
#include "../engine/runtime/draft_ops.h"
#include "../engine/runtime/draft_dense.h"
#include "../engine/runtime/draft_adapt.h"

using namespace crucible;

template <typename T> T* Put(const std::vector<T>& v) {
  T* p;
  HIP_OK(hipMalloc(&p, v.size() * sizeof(T)));
  HIP_OK(hipMemcpy(p, v.data(), v.size() * sizeof(T), hipMemcpyHostToDevice));
  return p;
}

template <typename T> std::vector<T> Get(T* p, int n) {
  std::vector<T> v(n);
  HIP_OK(hipMemcpy(v.data(), p, n * sizeof(T), hipMemcpyDeviceToHost));
  return v;
}

std::vector<float> Linear(const std::vector<uint32_t>& tiles, const std::vector<__half>& scales,
                          const std::vector<float>& input, int O,
                          const std::vector<int>& selected = {}) {
  constexpr int I = 2560, K = 5;
  LinJob j{};
  j.tiles = Put(tiles);
  j.suh = Put(std::vector<__half>(I, __float2half(0.01f)));
  j.svh = Put(scales);
  j.src = Put(input);
  j.xh = Put(std::vector<__half>(I));
  j.partial = Put(std::vector<float>(O));
  j.dst = Put(std::vector<float>(O));
  j.I = I; j.O = O; j.K = K; j.S = 1; j.R = 1;
  j.lds = I; j.ldd = O; j.epi = kEpiStore;
  auto* job = Put(std::vector<LinJob>{j});
  std::vector<GemvItem> items;
  std::vector<XfItem> in, out;
  for (int b = 0; b < O / 16; b += 8) items.push_back({0, 0, b, 8});
  for (int b = 0; b < I / 128; ++b) in.push_back({0, 0, b});
  for (int b = 0; b < O / 128; ++b) out.push_back({0, 0, b});
  auto* di = Put(in);
  auto* dm = Put(items);
  auto* dout = Put(out);
  LinInKernel<<<(in.size() + 7) / 8, 256>>>(job, di, in.size());
  LinGemvKernel<1><<<items.size(), 256, (I / 16) * 32>>>(job, dm, items.size());
  LinOutKernel<<<(out.size() + 7) / 8, 256>>>(job, dout, out.size());
  if (!selected.empty()) {
    Lin head{I, O, K, const_cast<uint32_t*>(j.tiles), const_cast<__half*>(j.suh), const_cast<__half*>(j.svh)};
    auto* packed = PackDraftDense(head, selected);
    DraftDenseGemvKernel<<<(selected.size() + 7) / 8, 256>>>(packed, j.xh, j.dst, selected.size());
    return Get(j.dst, selected.size());
  }
  return Get(j.dst, O);
}

int main() {
  constexpr int O = 1024, words = 128 * 2560 * 5 / 32;
  std::mt19937 rng(42);
  std::vector<uint32_t> tiles(words * O / 128);
  for (auto& x : tiles) x = rng();
  std::vector<__half> scales(O);
  for (auto& x : scales) x = __float2half((int(rng() % 200) - 100) / 100.f);
  std::vector<float> input(2560);
  for (auto& x : input) x = (int(rng() % 200) - 100) / 100.f;
  const auto full = Linear(tiles, scales, input, O);
  Lin source{2560, O, 5, Put(tiles), nullptr, Put(scales)};
  BlockDraftHead adaptive(source, {4, 6}, true);
  const int observed[] = {130, 131, 600, 130};
  assert(adaptive.Observe(observed, 4));
  assert(!adaptive.Observe(observed, 4));
  assert(adaptive.Weights().O == 3 * 128);
  assert(Get(adaptive.Mapping(), 3) == std::vector<int>({4, 6, 1}));
  const auto packed = Get(adaptive.Weights().tiles, 3 * words);
  assert(std::memcmp(packed.data() + 2 * words, tiles.data() + words, words * sizeof(uint32_t)) == 0);
  const std::vector<int> blocks{1, 4, 6};
  std::vector<uint32_t> subtiles;
  std::vector<__half> subscales;
  for (int b : blocks) {
    subtiles.insert(subtiles.end(), tiles.begin() + b * words, tiles.begin() + (b + 1) * words);
    subscales.insert(subscales.end(), scales.begin() + b * 128, scales.begin() + (b + 1) * 128);
  }
  const auto subset = Linear(subtiles, subscales, input, blocks.size() * 128);
  for (size_t i = 0; i < blocks.size(); ++i)
    assert(std::memcmp(subset.data() + i * 128, full.data() + blocks[i] * 128, 128 * sizeof(float)) == 0);

  const std::vector<int> selected{1, 128, 333, 999};
  const auto dense = Linear(tiles, scales, input, O, selected);
  float error = 0.f, norm = 0.f;
  for (int i = 0; i < (int)selected.size(); ++i) {
    const float delta = dense[i] - full[selected[i]];
    error += delta * delta;
    norm += full[selected[i]] * full[selected[i]];
  }
  assert(std::sqrt(error / norm) < 0.001f);
  std::printf("fp16 draft-head relative error %.6f\n", std::sqrt(error / norm));

  std::vector<int> sample_blocks(128);
  for (int i = 0; i < (int)sample_blocks.size(); ++i) sample_blocks[i] = (127 - i) * 2;
  auto* db = Put(sample_blocks);
  auto* dc = Put(std::vector<DraftChoice>(1));
  std::vector<int> token_ids(16384);
  for (int i = 0; i < (int)token_ids.size(); ++i) token_ids[i] = i * 3;
  auto* dt = Put(token_ids);
  for (int trial = 0; trial < 50; ++trial) {
    const bool individual = trial % 3 == 2, mapped = trial % 3 != 1;
    std::vector<float> logits(mapped ? sample_blocks.size() * 128 : kVocab);
    for (auto& x : logits) x = (int(rng() % 200) - 100) / 10.f;
    if (trial == 0) std::fill(logits.begin(), logits.end(), 1.f);
    auto* dl = Put(logits);
    DraftChoiceKernel<<<1, 1024>>>(dl, logits.size(), individual ? dt : mapped ? db : nullptr, dc, individual);
    const auto got = Get(dc, 1)[0];
    std::vector<int> order(logits.size());
    for (int i = 0; i < (int)order.size(); ++i) order[i] = i;
    const auto token_id = [&](int i) {
      return individual ? token_ids[i] : mapped ? sample_blocks[i / 128] * 128 + i % 128 : i;
    };
    std::sort(order.begin(), order.end(), [&](int a, int b) {
      return logits[a] > logits[b] || (logits[a] == logits[b] && token_id(a) < token_id(b));
    });
    const int token = individual ? token_ids[order[0]] : mapped ? sample_blocks[order[0] / 128] * 128 + order[0] % 128 : order[0];
    assert(got.token == token);
    assert(got.margin == logits[order[0]] - logits[order[1]]);
    HIP_OK(hipFree(dl));
  }
  std::puts("PASS: retained EXL3 logits bitwise exact; mapped top-2 and ties (50 trials)");
}
