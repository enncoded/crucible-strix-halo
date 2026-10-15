#define ENGINE_LIBRARY
#include "../engine/engine.hip"
#include <filesystem>
#include <cassert>
#include "statecache_options.h"

struct Observation { std::vector<int> ids; std::vector<float> logits; };
Observation Continue(Engine& e, const std::vector<int>& ids, int rows, int replay, bool bulk,
                     const std::string& dump) {
  Observation out;
  for (size_t i = 0; i < ids.size();) {
    const int r = std::min<size_t>(rows, ids.size() - i);
    auto y = bulk ? e.Prefill(ids.data() + i, r, dump) : e.Forward(ids.data() + i, r, replay, r, dump);
    replay = 0;
    out.ids.insert(out.ids.end(), y.begin(), y.end());
    const auto logits = e.CopyLogits(bulk ? 1 : r);
    out.logits.insert(out.logits.end(), logits.begin(), logits.end());
    i += r;
  }
  return out;
}
int main(int argc, char** argv) {
  const auto rows = SnapshotPrefill(argc, argc == 4 ? argv[3] : nullptr,
                                    [](const char* raw) { return GenNumber<int>(raw); });
  const auto dense = SnapshotDense(std::getenv("CRUCIBLE_STATECACHE_GATE_DENSE"),
                                    [](const char* raw) { return GenNumber<int>(raw); });
  if (!rows || !dense) return 2;
  const int prefill = *rows;
  // Primary gates use tools/gates.py. This harness tests snapshot state only.
  setenv("HSA_USERPTR_FOR_PAGED_MEM", "0", 0);
  std::ifstream file(argv[2]); std::stringstream text; text << file.rdbuf();
  const auto source = ParseIds(text.str());
  if (source.size() < 24) return 2;
  const std::string gate_root = "build/statecache-gate/" +
      std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
  std::printf("captures %s\n", gate_root.c_str());
  std::printf("snapshot prefill rows %d\n", prefill);
  Engine e(argv[1], 4096, prefill);
  float* stk = Zeros<float>((kMaxRows + 1) * kWide);
  float* carry = Zeros<float>(kWide);
  for (const int kind : {0, 1, 2, 3}) {
    e.Reset();
    std::vector<int> prefix;
    // A completed block in a partial tile makes omitted index-pool columns observable.
    const int length = kind == 3 ? 3077 : kind == 2 ? 64 : 14;
    for (int i = 0; i < length; ++i) prefix.push_back(source[i % source.size()]);
    Continue(e, prefix, kind >= 2 ? prefill : 8, 0, kind >= 2, "");
    int replay = 0;
    if (kind == 1) {
      e.Forward(source.data(), 3, 0, 1);
      e.Truncate(e.Length());
      replay = 2;
    }
    ConversationResume state{e.Length(), 0, replay, 0, false, {}, {}};
    const auto snapshot = ConversationStateIO::Save(e, e.ActiveSequence(), state, stk, carry);
    std::vector<int> suffix(source.begin() + 14, source.begin() + 24);
    const std::string root = gate_root + "/" + std::to_string(kind);
    std::filesystem::create_directories(root + "/a");
    std::filesystem::create_directories(root + "/b");
    const auto a = Continue(e, suffix, kind >= 2 ? 10 : 4, replay, kind >= 2, root + "/a");
    e.Reset();
    std::vector<int> decoy(snapshot->hist.size());
    for (size_t i = 0; i < decoy.size(); ++i) decoy[i] = source[(i + 8) % source.size()];
    Continue(e, decoy, 8, 0, false, "");
    // Full-commit forwards do not touch gpend_. Force a different pending window.
    e.Forward(source.data() + 9, 3, 0, 1);
    ConversationStateIO::Restore(e, e.ActiveSequence(), *snapshot, stk, carry);
    const auto audit = ConversationStateIO::Save(e, e.ActiveSequence(), state, stk, carry);
    if (audit->bytes != snapshot->bytes) return 1;
    const auto b = Continue(e, suffix, kind >= 2 ? 10 : 4, replay, kind >= 2, root + "/b");
    if (a.ids != b.ids || a.logits.size() != b.logits.size() ||
        std::memcmp(a.logits.data(), b.logits.data(), a.logits.size() * sizeof(float))) return 1;
    for (int layer = 0; layer <= kLayers; ++layer) {
      char name[64]; std::snprintf(name, sizeof(name), "/h_%02d.f32", layer);
      std::ifstream fa(root + "/a" + name, std::ios::binary), fb(root + "/b" + name, std::ios::binary);
      const std::string ba(std::istreambuf_iterator<char>(fa), {}), bb(std::istreambuf_iterator<char>(fb), {});
      if (ba != bb || ba.empty()) return 1;
    }
    std::printf("PASS statecache kind=%d, 49 dumps, %zu logits, %zu IDs\n", kind, a.logits.size(), a.ids.size());
  }
  GenOptions o;
  o.ids = source; o.n = 64; o.ctx = 4096;
  for (int mode : {0, 1, 2, 3}) {
    if (mode == 3 && !*dense) {
      std::puts("SKIP optional fp16 dense draft head: explicit benchmark opt-out");
      continue;
    }
    const int spec = mode == 0 ? 0 : 2;
    const bool sampled = mode >= 2;
    std::vector<int> selection;
    if (sampled) {
      for (int token : source) selection.push_back(token / 128);
      std::sort(selection.begin(), selection.end());
      selection.erase(std::unique(selection.begin(), selection.end()), selection.end());
    }
    o.n = 4; o.spec = spec; o.sampling.temperature = sampled ? 0.7 : 0;
    o.sampling.top_k = 17; o.sampling.seed = 18073;
    o.adapt = sampled ? 1 : 0;
    ConversationHooks hooks;
    std::shared_ptr<const ConversationSnapshot> snapshot;
    hooks.save = [&](const ConversationResume& state, float* s, float* c) {
      snapshot = ConversationStateIO::Save(e, e.ActiveSequence(), state, s, c);
    };
    o.conversation = &hooks;
    RunGen(e, o, selection);
    if (!snapshot || snapshot->hist.size() != e.Length()) {
      // An accepted window can exceed the returned cap. Retain a prompt-only turn instead.
      o.n = 1;
      RunGen(e, o, selection);
    }
    if (!snapshot || snapshot->hist.size() != e.Length()) return 1;
    if (spec && snapshot->resume.mpos <= 0) return 1;
    o.ids = snapshot->hist;
    o.ids.insert(o.ids.end(), source.begin() + 8, source.begin() + 16);
    o.n = 32;
    hooks.save = {};
    std::pair<std::vector<float>, std::vector<int>> reference_draft;
    const auto probe_draft = [&](float* s, float* c) {
      auto tokens = snapshot->resume.pending_tokens;
      if (tokens.empty()) {
        HIP_OK(hipMemcpyAsync(s, c, sizeof(float) * kWide, hipMemcpyDeviceToDevice, 0));
        tokens = {o.ids[snapshot->hist.size()]};
      } else tokens.back() = o.ids[snapshot->hist.size()];
      // Only future cache positions change. The controller rewrites them before use.
      e.Mtp(s, tokens.data(), int(tokens.size()), snapshot->resume.mpos);
      return e.DraftLogits(int(tokens.size()));
    };
    hooks.restore = [&](float* s, float* c) {
      if (spec) reference_draft = probe_draft(s, c);
      return snapshot->resume;
    };
    if (mode == 3) {
      o.adapt = 0; o.draft_dense = true;
      selection = source;
      std::sort(selection.begin(), selection.end());
      selection.erase(std::unique(selection.begin(), selection.end()), selection.end());
    }
    const auto continued = RunGen(e, o, selection);
    GenOptions other;
    other.ids.resize(snapshot->hist.size());
    for (size_t i = 0; i < other.ids.size(); ++i) other.ids[i] = source[source.size() - 1 - i % source.size()];
    other.n = 2; other.ctx = 512; other.spec = spec;
    RunGen(e, other, {});
    hooks.restore = [&](float* s, float* c) {
      const auto state = ConversationStateIO::Restore(e, e.ActiveSequence(), *snapshot, s, c);
      const auto audit = ConversationStateIO::Save(e, e.ActiveSequence(), state, s, c);
      if (audit->bytes != snapshot->bytes) throw std::runtime_error("restored state bytes differ");
      const auto target = e.CopyLogits(1);
      if (std::memcmp(target.data(), state.logits.data(), target.size() * sizeof(float)))
        throw std::runtime_error("restored target logits differ");
      if (spec) {
        const auto draft = probe_draft(s, c);
        if (draft.second != reference_draft.second || draft.first.size() != reference_draft.first.size() ||
            std::memcmp(draft.first.data(), reference_draft.first.data(), draft.first.size() * sizeof(float)))
          throw std::runtime_error("restored MTP draft logits differ");
      }
      return state;
    };
    const auto restored = RunGen(e, o, selection);
    if (continued.gen != restored.gen) return 1;
    std::printf("PASS statecache live continuation sampled=%d dense=%d spec=%d mpos=%d replay=%d: 32 IDs match\n",
                sampled, o.draft_dense, spec, snapshot->resume.mpos, snapshot->resume.replay);
    if (spec) std::puts("PASS MTP draft mapping and raw logits match bitwise");
    o.ids = source; o.adapt = 0; o.draft_dense = false;
  }
  for (bool sampled : {false, true}) {
    GenOptions capped;
    capped.ids.resize(160);
    for (size_t i = 0; i < capped.ids.size(); ++i) capped.ids[i] = source[i % source.size()];
    capped.n = 32; capped.ctx = 4096; capped.spec = 2; capped.mtp_prefix = 256;
    capped.sampling.temperature = sampled ? 0.7 : 0;
    capped.sampling.top_k = 17;
    ConversationHooks hooks;
    hooks.boundary = 16;
    hooks.split_at_boundary = true;  // this check needs an unaligned snapshot at the boundary; fresh runs split too
    std::shared_ptr<const ConversationSnapshot> snapshot;
    hooks.save = [&](const ConversationResume& state, float* s, float* c) {
      if (state.length == 16) snapshot = ConversationStateIO::Save(e, e.ActiveSequence(), state, s, c);
    };
    capped.conversation = &hooks;
    RunGen(e, capped, {});
    if (!snapshot || snapshot->resume.mpos != 15) return 1;
    capped.ids.resize(320);
    for (size_t i = 160; i < capped.ids.size(); ++i) capped.ids[i] = source[i % source.size()];
    hooks.save = {};
    for (uint64_t seed : {1, 18073, 49123}) {
      capped.sampling.seed = seed;
      hooks.restore = {};
      const auto fresh = RunGen(e, capped, {});
      hooks.restore = [&](float* s, float* c) {
        return ConversationStateIO::Restore(e, e.ActiveSequence(), *snapshot, s, c);
      };
      const auto hit = RunGen(e, capped, {});
      if (fresh.gen != hit.gen) throw std::runtime_error("bounded-MTP cache IDs differ from fresh IDs");
      std::printf("PASS bounded-MTP sampled=%d seed=%llu: %zu cached/fresh IDs match\n",
                  sampled, (unsigned long long)seed, hit.gen.size());
    }
  }
  return 0;
}
