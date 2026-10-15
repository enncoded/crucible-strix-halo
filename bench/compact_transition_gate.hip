// Supplemental snapshot coverage. Primary exactness still uses tools/gates.py.
// The rename is kept for builds against engine sources older than train 20 (an unguarded main in gen_cli.h).
#define ENGINE_LIBRARY
#define main CrucibleCliMain
#ifndef CRUCIBLE_ENGINE_HEADER
#define CRUCIBLE_ENGINE_HEADER "../engine/engine.hip"
#endif
#include CRUCIBLE_ENGINE_HEADER
#undef main
#include "compact_transition.h"

struct CompactObservation {
  std::vector<int> ids;
  std::vector<float> logits;
};

int main(int argc, char** argv) {
  try {
    if (argc != 3 || !CompactStateFlag(std::getenv("CRUCIBLE_COMPACT_STATE"))) return 2;
    std::ifstream input(argv[2]);
    if (!input) throw std::runtime_error("cannot open prompt IDs");
    std::stringstream text; text << input.rdbuf();
    const auto source = ParseIds(text.str());
    if (source.size() < 24 || !ValidTokenIds(source, kVocab))
      throw std::runtime_error("the probe needs at least 24 valid prompt IDs");
    constexpr int context = 4096, rows = 4096, prefix = 3077, continuation = 32;
    Engine engine(argv[1], context, rows);
    GenOptions initial;
    initial.ctx = context; initial.prefill_rows = rows;
    // RunGen drops speculation when n == 1, and only a speculative prompt saves an MTP position.
    initial.spec = 2; initial.n = 2;
    for (int i = 0; i < prefix; ++i) initial.ids.push_back(source[size_t(i) % source.size()]);
    ConversationHooks hooks;
    // The default tail save would replace the prompt snapshot with the generated tail.
    hooks.save_generated_tail = false;
    std::shared_ptr<const ConversationSnapshot> snapshot;
    hooks.save = [&](const ConversationResume& resume, float* stacks, float* carry) {
      snapshot = ConversationStateIO::Save(engine, engine.ActiveSequence(), resume, stacks, carry);
    };
    initial.conversation = &hooks;
    GenOptions resumed;
    resumed.ctx = context; resumed.prefill_rows = rows;
    resumed.spec = 2; resumed.n = continuation;
    ConversationHooks resumed_hooks;
    resumed.conversation = &resumed_hooks;
    const auto result = CompactAcrossStreams(engine,
      [&] {
        RunGen(engine, initial, {});
        if (!snapshot || snapshot->hist.size() != size_t(prefix) || snapshot->resume.mpos <= 0)
          throw std::runtime_error("missing speculative prompt snapshot");
        resumed.ids = snapshot->hist;
        resumed.ids.insert(resumed.ids.end(), source.begin() + 8, source.begin() + 16);
        return snapshot;
      },
      [&](const auto& saved) {
        resumed_hooks.restore = [&](float* stacks, float* carry) {
          const auto resume = ConversationStateIO::Restore(engine, engine.ActiveSequence(), *saved, stacks, carry);
          RequireSameSnapshot(*saved, *ConversationStateIO::Save(engine, engine.ActiveSequence(), resume, stacks, carry));
          return resume;
        };
        const auto generated = RunGen(engine, resumed, {});
        return CompactObservation{generated.gen, engine.CopyLogits(1)};
      },
      [&] {
        GenOptions serial;
        serial.ctx = context; serial.prefill_rows = rows; serial.n = 2; serial.spec = 0;
        serial.ids.assign(source.rbegin(), source.rbegin() + 24);
        RunGen(engine, serial, {});
      },
      [&] { return engine.ActiveSequence().qc.contains(kLayers); });
    if (result.first.ids.size() != continuation || result.first.ids != result.second.ids ||
        result.first.logits.size() != result.second.logits.size() ||
        std::memcmp(result.first.logits.data(), result.second.logits.data(), result.first.logits.size() * sizeof(float)))
      throw std::runtime_error("cross-stream continuation IDs or logits differ");
    std::printf("PASS compact cross-stream snapshot promotion: prefix=%d mpos=%d, %d IDs and copied head logits match\n",
                prefix, snapshot->resume.mpos, continuation);
    std::puts("PASS same-engine serial-to-speculative transition into an initially absent MTP cache");
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "compact transition: %s\n", error.what());
    return 2;
  }
}
