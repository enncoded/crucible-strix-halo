// Optional greedy multi-stream CLI. The serving worker and sampler do not use this path yet.
#pragma once

#include <filesystem>
#include <numeric>

#include "batch_options.h"

inline bool HasBatchOptions(int argc, char** argv) {
  for (int i = 3; i < argc; i += 2)
    if (std::string_view(argv[i]) == "--streams" || std::string_view(argv[i]) == "--batch-list") return true;
  return false;
}

inline int PrepareBatchDump(const BatchOptions& options) {
  if (!options.dump.empty()) {
    try {
      for (int s = 0; s < options.streams; ++s) {
        const auto dir = std::filesystem::path(options.dump) / ("stream_" + std::to_string(s));
        std::filesystem::create_directories(dir);
        for (const auto& entry : std::filesystem::directory_iterator(dir))
          if (entry.path().filename().string().starts_with("h_")) {
            std::fprintf(stderr, "batch dump directory already contains residual files\n");
            return 2;
          }
      }
    } catch (const std::filesystem::filesystem_error& failure) {
      std::fprintf(stderr, "cannot prepare batch dump directories: %s\n", failure.what());
      return 2;
    }
  }
  return 0;
}

inline double PrefillBatch(Engine& eng, const BatchOptions& options, std::vector<int>& tokens) {
  const double prefill_start = Now();
  for (int s = 0; s < options.streams; ++s) {
    eng.SelectStream(s);
    eng.Reset();
    const auto& prompt = options.prompts[s];
    const std::string dump = options.dump.empty() ? "" : options.dump + "/stream_" + std::to_string(s);
    const int chunk = options.prefill_rows ? options.prefill_rows : options.chunk;
    for (size_t i = 0; i < prompt.size(); i += chunk) {
      const int R = (int)std::min<size_t>(chunk, prompt.size() - i);
      tokens[s] = options.prefill_rows ? eng.Prefill(prompt.data() + i, R, dump).back()
                                      : eng.Forward(prompt.data() + i, R, 0, R, dump).back();
    }
  }
  return Now() - prefill_start;
}

inline void PrintBatch(const BatchOptions& options, const std::vector<std::vector<int>>& generated,
                       double prefill_s, double decode_s) {
  size_t prompt_tokens = 0;
  for (const auto& prompt : options.prompts) prompt_tokens += prompt.size();
  for (int s = 0; s < options.streams; ++s) {
    std::printf("stream %d ids:", s);
    for (int token : generated[s]) std::printf(" %d", token);
    std::printf("\n");
  }
  const size_t decoded = (size_t)options.streams * (options.n - 1);
  const double aggregate = decoded == 0 ? 0 : decoded / decode_s;
  std::printf("batch %d streams, prefill %zu tok in %.3f s, decode %zu tok in %.3f s "
              "(%.2f aggregate tok/s, %.2f per-stream tok/s)\n", options.streams, prompt_tokens,
              prefill_s, decoded, decode_s, aggregate, aggregate / options.streams);
}

inline int BatchCommand(int argc, char** argv) {
  BatchOptions options;
  const std::string error = ParseBatchOptions(argc, argv, options);
  if (!error.empty()) {
    std::fprintf(stderr, "%s\n", error.c_str());
    return 2;
  }
  if (const int result = PrepareBatchDump(options)) return result;
  Engine eng(argv[2], options.ctx, options.prefill_rows);
  eng.ConfigureStreams(options.streams);
  eng.WarmPlans();
  std::vector<int> streams(options.streams), tokens(options.streams);
  std::iota(streams.begin(), streams.end(), 0);
  const double prefill_s = PrefillBatch(eng, options, tokens);
  std::vector<std::vector<int>> generated(options.streams);
  for (int s = 0; s < options.streams; ++s) generated[s].push_back(tokens[s]);
  const double decode_start = Now();
#if CRUCIBLE_DEBUG_NONFINITE & 4
  // Inject probe: the NaN goes into one stream's row of a packed decode pass. The pass ends with the
  // nonfinite error; the other streams' dumps up to that pass must equal a run without the NaN.
  eng.SetInject(options.inject);
  try {
#endif
  for (int step = 1; step < options.n; ++step) {
    tokens = eng.ForwardBatch(tokens, streams, options.dump);
    for (int s = 0; s < options.streams; ++s) generated[s].push_back(tokens[s]);
  }
#if CRUCIBLE_DEBUG_NONFINITE & 4
  } catch (const std::invalid_argument& error) {
    if (!std::string_view(error.what()).starts_with("nonfinite model logits")) throw;
    PrintBatch(options, generated, prefill_s, Now() - decode_start);
    std::printf("probe_error %s\n", error.what());
    std::fprintf(stderr, "error: %s\n", error.what());
    return 3;
  }
#endif
  const double decode_s = Now() - decode_start;
  PrintBatch(options, generated, prefill_s, decode_s);
  return 0;
}
