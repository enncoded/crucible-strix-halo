// Exercise retained-prefix replay after a GEMM prompt and a full trunk restore.
#define CRUCIBLE_QSA_NO_MAIN
#include "qsa_engine_long.hip"

int main(int argc, char** argv) {
  if (argc != 7) {
    std::fprintf(stderr, "usage: prefill_state_gates CKPT IDS ROWS CTX STATE OUTPUT_TOKENS\n");
    return 2;
  }
  if (setenv("HSA_USERPTR_FOR_PAGED_MEM", "0", 0) != 0) {
    std::perror("cannot set HIP host allocation policy");
    return 1;
  }
  const auto ids = ReadIds(argv[2]);
  const int rows = std::stoi(argv[3]), ctx = std::stoi(argv[4]), count = std::stoi(argv[6]);
  Check(rows >= 9 && rows <= kMaxPrefillRows, "prefill row capacity");
  Check(ctx > 0 && ctx <= 262144 && RequestFitsContext(ids.size(), count, ctx), "request context");
  Check(count > 1, "at least two output tokens");
  Engine engine(argv[1], ctx, rows);
  engine.Reset();
  int next = -1;
  for (size_t i = 0; i < ids.size(); i += rows) {
    const int r = (int)std::min<size_t>(rows, ids.size() - i);
    const auto out = engine.Prefill(ids.data() + i, r, "", i + r == ids.size());
    if (!out.empty()) next = out[0];
  }
  Check(next >= 0, "prefill next token");
  EngineStateIO::Transfer(engine, argv[5], true, next);
  VerifyRestored(engine, argv[5], count, ctx);
  std::printf("PASS GEMM prefill state: %zu prompt tokens, %d rows per chunk\n", ids.size(), rows);
  return 0;
}
