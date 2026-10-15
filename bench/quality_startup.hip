// Run the unchanged CLI with test-only allocation tracking and migration settling.
#include "quality_settle.h"

template <class T>
hipError_t QualityAlloc(T** pointer, size_t bytes) {
  const hipError_t result = hipMalloc(pointer, bytes);
  if (result == hipSuccess) crucible::TraceBuffer(*pointer,bytes);
  return result;
}

namespace { class Engine; void QualityReady(Engine& engine); }
#define hipMalloc QualityAlloc
#define CRUCIBLE_BENCH_READY(engine) QualityReady(engine)
#define ENGINE_LIBRARY
#include "../engine/engine.hip"
#undef hipMalloc
#undef CRUCIBLE_BENCH_READY

namespace {
struct EngineStateIO {
  static void Prepare(Engine& engine) {
    for (int rows = 1; rows <= kMaxRows; ++rows) engine.PlansFor(rows);
  }
};
void QualityReady(Engine& engine) {
  CounterSnapshot("loaded");
  if (std::getenv("CRUCIBLE_SETTLE")) {
    EngineStateIO::Prepare(engine);
    TouchAndSettle();
  }
  CounterSnapshot("ready");
}
}

int main(int argc, char** argv) {
  const int result = CrucibleCliMain(argc,argv);
  CounterSnapshot("done");
  return result;
}
