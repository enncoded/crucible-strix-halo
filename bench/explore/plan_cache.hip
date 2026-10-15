// Build many exact-row plans without the cost of full prompt forwards.
#define ENGINE_LIBRARY
#include "../../engine/engine.hip"

namespace {
struct EngineStateIO {
  static size_t Bytes(const Engine& e) {
    size_t bytes = 0;
    for (const auto& [rows, entry] : e.prefill_plans_)
      for (const auto& layer : entry.second->layers)
        bytes += layer.input.DeviceBytes() + layer.output.DeviceBytes();
    return bytes;
  }
  static void Build(Engine& e, int rows) {
    const auto& p = e.PrefillPlansFor(rows);
    size_t bytes = 0, free = 0, total = 0;
    for (const auto& layer : p.layers)
      bytes += layer.input.DeviceBytes() + layer.output.DeviceBytes();
    HIP_OK(hipMemGetInfo(&free, &total));
#ifdef CRUCIBLE_PLAN_BYTE_CAP
    if (e.prefill_plan_bytes_ != Bytes(e) || Bytes(e) > kPrefillPlanBytes)
      throw std::runtime_error("plan byte accounting or budget failed");
#endif
    std::printf("{\"rows\":%d,\"plan_bytes\":%zu,\"cached_count\":%zu,"
                "\"cached_bytes\":%zu,\"device_free_bytes\":%zu}\n",
                rows, bytes, e.prefill_plans_.size(), Bytes(e), free);
    std::fflush(stdout);
  }
};
}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) return 2;
  if (!SetRuntimeMemoryDefaults()) return 1;
  Engine engine(argv[1], 32768, 4096);
  for (int rows : {24, 256, 1024, 2048}) EngineStateIO::Build(engine, rows);
  for (int i = 0; i < 24; ++i) EngineStateIO::Build(engine, 4096 - i);
  EngineStateIO::Build(engine, 24);
  EngineStateIO::Build(engine, 4096);
  engine.SetPrefillTiles(PrefillTiles{});
  if (EngineStateIO::Bytes(engine) != 0) return 1;
  std::puts("cache clear: zero bytes");
}
