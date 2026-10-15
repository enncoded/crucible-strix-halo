#define ENGINE_LIBRARY
#include "../../engine/engine.hip"
#include "startup_trace.h"
#include <cstring>

int main(int argc, char** argv) {
  if (argc != 3 || std::strlen(argv[2]) != 1 || argv[2][0] < '0' || argv[2][0] > '4') {
    std::fprintf(stderr, "usage: startup_probe CKPT MODE (0 baseline, 1 SVM off, 2 threshold, 3 eager, 4 all)\n");
    return 2;
  }
  const int mode = argv[2][0] - '0';
  // Each mode starts a new process. ROCm reads these flags during initialization.
  const bool svm_off = mode == 1 || mode == 4;
  const bool threshold = mode == 2 || mode == 4;
  const bool eager = mode == 3 || mode == 4;
  if (setenv("HSA_USERPTR_FOR_PAGED_MEM", "0", 1) ||
      setenv("HSA_USE_SVM", svm_off ? "0" : "1", 1) ||
      setenv("GPU_PINNED_MIN_XFER_SIZE", threshold ? "65536" : "1", 1) ||
      setenv("CRUCIBLE_PROBE_EAGER", eager ? "1" : "0", 1) ||
      setenv("CRUCIBLE_WARMUP", "1", 1)) {
    std::perror("startup probe environment");
    return 2;
  }
  std::printf("{\"mode\":%d,\"svm_off\":%s,\"threshold_64gib\":%s,\"eager\":%s}\n",
              mode, svm_off ? "true" : "false", threshold ? "true" : "false", eager ? "true" : "false");
  const auto total = ProbeNow();
  const auto init = ProbeNow();
  HIP_OK(hipInit(0));
  ProbePhase("hip_init", init);
  Engine engine(argv[1], 32768, 4096);
  ProbePhase("engine_ready", total);
  const auto resident = ProbeNow();
  engine.WarmPlans();
  HIP_OK(hipDeviceSynchronize());
  ProbePhase("resident_plans", resident);
  ProbePhase("resident_ready", total);
  ProbeCounters();
  return 0;
}
