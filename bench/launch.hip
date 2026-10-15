// Kernel-boundary costs on gfx1151: what a dependent kernel chain loses
// between kernels, measured on the GPU's own 100 MHz wall clock.
//
//   launch   host throughput, device-side gaps (plain stream and HIP graph)
#include "common.h"

__device__ uint32_t g_done;

// Records the first-wave start and last-block end of each kernel in a chain.
__global__ void Stamp(uint64_t* stamps, int idx, int spin_ns) {
  if (threadIdx.x == 0 && blockIdx.x == 0) stamps[2 * idx] = wall_clock64();
  if (spin_ns > 0 && threadIdx.x == 0) {
    uint64_t t0 = wall_clock64();
    while ((wall_clock64() - t0) * 10 < (uint64_t)spin_ns) {
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    uint32_t prev = atomicAdd(&g_done, 1u);
    if (prev == gridDim.x - 1) {
      stamps[2 * idx + 1] = wall_clock64();
      g_done = 0;
    }
  }
}

__global__ void Empty() {}

static void Report(const char* name, const std::vector<uint64_t>& s, int n) {
  std::vector<double> gaps, durs;
  for (int i = 1; i < n; ++i) gaps.push_back((s[2 * i] - s[2 * i - 1]) * 10.0);
  for (int i = 0; i < n; ++i) durs.push_back((s[2 * i + 1] - s[2 * i]) * 10.0);
  std::sort(gaps.begin(), gaps.end());
  std::printf("%-34s gap p10 %.0f ns, p50 %.0f ns, p90 %.0f ns; kernel span p50 %.0f ns\n",
              name, gaps[gaps.size() / 10], gaps[gaps.size() / 2],
              gaps[gaps.size() * 9 / 10], bench::Median(durs));
}

int main() {
  const int kN = 2000;
  uint64_t* stamps;
  HIP_CHECK(hipMalloc(&stamps, 2 * kN * sizeof(uint64_t)));
  hipStream_t st;
  HIP_CHECK(hipStreamCreate(&st));
  bench::EventTimer t;

  for (int i = 0; i < 100; ++i) hipLaunchKernelGGL(Empty, 1, 64, 0, st);
  HIP_CHECK(hipStreamSynchronize(st));
  t.Begin(st);
  for (int i = 0; i < 20000; ++i) hipLaunchKernelGGL(Empty, 1, 64, 0, st);
  std::printf("empty kernel, host launch throughput: %.2f us/launch\n",
              t.EndMs(st) * 1e3 / 20000);

  int cus = bench::NumCUs();
  for (int blocks : {1, cus, 4 * cus}) {
    for (int spin : {0, 2000}) {
      std::vector<uint64_t> h(2 * kN);
      for (int i = 0; i < kN; ++i)
        hipLaunchKernelGGL(Stamp, blocks, 256, 0, st, stamps, i, spin);
      HIP_CHECK(hipStreamSynchronize(st));
      HIP_CHECK(hipMemcpy(h.data(), stamps, h.size() * 8, hipMemcpyDeviceToHost));
      char name[96];
      std::snprintf(name, sizeof name, "stream, %d blocks, %d ns work", blocks, spin);
      Report(name, h, kN);

      hipGraph_t g;
      hipGraphExec_t ge;
      HIP_CHECK(hipStreamBeginCapture(st, hipStreamCaptureModeGlobal));
      for (int i = 0; i < kN; ++i)
        hipLaunchKernelGGL(Stamp, blocks, 256, 0, st, stamps, i, spin);
      HIP_CHECK(hipStreamEndCapture(st, &g));
      HIP_CHECK(hipGraphInstantiate(&ge, g, nullptr, nullptr, 0));
      HIP_CHECK(hipGraphLaunch(ge, st));
      HIP_CHECK(hipStreamSynchronize(st));
      t.Begin(st);
      HIP_CHECK(hipGraphLaunch(ge, st));
      float ms = t.EndMs(st);
      HIP_CHECK(hipMemcpy(h.data(), stamps, h.size() * 8, hipMemcpyDeviceToHost));
      std::snprintf(name, sizeof name, "graph,  %d blocks, %d ns work", blocks, spin);
      Report(name, h, kN);
      std::printf("%-34s %.2f us per kernel end to end\n", "", ms * 1e3 / kN);
      HIP_CHECK(hipGraphExecDestroy(ge));
      HIP_CHECK(hipGraphDestroy(g));
    }
  }
  return 0;
}
