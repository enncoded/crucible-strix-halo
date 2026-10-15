// Cross-workgroup synchronization inside one persistent kernel: the cost a
// megakernel pays per dependency instead of a kernel boundary.
//
//   sync   ping-pong one-way latency between two workgroups; grid barrier cost
#include "common.h"

enum Mode { kAcquireSpin = 0, kRelaxedSpinThenFence = 1 };

template <int MODE>
__device__ inline uint32_t WaitAtLeast(uint32_t* p, uint32_t v) {
  uint32_t x;
  if constexpr (MODE == kAcquireSpin) {
    while ((x = __hip_atomic_load(p, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT)) < v) {
    }
  } else {
    while ((x = __hip_atomic_load(p, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT)) < v) {
    }
    __atomic_thread_fence(__ATOMIC_ACQUIRE);  // agent-scope acquire once
  }
  return x;
}

// Block 0 and block `peer` take turns incrementing a counter.
template <int MODE>
__global__ void PingPong(uint32_t* flag, int peer, int iters, uint64_t* out) {
  if (threadIdx.x != 0) return;
  int me = blockIdx.x;
  if (me != 0 && me != peer) return;
  uint64_t t0 = wall_clock64();
  for (int i = 0; i < iters; ++i) {
    uint32_t turn = 2 * i + (me == 0 ? 0 : 1);
    WaitAtLeast<MODE>(flag, turn);
    __hip_atomic_store(flag, turn + 1, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
  }
  if (me == 0) out[0] = wall_clock64() - t0;
}

// Sense-free counting barrier: arrival counter grows by gridDim.x per round.
template <int MODE>
__global__ void Barrier(uint32_t* counter, int rounds, uint64_t* out) {
  uint64_t t0 = wall_clock64();
  for (int r = 1; r <= rounds; ++r) {
    __syncthreads();
    if (threadIdx.x == 0) {
      __hip_atomic_fetch_add(counter, 1u, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
      WaitAtLeast<MODE>(counter, r * gridDim.x);
    }
    __syncthreads();
  }
  if (blockIdx.x == 0 && threadIdx.x == 0) out[0] = wall_clock64() - t0;
}

int main() {
  uint32_t* flag;
  uint64_t* out;
  HIP_CHECK(hipMalloc(&flag, 256));
  HIP_CHECK(hipMalloc(&out, 64));
  const int kIters = 20000;
  int cus = bench::NumCUs();
  for (int mode = 0; mode < 2; ++mode) {
    std::vector<double> one_way;
    for (int peer = 1; peer < cus; peer += 2) {
      HIP_CHECK(hipMemset(flag, 0, 4));
      auto k = mode == 0 ? PingPong<0> : PingPong<1>;
      // One block per CU (cus blocks of 64 threads) so peers sit on other CUs.
      hipLaunchKernelGGL(k, cus, 64, 0, nullptr, flag, peer, kIters, out);
      HIP_CHECK(hipDeviceSynchronize());
      uint64_t ticks;
      HIP_CHECK(hipMemcpy(&ticks, out, 8, hipMemcpyDeviceToHost));
      one_way.push_back(ticks * 10.0 / (2.0 * kIters));
    }
    std::sort(one_way.begin(), one_way.end());
    std::printf("ping-pong one-way, %s: min %.0f ns, median %.0f ns, max %.0f ns\n",
                mode == 0 ? "acquire spin      " : "relaxed spin+fence",
                one_way.front(), one_way[one_way.size() / 2], one_way.back());
  }
  const int kRounds = 5000;
  for (int mode = 0; mode < 2; ++mode)
    for (int blocks : {cus / 2, cus, 2 * cus, 4 * cus}) {
      HIP_CHECK(hipMemset(flag, 0, 4));
      auto k = mode == 0 ? Barrier<0> : Barrier<1>;
      hipLaunchKernelGGL(k, blocks, 256, 0, nullptr, flag, kRounds, out);
      HIP_CHECK(hipDeviceSynchronize());
      uint64_t ticks;
      HIP_CHECK(hipMemcpy(&ticks, out, 8, hipMemcpyDeviceToHost));
      std::printf("grid barrier, %3d blocks x 256, %s: %.0f ns per barrier\n", blocks,
                  mode == 0 ? "acquire spin      " : "relaxed spin+fence",
                  ticks * 10.0 / kRounds);
    }
  return 0;
}
