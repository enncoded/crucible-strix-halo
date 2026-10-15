// Probe: does the 32 MB MALL absorb producer -> consumer activation traffic?
// Prefill moves ~239 GB of activations per 4K chunk between kernels. Ring
// buffers that stay inside the MALL only help if a written line can be
// overwritten or read back before it goes to DRAM.
//
//   W   write-only kernels over a ring of S bytes, repeated to 4 GiB written
//   WR  a write kernel then a read kernel over the same S-byte slot,
//       slots rotating over a ring of S*slots bytes, 4 GiB each way
//   R   read-only kernels over a ring of S bytes (reference: MALL re-read)
//
// Store variants: plain stores, and stores with the nontemporal hint.
// Usage: mall_write
#include "../common.h"

namespace {

template <bool NT>
__global__ void __launch_bounds__(256) Write(uint4* p, size_t n, uint32_t seed) {
  const size_t stride = (size_t)gridDim.x * blockDim.x;
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
    const uint4 v = make_uint4(seed, (uint32_t)i, seed ^ 0x55u, (uint32_t)(i >> 7));
    typedef unsigned int U32x4 __attribute__((ext_vector_type(4)));
    if constexpr (NT) {
      const U32x4 w = {v.x, v.y, v.z, v.w};
      __builtin_nontemporal_store(w, reinterpret_cast<U32x4*>(p + i));
    } else {
      p[i] = v;
    }
  }
}

// One kernel, `passes` passes over the same S bytes (no launch gaps).
__global__ void __launch_bounds__(256) WriteLoop(uint4* p, size_t n, int passes) {
  const size_t stride = (size_t)gridDim.x * blockDim.x;
  for (int k = 0; k < passes; ++k)
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride)
      p[i] = make_uint4(k, (uint32_t)i, k ^ 0x55, (uint32_t)(i >> 7));
}

__global__ void __launch_bounds__(256) Read(const uint4* p, size_t n, uint32_t* sink) {
  const size_t stride = (size_t)gridDim.x * blockDim.x;
  uint32_t acc = 0;
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
    const uint4 v = p[i];
    acc ^= v.x + v.y + v.z + v.w;
  }
  if (acc == 0x12345678u) sink[0] = acc;
}

constexpr size_t kTotal = size_t(4) << 30;
constexpr int kGrid = 640;  // 16 blocks per WGP-pair; grid-stride

}  // namespace

int main() {
  uint8_t* buf;
  uint32_t* sink;
  const size_t cap = size_t(2) << 30;
  HIP_CHECK(hipMalloc(&buf, cap));
  HIP_CHECK(hipMalloc(&sink, 64));
  HIP_CHECK(hipMemset(buf, 0, cap));
  const size_t MiB = size_t(1) << 20;
  for (size_t S : {4 * MiB, 8 * MiB, 16 * MiB, 24 * MiB, 32 * MiB, 48 * MiB, 64 * MiB, 256 * MiB}) {
    const int reps = (int)(kTotal / S);
    // Ring of slots covering 2 GiB for the linear reference; S-sized ring
    // of 1 slot for the resident case.
    for (int nt = 0; nt < 2; ++nt) {
      bench::EventTimer t;
      t.Begin();
      for (int r = 0; r < reps; ++r) {
        uint4* p = reinterpret_cast<uint4*>(buf);
        if (nt) hipLaunchKernelGGL(Write<true>, kGrid, 256, 0, nullptr, p, S / 16, (uint32_t)r);
        else hipLaunchKernelGGL(Write<false>, kGrid, 256, 0, nullptr, p, S / 16, (uint32_t)r);
      }
      const float ms = t.EndMs();
      std::printf("W  ring %4zu MiB %s: %7.1f GB/s (%.2f us per kernel)\n", S / MiB, nt ? "nt   " : "plain",
                  kTotal / ms / 1e6, ms * 1e3 / reps);
    }
    {
      bench::EventTimer t;
      t.Begin();
      hipLaunchKernelGGL(WriteLoop, kGrid, 256, 0, nullptr, reinterpret_cast<uint4*>(buf), S / 16, reps);
      const float ms = t.EndMs();
      std::printf("W1 ring %4zu MiB one kernel: %7.1f GB/s\n", S / MiB, kTotal / ms / 1e6);
    }
    for (int ring : {1, 0}) {  // 1: same slot every time; 0: walk 2 GiB
      bench::EventTimer t;
      t.Begin();
      const size_t slots = ring ? 1 : cap / S;
      for (int r = 0; r < reps; ++r) {
        uint4* p = reinterpret_cast<uint4*>(buf + (size_t)(r % slots) * S);
        hipLaunchKernelGGL(Write<false>, kGrid, 256, 0, nullptr, p, S / 16, (uint32_t)r);
        hipLaunchKernelGGL(Read, kGrid, 256, 0, nullptr, p, S / 16, sink);
      }
      const float ms = t.EndMs();
      std::printf("WR %s %4zu MiB slots: %7.1f GB/s moved (%.2f us per write+read pair)\n",
                  ring ? "same-slot" : "walk-2GiB", S / MiB, 2.0 * kTotal / ms / 1e6, ms * 1e3 / reps);
    }
    {
      bench::EventTimer t;
      t.Begin();
      for (int r = 0; r < reps; ++r)
        hipLaunchKernelGGL(Read, kGrid, 256, 0, nullptr, reinterpret_cast<uint4*>(buf), S / 16, sink);
      const float ms = t.EndMs();
      std::printf("R  ring %4zu MiB      : %7.1f GB/s\n", S / MiB, kTotal / ms / 1e6);
    }
  }
  return 0;
}
