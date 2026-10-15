// Load-to-use latency by working-set size (L0/L1/L2/MALL/DRAM), and the cost
// of GPU page-table misses when a chase touches one line per page.
//
//   latency [stride_bytes]   default 128 (one line per hop, random order)
#include <numeric>
#include <random>

#include "common.h"

__global__ void Chase(const uint64_t* __restrict__ buf, uint64_t start, int hops,
                      uint64_t* out) {
  if (threadIdx.x != 0) return;
  uint64_t i = start;
  for (int h = 0; h < 2048; ++h) i = buf[i];  // warm TLB/cache path once
  uint64_t t0 = wall_clock64();
  for (int h = 0; h < hops; ++h) i = buf[i];
  uint64_t t1 = wall_clock64();
  out[0] = t1 - t0;
  out[1] = i;
}

int main(int argc, char** argv) {
  size_t stride = argc > 1 ? std::strtoull(argv[1], nullptr, 10) : 128;
  size_t max_ws = argc > 2 ? std::strtoull(argv[2], nullptr, 10) : (4ull << 30);
  uint64_t* out;
  HIP_CHECK(hipMalloc(&out, 64));
  std::printf("# pointer chase, one hop per %zu bytes, random cyclic order\n", stride);
  std::printf("working_set_bytes ns_per_hop\n");
  std::mt19937_64 rng(42);
  for (size_t ws = 16 << 10; ws <= max_ws; ws *= 2) {
    size_t nodes = ws / stride;
    if (nodes < 8) continue;
    size_t words = ws / 8;
    std::vector<uint64_t> h(words, 0);
    std::vector<uint64_t> order(nodes);
    std::iota(order.begin(), order.end(), 0);
    // Sattolo: a single cycle through every node.
    for (size_t k = nodes - 1; k > 0; --k) {
      size_t j = rng() % k;
      std::swap(order[k], order[j]);
    }
    size_t step = stride / 8;
    for (size_t k = 0; k < nodes; ++k)
      h[order[k] * step] = order[(k + 1) % nodes] * step;
    uint64_t* d;
    HIP_CHECK(hipMalloc(&d, ws));
    HIP_CHECK(hipMemcpy(d, h.data(), ws, hipMemcpyHostToDevice));
    // Each rep continues the cycle where the last stopped and walks at least
    // the whole cycle, so the touched footprint really is the working set.
    int hops = (int)std::min<size_t>(4000000, std::max<size_t>(20000, nodes));
    std::vector<double> ns;
    uint64_t pos = order[0] * step;
    for (int r = 0; r < 3; ++r) {
      hipLaunchKernelGGL(Chase, 1, 32, 0, nullptr, d, pos, hops, out);
      HIP_CHECK(hipDeviceSynchronize());
      uint64_t res[2];
      HIP_CHECK(hipMemcpy(res, out, 16, hipMemcpyDeviceToHost));
      ns.push_back(res[0] * 10.0 / hops);
      pos = res[1];
    }
    std::printf("%zu %.1f\n", ws, bench::Median(ns));
    HIP_CHECK(hipFree(d));
  }
  return 0;
}
