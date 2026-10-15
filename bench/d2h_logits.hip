// D2H cost of sampled-logit copies (1 and 4 rows of 248,320 fp32), into pageable and pinned host memory.
#include <hip/hip_runtime.h>
#include <chrono>
#include <cstdio>
#include <vector>
#include <algorithm>
int main() {
  const size_t V = 248320; float* d; hipMalloc(&d, 8 * V * 4); hipMemset(d, 0, 8 * V * 4);
  std::vector<float> h(8 * V); float* p; hipHostMalloc(&p, 8 * V * 4);
  for (int rows : {1, 4}) for (int pinned : {0, 1}) {
    double best = 1e9;
    for (int it = 0; it < 50; ++it) {
      hipDeviceSynchronize();
      auto t0 = std::chrono::steady_clock::now();
      hipMemcpy(pinned ? p : h.data(), d, rows * V * 4, hipMemcpyDeviceToHost);
      best = std::min(best, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
    }
    std::printf("rows %d %s: %.3f ms\n", rows, pinned ? "pinned" : "pageable", best);
  }
}
