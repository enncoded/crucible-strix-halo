// Fixed host buffers must survive repeated asynchronous transfers and reuse.
#include <algorithm>
#include <cassert>
#include <type_traits>
#include "../engine/gfx1151/pinned_buffer.h"

using namespace crucible;

__global__ void DelayCopy() {
  // gfx1151 clock64() reads a wrapping 20-bit counter. A bounded instruction
  // loop avoids a signed elapsed-time test that can never finish after wrap.
  for (int i = 0; i < 1024; ++i) __builtin_amdgcn_s_sleep(1);
}

int main() {
  setenv("HSA_USERPTR_FOR_PAGED_MEM", "0", 0);
  static_assert(!std::is_copy_constructible_v<PinnedBuffer<int, kMaxRows>>);
  PinnedBuffer<int, kMaxRows> input, output;
  PinnedBuffer<int64_t, kMaxRows * 16> rows;
  int *device;
  HIP_OK(hipMalloc(&device, kMaxRows * sizeof(int)));
  int* const original = input.data();
  for (int repeat = 0; repeat < 64; ++repeat) {
    for (int count : {1, 3, kMaxRows}) {
      for (int i = 0; i < count; ++i) input.data()[i] = repeat * 100 + i;
      HIP_OK(hipMemcpyAsync(device, input.data(), count * sizeof(int), hipMemcpyHostToDevice));
      HIP_OK(hipMemcpyAsync(output.data(), device, count * sizeof(int), hipMemcpyDeviceToHost));
      HIP_OK(hipStreamSynchronize(0));
      assert(std::equal(input.data(), input.data() + count, output.data()));
      assert(input.data() == original);
    }
  }
  HIP_OK(hipFree(device));
  // No output synchronization separates these uploads. Reuse must wait for
  // the earlier DMA, including on different streams.
  PinnedInputBuffer<int, 1> staging;
  hipStream_t streams[2];
  for (auto& stream : streams) HIP_OK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
  HIP_OK(hipMalloc(&device, 64 * sizeof(int)));
  for (int i = 0; i < 64; ++i) {
    auto stream = streams[i % 2];
    DelayCopy<<<1, 1, 0, stream>>>();
    staging.CopyTo(device + i, &i, 1, stream);
  }
  HIP_OK(hipDeviceSynchronize());
  int values[64];
  HIP_OK(hipMemcpy(values, device, sizeof(values), hipMemcpyDeviceToHost));
  for (int i = 0; i < 64; ++i) assert(values[i] == i);
  for (auto stream : streams) HIP_OK(hipStreamDestroy(stream));
  HIP_OK(hipFree(device));
  std::puts("fixed pinned transfers: PASS");
}
