// Checks whether gfx1151 raw buffer loads include soffset in the OOB check.
#include <hip/hip_runtime.h>
#include <cstdio>
#include "../engine/exl3/kernels/exl3_gemv.h"

__global__ void Probe(const uint32_t* base, uint32_t* out) {
  const auto rs = crucible::StreamRsrc(base, 4096);
  const int lane = threadIdx.x;
  // Lane 0: voffset in range, soffset past num_records. Lane 1: voffset past.
  out[lane*3+0] = __builtin_amdgcn_raw_buffer_load_b32(rs, lane*4, 0, 0);
  out[lane*3+1] = __builtin_amdgcn_raw_buffer_load_b32(rs, lane*4, 4096, 0);
  out[lane*3+2] = __builtin_amdgcn_raw_buffer_load_b32(rs, 4096+lane*4, 0, 0);
}

int main() {
  uint32_t *buf, *out;
  hipMalloc(&buf, 16384); hipMalloc(&out, 96*4);
  uint32_t host[4096];
  for (int i = 0; i < 4096; ++i) host[i] = 0x1000+i;
  hipMemcpy(buf, host, 16384, hipMemcpyHostToDevice);
  Probe<<<1,32>>>(buf, out);
  uint32_t r[96];
  hipMemcpy(r, out, sizeof r, hipMemcpyDeviceToHost);
  for (int l = 0; l < 2; ++l)
    std::printf("lane %d in=%#x soffset_past=%#x voffset_past=%#x\n", l, r[l*3], r[l*3+1], r[l*3+2]);
  std::printf("soffset_checked=%d\n", r[1] == 0);
}
