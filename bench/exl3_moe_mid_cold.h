#pragma once
#include "common.h"
// Same strict 256MiB read eviction used by exl3_gemm_common.h.
namespace gm {
struct ColdCache {
  uint32_t *data, *sum;
  size_t words;
  void Free() const {HIP_CHECK(hipFree(data));HIP_CHECK(hipFree(sum));}
};
__global__ void EvictKernel(const uint32_t* src,uint32_t* dst,size_t words) {
  const int tid=blockIdx.x*blockDim.x+threadIdx.x;uint32_t sum=0;
  for(size_t i=tid;i<words/4;i+=gridDim.x*blockDim.x) {
    const uint4 v=reinterpret_cast<const uint4*>(src)[i];sum^=v.x^v.y^v.z^v.w;
  }
  dst[tid]=sum;
}
inline ColdCache MakeColdCache() {
  ColdCache c{nullptr,nullptr,64u*1024*1024};
  HIP_CHECK(hipMalloc(&c.data,c.words*4));HIP_CHECK(hipMalloc(&c.sum,320*256*4));
  HIP_CHECK(hipMemset(c.data,0x51,c.words*4));return c;
}
inline void Flush(const ColdCache* c) {
  EvictKernel<<<320,256>>>(c->data,c->sum,c->words);HIP_CHECK(hipGetLastError());
}
}
