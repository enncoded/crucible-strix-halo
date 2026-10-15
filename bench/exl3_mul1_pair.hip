// Check every reachable byte-sum pair and every16-bit state against the old codec.
#include "common.h"
#define EXL3_MUL1_SAD_HI
#include "../engine/exl3/kernels/exl3_gemv.h"
using namespace crucible;
__device__ __forceinline__ __half2 OldPair(uint32_t p0,uint32_t p1) {
  const uint32_t s0=__builtin_amdgcn_sad_u8(p0,0u,0x6400u);
  const uint32_t s1=__builtin_amdgcn_sad_u8(p1,0u,0x6400u);
  const uint32_t q=__builtin_amdgcn_perm(s1,s0,0x05040100u);
  const __half2 h=*reinterpret_cast<const __half2*>(&q);
  const __half2 inv=__halves2half2(__ushort_as_half(0x1eee),__ushort_as_half(0x1eee));
  const __half2 bias=__halves2half2(__ushort_as_half(0xc931),__ushort_as_half(0xc931));
  return __hfma2(h,inv,bias);
}
__device__ uint32_t SumWord(uint32_t sum) {
  uint32_t w=0;
#pragma unroll
  for(int b=0;b<4;++b) {const uint32_t x=min(sum,255u);w|=x<<(b*8);sum-=x;}
  return w;
}
__global__ void PairCheck(unsigned* bad) {
  const uint32_t i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<1021u*1021u) {
    const uint32_t p0=SumWord(i/1021),p1=SumWord(i%1021);
    const auto a=OldPair(p0,p1),b=Mul1Pair(p0,p1);
    if(*reinterpret_cast<const uint32_t*>(&a)!=*reinterpret_cast<const uint32_t*>(&b))atomicAdd(bad,1);
  }
  const uint32_t masks[6]={0u,1u,0x1555u,0x7fffu,0x8000u,0xffffu};
  if(i<65536)for(int j=0;j<6;++j) {
    const uint32_t p0=i*kMul1,p1=(i^masks[j])*kMul1;
    const auto a=OldPair(p0,p1),b=Mul1Pair(p0,p1);
    if(*reinterpret_cast<const uint32_t*>(&a)!=*reinterpret_cast<const uint32_t*>(&b))atomicAdd(bad,1);
  }
}
int main() {
  unsigned* bad;HIP_CHECK(hipMalloc(&bad,4));HIP_CHECK(hipMemset(bad,0,4));
  PairCheck<<<(1021u*1021u+255)/256,256>>>(bad);HIP_CHECK(hipGetLastError());
  unsigned n;HIP_CHECK(hipMemcpy(&n,bad,4,hipMemcpyDeviceToHost));HIP_CHECK(hipFree(bad));
  std::printf("byte_sum_pairs=1042441 state_pairs=393216 mismatches=%u\n",n);
  return n?1:0;
}
