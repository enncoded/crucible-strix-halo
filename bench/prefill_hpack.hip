// Bound the optional H pack cost before selecting a custom copy kernel.
#include "prefill_common.h"
#include <cstring>
using namespace crucible;
using namespace prefill_bench;

template<int UNROLL>
__global__ void PackHRows(const float* src,float* dst,int rows,int stride) {
  using U4=uint32_t __attribute__((ext_vector_type(4)));
  const size_t count=(size_t)rows*kWide/4,step=(size_t)gridDim.x*blockDim.x;
  for (size_t begin=(size_t)blockIdx.x*blockDim.x+threadIdx.x;begin<count;begin+=UNROLL*step) {
    if (begin+(UNROLL-1)*step<count) {
      U4 values[UNROLL];
#pragma unroll
      for (int u=0;u<UNROLL;++u) {
        const size_t i=begin+u*step;
        values[u]=*reinterpret_cast<const U4*>(src+(i/(kWide/4))*stride+4*(i%(kWide/4)));
      }
#pragma unroll
      for (int u=0;u<UNROLL;++u) reinterpret_cast<U4*>(dst)[begin+u*step]=values[u];
    } else {
#pragma unroll
      for (int u=0;u<UNROLL;++u) {
        const size_t i=begin+u*step;
        if (i<count) reinterpret_cast<U4*>(dst)[i]=
            *reinterpret_cast<const U4*>(src+(i/(kWide/4))*stride+4*(i%(kWide/4)));
      }
    }
  }
}

int main() {
  constexpr int stride=kWide+64;
  for (int R : {1,8,9,33,1024,4096}) {
    const auto expected=Random((size_t)R*kWide,.3f,101);
    std::vector<float> padded((size_t)R*stride,123.f);
    for (int r=0;r<R;++r) std::copy_n(expected.data()+(size_t)r*kWide,kWide,padded.data()+(size_t)r*stride);
    Buffer<float> source(padded),target(expected.size());
    const auto copy=[&] { HIP_OK(hipMemcpy2DAsync(target.p,kWide*sizeof(float),source.p,stride*sizeof(float),
                                              kWide*sizeof(float),R,hipMemcpyDeviceToDevice,0)); };
    copy(); CheckLaunch();
    if (target.Get()!=expected) return 1;
    const double bytes=2.*R*kWide*sizeof(float),us=Time(copy,32),gbps=bytes/us/1e3;
    std::printf("R%d hipMemcpy2DAsync %.3f us %.1f GB/s %.2f%% DRAM\n",R,us,gbps,gbps/245.5*100);
    for (int grid : {20,40,80,256}) {
      const auto kernel=[&] { PackHRows<16><<<grid,128>>>(source.p,target.p,R,stride); };
      kernel(); CheckLaunch();
      const auto got=target.Get();
      if (std::memcmp(expected.data(),got.data(),expected.size()*sizeof(float))) return 1;
      const double time=Time(kernel,32),rate=bytes/time/1e3;
      std::printf("R%d grid%d u16 %.3f us %.1f GB/s %.2f%% DRAM\n",R,grid,time,rate,rate/245.5*100);
    }
  }
  std::puts("PASS: all H packs match bit for bit");
}
