// Compare the fp16-only input transform with the existing fp32-exact path.
#include "../engine/exl3/prefill_linear.h"

#include <cstring>
#include <vector>

using namespace crucible;

static void Check(bool ok, const char* message) {
  if (!ok) { std::fprintf(stderr,"FAIL: %s\n",message); std::exit(1); }
}

template<class T> static T* Device(const std::vector<T>& values) {
  T* pointer=nullptr;
  HIP_OK(hipMalloc(&pointer,values.size()*sizeof(T)));
  HIP_OK(hipMemcpy(pointer,values.data(),values.size()*sizeof(T),hipMemcpyHostToDevice));
  return pointer;
}

static void Test(int rows, bool packed) {
  constexpr int columns=2560;
  const size_t count=(size_t)rows*columns;
  std::vector<__half> half(count), scale(columns);
  std::vector<float> wide(count);
  for (size_t i=0;i<count;++i) {
    half[i]=__float2half((float)((int)(i*131%4093)-2046)/512.f);
    wide[i]=__half2float(half[i]);
  }
  for (int i=0;i<columns;++i) scale[i]=__float2half((float)((i*71%509)-254)/128.f);
  auto* xh=Device(half); auto* xf=Device(wide); auto* su=Device(scale);
  __half *before=nullptr,*after=nullptr;
  HIP_OK(hipMalloc(&before,count*sizeof(__half))); HIP_OK(hipMalloc(&after,count*sizeof(__half)));
  LinJob job{}; job.src=xf; job.suh=su; job.xh=before;
  job.I=columns; job.O=128; job.K=5; job.R=rows; job.lds=columns;
  auto* jobs=Device(std::vector<LinJob>{job});
  std::vector<XfItem> list;
  for (int r=0;r<rows;++r) for (int b=0;b<columns/128;++b) list.push_back({0,r,b});
  auto* items=Device(list); const int n=list.size();
  if (packed) PrefillLinInPackedKernel<<<(n+7)/8,256>>>(jobs,items,n);
  else LinInKernel<<<(n+7)/8,256>>>(jobs,items,n);
  job.xh=after; HIP_OK(hipMemcpy(jobs,&job,sizeof(job),hipMemcpyHostToDevice));
  if (packed) PrefillLinInHalfKernel<true><<<(n+7)/8,256>>>(jobs,items,n,xh);
  else PrefillLinInHalfKernel<false><<<(n+7)/8,256>>>(jobs,items,n,xh);
  std::vector<__half> a(count),b(count);
  HIP_OK(hipMemcpy(a.data(),before,count*sizeof(__half),hipMemcpyDeviceToHost));
  HIP_OK(hipMemcpy(b.data(),after,count*sizeof(__half),hipMemcpyDeviceToHost));
  Check(std::memcmp(a.data(),b.data(),count*sizeof(__half))==0,"input transform differs");
  for (void* p : {(void*)xh,(void*)xf,(void*)su,(void*)before,(void*)after,(void*)jobs,(void*)items})
    HIP_OK(hipFree(p));
}

int main() {
  for (int rows : {1,8,9,24,33,1024}) for (bool packed : {false,true}) Test(rows,packed);
  std::puts("PASS: fp16 source transform, packed and row-major, 1/8/9/24/33/1024 rows");
}
