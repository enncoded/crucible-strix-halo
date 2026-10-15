// Compile-only target for the exact K5 decoder's instruction count.
#include "kernels/exl3_gemv.h"
using namespace crucible;
int main() {
  hipLaunchKernelGGL((Exl3GemvKernel<5, 1, kMulTrue16, 8>), 1, 256, 5120, nullptr,
                     static_cast<const Exl3Job*>(nullptr), static_cast<const Exl3Item*>(nullptr), 0);
  return 0;
}
