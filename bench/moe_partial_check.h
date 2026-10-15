#pragma once
#include <bit>
#include <cstdint>
#include <cstring>
#include <span>

namespace crucible_bench {
// Both halves are NaN in half mode. Each live cell must replace the poison.
template<bool HALF,bool CANDIDATE> constexpr uint32_t PartialPoison() {
  if constexpr(HALF)return CANDIDATE?0x7e017e01u:0x7e007e00u;
  else return CANDIDATE?0x7fc00001u:0x7fc00000u;
}
inline bool PartialShape(std::span<const float> data,std::span<const int> rows,int width) {
  if(width<=0||data.empty())return false;
  size_t expected=0;
  for(int count:rows) {
    if(count<0||(size_t)count>(data.size()-expected)/(size_t)width)return false;
    expected+=(size_t)count*width;
  }
  return expected==data.size();
}
template<bool HALF> bool PartialCovered(std::span<const float> data,
    std::span<const int> rows,int width,uint32_t poison) {
  if(!PartialShape(data,rows,width))return false;
  size_t base=0;
  for(int count:rows) {
    const size_t cells=(size_t)count*width;
    for(size_t i=0;i<cells;++i) {
      const uint32_t word=std::bit_cast<uint32_t>(data[base+(HALF?i/2:i)]);
      const uint32_t value=HALF?((word>>((i&1)*16))&0xffffu):word;
      const uint32_t mask=HALF?0x7c00u:0x7f800000u;
      if((value&mask)==mask)return false;
    }
    if constexpr(HALF) {
      // Half partials occupy the first half of each expert's float allocation.
      for(size_t byte=cells*2;byte<cells*4;++byte) {
        const uint32_t word=std::bit_cast<uint32_t>(data[base+byte/4]);
        if(((word>>((byte&3)*8))&255u)!=((poison>>((byte&3)*8))&255u))return false;
      }
    }
    base+=cells;
  }
  return true;
}
template<bool HALF> bool PartialMatches(std::span<const float> reference,
    std::span<const float> candidate,std::span<const int> rows,int width) {
  if(!PartialCovered<HALF>(reference,rows,width,PartialPoison<HALF,false>())||
     !PartialCovered<HALF>(candidate,rows,width,PartialPoison<HALF,true>()))return false;
  size_t base=0;
  for(int count:rows) {
    const size_t cells=(size_t)count*width;
    if(std::memcmp(reference.data()+base,candidate.data()+base,cells*(HALF?2:4)))return false;
    base+=cells;
  }
  return true;
}
}  // namespace crucible_bench
