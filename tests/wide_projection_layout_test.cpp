#include <cassert>
#include "../bench/exact_projection_geometry.h"
using namespace crucible::wide_probe;
int main() {
  assert(!FullWaves(0));
  assert(!FullWaves(16));
  assert(!FullWaves(129));
  assert(FullWaves(128));
  assert(FullWaves(16384));
  assert(!CompleteChunks(0));
  assert(!CompleteChunks(-8));
  for (int span : {5,20,40,48,80,160}) assert(CompleteChunks(span)==(span%8==0));
  for (int rows : {8,16,32}) for (int item=0;item<99;++item) for (int stripe=0;stripe<rows/8;++stripe) {
    const int block=item*(rows/8)+stripe;
    assert(InterleavedItem(block,rows)==item);
    assert(InterleavedRow(block,rows)==stripe*8);
  }
  for (int rows : {8,16,32}) for (int splits : {1,2,4,8})
    for (int columns : {128,2560,16384})
      for (int split=0;split<splits;++split) for (int stripe=0;stripe<rows;stripe+=8)
        for (int local=0;local<8;++local) {
          const size_t base=StripePartialBase(split,rows,stripe,columns);
          assert(base+size_t(split*8+local)*columns==size_t(split*rows+stripe+local)*columns);
        }
  for (int rows : {8,16,32}) {
    assert(RollingLdsBytes(rows)==size_t(rows)*256);
    for (int span : {5,20,40,48,80,160}) for (int row=0;row<rows;++row)
      for (int chunk=0;chunk<span;chunk+=8) for (int tile=0;tile<8 && chunk+tile<span;++tile)
        for (int word=0;word<8;++word) {
          const size_t shared=RollingSlot(row,tile,word);
          assert(shared<size_t(rows)*64);
          assert(shared/64==size_t(row));
          assert((shared%64)/8==size_t(tile));
          assert(shared%8==size_t(word));
        }
  }
}
