"""Compile the clipped probe loop against a CPU engine, without HIP."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).parents[1]
DRIVER = r'''
#include "bench/long_probe_runtime.h"
#include <cassert>
#include <limits>
struct Engine {
  size_t position = 0;
  bool bad_ids = false;
  std::vector<float> buffer;
  void Reset() { position = 0; }
  const float* logits() { return buffer.data(); }
  std::vector<int> Forward(const int*,int rows,int,int) {
    buffer.assign(rows*4,0);
    std::vector<int> result;
    for(int row=0;row<rows;++row) {
      int top=(position+row)%4;
      buffer[row*4+top]=10;
      result.push_back(bad_ids ? (top+1)%4 : top);
    }
    position+=rows;
    return result;
  }
  std::vector<int> Prefill(const int*,int rows,const std::string&,bool predict) {
    position+=rows;
    if(!predict) return {};
    buffer.assign(4,0); int top=(position-1)%4;buffer[top]=10;
    return {bad_ids ? (top+1)%4 : top};
  }
};
int main() {
  using namespace crucible::longprobe;
  Engine engine;
  std::vector<int> input(20,1), points{7,15,19};
  auto copy=[](const float* src,size_t n){return std::vector<float>(src,src+n);};
  std::vector<int> captured;
  auto emit=[&](int point,int id,const std::vector<float>& values){
    assert(id==point%4); assert(values.size()==4);captured.push_back(point);
  };
  const auto control=Capture(engine,input,points,8,false,4,copy,emit);
  assert((captured==points));assert((control==std::vector<Block>{{0,8},{8,8},{16,4}}));
  captured.clear();
  const auto clipped=Capture(engine,input,points,16,true,4,copy,emit);
  assert((clipped==control));assert((captured==points));
  captured.clear();
  const std::vector<int> final{19};
  const auto production=Capture(engine,input,final,16,true,4,copy,emit);
  assert((production==std::vector<Block>{{0,16},{16,4}}));assert(captured.back()==19);
  engine.bad_ids=true;
  try {Capture(engine,input,points,8,false,4,copy,emit);assert(false);}catch(const std::runtime_error&){}
  engine.bad_ids=false;
  auto missing=[](const float* src,size_t n){return std::vector<float>(src,src+n-1);};
  try {Capture(engine,input,points,16,true,4,missing,emit);assert(false);}catch(const std::runtime_error&){}
  auto nonfinite=[](const float* src,size_t n){auto out=std::vector<float>(src,src+n);out[0]=std::numeric_limits<float>::quiet_NaN();return out;};
  try {Capture(engine,input,points,16,true,4,nonfinite,emit);assert(false);}catch(const std::runtime_error&){}
  for(const std::vector<int>& invalid : std::vector<std::vector<int>>{{},{7},{15,7,19},{7,7,19},{-1,19},{20}})
    try {Capture(engine,input,invalid,8,false,4,copy,emit);assert(false);}catch(const std::runtime_error&){}
}
'''


class RuntimeTests(unittest.TestCase):
    def test_clipped_and_production_schedule_and_argmax_proof(self):
        compiler = shutil.which('g++')
        if compiler is None: self.skipTest('g++ unavailable')
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)/'driver.cpp'
            source.write_text(DRIVER)
            output = Path(directory)/'driver'
            subprocess.run([compiler,'-std=c++20','-Wall','-Wextra','-Werror','-I',str(ROOT),str(source),'-o',str(output)],check=True)
            subprocess.run([str(output)],check=True)


if __name__ == '__main__': unittest.main()
