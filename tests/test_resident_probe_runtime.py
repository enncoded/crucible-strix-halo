"""CPU proof that resident capture roles share one engine and reset each request."""
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]

class ResidentRuntimeTest(unittest.TestCase):
    def test_one_engine_per_role_and_fresh_sequence_per_capture(self):
        source=r'''
#include "bench/resident_probe_runtime.h"
#include <cassert>
#include <vector>
struct Engine { int resets=0; void Reset() { ++resets; } };
int main() {
 Engine e;
 const std::vector<int> jobs{1,2,3};
 std::vector<int> calls;
 crucible::residentprobe::Run(e,jobs,
  [&](Engine& engine,int job) { assert(engine.resets==int(calls.size())+1); calls.push_back(job*10); },
  [&](Engine& engine,int job) { assert(engine.resets==int(calls.size())+1); calls.push_back(job*10+1); });
 assert((calls==std::vector<int>{10,11,20,21,30,31}));
 assert(e.resets==6);
}
'''
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory); (path/'test.cpp').write_text(source)
            subprocess.run(['g++','-std=c++20','-Wall','-Wextra','-Werror','-I',str(ROOT),
                            str(path/'test.cpp'),'-o',str(path/'test')],check=True,capture_output=True)
            subprocess.run([str(path/'test')],check=True,capture_output=True)

if __name__=='__main__': unittest.main()
