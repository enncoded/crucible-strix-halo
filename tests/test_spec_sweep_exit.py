"""CPU-only exit-status checks for the resident speculation sweep."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).parents[1]
FAKE = r'''
#pragma once
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>
constexpr int kMaxRows=8;
struct Steps{void Print(FILE*)const{}};
struct SamplingOptions{double temperature=0;int top_k=0;};
struct GenOptions{int ctx=32768,n=3,spec=0,lookup=0,head_auto=0;bool steps=false,bad=false;SamplingOptions sampling;};
struct HeadPolicy{long Steps(bool)const{return 0;} bool IncumbentFull()const{return false;}};
inline double Now(){return 0;}
struct GenResult{std::vector<int> gen;double prefill_s=0,decode_s=1;long forwards=1,accepted=0,proposed=0,lookup_steps=0,lookup_accepted=0;Steps steps;HeadPolicy head_policy;};
struct Engine{
 Engine(const char*,int,int=0){} void WarmPlans(){} void CountExperts(bool){}
 void EnableProfile(bool){} void Report(int){} std::vector<unsigned long long> ExpertCounts(){return std::vector<unsigned long long>(4*(kMaxRows+1));}
};
inline bool ParseGenOptions(int argc,char**argv,int first,GenOptions&o){
 for(int i=first;i+1<argc;i+=2){std::string k=argv[i],v=argv[i+1];if(k=="--spec")o.spec=std::stoi(v);if(k=="--bad")o.bad=v=="1";if(k=="--n")o.n=std::stoi(v);}
 return true;
}
inline int CheckGenOptions(const GenOptions&,std::vector<int>&){return 0;}
inline GenResult RunGen(Engine&,const GenOptions&o,const std::vector<int>&){return {{o.bad ? 7 : 1,2,3}};}
'''


class SweepExitTests(unittest.TestCase):
    def test_mismatch_is_nonzero_and_rows_are_preserved(self):
        compiler = shutil.which('g++')
        if compiler is None: self.skipTest('g++ unavailable')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            header = root/'fake.h'
            header.write_text(FAKE)
            binary = root/'sweep'
            subprocess.run([compiler,'-std=c++20','-x','c++',f'-DCRUCIBLE_ENGINE_HEADER="{header}"',
                            str(ROOT/'bench/spec_sweep.hip'),'-o',str(binary)],check=True,capture_output=True)
            config = root/'requests'
            reference = 'serial --ids-file fixture --n 3 --spec 0\n'
            for bad in (0,1):
                config.write_text(reference+f'spec --ids-file fixture --n 3 --spec 2 --bad {bad}\n'+
                                  'tail --ids-file fixture --n 3 --spec 2\n')
                result = subprocess.run([str(binary),'fake-checkpoint',str(config)],capture_output=True,text=True)
                self.assertEqual(result.returncode,bad)
                self.assertEqual(sum(line.startswith('== ') for line in result.stdout.splitlines()),3)
                self.assertEqual('MISMATCH' in result.stdout,bool(bad))
                self.assertIn('== tail:',result.stdout)


if __name__ == '__main__': unittest.main()
