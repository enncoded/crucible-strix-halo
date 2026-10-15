#include "../bench/moe_pair_stats.h"
#include <cassert>
#include <limits>

using crucible_bench::PairStats;
using crucible_bench::PairSample;

int main() {
  const std::vector<PairSample> drift{{14,13,12,11},{10,9,8,7}};
  const auto unchanged=drift;
  const auto canceled=PairStats(drift);
  assert(canceled.delta==0);
  assert(canceled.baseline==canceled.candidate);
  assert(drift==unchanged);
  const auto gain=PairStats({{14,12,11,11},{10,8,8,10},{6,5,5,6}});
  assert(gain.baseline==10&&gain.candidate==8);
  assert(gain.delta==1&&gain.minimum_delta==1&&gain.maximum_delta==2);
  int rejected=0;
  try { (void)PairStats({}); } catch(const std::invalid_argument&) { ++rejected; }
  for(double bad:{0.0,-1.0,std::numeric_limits<double>::infinity(),
                  std::numeric_limits<double>::quiet_NaN()}) {
    try { (void)PairStats({{1,bad,1,1}}); } catch(const std::invalid_argument&) { ++rejected; }
  }
  assert(rejected==5);
}
