// Keep plan offsets fixed while the shared expert starts before lighter work.
#pragma once
#include <algorithm>
#include <vector>

namespace crucible {
template <class Item>
std::vector<Item> MoeSharedFirstItems(const std::vector<Item>& items,int shared_job) {
  auto result=items;
  if(shared_job>=0) {
    const auto first=std::find_if(result.begin(),result.end(),[&](const Item& item){return item.job>=shared_job;});
    std::rotate(result.begin(),first,result.end());
  }
  return result;
}
}  // namespace crucible
