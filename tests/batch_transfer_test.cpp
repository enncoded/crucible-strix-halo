#include "../engine/runtime/batch_inputs.h"

#include <cassert>
#include <future>
#include <utility>
#include <vector>

crucible::BatchHostInputs Prepare(const std::vector<int>& tokens) {
  std::vector<int64_t> rows(tokens.size() * 16);
  for (size_t i = 0; i < rows.size(); ++i) rows[i] = 42'000'000'000ll + i;
  return crucible::BatchHostInputs(tokens, std::move(rows));
}

int main() {
  for (int count = 1; count <= 8; ++count) {
    const std::vector<int> tokens(count, 123);
    const auto inputs = Prepare(tokens);
    std::promise<void> release;
    auto transfer = std::async(std::launch::async,
        [tokens = inputs.tokens(), rows = inputs.rows(), ready = release.get_future()]() mutable {
          ready.wait();
          return std::pair(std::vector<int>(tokens.begin(), tokens.end()),
                           std::vector<int64_t>(rows.begin(), rows.end()));
        });
    // Queued copies retain raw spans, not ownership. Reuse freed preparation allocations before DMA.
    const std::vector<std::vector<int64_t>> churn(128, std::vector<int64_t>(count * 16, -1));
    release.set_value();
    const auto copied = transfer.get();
    assert(copied.first == tokens && copied.second.size() == size_t(count * 16));
    for (size_t i = 0; i < copied.second.size(); ++i) assert(copied.second[i] == 42'000'000'000ll + i);
  }
}
