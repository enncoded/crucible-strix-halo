#include "../engine/layers/cache_storage.h"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <map>
#include <vector>
using namespace crucible;
struct Cache { uint16_t* k; uint16_t* v; float* ik_raw; float* ik_pool; int raw_slots; };

void ResetTest(int context, int slots) {
  const std::array<size_t, 4> expected = {size_t(context) * 1024, size_t(context) * 1024,
      size_t(slots ? slots : context) * 512, size_t((context + 127) / 128) * 16384};
  std::array<std::vector<uint8_t>, 4> buffers;
  for (size_t i = 0; i < buffers.size(); ++i) buffers[i].resize(expected[i], 37);
  const Cache descriptor{reinterpret_cast<uint16_t*>(buffers[0].data()),
      reinterpret_cast<uint16_t*>(buffers[1].data()), reinterpret_cast<float*>(buffers[2].data()),
      reinterpret_cast<float*>(buffers[3].data()), slots};
  std::map<int, Cache> caches;
  size_t calls = 0;
  const auto clear = [&](void* pointer, size_t bytes) {
    assert(calls < 4 && pointer == buffers[calls].data() && bytes == expected[calls]);
    std::memset(pointer, 0, bytes); ++calls;
  };
  assert(!ResetQsaCache(caches, kLayers, context, clear) && calls == 0);
  caches[kLayers] = descriptor;
  assert(ResetQsaCache(caches, kLayers, context, clear) && calls == 4);
  for (const auto& buffer : buffers)
    assert(std::all_of(buffer.begin(), buffer.end(), [](uint8_t value) { return value == 0; }));
  caches[kLayers].raw_slots = 7;
  bool rejected = false;
  try { ResetQsaCache(caches, kLayers, context, clear); }
  catch (const std::invalid_argument&) { rejected = true; }
  assert(rejected && calls == 4);
}
int main() {
  for (int context : {1, 13, 80, 192}) {
    ResetTest(context, 0);
    ResetTest(context, kQsaRawSlots);
  }
}
