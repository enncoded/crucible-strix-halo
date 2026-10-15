// Mapping ownership is independent of HIP and can be checked by the host kernel.
#ifndef SAFETENSORS_HEADER
#define SAFETENSORS_HEADER "../engine/exl3/safetensors.h"
#endif
#include SAFETENSORS_HEADER
#include <cassert>
#include <filesystem>
#include <type_traits>
#ifdef __APPLE__
#include <mach/mach.h>
#include <mach/mach_vm.h>
#endif

static_assert(!std::is_copy_constructible_v<st::Checkpoint>);
static_assert(!std::is_copy_assignable_v<st::Checkpoint>);
static_assert(!std::is_move_constructible_v<st::Checkpoint>);
static_assert(!std::is_move_assignable_v<st::Checkpoint>);

static void Shard(const std::filesystem::path& path, const char* tensor) {
  const std::string header = std::string("{\"") + tensor +
      "\":{\"dtype\":\"F16\",\"shape\":[4],\"data_offsets\":[0,8]}}";
  std::ofstream file(path, std::ios::binary);
  assert(file);
  const uint64_t size = header.size();
  uint8_t prefix[8];
  for (int i = 0; i < 8; ++i) prefix[i] = uint8_t(size >> (8 * i));
  file.write(reinterpret_cast<const char*>(prefix), sizeof(prefix));
  file.write(header.data(), header.size());
  const char payload[8] = {};
  file.write(payload, sizeof(payload));
  file.close();
  assert(file);
}

static bool Mapped(uintptr_t address, [[maybe_unused]] size_t page_size) {
#ifdef __APPLE__
  // mincore success alone does not prove a Darwin region exists. Query its map.
  mach_vm_address_t region = address;
  mach_vm_size_t size = 0;
  vm_region_basic_info_data_64_t info{};
  mach_msg_type_number_t count = VM_REGION_BASIC_INFO_COUNT_64;
  mach_port_t object = MACH_PORT_NULL;
  const kern_return_t result = mach_vm_region(mach_task_self(), &region, &size,
      VM_REGION_BASIC_INFO_64, reinterpret_cast<vm_region_info_t>(&info), &count, &object);
  if (object != MACH_PORT_NULL) mach_port_deallocate(mach_task_self(), object);
  assert(result == KERN_SUCCESS || result == KERN_INVALID_ADDRESS);
  return result == KERN_SUCCESS && region <= address && size > address - region;
#else
  unsigned char residency;
  errno = 0;
  const int result = mincore(reinterpret_cast<char*>(address), page_size, &residency);
  if (result == 0) return true;
  assert(errno == ENOMEM);
  return false;
#endif
}

int main() {
  char path[] = "/tmp/crucible-checkpoint-raii-XXXXXX";
  assert(mkdtemp(path));
  const std::filesystem::path root(path);
  Shard(root / "one.safetensors", "a");
  Shard(root / "two.safetensors", "b");
  {
    std::ofstream index(root / "model.safetensors.index.json");
    index << "{\"weight_map\":{\"a\":\"one.safetensors\",\"b\":\"two.safetensors\"}}";
    assert(index);
  }
  const long system_page = sysconf(_SC_PAGESIZE);
  assert(system_page > 0);
  const size_t page_size = size_t(system_page);
  uintptr_t first, second;
  {
    st::Checkpoint checkpoint(root.string());
    const uintptr_t a = reinterpret_cast<uintptr_t>(checkpoint.Data("a"));
    const uintptr_t b = reinterpret_cast<uintptr_t>(checkpoint.Data("b"));
    first = (a / page_size) * page_size;
    second = (b / page_size) * page_size;
    assert(first != second);
    assert(Mapped(first, page_size));
    assert(Mapped(second, page_size));
  }
  assert(!Mapped(first, page_size));
  assert(!Mapped(second, page_size));
  std::filesystem::remove_all(root);
  std::puts("PASS: both shard mappings are released at checkpoint destruction");
}
