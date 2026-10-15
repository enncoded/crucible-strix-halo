#include "../engine/serve/serve_settings.h"
#include <cassert>
#include <string>
#include <vector>
using namespace crucible;

static ServeSettings Settings(std::initializer_list<const char*> args) {
  std::vector<std::string_view> options;
  for (const char* value : args) options.emplace_back(value);
  return ParseServeSettings(options);
}

static void Rejected(std::initializer_list<const char*> args) {
  try { Settings(args); assert(false); }
  catch (const std::runtime_error&) {}
}

int main() {
  const auto defaults = Settings({});
  assert(defaults.ctx == 32768 && defaults.cache_mib == 0 && defaults.prefill_rows == kServePrefillAuto);
  const auto cached = Settings({"--conversation-cache-mib", "1024", "--prefill", "24", "--mtp-prefix", "8"});
  assert(cached.cache_mib == 1024 && cached.prefill_rows == 24 && cached.policy.mtp_prefix == 8);
  assert(Settings({"--conversation-cache-mib", "28672"}).cache_mib == 28672);
  assert(defaults.policy.spec == 7 && defaults.policy.draft_q == 0.3f);
  assert(defaults.policy.draft_head == "64k" && defaults.policy.mtp_prefix == 256);
  assert(defaults.policy.draft_adapt == -1 && !defaults.exact_retry);
  assert(Settings({"--conversation-cache-exact-retry", "1"}).exact_retry);
  assert(!Settings({"--conversation-cache-exact-retry", "0"}).exact_retry);
  assert(!defaults.session_replay && Settings({"--session-replay", "1"}).session_replay);
  assert(!Settings({"--session-replay", "0"}).session_replay);
  Rejected({"--session-replay", "2"});
  assert(defaults.host_reserve_mib == 8192);
  assert(Settings({"--conversation-cache-mib", "64", "--host-reserve-mib", "2048"}).host_reserve_mib == 2048);
  Rejected({"--conversation-cache-mib", "64", "--host-reserve-mib", "1023"});
  Rejected({"--conversation-cache-mib", "64", "--host-reserve-mib", "131073"});
  const auto serial = Settings({"--spec", "0", "--draft-q", "0", "--draft-head", "full", "--mtp-prefix", "0",
                               "--draft-adapt", "0", "--ctx", "32"});
  assert(serial.ctx == 32 && serial.policy.spec == 0 && serial.policy.draft_q == 0);
  assert(serial.policy.draft_head == "full" && serial.policy.mtp_prefix == 0 && serial.policy.draft_adapt == 0);
  const auto request = ParseServeRequest(R"({"ids":[1],"max_tokens":1})", 32);
  assert(request.spec == 7 && request.draft_q == 0.3f && request.draft_head == "64k" && request.mtp_prefix == 256);
  const auto inherited = ParseServeRequest(R"({"ids":[1],"max_tokens":1})", 32, serial.policy);
  assert(inherited.spec == 0 && inherited.mtp_prefix == 0 && inherited.draft_head == "full");
  const auto override = ParseServeRequest(R"({"ids":[1],"max_tokens":1,"spec":2,"draft_head":"full","draft_q":0,"mtp_prefix":0,"draft_adapt":0})", 32);
  assert(override.spec == 2 && override.draft_head == "full" && override.draft_q == 0 && override.mtp_prefix == 0);
  for (auto args : {std::initializer_list<const char*>{"--spec", "8"}, {"--ctx", "32junk"}, {"--spec", "-1"},
                    {"--draft-q", "nan"}, {"--draft-q", "1.1"}, {"--draft-head", "bad"}, {"--mtp-prefix", "262145"},
                    {"--mtp-prefix", "-1"}, {"--draft-adapt", "2"}, {"--draft-head", "full", "--draft-adapt", "1"},
                    {"--spec"}, {"--unknown", "0"}, {"--spec", "2", "--spec", "3"},
                    {"--conversation-cache-mib", "-1"}, {"--conversation-cache-mib", "131073"},
                    {"--prefill", "-2"}, {"--prefill", "4097"},
                    {"--conversation-cache-exact-retry", "-1"}, {"--conversation-cache-exact-retry", "2"},
                    {"--conversation-cache-exact-retry", "1.0"}, {"--conversation-cache-exact-retry"},
                    {"--conversation-cache-exact-retry", "0", "--conversation-cache-exact-retry", "1"}}) Rejected(args);
  for (const char* field : {"\"draft_q\":1.1", "\"draft_q\":-1", "\"draft_head\":\"bad\"", "\"mtp_prefix\":-1",
                           "\"mtp_prefix\":262145", "\"draft_adapt\":2", "\"draft_head\":\"full\",\"draft_adapt\":1"}) {
    try { ParseServeRequest(std::string("{\"ids\":[1],\"max_tokens\":1,") + field + "}", 32); assert(false); }
    catch (const std::runtime_error&) {}
  }
}
