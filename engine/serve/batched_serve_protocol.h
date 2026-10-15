// Commands are synchronous. One frontend worker owns all request state changes.
#pragma once
#include "serve_protocol.h"
#include <optional>

namespace crucible {
struct BatchedServeCommand {
  std::string op;
  uint64_t request_id = 0;
  std::vector<uint64_t> request_ids;
  std::optional<ServeRequest> request;
  int every = 0;  // interleaved prefill: a verify pass for request_ids after every `every` layers of the chunk
};

inline BatchedServeCommand ParseBatchCommand(const std::string& line, int context) {
  st::Scanner scanner(line, "batch command", true);
  BatchedServeCommand command;
  std::set<std::string> fields;
  scanner.Expect('{');
  while (!scanner.Peek('}')) {
    const auto key = scanner.Str();
    if (!fields.insert(key).second) throw std::runtime_error("duplicate batch command field");
    scanner.Expect(':');
    if (key == "op") command.op = scanner.Str();
    else if (key == "request_id") command.request_id = scanner.Uint();
    else if (key == "every") {
      const uint64_t every = scanner.Uint();
      if (every < 1 || every > 47) throw std::runtime_error("prefill interval must be 1 to 47 layers");
      command.every = int(every);
    }
    else if (key == "request") {
      const size_t start = scanner.pos();
      scanner.Skip();
      // Batch requests must state spec; a missing field keeps the invalid -1 and fails the policy check.
      DecodePolicy explicit_spec;
      explicit_spec.spec = -1;
      command.request = ParseServeRequest(line.substr(start, scanner.pos() - start), context, explicit_spec);
    } else if (key == "request_ids") {
      scanner.Expect('[');
      while (!scanner.Peek(']')) {
        if (command.request_ids.size() >= kMaxRows) throw std::runtime_error("batch command exceeds eight rows");
        command.request_ids.push_back(scanner.Uint());
        scanner.Separator(']');
      }
      scanner.Expect(']');
    } else throw std::runtime_error("unknown batch command field");
    scanner.Separator('}');
  }
  scanner.Expect('}');
  scanner.Finish();
  if (command.op == "open") {
    if (fields != std::set<std::string>{"op", "request_id", "request"} || !command.request_id ||
        !command.request)
      throw std::runtime_error("open requires a positive request ID and an explicit spec request");
  } else if (command.op == "step" || command.op == "verify_step") {
    const auto& ids = command.request_ids;
    if (fields != std::set<std::string>{"op", "request_ids"} || ids.empty() ||
        std::set<uint64_t>(ids.begin(), ids.end()).size() != ids.size() ||
        std::find(ids.begin(), ids.end(), 0) != ids.end())
      throw std::runtime_error("step and verify_step require unique positive request IDs");
  } else if (command.op == "prefill" && fields.contains("request_ids")) {
    const auto& ids = command.request_ids;
    if (fields != std::set<std::string>{"op", "request_id", "request_ids", "every"} || !command.request_id ||
        ids.empty() || std::set<uint64_t>(ids.begin(), ids.end()).size() != ids.size() ||
        std::find(ids.begin(), ids.end(), 0) != ids.end() ||
        std::find(ids.begin(), ids.end(), command.request_id) != ids.end())
      throw std::runtime_error("interleaved prefill requires its request ID, other unique decoders, and every");
  } else if (command.op == "stage") {
    // The disk tier (state_disk.h): start the read of a disk state for this request before its open.
    if (fields != std::set<std::string>{"op", "request"} || !command.request)
      throw std::runtime_error("stage requires a request and no request ID");
  } else if (command.op == "close" || command.op == "prefill" || command.op == "spec_step") {
    if (fields != std::set<std::string>{"op", "request_id"} || !command.request_id)
      throw std::runtime_error("close and prefill require a positive request ID");
  } else throw std::runtime_error("batch op must be open, prefill, step, verify_step, spec_step, stage, or close");
  return command;
}
}  // namespace crucible
