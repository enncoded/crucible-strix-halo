// Rows of one packed verify pass in batch serve (CRUCIBLE_PACKED_ROWS).
#pragma once
#include <stdexcept>
#include <string>
#include "../model/flashnext/geometry.h"

static_assert(crucible::kMaxRows == 8 && crucible::kPackedRows == 16, "the message names the limits");

namespace crucible {

// Decoders from which an automatic pass may use 16 rows. A sweep on
// 2026-10-09 measured 16 rows at -2 to -3% with 2 or 3 sessions and +4 to +26% with 4 to 8 sessions.
constexpr int kWideDecoders = 4;

// rows: the row cap of a pass. wide_from: 0 for a fixed cap; else passes with fewer decoders use kMaxRows.
struct PackedRowsSetting {
  int rows = kMaxRows, wide_from = 0;
  bool operator==(const PackedRowsSetting&) const = default;
};

// Unset or empty: automatic (kPackedRows from kWideDecoders decoders, else kMaxRows; BatchSession::SetPassRows
// falls back to kMaxRows where 16-row passes cannot run). Otherwise a plain decimal from kMaxRows to kPackedRows:
// a fixed cap, which needs an engine that runs it (Engine::MaxPassRows). The rows change only how many drafts
// each request proposes, never a request's output.
inline PackedRowsSetting PackedRowsOption(const char* value) {
  if (!value || !*value) return {kPackedRows, kWideDecoders};
  const std::string text(value);
  int rows = 0;
  for (const char c : text) {
    if (c < '0' || c > '9' || rows > kPackedRows) throw std::invalid_argument("CRUCIBLE_PACKED_ROWS must be 8 to 16");
    rows = rows * 10 + (c - '0');
  }
  if (rows < kMaxRows || rows > kPackedRows) throw std::invalid_argument("CRUCIBLE_PACKED_ROWS must be 8 to 16");
  return {rows, 0};
}

// The serve stderr line for the pass rows: empty for the default (automatic) and for 8 rows, so default stderr
// does not change.
inline std::string PackedRowsNotice(PackedRowsSetting setting) {
  if (setting.rows == kMaxRows || setting.wide_from) return {};
  return "serve: packed verify passes of up to " + std::to_string(setting.rows) + " rows\n";
}

}  // namespace crucible
