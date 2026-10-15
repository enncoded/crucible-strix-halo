// Rows one packed verify pass may hold.
#pragma once

#include "../model/flashnext/geometry.h"

namespace crucible {

// A static dense group of a 9..16-row pass runs only through the dense ring
// (LinRingRowsKernel): a full-span activation stage for 16 rows needs 80 KB
// of LDS. The ring reads front-order tiles, or engine-order tiles whose
// split spans are whole ring chunks.
struct PackedGroupShape {
  bool has_dst;        // dense projection (expert groups have none)
  bool front;          // the forward reads front-order tiles
  bool chunk_aligned;  // every job's split span is a multiple of lin_ring::kChunk
};

constexpr bool PackedGroupFits(PackedGroupShape g) { return g.has_dst && (g.front || g.chunk_aligned); }

// kPackedRows when every static group fits; kMaxRows otherwise.
constexpr int MaxPassRows(bool groups_fit) {
  return groups_fit ? kPackedRows : kMaxRows;
}

}  // namespace crucible
