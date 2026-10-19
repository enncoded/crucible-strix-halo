// Prefill row limits and the "pick for me" request. Host-only constants. engine.hip includes this header
// inside its anonymous namespace, before class Engine; the other modules read the names from there.
#pragma once

// Larger prefill chunks give each routed expert more rows per weight decode.
constexpr int kMaxPrefillRows = 4096;
constexpr int kPrefillCoreRows = 2048;
// Partial last-layer pruning starts at a multiple of this row count (the HC mix and MoE row tiles).
constexpr int kPrefillPruneAlign = 128;
// Rows of the tight stack buffer (Engine H_). With the padded bulk buffer (pf_H_, the default) every prefill launch
// and every bulk MTP warm-up runs there, so H_ holds only one packed verify pass; bulk readers copy their rows out
// with CopyPrefillStacks (192-row launches). Without it, bulk launches run in H_.
constexpr int TightStackRows(int capacity, bool padded_bulk, int packed_rows) {
  return padded_bulk && capacity > packed_rows ? packed_rows : capacity;
}
// Engine(ckpt, ctx, kPrefillAuto): GEMM prefill in kPrefillDefaultRows-row
// chunks when the weights allow it (best measured at 1K, 4K and 16K).
constexpr int kPrefillAuto = -1, kPrefillDefaultRows = 4096;
