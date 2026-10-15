// Prefill row limits and the "pick for me" request. Host-only constants. engine.hip includes this header
// inside its anonymous namespace, before class Engine; the other modules read the names from there.
#pragma once

// Larger prefill chunks give each routed expert more rows per weight decode.
constexpr int kMaxPrefillRows = 4096;
constexpr int kPrefillCoreRows = 2048;
// Partial last-layer pruning starts at a multiple of this row count (the HC mix and MoE row tiles).
constexpr int kPrefillPruneAlign = 128;
// Engine(ckpt, ctx, kPrefillAuto): GEMM prefill in kPrefillDefaultRows-row
// chunks when the weights allow it (best measured at 1K, 4K and 16K).
constexpr int kPrefillAuto = -1, kPrefillDefaultRows = 4096;
