# engine/exl3: charter

Rank 1 (format). A format swap replaces this module.

## Interface

- The EXL3 linear op (`Lin`): input transform, multi-matrix matvec and output
  transform (`exl3.h`). The format seam is at the linear op.
- Plans and work lists: `plans.h`, `balanced_items.h`, `decode_split.h`,
  `row_policy.h`, `front_order.h`, `lookahead.h`.
- Dense decode schedules: `exl3_ring.h`, `ring_rows.h`.
- Batched prefill plans: `prefill_linear.h`, `prefill_items.h`.
- Load support: `static_layout.h`, `load_tasks.h`, `decode_support.h` and the
  checkpoint container reader `safetensors.h` (from `bench/`).
- Device-only kernels live in `kernels/` (see its charter).

## Invariants

- Codebook decode is exact and stays inside the GEMV and GEMM inner loops.
  Do not separate it (structure advice, "Do not").
- Input scaling precedes the 128-element Hadamard transform. Output scaling
  follows the output transform (docs/EXL3_FORMAT.md).
- Each output element keeps one fixed contraction order for every row count.

## Allowed includes

- `engine/gfx1151/`, `engine/exl3/kernels/` and the leaves.
- Not `layers/`, `model/`, `runtime/`, `serve/` or `cli/`.

## Switches

The registry (`engine/switches.h`) is the truth for the names and the module of each switch.
Names have the prefix `CRUCIBLE_`, left out here.

- Exact-speed: `DECODE_RING`, `PF_FX_LEAD`, `PF_MOE_RM`, `PF_TILES`.
- Build macro: `MOE_WINDOW_PROBE`.

## Lever ledger (closed)

| Lever | Result | Evidence |
|---|---|---|
| Lookahead phase caps (`LOOKAHEAD_PHASE`) | Killed: bitwise, but no setting saved 0.5 ms per forward | development notes (not public) |
| A lookup-table mul1 decode | 128 KiB table exceeds 64 KiB LDS | development notes (not public) |
| Fold the mul1 scale into WMMA | Inexact per-weight fp16 rounding | docs/EXL3_FORMAT.md section 3 |
| `v_sad_hi_u8` pair decode in the MoE body | No gain in the full prefill MoE body | development notes (not public) |
| A second expert copy in a cheaper format | No memory for it (about 41 GB spare) | development notes (not public) |
