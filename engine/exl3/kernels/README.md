# engine/exl3/kernels: charter

Device-only EXL3 kernels (rank 1). The ledger is in `engine/exl3/README.md`.

## Interface

- Trellis extraction, codebook decode and ordered GEMV (`exl3_gemv.h`).
- WMMA prefill GEMM bodies: `exl3_gemm.h`, `exl3_gemm_pair.h`,
  `exl3_moe_window.h`.
- Bench-only variants are not in this module. The move puts them in
  `bench/kernels/` (design decision, Oct 10).

## Invariants

- A kernel header holds device code and its launch shapes only. Host plans
  stay in `engine/exl3/`.
- Every kernel keeps the contraction order of `exl3_gemv.h` for its rows.

## Allowed includes

- Other kernel headers of the same or a lower rank, `engine/gfx1151/` and the
  leaves. Never host engine code. No header breaks this rule today.

## Switches

- None read here. Switches select kernels in the host code.
- A kernel header may include the leaf `engine/switches.h`.
