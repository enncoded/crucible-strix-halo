# engine/layers/kernels: charter

Device-only layer kernels (rank 2). The ledger is in
`engine/layers/README.md`.

## Interface

- Chunk-64 GDN prefill (`prefill_gdn.h`) and the ordered coefficient load
  (`persistent_gdn_load.h`). The bench-only `prefill_gdn_packed.h` moves to
  `bench/kernels/`.
- QSA prefill: `prefill_qsa.h`, `prefill_qsa_score*.h`,
  `prefill_qsa_select.h`, `prefill_qsa_union.h`.
- Multi-row HC (`hc_multirow.h`), expert-major MoE dispatch
  (`prefill_moe.h`) and the vision tower (`vision.h`).

## Invariants

- A kernel header holds device code and its launch shapes only.
- Decode and speculative replay keep their own paths; prefill kernels do not
  change them.

## Allowed includes

- Kernel headers of the same or a lower rank (`engine/exl3/kernels/`),
  `engine/gfx1151/` and the leaves. Never host engine code.
- Six includes of host code are in the `tests/test_layering.py` baseline
  today (`ops.h` three times, `numerics.h`, `qsa_long.h`,
  `prefill_launch.h`). Each fix deletes one entry.

## Switches

- None read here. The registry lists the switches of these kernels under `layers` (see
  `engine/layers/README.md`). A kernel header may include the leaf `engine/switches.h`.
