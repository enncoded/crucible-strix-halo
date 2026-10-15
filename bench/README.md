# bench: charter

Microbenchmarks, probes and the benchmark suite.

## Interface

- `bench/*.hip`, `bench/*.cpp`: one program per probe. The Makefile builds
  each one (`make benchmarks`).
- `bench/suite/`: the resident benchmark suite (`run.py`, `resident.hip`).
- The checkpoint reader `safetensors.h` moves to `engine/exl3/` in the
  layout move (see `tools/layout_map.tsv`).
- `bench/kernels/` (after the move): bench-only kernel headers (module
  `bench`, rank 8 in `tools/layout.py`). Engine code never includes them.

## Invariants

- A probe proves a kernel change first. The
  full engine is the final confirmation only.
- A probe reuses engine headers. It does not copy them (structure rules, "No
  copies").
- Synthetic content only.

## Allowed includes

- Any engine or kernel header. Engine code never includes `bench/`
  (`tests/test_layering.py`).

## Switches

- A probe may read its own environment. Engine switches go through the
  registry.

## Lever ledger

- Closed levers belong to the engine module that a probe measures. See
  the module charters.
