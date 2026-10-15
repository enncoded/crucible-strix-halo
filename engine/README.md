# engine: charter

The host and device code of the Crucible engine.

## Interface

- `engine.hip` is the one translation unit: class `Engine` and the includes of
  every module. Its fragments are in `runtime/`. `main` is in
  `cli/gen_cli.h`, which `engine.hip` includes last. Do not split it into
  several translation units now (structure advice, "Do not").
- `make` builds it to `build/engine/crucible` (docs/BUILD.md).

## Modules

`tools/layout_map.tsv` gives every file one module. Ranks, low to high:

| Rank | Module | Charter |
|---:|---|---|
| 0 | `gfx1151/` | device helpers, WMMA, sync, pinned memory |
| 1 | `exl3/`, `exl3/kernels/` | the format: linear op, plans, decode kernels |
| 2 | `layers/`, `layers/kernels/` | GDN, QSA, HC, MoE route, PLE operators |
| 3 | `model/flashnext/` | config, weights, tensor names, state layout |
| 4 | `runtime/` | class `Engine` parts, decode, drafts, caches |
| 5 | `serve/` | serve protocols and batched sessions |
| 6 | `cli/` | `gen` and bench modes |
| 7 | `engine.hip` | the translation unit |

The layout move of train 18 is done. The module
charters are `README.md` in each module directory.

## Invariants

- Includes go down only. A kernel header includes only kernel headers,
  `gfx1151/` and the leaves. The leaves are `model/flashnext/geometry.h`,
  `switches.h` and `runtime/sha256.h`; they include only the standard
  library. `tests/test_layering.py` checks this; its baseline may only shrink.
- Bench-only kernel headers live in `bench/kernels/` (rank 8). Engine code
  never includes them.
- Every file has one row in `tools/layout_map.tsv` (ownership test).
- The numerics switches are on by default. `EXACT=1` selects the bitwise path
  (`docs/ARCHITECTURE.md`, Exactness rules).
- No new layer loop; no runtime model descriptor; no dynamic shapes.

## Switches

- Every `CRUCIBLE_*` switch has kind, default, module, issue and decide_by in
  the registry (`switches.h`). The engine has no raw `getenv` outside it.
- The registry has six kinds: numerics, exact-speed, layout, diagnostic, build
  and profile (a user profile input that changes the model on purpose).
- The registry is the truth for switch names and modules. Each module charter
  lists the switches of its registry module, without the prefix.

## Lever ledger

- See the module charters. Durable results go to `docs/<module>/<topic>.md`
  and the module charter links them.
