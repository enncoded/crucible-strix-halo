# engine/serve: charter

Rank 5 (serve).

## Interface

- Serial JSON-line serve (`serve.h`, `serve_protocol.h`,
  `serve_settings.h`). The parity check at `--streams 1` passed. The
  server (`python -m server`) now defaults to `--streams 1`; `--serial` keeps
  this path for one train. It is deleted in train 20.
- Batched serve (`batched_serve*.h`, `batch_session.h`): one caller owns the
  engine; the frontend selects ready requests.
- Batched caches: `batch_conversation_cache.h`, `batch_state_disk.h`,
  `batch_mtp_history.h`, `cache_status.h`.
- The Python frontend is `server/` (see its README).

## Invariants

- All request checks run before any state reset or device work.
- stdout carries protocol messages only.
- Each command completes before the frontend sends the next one.
- Planned extension points (`SnapshotPart`, `DraftSource`) go into
  `batched_serve.h` after an earlier change (structure advice item 7).

## Allowed includes

- Every lower module and the leaves. Not `cli/`. This holds for names too, not only for `#include`
  (the symbol ratchet in `tests/test_layering.py`): serve reads `GenOptions` and the draft
  rules from `runtime/gen_options.h` and `runtime/draft_accept.h`. The one remaining use of a `cli/`
  name is `RunGen` in `serve.h`, which goes with the serial serve.

## Switches

The registry (`engine/switches.h`) is the truth for the names and the module of each switch.
Names have the prefix `CRUCIBLE_`, left out here.

- Exact-speed: `PACKED_ROWS`.
- Diagnostic: `SERVE_PROFILE`, `SPEC_STATS`, `VERIFY_QSA_CHECK`.

## Lever ledger (closed)

- None recorded yet. Add each closed serve lever here with its issue.
