# ref: charter

CPU references: the EXL3 codec, the fp64 layer oracle, prefill and vision
references.

## Interface

- `exl3_ref.py`: reference EXL3 decode.
- `oracle.py`: fp64 layer reference for checks where bitwise Gate 1 cannot
  apply.
- `prefill_ref.py`, vision references and their tests.

## Invariants

- A reference follows `docs/FLASHNEXT_FORWARD.md` and `docs/EXL3_FORMAT.md`,
  not the engine code.
- CPU only. PyTorch is optional (`make test-reference`).

## Allowed dependencies

- Python standard library, numpy and the optional reference packages.

## Switches

- None.

## Lever ledger

- Not applicable.
