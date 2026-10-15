# tests: charter

CPU and GPU tests of the engine and the tools.

## Interface

- `tests/*_test.cpp`: CPU tests that `make test-cpu` builds and runs.
- `tests/*.hip`: GPU tests (`make gpu-tests`).
- `tests/test_*.py`: Python tests (`python -m unittest discover -s tests`).
- `tests/data/`: fixtures. The layout move does not rewrite them.

## Invariants

- TDD: a failing test first, then the fix.
- Ratchet tests have an explicit baseline that may only shrink. They fail on
  a new violation and on a stale entry. Today: `test_layering.py` (include
  direction and file ownership), `test_no_copies.py` (byte-identical twins in
  `tools/` and `deploy/`) and `test_near_copies.py` (line ratio 0.9 or more,
  `tests/data/near_copies.txt`, an earlier change).
- A test does not need the GPU unless its name or directory says so.
- Synthetic content only.

## Allowed includes

- Any engine, kernel or bench header.

## Switches

- A test sets the switches it checks. It does not depend on the caller's
  environment.

## Lever ledger

- Not applicable.
