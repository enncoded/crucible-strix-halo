# training: charter

Training code for the draft (MTP) head.

## Interface

- `mtp_tune/`: fine-tune and A/B scripts for the existing MTP block.

## Invariants

- Training changes proposals only. Verification keeps the output exact.
- GPU runs use `gpu-run` leases.

## Allowed dependencies

- Python packages that `mtp_tune/` names. Engine binaries for A/B runs.

## Switches

- None of its own. A/B runs set engine switches from the registry.

## Lever ledger

- Training-time-test fine-tune of the MTP block: open, after the decode
  megakernel (ranked table item 11).
