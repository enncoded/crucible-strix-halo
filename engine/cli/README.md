# engine/cli: charter

Rank 6 (command line).

## Interface

- `gen` (`gen_cli.h`, `RunGen`), its `--jobs` driver (`gen_jobs.h`,
  `gen_job_state.h`) and the greedy multi-stream mode (`batch_cli.h`,
  `batch_options.h`).
- The argument parser and checks (`ParseGenOptions`, `CheckGenOptions`) of `GenOptions`. The options
  struct itself is in `runtime/gen_options.h`, because serve reads it too.
- `main` and the `serve` dispatch (`gen_cli.h`): `ServeCommand` starts the
  serial serve (`serve/serve.h`, deleted in train 20). `TryBatchedServeCommand`
  (`serve/batched_serve.h`) starts the batched serve.
- The prefill bench mode (`prefill_bench.h`, included after class `Engine`).
- `bench/suite/resident.hip` reuses `RunGen`. It does not implement another
  engine.

## Invariants

- Parse and check every option before the model allocates device memory.
- A CLI mode adds no forward path. It calls the runtime.

## Allowed includes

- Every lower module and the leaves.

## Switches

The registry (`engine/switches.h`) is the truth for the names and the module of each switch.
Names have the prefix `CRUCIBLE_`, left out here.

- Diagnostic: `PFBENCH_PASSES`, `PF_OUT`, `PROFILE`.

## Lever ledger (closed)

- None. CLI changes do not change speed.
