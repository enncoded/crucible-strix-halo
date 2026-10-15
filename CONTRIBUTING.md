# Contributing

Crucible is an exact inference engine for one model family on one GPU architecture.
Read [the architecture](docs/ARCHITECTURE.md) and [the build guide](docs/BUILD.md) first.

## License

The engine uses the Apache-2.0 license (see `LICENSE` and `NOTICE`).
A contribution enters under the same license.
Do not copy third-party code without its license and its required notices.
Do not include model weights, personal prompts, credentials or private machine details.

## Before a change

- Open an issue before a large design change.
- Write the expected result and the execution path that the change affects.
- For a performance change, write the hardware ceiling that applies and the measured percentage of it.
- Look for an existing implementation or measurement before you add a new path.

## Build and test

Use C++20 and the existing HIP style.
Write comments that explain the numerical order and the safety assumptions.

Write a failing test before a bug fix or a new feature.
Then write the smallest change that makes the test pass.
Run the CPU suite:

```sh
make -j2 test-cpu
```

Use the sanitizers for a change to state, ownership or a parser:

```sh
make -B -j2 test-cpu SANITIZERS='-fsanitize=address,undefined -fno-omit-frame-pointer'
```

A skipped test is not a pass. Do not report it as a pass.

## The exactness rule

A change must not change the output of a request, unless the change is a declared numerics change.

- Speculation, batches, caches and the load of other sessions must not change the token IDs.
- Run the gates of `tools/gates.py` for an engine change, against a named baseline binary
  that is not the candidate.
- Run both checkpoint packs when a numerical behavior changes.
- A lossy numerics change must be behind a `kNumerics` switch. `CRUCIBLE_EXACT=1` must turn it off.
  It needs the approval of a maintainer.

Keep the complete gate report and its artifacts.

## Measurements

Give each before and after value with the prompt, the context, the generation policy and the number of runs.
Give the hardware ceiling and the percentage.
Keep throughput, time to first token, load time and network latency apart.
Do not compare different prompts or acceptance policies as one speedup.

## Pull requests

Include these items:

- The problem and the intended behavior.
- The changed execution paths and the numerical assumptions.
- The test commands, the results, the skips and the known risks.
- The gate results for an engine change.
- The before and after measurements for a performance change.
- The third-party sources and notices, if there are any.

Use commit subjects such as `fix: validate context bounds` or `perf: share expert weight reads`.

## Report a bug

Open an issue with these items:

- The commit of Crucible and the checkpoint pack.
- The GPU, the kernel and the ROCm version.
- The command or the HTTP request, and the server log lines around the error.
- The expected result and the actual result.

Remove personal paths, host names, prompts and tokens from the attachments.
Do not put a security problem in a public issue.
Use the private vulnerability reporting of the repository.

## Documentation

Write short, direct sentences. Use ASD-STE100 Simplified Technical English where you can.
Do not use em dashes.
Keep measured results, model estimates and untested expectations apart.
