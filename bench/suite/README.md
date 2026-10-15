# Reproducible benchmark suite

## Fixed inputs

`prompts.json` defines two code, two prose, two copy-edit, and three chat prompts.
One chat prompt uses Spanish to expose vocabulary bias.
The long fixtures contain exactly 4,096, 16,384, and 65,536 IDs.
They repeat service records, retain the final question, and preserve the chat suffix.
These are throughput fixtures, not broad quality tests or realistic long-context distributions.

`manifest.json` records ID counts and SHA-256 hashes.
The committed IDs come from `tools/tok.py --chat` with an empty think block.
Both tested checkpoint tokenizers have the same SHA-256 hash.
The runner rejects a different tokenizer.

Regenerate the IDs with the checkpoint tokenizer. This operation uses only the CPU.

```bash
python3 bench/suite/generate.py "$CKPT"
python3 -m unittest discover -s bench/suite -p 'test_*.py'
```

## Build the resident harness

The source tree must provide `RunGen`, `GenOptions`, and `Engine::WarmPlans`.
Main `e0bf131` and speculative-decoding branch `44450ab` provide these interfaces.
The harness includes the selected engine directly. It does not change the timed decode loop.

Build against a source worktree on the GPU host.

```bash
python3 bench/suite/run.py build \
  --source /path/to/source-worktree \
  --output build/bench/suite-resident
```

The compiler command uses ROCm, C++20, and gfx1151.
The build writes a `.build.json` file beside the binary.
Keep this file with the binary. The runner checks its binary hash.

For a `git archive` export, record its exact origin explicitly.

```bash
python3 bench/suite/run.py build \
  --source /path/to/export \
  --source-branch spec-branch --source-commit 44450ab \
  --output build/bench/suite-resident
```

Archive origins are operator-supplied. The build also hashes every engine and kernel source file.
Do not label an edited export as an unchanged commit.

## Run

Set `CKPT` to the complete checkpoint directory.
Run the Python runner without an outer GPU lock. It applies the required lock to each resident process.

```bash
python3 bench/suite/run.py run "$CKPT" \
  --binary build/bench/suite-resident \
  --output results/suite-short
```

The default run uses one code, chat, prose, and copy prompt.
It requests 160 output IDs with serial decode and depth-two speculation.
It performs one discarded warmup and one measured repetition per prompt and mode.
Each request starts with reset state. The checkpoint loads once for the entire process.

Each process runs under this command prefix:

`$WORK_DIR` and the `WORK_` variables come from `tools/work_paths.sh`.
Run `. tools/work_paths.sh` in a checkout before the commands.

```bash
$WORK_GPU_RUN -p 3 -- timeout -k 10 300
```

The queue helper takes the GPU lock. `tools/gpu-run` is a simple stand-in (see `tools/README.md`).
Use `--gpu-run /path/to/gpu-run` if the installed helper has a different path.
The queue waits up to one hour by default. Exit code 75 means it did not obtain the lock.
Use `--gpu-wait 7200` for a longer queue wait. The GPU process still has a 300-second timeout.
Use `--priority 2` for a priority-two run. Valid priorities are zero through three; the default remains three.
The report records the selected priority and the exact queued command.
The host snapshot uses the central gate allowlist for runtime values.
It records unknown `CRUCIBLE_*` names with `[REDACTED]` values.
The runner rejects every unknown name before it requests a GPU lease.
The worker checks the environment again before it starts the engine.
Unset unknown variables before a run. Keep the two recorded policy files unchanged.
The ten-second kill grace stops commands that ignore TERM.
After a failed wait, use a new output directory to preserve the failed attempt.

The runner estimates the duration before it starts.
It refuses estimates above 270 seconds, leaving time for shutdown below the hard limit.
This estimate is admission control, not a guarantee. A timeout remains a failed run.
Use fewer prompts for repeated measurements.

```bash
python3 bench/suite/run.py run "$CKPT" \
  --binary build/bench/suite-resident \
  --prompts code_fibonacci --repeat 3 \
  --output results/suite-code-repeat
```

## Long contexts

Select one long fixture at a time.
Use a measured lower bound for that binary's prefill rate with `--prefill-floor`.
The default 100 tok/s bound rejects 16K and 64K runs on the small-row baseline.
Do not raise the bound merely to bypass the time limit.
Fast bulk prefill support depends on the selected source tree's `RunGen` implementation.
The suite does not force a bulk prefill option into older engines.

The fixtures exist even when a selected binary cannot process them within the shared-run limit.
Main 64K is not run because small-row prefill exceeds the 300-second limit.
The historical 16K small-row result took 211.7 seconds.

Build the bulk candidate with its explicit API adapter.

```bash
python3 bench/suite/run.py build \
  --source /path/to/prefill-source --bulk-prefill \
  --output build/bench/suite-prefill
```

Compare one prefill request through the first target ID. This mode does not measure decode or speculation.

```bash
python3 bench/suite/run.py run "$CKPT" \
  --binary build/bench/suite-main \
  --compare-binary build/bench/suite-prefill --compare-prefill-rows 4096 \
  --mode prefill --tokens 1 --warmup 0 --prompts long_4096 \
  --prefill-floor 77 --prefill-ceiling 2310 --output results/prefill-4k
```

Use `long_16384` for the 16K comparison. Run 64K on the bulk candidate only.

```bash
python3 bench/suite/run.py run "$CKPT" \
  --binary build/bench/suite-prefill --prefill-rows 4096 \
  --mode prefill --tokens 1 --warmup 0 --prompts long_65536 \
  --prefill-floor 1000 --prefill-ceiling 2310 --output results/prefill-64k
```

The candidate `577b645` supports the 3.05 checkpoint only. It rejects shared expert K6 in the 4.05 checkpoint.
The bulk row count is an implementation setting. Compare mode still requires matching prompt IDs and target output IDs.

## Post-push regression

Build the resident harness from the published main source.
The regression mode uses two short prompts and one 4K prefill request.
One checkpoint load serves nine requests, including short warmups.
The expected GPU time is below two minutes. Queue wait is separate.

```sh
SHA=$(git rev-parse --short HEAD)
python3 bench/suite/run.py build --source . --bulk-prefill \
  --output build/bench/suite-regression-$SHA
python3 bench/suite/run.py regression "$CKPT" \
  --binary build/bench/suite-regression-$SHA \
  --baseline bench/suite/results/2026-10-03-main-c286bda \
  --noise-percent 5 --gpu-wait 7200 \
  --output results/post-push-$SHA
```

The default prompts are `code_fibonacci`, `copy_python`, and `long_4096`.
The short requests use 160 output IDs with serial decode and depth-two speculation.
The 4K request stops at the first target ID.
The tool requires exact output IDs against the frozen published report.
The tool also requires serial and speculative IDs to match each other.
It checks checkpoint metadata and the baseline report hashes before GPU work.
The default threshold permits a five-percent rate decrease or TTFT increase.
A larger change returns nonzero and writes `FAIL` in `regression.json`.
Missing measurements, changed IDs, changed binaries, or failed processes also fail.
Use a new output directory for each attempt.

The queue uses priority three, a 7,200-second wait, and a 300-second timeout.
This performance check does not replace the shared correctness gates.
Advance the baseline only after reviewing a published main measurement.
Do not use an unpublished candidate as the baseline.

## Repetition spread

Full-suite qualification includes four short prompts and the 4K, 16K, and 64K prefill fixtures.
Run three separate resident sessions for each group.
Each short session uses the default warmup and measured repetition.
Each long session uses one measured request per fixture, without warmups.
This structure remains below the shared execution limit without raising admission estimates.

```sh
for REP in 1 2 3; do
  python3 bench/suite/run.py run "$CKPT" \
    --binary "$BINARY" --prefill-rows 4096 --prefill-ceiling 2310 \
    --gpu-wait 7200 --output results/repeat-$REP-short
  python3 bench/suite/run.py run "$CKPT" \
    --binary "$BINARY" --mode prefill --tokens 1 --warmup 0 \
    --prompts long_4096,long_16384,long_65536 \
    --prefill-rows 4096 --prefill-floor 1000 --prefill-ceiling 2310 \
    --gpu-wait 7200 --output results/repeat-$REP-long
 done
python3 bench/suite/spread.py results/repeat-{1,2,3}-{short,long}/a/results.json \
  --samples 3 --expect-full-suite --output results/full-spread.json
```

The CPU summary reports min, median, and max for all three measurements.
It rejects missing repetitions, mismatched request settings, unequal IDs, and different checkpoint or binary identities.
Renamed report copies cannot count as separate measurements.
It rejects repeated report content and repeated queued run identities.
It records every input report hash. Keep the raw reports beside the summary.

## Conversation TTFT through serve

The `conversations/` fixtures contain three scripted tool-use turns near 4K, 16K, and 30K final tokens.
Each later turn adds about 1,500 tokens. The manifest records stable cache boundaries and exact ID hashes.
Use a state-cache engine that accepts `cache_prefix_tokens` and `cache_control`.

```bash
python3 bench/suite/run.py serve "$CKPT" \
  --engine build/engine/crucible --source-commit COMMIT \
  --fixture 4k --cache-control auto --output results/serve-4k-auto
```

Run again with `--cache-control bypass` and a fresh output path for an uncached comparison.
For an older engine, add `--legacy-protocol --cache-control bypass`.
The runner reuses `server.backend.EngineBackend` for the resident JSON-lines transport.
It requests one target ID per turn, records complete IDs, and saves each raw event.
Compare those IDs before accepting a cached speedup.
Histories use fixed assistant replies, not earlier timed outputs. These fixtures do not measure answer quality.

Wall TTFT includes request dispatch and token transport, but excludes model load and GPU queue time.
The report keeps the engine's cache-hit, cached-token, and prefill-token counters without assuming a hit.
Admission estimates all turns without cache reuse. The default floor is 77 tok/s.
Use a measured faster `--prefill-floor` for large cases, not an assumed cache speedup.
The same priority-three queue and 300-second execution limit apply.

The resident full-head suite now pins draft selection, adaptation, Q, lookup, and overlap explicitly.
New engine CLI defaults cannot silently change its roofline model.

## Outputs

Each binary gets its own subdirectory, `a` or `b`.
The runner writes these artifacts:

- `results.json`: timings, counters, output IDs, source identity, hashes, and ceilings.
- `results.md`: a Markdown table with the same measured rows.
- `stdout.jsonl`: all raw result records, including discarded warmups.
- `stderr.log`: engine diagnostics.
- `requests.txt`: exact resident requests.
- `host.json`: memory state captured inside the lock, before model load.

The host record includes `/proc/meminfo`, `/proc/buddyinfo`, and the effective HSA memory policy.
The harness defaults `HSA_USERPTR_FOR_PAGED_MEM` to `0`. An explicit existing value remains unchanged.
The build record includes compiler details and source hashes.
Checkpoint identity includes its path, tokenizer hash, config hash, and quantization-manifest hash.
It does not hash every large weight shard. Preserve the checkpoint revision separately.

TTFT means engine prefill through the first target ID.
It excludes checkpoint load, request setup, tokenization, transport, and queue time.
Decode tok/s uses the remaining `N - 1` IDs.
Acceptance counts all proposed and accepted drafts, including a possible final overshoot beyond the requested length.

The runner checks complete output IDs, not only hashes.
It rejects missing rows, duplicate rows, wrong lengths, failed processes, and serial/speculative differences.
These checks cover Gate 3 on the selected prompts. They do not replace the other engine gates.

## Ceilings

The known 3.05 checkpoint gets the 53.19 tok/s serial weight-model ceiling.
Unknown checkpoint names do not inherit that ceiling.
Use `--serial-ceiling`, `--prefill-ceiling`, or `--spec-ceiling` only with a documented model for the tested workload.
Plain fixed-depth 3.05 speculation gets the documented representative expert-union weight model.
The model uses the actual delivered IDs per forward, but its expert union comes from separate measurements.
It excludes KV/state traffic, activation traffic, and launch gaps.
An explicit `--spec-ceiling` replaces this estimate.
Other unknown ceilings remain `null` in JSON and `n/a` in Markdown.
The 2,310 tok/s prefill reference also adds a separate percentage against the 3,866 tok/s fused-design estimate.
See [benchmark method](../../docs/BENCHMARKS.md) for units and assumptions.

Add these estimates to an older report without changing its measurements.

```bash
python3 bench/suite/run.py annotate results/old/results.json \
  --output results/old/with-ceilings.json
```

## Compare binaries or branches

Build one harness against each source worktree. This permits branch comparisons without changing either branch.
Run the binaries in sequence, with separate model loads and lock intervals.

```bash
python3 bench/suite/run.py run "$CKPT" \
  --binary build/bench/suite-a \
  --compare-binary build/bench/suite-b \
  --output results/suite-compare
```

Compare saved reports without a GPU.

```bash
python3 bench/suite/run.py compare \
  results/suite-compare/a/results.json \
  results/suite-compare/b/results.json \
  --output results/suite-compare/delta.json
```

The comparison rejects different request settings, prompt hashes, checkpoint metadata, or output IDs.
It writes both JSON changes and a Markdown table.
Positive throughput change is faster. Negative TTFT change is faster.
Repeat an A/B test in B/A order before attributing small changes to code.
Host memory state and cache state can change between lock intervals.

## Full engine gates

`bench/spec_gates.sh` checks Gates 1(a), 2, and 3 using the existing resident sweep.
Run it under the shared lock. Supply a trusted baseline for the same checkpoint and prompt.
It requires exactly 49 nonempty dump files and every expected sweep verdict.
A failed process, missing result, or mismatch returns a failure status.
It creates a new artifact directory and does not delete earlier gate data.
Gate 1(b) oracle checks are a separate procedure.
