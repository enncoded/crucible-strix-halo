# Multi-session agent benchmark

This benchmark measures concurrent coding-session traffic through the OpenAI streaming API.
TTFT is the first metric. CPU mock results are tests, not GPU performance evidence.

## Workload

- Use K = 1, 2, 4, or 8 sessions.
- Use a common system-and-tools prefix of about 12K tokens.
- Append public synthetic code through tool messages. No private repository data is used.
- Cycle suffix targets through 256, 1K, 4K, and 16K tokens.
- Request 200..800 output tokens per turn. The default is 400.
- Grow each history toward a 64K or 128K context limit, including its output budget.
- Record actual token counts. Complete code records can differ from a suffix target.
- Clip the last suffix when needed. Keep space for output and message framing.
- Use fixed seeds and teacher-forced histories. Timed model output does not change later requests.

Only the first requests start at a common barrier. Each session submits its next turn after its previous response completes.
This permits one session's prefill to interrupt another session's decode.

## CPU preparation

Run these commands from the repository root. The preparation command needs `tokenizers` and the checkpoint template dependencies.

```bash
python -m unittest discover -s bench/agents/tests -p 'test_*.py'
python -m bench.agents.runner prepare /path/to/checkpoint \
  --sessions 4 --context 65536 --tokens 400 --output /tmp/agents-k4-64k.json
```

Use `--legacy-metrics` for a server that does not accept `crucible_metrics`.
Unknown server measurements remain null. Do not infer phase times from client TTFT.

## GPU baseline plan

The serving code owns the opt-in timing contract.
Measure these two named baselines separately:

1. Single-stream serve with `--prefill 4096 --conversation-cache-mib 4096`, using FIFO turns.
2. Batch serve with `--streams B`, the same prefill grid, and the same cache budget.
   Main train 5 supports greedy and sampled speculation, prefill, and conversation caching.

The runner connects to an existing endpoint. It does not start a GPU process or change a persistent service.
All GPU server work must run through `gpu-run`, with the approved process timeout.
Split measurements by K and context. Ask the parent before a request over 240 seconds.
Do not use the C5 KL timeout exception for serving measurements.

After an approved server starts, run:

```bash
python -m bench.agents.runner run /tmp/agents-k4-64k.json \
  --url http://127.0.0.1:8080/v1/chat/completions \
  --label fifo-prefill4096-cache4096 \
  --server-record /path/to/build-record.json --output /tmp/agents-k4-64k-report.json
```

Use a new output path for each run. The report binds the plan, server record, and measurement tool hashes.
Supply `--decode-ceiling` only when a documented ceiling applies to that baseline.
Do not multiply a shared GPU ceiling by K. Batched rooflines remain unknown unless a model defines them.

## Measurements

- Client TTFT starts before HTTP submission and ends at the first streamed token ID.
- Empty role chunks and empty text do not substitute for a token ID.
- The reader preserves every ID and its client arrival time.
- Optional `crucible_t` values use the server clock. Do not mix client and server timestamps.
- Final `usage.crucible_metrics` supplies queue, restore, prefill, between-chunk, and post-prefill observations.
- Server TTFT equals queue + prefill + between-chunk + post-prefill time when all fields exist.
- Restore time is nested in prefill or between-chunk time. Do not add it a second time.
- Prefill chunks identify token gaps that overlap another session's prefill.
- Truncated chunk observations cannot prove complete interference statistics.
- The summary uses nearest-rank p50, p95, and p99 values. It preserves the sample count.
- Per-session decode uses N-1 tokens divided by the first-to-last token interval.
- Aggregate generation uses all output tokens divided by first submission through last completion.
- The aggregate interval includes queue and prefill time. It is not a decode-only measurement.
- Cache hit rates use observed cached-token counts. Restoration totals require explicit server measurements.
- KV capacity and use come from the last observed server memory record. They are not estimated from K or context limits.
- Request rejections and errors remain in the report. A failed session does not submit its remaining turns.

## Agent-shaped matrix

`bench.agents.coding` prepares fifteen turns with a 12K shared prefix and tool results from 0.5K to 8K tokens.
It supports greedy and sampled requests (temperature 0.6, top_p 0.95, top_k 20).
`bench.agents.matrix` queues one temporary server lease at a time, at priority 2.
A startup probe selects the cache budget from MemAvailable minus nine GiB, with a 16 GiB maximum.
The protocol tee preserves engine stdin and stdout. Its extra stdout hop is a measurement condition.
The lease checks port ownership, its own ready event, the live child, and matching engine done events.
Raw done events supply snapshot bytes. A one-second sampler records the MemAvailable floor.
Slot bytes use the pinned full-storage formula. They are source-derived values, not a dynamic pool measurement.
K4/K8 rows use short segments. Later segment starts remain a separate cold-start category.
Do not label segmented rows as uninterrupted fifteen-turn sessions.

## Current limits

The CPU tests cover K 1/2/4/8 through a mock HTTP engine with real SSE responses.
The first train-5 GPU pilot passes. The full matrix and measured KV pool values remain pending.
Train 5 does not report memory fields in the observed done events.
The server trace ring integration remains pending. Long batch prefills can exceed the per-request chunk limit.
The current report does not measure process RSS or physical system memory. Missing memory fields remain null.
