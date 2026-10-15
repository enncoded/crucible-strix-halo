# Crucible

Crucible is an inference engine for **EXL3** checkpoints of **Qwen3.8-Flash-Next** on **AMD Strix Halo**
(**Ryzen AI Max+ 395**, **Radeon 8060S**, **gfx1151**) with **ROCm/HIP** kernels in C++20.
It runs the **Qwen 3.8 Flash Next** model family on one unified-memory APU. It does not use another
engine for inference. It supports local chat, streamed responses, function calls, and exact
speculative decoding.

This is research software for one GPU architecture, not a general model backend or a production service.
The target is the measured hardware ceiling, not another engine's speed.

[Progress report](https://srv.jaedyn.me/strix-halo-engine-optimization/index.html) · [Measurements](docs/BENCHMARKS.md) ·
[Architecture](docs/ARCHITECTURE.md) · [Build guide](docs/BUILD.md)

**These numbers are provisional.** They come from the working records of the development history
(up to 2026-10-15), not from a release run. A release run replaces them. Each row names its date and
its number of runs. A row is not a measurement of a later build.

## Measured results

The test system has a Ryzen AI Max+ 395, Radeon 8060S (gfx1151), 128 GB shared LPDDR5X-8000.
Its measured ceilings are **245.5 GB/s cold reads** and **48.2 TFLOPS f16 WMMA**.
See [chip calibration](docs/CALIBRATION.md).

| Workload | Measured rate | Hardware ceiling | Percent of ceiling | Runs | Date |
|---|---:|---:|---:|---:|---|
| Serial decode, 3.05 bpw | 38.79 tok/s | 53.19 tok/s | 72.9% | 2 | 2026-10-04 |
| Serial decode, 4.05 bpw | 33.71 tok/s | 45.84 tok/s | 73.5% | 2 | 2026-10-04 |
| MTP, two drafts, 3.05 bpw | 68.02 tok/s | acceptance-dependent | n/a | 1 | 2026-10-03 |
| 4,096-token prefill, 3.05 bpw | 1,566.2 tok/s | 2,310 tok/s | 67.8% | 2 | 2026-10-04 |
| 16,384-token prefill, 3.05 bpw | 1,526.9 tok/s | 2,310 tok/s | 66.1% | 2 | 2026-10-04 |
| 4,096-token prefill, 4.05 bpw | 1,511.3 tok/s | 2,289 tok/s | 66.0% | 2 | 2026-10-04 |
| 16,384-token prefill, 4.05 bpw | 1,493.1 tok/s | 2,289 tok/s | 65.2% | 2 | 2026-10-04 |
| Eight streams in total, 3.05 bpw | 104.24 tok/s | no single ceiling | n/a | 1 | 2026-10-03 |
| Eight streams in total, 4.05 bpw | 90.83 tok/s | no single ceiling | n/a | 1 | 2026-10-03 |

The serial ceiling is the weight-read ceiling: 4.615 GB of logical weight reads for each token at
3.05 bits per weight, and 5.356 GB at 4.05 bits per weight.
The prefill ceiling is an estimate for a 4,096-token prompt with EXL3 trellis decode in the current
data path.
The percentage is a model ratio. It is not a measured GPU utilization.
Prefill timing ends at the first target ID. It excludes model load, tokenization, queue time, and
HTTP transport.

## Same-harness comparison with Halogen and Gufo

| Engine | Version | Date | Cold 32K prefill | Cold 130K prefill | Decode after a 32K prompt |
|---|---|---|---:|---:|---:|
| Crucible | development build of 2026-10-08 (3.05 bits per weight) | 2026-10-10 | 22.4 s | 88.3 s | 79 to 80 tok/s |
| Halogen | 0.16.2 | 2026-10-10 | 18.6 s | 75.0 s | 53 to 55 tok/s |
| Crucible | development build of 2026-10-06 (3.05 bits per weight) | 2026-10-07 | 22.4 s | 94.8 s | not measured in this run |
| Halogen | 0.16.2 | 2026-10-07 | 18.2 s | 74.0 s | not measured in this run |
| Gufo | 0.9.0 (GGUF UD-Q4_K_XL, MTP on) | 2026-10-07 | 23.8 s | 97.7 s | not measured in this run |

**Thank you to the authors of Halogen and of Gufo.** Their engines run on the same hardware with
their own weight formats, and they made this comparison possible.

How these runs were made: one machine, the same request harness
(`tools/h2h_decode.py`), synthetic prompts, a fresh nonce in each prompt so that every prefill is
cold, greedy decoding, and thinking off. Each engine runs the checkpoint and the flags that its own
authors recommend. The engine and the weight format differ, so a rate difference is not an
engine-only comparison. Halogen ran at about 5% higher GPU clock in the 2026-10-10 run.
Gufo 0.9.0 ran on 2026-10-07, three days before the 2026-10-10 run, with its GGUF UD-Q4_K_XL
pack. That run used an older Crucible build, so its rows show that build.
These comparison numbers are provisional in the same way as the table above.

## What it is

- One engine for one model family and one GPU architecture. No fallback path.
- **Exact speculative decoding.** The target verifies every draft. A rejected suffix never becomes
  committed state.
- **A keyed sampling law.** A seeded request gives the same IDs at any speculation depth, and in a
  packed batch pass.
- **Measured hardware ceilings.** Most speed claims name the ceiling and the percentage of it.
  Speculative decoding and batching have no single ceiling.
- A local OpenAI-compatible HTTP frontend with chat, streaming, tools, and up to eight sessions on
  one engine.

## Quick start

### 1. Build

```sh
make
```

The build needs the [software below](#hardware-and-software). It writes `build/engine/crucible`.

### 2. Download the model

Read the [model license note](#model-license) first.

```sh
PACK=3.05bpw_h5_ng5
CKPT="$PWD/models/Qwen3.8-Flash-Next-exl3-$PACK"
hf download turboderp/Qwen3.8-Flash-Next-exl3 --revision "$PACK" --local-dir "$CKPT"
```

The supported packs are in [turboderp/Qwen3.8-Flash-Next-exl3](https://huggingface.co/turboderp/Qwen3.8-Flash-Next-exl3).
Use revision `4.05bpw_h6_ng6` for the quality pack.
Keep every tokenizer file, weight shard, n-gram table, and MTP mixer patch.

### 3. Run

```sh
python -m server "$CKPT" --engine build/engine/crucible --ctx 32768
```

The server binds to `127.0.0.1:8790`. It has no authentication. Do not expose it to a network.
Send a request from another terminal:

```sh
curl -sS http://127.0.0.1:8790/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"crucible","messages":[{"role":"user","content":"Say hello."}],
       "temperature":0,"max_tokens":32}'
```

The API provides `/v1/chat/completions`, `/v1/completions`, `/v1/models` and `/tokenize`.
Use `stream: true` for server-sent events. Add OpenAI `tools` definitions for function calls.
See [the server guide](server/README.md) for the options and the limits.

For direct generation without the server:

```sh
python tools/tok.py "$CKPT" encode "Write a Python function that adds two numbers." --chat > prompt.ids
build/engine/crucible gen "$CKPT" --ids-file prompt.ids --n 160 --spec 2 --ctx 4096
```

## Hardware and software

- Ryzen AI Max+ 395, Radeon 8060S (gfx1151), 128 GB shared LPDDR5X-8000.
- Linux. The tested system is Arch Linux with kernel 7.2.7 and ROCm 7.2.4.
- A C++20 HIP compiler, GNU Make, and POSIX threads. The CPU test suite needs
  GCC 14 or newer, or Clang 18 or newer.
- Python 3.12 or newer. The CPU dependencies are in `requirements-cpu.txt`.
- Enough disk space for one complete checkpoint and the build output.

The 3.05 bits-per-weight payload alone is about 84.45 GB. Use one model instance. An ordinary
discrete desktop GPU does not have enough memory for it.
The 4.05 bits-per-weight pack needs more memory than the 3.05 pack.
Set `CRUCIBLE_PLE_HOST=1` to keep the PLE n-gram table in host memory, at 32.6 GB (3.05) or
39.0 GB (4.05).

## Model license

The engine uses Apache-2.0. **The model weights use other terms.**
Both supported EXL3 packs use the Qwen Community License 1.0.
Commercial use has additional conditions.
Read [the model license note](docs/THIRD_PARTY.md#model-weights) before you download or use a pack.
The license covers the model artifacts. It does not cover this engine. This engine does not include
the weights.

## Tests

```sh
make -j2 test-cpu
```

The CPU suite builds the test and benchmark programs and runs the offline tests. It needs no GPU.
GPU checks are manual. They need the qualified machine, both checkpoints, and the gate tool
`tools/gates.py`.
See [all build and test targets](docs/BUILD.md).

## Documentation

- [Architecture](docs/ARCHITECTURE.md) and [the Flash-Next forward specification](docs/FLASHNEXT_FORWARD.md)
- [EXL3 format](docs/EXL3_FORMAT.md) and [checkpoint metadata](docs/EXL3_CHECKPOINT_METADATA.json)
- [Build and test guide](docs/BUILD.md)
- [Calibration](docs/CALIBRATION.md) and [benchmark history](docs/BENCHMARKS.md)
- [Exactness rules and gates](docs/ARCHITECTURE.md#exactness-rules)
- [Server guide](server/README.md), [OpenAI clients](docs/OPENAI_CLIENTS.md) and [session replay](docs/SESSION_REPLAY.md)
- [Third-party terms](docs/THIRD_PARTY.md)
- Progress report: [https://srv.jaedyn.me/strix-halo-engine-optimization/index.html](https://srv.jaedyn.me/strix-halo-engine-optimization/index.html)

## Limitations

- Only the named model packs and gfx1151 are qualified.
- Image input is opt-in (`--vision`). Video is not supported. There is no general model loader.
- The HTTP frontend has no authentication, no TLS, and no tenant isolation. Keep it on loopback.
- The conversation cache is opt-in and stays in host memory. A restart clears it.
- Long contexts need a large amount of memory. A long-context timing is not a quality test.
- Generated tool arguments can be invalid. The server reports the error. It never runs a tool.

## License

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
The ExLlamaV3 notice is in `NOTICE`.
