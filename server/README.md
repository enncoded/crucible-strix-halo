# Crucible local server

This server provides a small OpenAI-compatible API.
It supports text chat and text completions with greedy or exact sampled decoding.
It binds only to `127.0.0.1`, on port 8790 unless `--port` selects another port.
It has no authentication and must remain local.
It is not a persistent service.

## Agent clients

See [OpenAI and Pi connections](../docs/OPENAI_CLIENTS.md) for client settings and CPU compatibility checks.
Use Chat Completions with text input, streamed usage, and validated function calls.

## Run a local server

Follow the [build guide](../docs/BUILD.md) to install Python dependencies.
Run these commands from the repository root.
Use a complete checkpoint from the [download guide](../README.md#download-a-model).

```bash
make
CKPT="$PWD/models/Qwen3.8-Flash-Next-exl3-3.05bpw_h5_ng5"
python -m server "$CKPT" --engine build/engine/crucible --ctx 32768
```

These commands assume exclusive use of your own GPU.
On a shared GPU, start each command through `tools/gpu-run` with a timeout.
Stop the server with Ctrl-C after the test.
Do not run another full engine while this server uses the GPU.

GEMM prefill keeps a 1.29 GB copy of the HC weights for speed.
It makes prefill about 44 ms faster per 4K tokens. The results do not change.

Send a chat request from another terminal:

```bash
curl -sS -N http://127.0.0.1:8790/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"crucible","messages":[{"role":"user","content":"Say hello."}],
       "temperature":0,"max_tokens":32,"stream":true,
       "stream_options":{"include_usage":true}}'
```

Set `stream` to `false` for one JSON response.
Use `GET /v1/models` to get the loaded model ID.
Use `/v1/completions` with a `prompt` string for plain text.
A completion prompt can also contain one list of token IDs.
Batches are not supported.

### Tokenize and apply the template (`POST /tokenize`, `POST /v1/tokenize`, `POST /apply-template`)

These utility paths use the OpenAI-compatible request shapes. They use the CPU only: no engine command, no queue entry, no
slot, and no cache change. A success writes no log line; a refusal writes one `crucible_error` line, as on chat.

- `/tokenize` and `/v1/tokenize` with `{"content": "text"}` return `{"tokens": [...]}`: the IDs of a
  `/v1/completions` prompt with the same text (special-token text is parsed, as in a raw prompt). A list of token
  IDs comes back unchanged, and an empty text gives an empty list. `add_special` and `parse_special` are accepted
  and change nothing (the model has no BOS token). `with_pieces: true` gets HTTP 400, and so does a body with both
  `content` and `messages`.
- `/tokenize` with `{"messages": [...]}` and any chat field (`tools`, `tool_choice`, thinking fields, `model`)
  returns the IDs that `/v1/chat/completions` prefills for that request, with the server flags.
- `/apply-template` with `{"messages": [...]}` and any chat field returns `{"prompt": "..."}`: the chat prompt text
  that the template renders for that request. An image part shows as one placeholder; the prefill expands it.

### Token count (`POST /v1/messages/count_tokens`)

Gateways that use the Anthropic Messages format ask for the input token count before each request. The server
answers `200 {"input_tokens": N}`. N is the number of prompt IDs that `/v1/chat/completions` prefills for the same
conversation with the same server flags (template, tools, `--default-thinking`, `--vision`). The count calls the same
prompt function as the chat path (`Service.prompt_ids` in `server/http.py`). It uses the CPU only: no engine command,
no queue entry, no slot, and no cache change. A query string such as `?beta=true` is accepted.

The request body is an Anthropic Messages body. `server/anthropic.py` maps it to a chat request:

| Anthropic field | Chat request |
|---|---|
| `system` (text or text blocks), or a `system` message | the first `system` message (blocks become text parts) |
| user `text` and `image` blocks, or string content | one `user` message; an image needs `--vision`, as in chat |
| user `tool_result` blocks | one `tool` message each, before the user's text, with `tool_call_id` = `tool_use_id`; text blocks are joined; `is_error: true` adds the text `Error: ` at the start |
| assistant `text` blocks | `content` (joined) |
| assistant `thinking` blocks | `reasoning_content` (joined); `redacted_thinking` is ignored |
| assistant `tool_use` blocks | `tool_calls` (`id`, `name`, `arguments` = JSON of `input`) |
| `tools` (`name`, `description`, `input_schema`) | function tools with `parameters` = `input_schema`; server tools are refused |
| `tool_choice` `auto`, `none`, `any`, `{"type":"tool","name":...}` | `auto`, `none`, `required`, the named function |
| `disable_parallel_tool_use: true` | `parallel_tool_calls: false` |
| `thinking` `{"type":"disabled"}`; `enabled` or `adaptive` | thinking off; thinking on (`budget_tokens` does not change the prompt) |
| `output_config.effort` `low`, `medium`, `high`, `max` | `reasoning_effort` with the same name: template levels `low`, `medium`, `xhigh`, `xhigh` |
| no `thinking` and no effort | the server default (`--default-thinking`) |
| `max_tokens`, `stream`, `temperature`, `top_p`, `top_k`, `stop_sequences`, `metadata`, `service_tier`, `context_management`, `cache_control`, `container` | ignored (another unknown field gets HTTP 400, because it could change the prompt) |
| `cache_control` and `citations` on blocks and tools | ignored |

Errors are the same as on the chat path, with the server's error object: an unknown model gets HTTP 404, a bad
shape or an unknown field HTTP 400, and a body above `--max-body-mib` HTTP 413. A refusal writes one
`crucible_error` line with a reason code only. A successful count writes no log line.

The server protects itself against a bad client:

- It accepts at most 256 connections at one time. Above that, a new connection gets HTTP 503.
- A client must send the request head and body in 60 s. Then the server closes the connection.
- A request that does not fit the engine line limit (2 MiB) gets HTTP 413.
- With `--serial`, a stream that does not read its output for 30 s stops with an error.
- `--max-requests-per-client N` limits the open requests of one client address (HTTP 429). It is off by default,
  because all clients on the loopback address share one address.
- The log line `crucible_engine_error` holds only known engine messages. Another engine text appears as `other`, with
  its length and a sha256 prefix. `--log-engine-errors` also logs that text. Use it only with your own data, because
  the text could quote request content.
- The server passes only the allowed `CRUCIBLE_*` switches to the engine (`ENGINE_KEEP_SWITCHES` in
  `server/backend.py`): kernel and layout switches, and switches that print numbers only (timings, draft counts,
  kill-check counts). It removes every other `CRUCIBLE_*` name, for example the capture, dump and interleave check
  switches, and names the removed switches once on stderr.
  The `kServe` column of the switch registry `engine/switches.h` is the source of this set. A CPU test checks it.
  Serve never writes content-derived data to disk.
- Special-token strings in client content are text (RA-8). This applies to message content, tool results, tool call
  arguments, tool definitions and the text parts of multi-part content. For example, a tool result that contains
  `<|im_end|>` or `<tool_call>` encodes to the characters that it shows. Only the chat template writes control
  tokens. A prompt with no such string gets the same IDs as before, so the cache, session replay and slot resume do
  not change. The `/v1/completions` prompt is raw model input: its special-token strings stay control tokens.
  An added token that the model wrote itself (for example in a tool-call fallback) keeps its ID when the client
  sends the returned assistant text back unchanged, so session replay still hits. The server remembers only a hash
  and token offsets of such texts. A text that the model generated in two different ways stays text. Because of
  this memory, one fixed request body can give other prompt IDs after a restart, as with session replay (main's
  decision). The token count and the tokenize paths follow the same memory.
  `--content-special-tokens parse` sets the old rule. The startup line names the mode
  (`Crucible is ready at ... (content special tokens: text)`). `server/content_tokens.py` has the method.

## Supported request options

- `temperature`: a finite number from `0` through `2`, default `0`.
- `top_k`: an integer from `0` through `248320`; `0` disables this filter.
- `top_p`: a finite number from `0` through `1`, default `1`.
- `min_p`: a finite number from `0` through `1`, default `0`.
- `seed`: an unsigned 64-bit integer. An omitted or null seed uses fresh system entropy.
  Every response and stream chunk reports the selected value as `seed`.
- `max_tokens` or `max_completion_tokens`: a positive integer.
- `stop`: one string or at most four strings, each with 1 to 4096 UTF-8 bytes.
  Null, an empty list, or an omitted field disables text stops.
- `stream`: `true` or `false`.
- `stream_options.include_usage`: include the final SSE usage event.
- `spec`: draft depth from `0` through `7`, default `7`; `0` selects serial decode.
- `draft_q`: the draft confidence limit from `0` through `1`, default `0.3`; `0` disables it.
- `draft_head`: `64k` by default; use `full` for the full vocabulary.
- `mtp_prefix`: auxiliary warmup limit from `0` through `262144`; `0` warms the full prompt. The default comes from
  the server flag `--mtp-window N` (default `256`). Batched serve accepts `0` too. At an equal restore point, the
  window changes only the drafts, not the output IDs. In batched serve, the server window also sets the MTP
  cache size of each slot: window `W` above 0 gives `W + 33024` positions (at most the slot context), and `0`
  gives the slot context. A request that needs more MTP positions restarts its MTP at position 0. This changes only
  the drafts.
  A profile that uses the full MTP history (`mtp_prefix 0` on its requests) must start the server with
  `--mtp-window 0`. With the default window, such a request restarts its MTP above `256 + 33024` positions and saves
  no MTP snapshot, so its drafts lose the history.
- Draft vocabulary data is a server setting, off by default. `--draft-capture DIR` (an absolute path; the
  server makes it 0700) writes `DIR/draft-counts-YYYY-MM-DD.json` (0600) every 10 minutes and at stop: for each UTC
  day and turn class (`chat`, `chat+think`, `tool`, `tool+think`), how often each 128-token block occurs in the
  generated output. The file holds numbers only: no text, no prompt, no request ID, no time finer than the day.
  `--draft-capture-mode tokens` writes a token ID spool for the machine-only training pipeline instead. Only that
  exact value turns it on, and it needs the user's explicit approval. `--draft-capture-max-gib N` (tokens mode
  only) limits the spool. DIR must be the server user's own directory, closed to group and other, and not a
  symlink; one server at a time writes it (`DIR/.lock`); file times are set to 00:00 UTC of the day. A capture
  fault never fails a request.
  Export floor: the per-day files never leave the serving machine. Any export sums 7 or more days and drops each class with
  fewer than 20 sequences.
- `draft_adapt`: optional `0` or `1`; the default adapts the built-in 64K head.
  Value `1` requires the `64k` head.
- Prompt lookup is a server setting, not a request option. The server flag `--lookup N` sets the depth
  (`0` to `7`, default `7`; minimum match 8 tokens). A speculative request then also proposes earlier prompt or
  output tokens that follow the same 8 or more tokens. Use `--lookup 0` to turn it off. The output does not change.
- `enable_thinking`: `true` or `false`. The default comes from `--default-thinking` (`off` unless set).
- `chat_template_kwargs`: `enable_thinking`, `preserve_thinking`, and `reasoning_effort` only.
- `reasoning_effort`: `none`, `minimal`, `low`, `medium`, `high`, `xhigh`, or `max`.
  The server maps `minimal` to `low`, and `high` and `max` to `xhigh`.
  Without an explicit `enable_thinking`, a level turns thinking on and `none` turns it off.
- `n`: only `1`.
- `return_token_ids`: an optional diagnostic extension.

The diagnostic extension adds `token_ids` to a non-streamed choice.
It adds `token_id` to each streamed token choice.
It includes terminal stop IDs.

With thinking on, the server splits the output at `</think>`, as other engines and Halogen do:

- The text before the tag goes to `message.reasoning_content` (streamed as `delta.reasoning_content`).
- The text after the tag goes to `content`. Tool calls come only from this part.
- `usage.completion_tokens_details.reasoning_tokens` counts the tokens before and including the tag.
- If `max_tokens` stops the reasoning before the tag, all text is reasoning and `finish_reason` is `length`.

The server keeps room for the answer, with Halogen's rule (Halogen 0.11 to 0.15):

- The answer room is `max(1024, 15% of max_tokens)` tokens, after `--clamp-max-tokens`.
  `--thinking-answer-room N` sets it to N tokens. `0` turns it off.
- A thinking budget can close the block earlier: `max_thinking_tokens`, or the aliases `thinking_budget_tokens`,
  `thinking_budget`, and `thinking_token_budget` (two different values give HTTP 400).
  `--max-thinking-tokens N` sets a server default. The request value wins.
- The block closes after `max(1, max_tokens - room)` generated tokens, or after the budget if it is smaller.
  The server then writes Qwen's budget sentence, `</think>`, and two newlines. These tokens count against `max_tokens`.
- When the whole close text fits, `usage.completion_tokens_details` has `reasoning_closed_at` (the token count) and
  `reasoning_closed_by` (`answer_room` or `max_thinking_tokens`).
- The close position depends only on the request, so it lands on the same token with any decode mode, layout, or load.

The engine has no forced-token command. At the close, the server stops the engine job and starts a new one with the
prompt, the generated reasoning, and the close text. The new job restores the prompt's snapshot and prefills the
reasoning again (about 1 s for each 1,300 tokens). Only turns that reach the close pay this cost.
The continuation depends only on the request. It is not bit-identical to one job that decodes the forced tokens,
because the second job computes the reasoning with prefill arithmetic.
That turn writes one `crucible_turn` line with the counts that the client sees and the first job's times,
so a log collector can join it to its own request record. `think_closed_at`, `think_closed_by`, and `cont_*` fields describe the
second job. `crucible_metrics` in the reply describes the second job.
Every `crucible_turn` line also carries numbers about the engine, the decode passes, the stall and the token
classes.
The reply has no `prompt_tokens_details`: the first job stops before it reports its cache count, and the second
job's count describes only the snapshot that the first job wrote.

The template replays assistant `reasoning_content` inside `<think>` for every turn, unless
`chat_template_kwargs.preserve_thinking` is `false`. Then it keeps only the reasoning after the last user turn.

The server renders the checkpoint's real chat template.
It accepts system, user, assistant, and tool messages.
It also accepts text content blocks, assistant `reasoning_content`, and assistant `tool_calls`.
With `--vision`, it also accepts image content parts. See [Image input](#image-input).
It rejects video, unsupported sampling controls, and unknown fields.

Generation stops at `<|im_end|>` (`248046`) or `<|endoftext|>` (`248044`).
The response omits these markers from text.
Usage counts include the generated stop token.
The prompt and requested output must fit the configured context.
Otherwise the server returns HTTP 400 with a message that contains "exceeds the context window".
Start the server with `--clamp-max-tokens` to lower `max_tokens` to the remaining context instead.
Request bodies are limited to 1 MiB (HTTP 413). `--max-body-mib N` raises the limit to N MiB (at most 64, or 1024 with `--vision`).
A 256K-token conversation of code can need more than 1 MiB.

Without `--streams`, the server uses the batch worker with one stream (`--streams 1`).
It processes one request at a time. The remaining requests wait in a queue.
`--streams N` (1 to 8) runs up to N requests at once.
`--serial` selects the old serial worker for one train, as a rollback (removed in train 20).
Do not use `--serial` with `--streams`. The old worker has a FIFO queue with a limit of 64 waiting requests.
The batch worker refuses `draft_head` `full`, `draft_adapt` 1 and `draft_q` 0 with HTTP 400. Only `--serial` accepts them.
A full queue returns HTTP 503.
A streamed request gets its HTTP 200 headers when the server accepts it, before the prefill starts.
While no token comes, the server sends the SSE comment `: keepalive` every 10 seconds.
SSE clients ignore comments, and the comments keep body timers (for example undici's 300 s timer) from firing.
An error after the headers ends the stream with an error chunk and `data: [DONE]`.
A connected client waits up to 3600 seconds for its first token, and up to 360 seconds between tokens.
The server checks the client connection every 0.25 seconds while no token comes.
If the client closed it, the server cancels the request and logs a `crucible_cancel` line.
The batch worker then stops the request before its next prefill chunk.
The serial worker (`--serial`) finishes a disconnected request internally before the next request starts.
This keeps the engine protocol synchronized.
Without `--conversation-cache-mib`, each request resets the sequence state and the server keeps no conversation state.
With it, the server keeps prompt snapshots in host memory, and session replay is on by default
([session replay](../docs/SESSION_REPLAY.md)). A restart clears the cache.

## Image input

Start the server with `--vision` to load the vision tower (about 0.3 GB).
Without `--vision`, a request with an image part gets HTTP 400 with a message that names the flag.
`--vision` also raises the default body limit to 256 MiB (`--max-body-mib` allows up to 1024 with `--vision`).

Send each image as an OpenAI `image_url` part with a base64 `data:` URL:

```json
{"role": "user", "content": [
  {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0..."}},
  {"type": "text", "text": "What does the red button say?"}]}
```

- User and tool messages can contain images. System and assistant messages cannot.
- A request can contain at most 256 images (`--vision-max-images`). The context window is the real limit:
  an agent client can send every earlier screenshot again on each turn. When the image tokens of a request pass the
  context (`--ctx`), the server refuses the request (HTTP 400, "exceeds the context window") before it decodes the
  next image.
- The server gives the engine each resized image as a private file (mode 0600) in a private directory
  (mode 0700) under `/dev/shm`. The server deletes the files when the engine has finished with the request.
  The engine reads only the files of images that it encodes; a cached image is not read.
- The server does not fetch `http` or `https` URLs. Send the pixels in the request.
- The server resizes each image as Halogen does: both sides become multiples of 32,
  with at least 256x256 and at most 2560x1440 pixels. `--vision-max-pixels N` (65536 to 4194304) changes the maximum.
- An image of H x W pixels after the resize uses (H/32) x (W/32) prompt tokens.
  A 1280x800 screenshot uses 1,000 tokens.
- A bad image gets HTTP 400 with the reason: not base64, not decodable, aspect ratio above 200:1,
  or above four times the maximum pixel count before the resize.
- The conversation cache keys each image by a SHA-256 of its resized pixels.
  Two different images never share a cache entry. A cached image is not encoded again.
- Text-only requests give the same tokens with and without `--vision`.
- Video parts are not supported.

## Tool calls

Chat requests accept `tools` with OpenAI function definitions.
The default `tool_choice` is `auto`.
Use `none`, `required`, or `{"type":"function","function":{"name":"add"}}` to select another policy.
Use `parallel_tool_calls: false` to allow at most one call in the model response.
These fields are not available on `/v1/completions`.

The server uses the checkpoint template and its typed XML format.
It returns OpenAI `tool_calls` with JSON argument strings and generated call IDs.
Its finish reason is `tool_calls` when the response contains calls.
Streaming sends each call in indexed `delta.tool_calls` entries while the model writes it.
The id, type and name go out when the function tag closes and the name passes `tool_choice` and `parallel_tool_calls`.
A parameter whose schema is only `"type": "string"` streams as JSON-escaped argument text.
Other parameters go out when they close and pass validation.
The closing `}` goes out when the whole call passes validation.
For a valid call, the joined argument text is byte-equal to the non-streamed `arguments`.
Use `--stream-tool-calls off` to send each call whole when the output ends.
A complete, well-formed call that fails only the schema goes out as a normal tool call with `finish_reason`
`tool_calls` (design decision, Oct 8). Examples: an unknown or missing property, a wrong type or enum value, a value
that is not JSON for its type (it goes out as the raw string), or a function that is not in `tools` (if its name
matches `[A-Za-z0-9_-]{1,64}`). The server never drops or changes a parameter. The client checks the call against
its own schema and sends the model the exact error. A tool with `strict: true` keeps the strict check.
stderr then has one `crucible_tool_pass` line with the call count and the reason codes only (not an error line).
Model output that is not a well-formed call comes back as content, with `tool_call_fallback: true` and a fixed reason
code in `usage.tool_call_invalid` (for example `incomplete_call`, `parameter_tag`, `unexpected_text`,
`duplicate_parameter`, `tool_choice`, `parallel_calls`).
If deltas of a call already went out, the stream ends with `finish_reason` `length`, not `stop`.
Clients such as Pi then do not run the partial call. Pi sends the model an error result and asks it to issue the call again.

The client must execute each function and send one tool result for each call ID.
Include the assistant message and its `tool_calls` before these results.
Use `{"role":"tool","tool_call_id":"call_...","content":"42"}` for a result.
Send all pending results before the next normal turn.
The server never executes a function.

Arguments must be finite JSON objects with unique keys.
The parser validates common types, required fields, bounds, and local schema references.
It does not implement complete JSON Schema or constrained decoding.
It does not execute untrusted schema regular expressions.
Schema processing has a shared limit of 16,384 traversal steps per request or output.
Input schemas share limits of 4,096 JSON nodes, 128 KiB of compact UTF-8 JSON, and 512 reference entries.
Nesting cannot exceed 64 levels; each reference chain cannot exceed 32 links.
Cyclic references are not supported.
The validator checks each schema node once and caches output matches by schema node and argument path.
Over-limit input returns HTTP 400 before queue submission.
A model output that does not parse as valid calls (malformed, over a limit, or against `tool_choice` or
`parallel_tool_calls`) returns as content, as in other engines, with finish_reason `stop` (or `length` at `max_tokens`).
The usage then has `tool_call_fallback: true`, and stderr has one `crucible_error` line with counts only.
HTTP 502 stays for engine failures. Grammar-constrained calls would prevent malformed calls.
Raw string values can be ambiguous when they contain literal structural end tags.
Prefer simple parameter types over string/nonstring unions.

## Engine JSON-line protocol

Start `crucible serve CKPT [--ctx N]` with exclusive GPU access.
The resident server uses the same decode defaults as `gen`.
Use `--spec 0`, `--draft-q 0`, `--draft-head full`, `--mtp-prefix 0`, or `--draft-adapt 0` to disable each feature.
Raw request fields override these startup settings for one request only.
The engine loads once and writes a `ready` event.
Send one JSON object per stdin line:

```json
{"ids":[248045,846,198],"max_tokens":32,"spec":2,"stop_ids":[248046,248044]}
```

The engine writes accepted token IDs immediately:

```json
{"event":"token","id":123}
```

A `done` event gives prompt usage, completion usage, timing, and the finish reason.
A malformed request produces an `error` event without stopping the resident engine.
The input limit is 2 MiB per line.
With `--vision 1`, a request can add `"images"` (at most 256): a list of objects with `offset` (first image token),
`height`, `width`, `digest` (64 hex characters), and `path` (an absolute path to a regular file of
height x width x 3 RGB bytes, not a symbolic link). The digest is the SHA-256 of the little-endian `uint32` pair
(height, width) followed by the pixels. The engine reads a file only when the image needs an encoder pass,
and it refuses pixels that do not match the digest. The engine never deletes the file; the caller does.
In batch mode, a file that fails its check after the open ends only its request: the `prefill` command returns
`{"event":"error","request_id":...,"slot_generation":...,"error":...,"recoverable":true}`, and the slot is free.
Each image needs (height/32) x (width/32) image pad tokens (`248056`) at its offset.
In batch mode, the `opened` event of an image request has `vision_slices`: the next that many `prefill` commands
each run one encoder slice (about 0.1 s or less), so other sessions can decode between them.
Only stdout carries protocol events.
The engine writes diagnostics to stderr.

Both modes use exact target verification.
Positive temperature applies top-k, top-p, and min-p in that order, then normalizes the probabilities.
Top-p retains the token that crosses its probability limit.
Min-p compares each probability with the largest probability.
Temperature zero ignores these filters and preserves greedy output.
Sampling uses a keyed draw: each draw depends only on the seed, the absolute token position, and the token ID.
Repeated identical requests with an explicit seed return identical token IDs.
Serial, speculative, and packed batch decoding give the same sampled token IDs for the same seed
(`engine/runtime/sampling.h`, keyed sampling law).

The CLI also supports optional draft policies and the same sampling parameters.
Use `--temperature`, `--top-k`, `--top-p`, `--min-p`, and `--seed`.
Positive-temperature CLI sampling does not support `--draft-overlap 1`.
The resident API uses exact target verification with the adaptive 64K proposal head by default.
It uses no lookup or asynchronous overlap.
Use `draft_head: "full"` to restore the full proposal vocabulary.
It emits verified IDs, never draft proposals.

## Tests

Run the CPU tests:

```bash
CRUCIBLE_TEST_CKPT="$CKPT" make test-cpu
```

Run these supplemental device tests only with exclusive GPU access.
On a shared GPU, start each command through `tools/gpu-run` with a 300-second timeout.

```bash
python -m server.tests.live_server_test "$CKPT"
python -m server.tests.live_tools_test "$CKPT"
```

These acceptance tests do not replace shared gate evidence.

It checks a real tool call, its result, streaming deltas, and exact greedy gen IDs.
It also checks explicit replay with the reported random seed.
It stops the temporary server and resident engine.

The original server test uses curl and stops its server in a cleanup block.
It checks exact dump files, exact output IDs, streaming, usage, queueing, and stop IDs.
It records readiness, time to first token, and decode speed under `results/`.

## Charter

- **Interface:** the local HTTP API (OpenAI and Anthropic shapes), the chat
  template, tokenization and tool-call checks. `EngineBackend` and
  `BatchedEngineBackend` talk to the engine's JSON-line protocol
  (`engine/serve/serve_protocol.h`, `engine/serve/batched_serve_protocol.h`).
- **Invariants:** binds only to `127.0.0.1`. A request's output depends only
  on the request. The serial backend is behind `--serial` in train 19 (rollback)
  and retires in train 20 with the serial engine serve.
- **Allowed dependencies:** `requirements-cpu.txt` and the engine binary.
  Model metadata comes from the engine's `ready` event, not from new
  literals.
- **Switches:** server flags. Engine switches that the server passes come
  from the registry.
- **Lever ledger (closed):** none recorded yet.
