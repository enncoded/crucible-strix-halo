# Agent conversation fixtures

These are scripted cache-performance fixtures, not model-generated conversations or a quality benchmark.
Each case has three request turns and a fixed history of user, assistant, and tool messages.
The system prompt includes tool definitions through the checkpoint's real Jinja template.
The code context contains deterministic cache tests.

The final request sizes are near 4K, 16K, and 30K tokens.
Each later request adds about 1,500 tokens.
Whole code records are preserved, so lengths need not equal the target exactly.

For each turn:

- `turn-N.json` contains raw messages, tools, and the thinking setting.
- `turn-N.ids.txt` contains the exact complete request IDs.
- `manifest.json` records both hashes, the token count, and the common prefix with the preceding request.
- `cache_prefix_tokens` marks the stable boundary before the assistant generation suffix.

The generator obtains that boundary with `add_generation_prompt=False`.
It asserts that those IDs form an exact prefix of the full generation request.
It uses `enable_thinking=False` and the checkpoint's own tool-call formatting.
The current HTTP chat API rejects tool messages. Use the direct engine protocol for these IDs.

The requested state-cache protocol is:

```json
{"ids":[1,2,3],"max_tokens":1,"spec":0,"stop_ids":[],"cache_prefix_tokens":2,"cache_control":"auto"}
```

Use `bypass` instead of `auto` for uncached comparisons.
Use `refresh` to write a fresh cache without reuse.
These cache fields require the statecache branch; the old serve parser rejects them.

Later histories use scripted assistant answers, not the IDs generated during timing.
The runner must record the actual cached prefix and misses. It must not claim every suffix is a cache hit.
Changing the template, thinking setting, or tools can change the exact prefix.

Regenerate on the test machine without a GPU.

`$WORK_DIR` and the `WORK_` variables come from `tools/work_paths.sh`.
Run `. tools/work_paths.sh` in a checkout before the commands.

```bash
$WORK_REFENV_PYTHON bench/suite/conversation_generate.py "$CKPT"
```
