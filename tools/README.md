# tools: charter

The gate runner, the build records, the layout rules and the measurement tools.

## Interface

- `gates.py`: the gate runner. Run `python3 tools/gates.py --help` for the options.
  Do not change it, or its helpers, in a tree where a gate runs.
- `layout.py` and `layout_map.tsv`: the module map and the include rules.
- `work_paths.py` and `work_paths.sh`: the work layout of the tools. `WORK_DIR` sets the root
  (default `~/crucible-work`).
- `gpu-run`: a simple stand-in for a GPU queue. It takes one lock, so that two GPU jobs never overlap.
  Copy it to `$WORK_DIR/bin/gpu-run`, or put a real queue program there with the same command line.
- `tok.py`: tokenize a prompt with the checkpoint tokenizer.
- `h2h_decode.py`: the request harness for a comparison of HTTP servers.

## Invariants

- No copies. Extend the shared tool. Do not copy a tool into a new directory.
- A tool that hashes the engine sources finds them recursively, so that a layout move needs no edit in it.
- Synthetic content only. No prompt text or transcripts in the outputs.

## Allowed includes and imports

- The Python standard library and `requirements-cpu.txt`. A tool can read any file of the repository.

## Switches

- Tool options are flags. A tool that sets engine switches lists them in its help text and uses the
  registry names (`engine/switches.h`).
