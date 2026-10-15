#!/usr/bin/env bash
# Run this script through gpu-run. Each process releases its model before the next process starts.
set -euo pipefail
. "$(dirname "$0")/../tools/work_paths.sh"
cd "$(dirname "$0")/.."
checkpoint=${1:?checkpoint required}
tag=${2:?unique tag required}
output="results/extended-$tag"
engine=${BATCH_ENGINE:-build/engine/crucible}
sparse=${BATCH_SPARSE_GATE:-build/bench/batch_sparse_gate}
mkdir -p "$output"
"$sparse" "$checkpoint" tests/data/draft/prose_science.ids tests/data/draft/chat_dinner.ids \
  > "$output/sparse.log" 2>&1
"$engine" gen "$checkpoint" --ids-file "$WORK_FIB_IDS" \
  --streams 8 --n 4 --ctx 32768 > "$output/streams.log" 2>&1
python3 - "$output/streams.log" "$output/prompts.ids" <<'PY'
import pathlib
import sys
lines = pathlib.Path(sys.argv[1]).read_text().splitlines()
streams = [line.split("ids:", 1)[1].split() for line in lines if line.startswith("stream ")]
assert len(streams) == 8 and all(len(row) == 4 for row in streams)
assert all(row == streams[0] for row in streams)
paths = ["tests/data/draft/code_json.ids", "tests/data/draft/prose_science.ids"]
pathlib.Path(sys.argv[2]).write_text("\n".join(pathlib.Path(p).read_text().strip() for p in paths) + "\n")
print("CLI streams gate: eight identical requests produce identical IDs")
PY
"$engine" gen "$checkpoint" --batch-list "$output/prompts.ids" --n 4 --ctx 4096 \
  > "$output/list.log" 2>&1
python3 - "$output/list.log" <<'PY'
import pathlib
import sys
lines = pathlib.Path(sys.argv[1]).read_text().splitlines()
streams = [line.split("ids:", 1)[1].split() for line in lines if line.startswith("stream ")]
assert len(streams) == 2 and all(len(row) == 4 for row in streams)
print("CLI batch-list gate: two different requests complete")
PY
grep 'sparse_batch_gate' "$output/sparse.log"
grep '^batch ' "$output/streams.log" "$output/list.log"
