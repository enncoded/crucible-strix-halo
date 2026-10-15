#!/bin/bash
# One dump lease after generation: turn spec_sweep logs into LIST shards and dump their stacks.
#   postgen.sh GENSET HELDOUT_FRACTION [GENSET HELDOUT_FRACTION ...]
set -u
cd "$(dirname "$0")/../.."
. training/mtp_tune/env.sh
while [ $# -ge 2 ]; do
  set_=$1 frac=$2; shift 2
  ~/crucible-work/mtp-data/.venv/bin/python -m training.mtp_tune.gen_to_list "$DATA/$set_" --heldout "$frac"
  for split in train heldout; do
    list="$DATA/$set_/gen-$split-000.list"
    [ -s "$list" ] || continue
    [ -d "$DATA/$set_/stk-gen-$split-000" ] && continue
    mkdir -p "$DATA/$set_/stk-gen-$split-000"
    build/bench/mtp_dump "$CK305" "$list" "$DATA/$set_/stk-gen-$split-000" --ctx 4096 2>&1 | grep -v "^prefill plan" | tail -2
  done
done
