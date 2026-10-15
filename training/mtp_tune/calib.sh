#!/bin/bash
# One calibration lease: margins and the Q-rule replay, stock head against a sidecar or run state,
# on the on-policy held-out sets (assistant rows) and the public held-out set.
#   calib.sh OUTDIR (--sidecar F | --state F)
set -u
cd "$(dirname "$0")/../.."
. training/mtp_tune/env.sh
out=$1; shift
mkdir -p "$out"
gen="$DATA/genh/stk-gen-heldout-000 $DATA/gen1/stk-gen-heldout-000"
for head in stock tuned; do
  extra=(); [ $head = tuned ] && extra=("$@")
  $PY -m training.mtp_tune.calib "$CK305" --heldout $gen --asst "${extra[@]}" --out "$out/gen-$head.json" | cut -c1-600
  $PY -m training.mtp_tune.calib "$CK305" --heldout $DATA/d1/stk-heldout-000 "${extra[@]}" --out "$out/pub-$head.json" | cut -c1-600
done
