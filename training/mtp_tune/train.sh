#!/bin/bash
# One training lease: resume RUN, train until the time budget, save, evaluate on held-out data.
# The first lease of a run also evaluates the stock head.
#   train.sh RUN SCOPE [train.py options]
set -u
cd "$(dirname "$0")/../.."
. training/mtp_tune/env.sh
run=$1 scope=$2
shift 2
data=$(ls -d $DATA/*/stk-train-* $DATA/*/stk-gen-train-* 2>/dev/null)
held=$DATA/d1/stk-heldout-000
budget=${SECONDS_BUDGET:-740}
if [ ! -f "$DATA/runs/$run/state.pt" ] && [ ! -f "$DATA/runs/$run/eval.jsonl" ]; then
  $PY -m training.mtp_tune.train "$CK305" "$DATA/runs/$run" --scope "$scope" --heldout $held --eval-only --stock "$@"
  budget=$((budget - 120))
fi
$PY -m training.mtp_tune.train "$CK305" "$DATA/runs/$run" --scope "$scope" --data $data \
  --heldout $held --eval --seconds $budget "$@"
# Keep a copy per lease, so later exports can pick any step.
step=$($PY -c "import torch,sys; print(torch.load(sys.argv[1], weights_only=False, map_location='cpu')['step'])" "$DATA/runs/$run/state.pt" 2>/dev/null)
[ -n "$step" ] && cp "$DATA/runs/$run/state.pt" "$DATA/runs/$run/state-$step.pt"
