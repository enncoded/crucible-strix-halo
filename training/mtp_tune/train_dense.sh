#!/bin/bash
# One dense-scope training lease (oracle-mtp recipe): public windows plus on-policy windows, loss on
# assistant rows, top-1 term, reach mask, level decay, two windows per update. Evaluates on the
# on-policy held-out sets (genh and gen1, held out by prompt).
#   train_dense.sh RUN INIT_STATE [train.py options]
set -u
cd "$(dirname "$0")/../.."
. training/mtp_tune/env.sh
run=$1 init=$2
shift 2
data=$(ls -d $DATA/d1/stk-train-*)
onp=$(ls -d $DATA/*/stk-gen-train-*)
held="$DATA/genh/stk-gen-heldout-000 $DATA/gen1/stk-gen-heldout-000"
$PY -m training.mtp_tune.train "$CK305" "$DATA/runs/$run" --scope all --init "$init" --data $data \
  --onpolicy $onp --onpolicy-frac 0.5 --mask --top1 0.4 --reach 3 --decay 0.8 --lr 1e-4 --lr-dense 2e-5 \
  --warmup 100 --total 400 --accum 2 --heldout $held --eval --eval-depth 7 --seconds ${SECONDS_BUDGET:-700} "$@"
step=$($PY -c "import torch,sys; print(torch.load(sys.argv[1], weights_only=False, map_location='cpu')['step'])" "$DATA/runs/$run/state.pt" 2>/dev/null)
[ -n "$step" ] && cp "$DATA/runs/$run/state.pt" "$DATA/runs/$run/state-$step.pt"
