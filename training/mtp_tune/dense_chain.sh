#!/bin/bash
# Dense pilot chain on the test machine, one queued lease at a time (plan steps 4 and 5):
# LEASES dense training leases from INIT, then requantization (native K and K6), then one evaluation lease.
#   dense_chain.sh RUN INIT LEASES [WAIT_PID]
# Expected GPU time: about 850 s per training lease, 600 s requant, 400 s evaluation.
set -u
cd "$(dirname "$0")/../.."
. training/mtp_tune/env.sh
run=$1 init=$2 leases=$3
[ -n "${4:-}" ] && while kill -0 "$4" 2>/dev/null; do sleep 30; done
R=$DATA/runs/$run
mkdir -p "$R"
for i in $(seq 1 "$leases"); do
  SECONDS_BUDGET=780 ~/crucible-work/bin/gpu-run -p 2 -w 14400 -- timeout -k 10 900 \
    training/mtp_tune/train_dense.sh "$run" "$init" > "$R/lease-$(date +%H%M%S).log" 2>&1
done
onp=$(ls -d $DATA/*/stk-gen-train-*)
cat > "$R/requant.sh" <<EOS
#!/bin/bash
cd $PWD && . training/mtp_tune/env.sh
\$PY -m training.mtp_tune.requant "\$CK305" "$R/state.pt" "$R/side-nat.safetensors" --data $onp
\$PY -m training.mtp_tune.requant "\$CK305" "$R/state.pt" "$R/side-k6.safetensors" --data $onp --k6 fe,fh,wq,wk,wv,wo
EOS
chmod +x "$R/requant.sh"
~/crucible-work/bin/gpu-run -p 2 -w 14400 -- timeout -k 10 900 "$R/requant.sh" > "$R/requant.log" 2>&1
held="$DATA/genh/stk-gen-heldout-000 $DATA/gen1/stk-gen-heldout-000"
cat > "$R/evals.sh" <<EOS
#!/bin/bash
cd $PWD && . training/mtp_tune/env.sh
E="\$PY -m training.mtp_tune.train \$CK305 $R/evals --heldout $held --eval-only --eval-depth 7"
\$E --stock
\$E --sidecar $DATA/runs/p1/side-508.safetensors
\$PY -m training.mtp_tune.train \$CK305 $R --scope all --heldout $held --eval-only --eval-depth 7
for s in nat k6; do [ -f $R/side-\$s.safetensors ] && \$E --sidecar $R/side-\$s.safetensors; done
EOS
chmod +x "$R/evals.sh"
~/crucible-work/bin/gpu-run -p 2 -w 14400 -- timeout -k 10 900 "$R/evals.sh" > "$R/evals.log" 2>&1
echo done > "$R/chain.done"
