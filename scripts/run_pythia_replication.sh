#!/usr/bin/env bash
# Cross-model replication on the pilot dataset with EleutherAI/pythia-410m.
#
# This is not just "try a second model". The vocabulary sanity check made a
# specific, falsifiable prediction: pythia-410m fails the `matte` minimal pairs
# (0.40, below chance) while gpt2 passes them (0.80), and both handle the
# colour terms at 1.00. If that check measures anything real, pythia should
# lose ground to gpt2 on the *finish* targets specifically, and not on colour.
#
# If instead pythia matches gpt2 on finish, the vocabulary check is not
# predictive and the model-selection argument in docs/DESIGN.md is weakened.
# Either outcome is worth having.
set -u
cd "$(dirname "$0")/.." || exit 1
LOG=results/pythia_replication.log
: > "$LOG"
say() { echo "=== $* ===" | tee -a "$LOG"; }
run() { echo "--- $*" >> "$LOG"; "$@" >> "$LOG" 2>&1; echo "exit=$?" >> "$LOG"; }

until grep -q "LENGTH CONTROL COMPLETE" results/length_control.log 2>/dev/null; do
    sleep 60
done
say "length control finished; starting replication"

say "extract pilot activations with pythia-410m"
run python -u scripts/03_extract.py --dataset pilot --model EleutherAI/pythia-410m \
    --conditions natural shuffled

say "probe"
run python -u scripts/04_probe.py --dataset pilot --model EleutherAI/pythia-410m \
    --layer-stride 2 --skip-mdl --max-train 6000 --max-test 3000 --mlp-hidden 64

say "MDL with regularisation sweep"
run python -u scripts/11_refresh_mdl.py --dataset pilot --model EleutherAI/pythia-410m

say "state persistence"
run python -u scripts/10_persistence.py --dataset pilot --model EleutherAI/pythia-410m

say "geometry"
run python -u scripts/05_geometry.py --dataset pilot --model EleutherAI/pythia-410m \
    --max-samples 6000 --n-trajectories 6

say "causal intervention (finish)"
run python -u scripts/06_intervene.py --dataset pilot --model EleutherAI/pythia-410m \
    --entity lips --other-entity eyes --attribute finish --trials 120
say "causal intervention (color)"
run python -u scripts/06_intervene.py --dataset pilot --model EleutherAI/pythia-410m \
    --entity lips --other-entity eyes --attribute color --trials 120

say "intervention statistics"
run python -u scripts/12_intervention_stats.py --dataset pilot \
    --model EleutherAI/pythia-410m

say "figures"
run python -u scripts/08_figures.py --dataset pilot --model EleutherAI/pythia-410m

say "report"
run python -u scripts/09_report.py --dataset pilot --model EleutherAI/pythia-410m

say "REPLICATION COMPLETE"
