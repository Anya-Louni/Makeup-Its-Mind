#!/usr/bin/env bash
# Runs phases 3-6 on the pilot once probing has produced its results file,
# then re-extracts with mean pooling and re-probes for the fair comparison.
set -u
cd "$(dirname "$0")/.." || exit 1
LOG=results/pilot_rest.log
: > "$LOG"

say() { echo "=== $* ===" | tee -a "$LOG"; }

until [ -f results/probe_pilot_gpt2.json ]; do sleep 15; done
say "probing finished"

say "phase 3: geometry"
python -u scripts/05_geometry.py --dataset pilot --model gpt2 \
    --max-samples 6000 --n-trajectories 12 >> "$LOG" 2>&1
echo "geometry exit=$?" >> "$LOG"

say "phase 5: capacity ablation"
python -u scripts/07_capacity_ablation.py --dataset pilot --model gpt2 \
    --capacities 8 32 128 512 --max-train 4000 --max-test 2000 >> "$LOG" 2>&1
echo "capacity exit=$?" >> "$LOG"

say "phase 4: causal intervention (lips.finish)"
python -u scripts/06_intervene.py --dataset pilot --model gpt2 \
    --entity lips --other-entity eyes --attribute finish --trials 120 \
    >> "$LOG" 2>&1
echo "intervene finish exit=$?" >> "$LOG"

say "phase 4: causal intervention (lips.color)"
python -u scripts/06_intervene.py --dataset pilot --model gpt2 \
    --entity lips --other-entity eyes --attribute color --trials 120 \
    >> "$LOG" 2>&1
echo "intervene color exit=$?" >> "$LOG"

say "phase 6: figures"
python -u scripts/08_figures.py --dataset pilot --model gpt2 >> "$LOG" 2>&1
echo "figures exit=$?" >> "$LOG"

say "report"
python -u scripts/09_report.py --dataset pilot --model gpt2 >> "$LOG" 2>&1
echo "report exit=$?" >> "$LOG"

say "re-extract pilot with mean pooling stored"
python -u scripts/03_extract.py --dataset pilot --model gpt2 \
    --conditions natural shuffled >> "$LOG" 2>&1
echo "reextract exit=$?" >> "$LOG"

say "probe with mean pooling (fair comparison with the pooled baseline)"
python -u scripts/04_probe.py --dataset pilot --model gpt2 --pooling mean \
    --max-train 6000 --max-test 3000 --mlp-hidden 64 >> "$LOG" 2>&1
echo "probe mean exit=$?" >> "$LOG"

say "ALL PILOT PHASES COMPLETE"
