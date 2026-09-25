#!/usr/bin/env bash
# Every phase on the full 4-entity / 3-attribute dataset.
# Waits for the probing run and the shuffled extraction to finish first.
set -u
cd "$(dirname "$0")/.." || exit 1
LOG=results/full_all.log
: > "$LOG"
say() { echo "=== $* ===" | tee -a "$LOG"; }
run() { echo "--- $* " >> "$LOG"; "$@" >> "$LOG" 2>&1; echo "exit=$? ($1 $2 $3)" >> "$LOG"; }

until [ -f results/probe_full_gpt2.json ]; do sleep 30; done
say "probing finished"
until grep -q "done in" results/extract_full.log && \
      [ "$(grep -c 'done in' results/extract_full.log)" -ge 2 ]; do sleep 30; done
say "shuffled extraction finished"

say "add the shuffled-order comparison"
run python -u scripts/13_add_shuffled.py --dataset full --model gpt2

say "MDL with regularisation sweep"
run python -u scripts/11_refresh_mdl.py --dataset full --model gpt2 --max-train 8000

say "state persistence"
run python -u scripts/10_persistence.py --dataset full --model gpt2 --max-train 8000

# Geometry across depth: this is where the binding result lives.
for L in 1 6 11; do
    say "geometry at layer $L"
    run python -u scripts/05_geometry.py --dataset full --model gpt2 \
        --layer "$L" --max-samples 8000 --n-trajectories 8
done

# Capacity ablation on one target per attribute (the full 12 would take hours
# and the question -- is the gain capacity or structure -- is per-attribute).
say "capacity ablation"
run python -u scripts/07_capacity_ablation.py --dataset full --model gpt2 \
    --targets lips.color lips.finish lips.coverage eyes.color \
    --capacities 8 32 128 512 --max-train 6000 --max-test 3000

# Causal interventions. Every attribute, and two disjoint region pairs so the
# cross-entity control is not always the same two regions.
say "intervention lips.color (control: eyes)"
run python -u scripts/06_intervene.py --dataset full --model gpt2 \
    --entity lips --other-entity eyes --attribute color --trials 150
say "intervention lips.finish (control: eyes)"
run python -u scripts/06_intervene.py --dataset full --model gpt2 \
    --entity lips --other-entity eyes --attribute finish --trials 150
say "intervention lips.coverage (control: eyes)"
run python -u scripts/06_intervene.py --dataset full --model gpt2 \
    --entity lips --other-entity eyes --attribute coverage --trials 150
say "intervention cheeks.color (control: skin)"
run python -u scripts/06_intervene.py --dataset full --model gpt2 \
    --entity cheeks --other-entity skin --attribute color --trials 150
say "intervention skin.finish (control: cheeks)"
run python -u scripts/06_intervene.py --dataset full --model gpt2 \
    --entity skin --other-entity cheeks --attribute finish --trials 150

say "intervention statistics"
run python -u scripts/12_intervention_stats.py --dataset full --model gpt2

say "figures"
run python -u scripts/08_figures.py --dataset full --model gpt2

say "report"
run python -u scripts/09_report.py --dataset full --model gpt2

say "FULL DATASET COMPLETE"
