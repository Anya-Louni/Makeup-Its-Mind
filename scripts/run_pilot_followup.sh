#!/usr/bin/env bash
# Runs after run_pilot_rest.sh: corrects the stale MDL numbers, repeats the
# geometry at the layers that actually matter, and regenerates figures/report.
set -u
cd "$(dirname "$0")/.." || exit 1
LOG=results/pilot_followup.log
: > "$LOG"
say() { echo "=== $* ===" | tee -a "$LOG"; }

# wait for the main chain to finish
until grep -q "ALL PILOT PHASES COMPLETE" results/pilot_rest.log 2>/dev/null; do
    sleep 30
done
say "main chain finished"

say "recompute MDL with the converged probe (last-token)"
python -u scripts/11_refresh_mdl.py --dataset pilot --model gpt2 >> "$LOG" 2>&1
echo "mdl refresh exit=$?" >> "$LOG"

say "recompute MDL for the mean-pooled probe results"
python -u scripts/11_refresh_mdl.py --dataset pilot --model gpt2 \
    --pooling mean >> "$LOG" 2>&1
echo "mdl refresh mean exit=$?" >> "$LOG"

say "persistence with mean pooling"
python -u scripts/10_persistence.py --dataset pilot --model gpt2 \
    --pooling mean >> "$LOG" 2>&1
echo "persistence mean exit=$?" >> "$LOG"

# The default geometry run used the median of the four best layers (6), which
# is nobody's best. Repeat where the state is actually most decodable.
for L in 1 11; do
    say "geometry at layer $L"
    python -u scripts/05_geometry.py --dataset pilot --model gpt2 \
        --layer "$L" --max-samples 6000 --n-trajectories 6 >> "$LOG" 2>&1
    echo "geometry L$L exit=$?" >> "$LOG"
done

say "figures"
python -u scripts/08_figures.py --dataset pilot --model gpt2 >> "$LOG" 2>&1
echo "figures exit=$?" >> "$LOG"

say "report"
python -u scripts/09_report.py --dataset pilot --model gpt2 >> "$LOG" 2>&1
echo "report exit=$?" >> "$LOG"

say "FOLLOWUP COMPLETE"
