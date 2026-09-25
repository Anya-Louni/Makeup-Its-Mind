#!/usr/bin/env bash
# Length control: is the colour collapse caused by entity count or text length?
#
# The pilot (2 entities, ~6.5 sentences) showed partial colour binding.
# The full set (4 entities, ~13 sentences) shows none: colour selectivity
# +0.019/-0.002/+0.011/+0.037 and cross-entity transfer equal to within-entity.
# Two things changed at once, so the headline claim -- that binding degrades
# with the number of competing entities -- is confounded with narrative length.
#
# This isolates it: 2 entities, but narrative length matched to the 4-entity
# set (10-16 sentences). Three outcomes, all informative:
#   colour works  -> entity count is the cause; the binding claim stands
#   colour fails  -> length/distance is the cause; the claim must be withdrawn
#   partial       -> both contribute, and the pilot-vs-full gap is not
#                    attributable to entity count alone
set -u
cd "$(dirname "$0")/.." || exit 1
LOG=results/length_control.log
: > "$LOG"
say() { echo "=== $* ===" | tee -a "$LOG"; }
run() { echo "--- $*" >> "$LOG"; "$@" >> "$LOG" 2>&1; echo "exit=$?" >> "$LOG"; }

until grep -q "FULL DATASET COMPLETE" results/full_all.log 2>/dev/null; do
    sleep 60
done
say "full-dataset chain finished; starting the length control"

# 2 entities, 3 attributes, but the 4-entity set's narrative length.
say "generate long2 (2 entities, 10-16 sentences)"
run python -u scripts/02_generate_dataset.py --name long2 \
    --entities lips eyes --attributes color finish coverage \
    --n 1800 --min-sentences 10 --max-sentences 16

say "extract"
run python -u scripts/03_extract.py --dataset long2 --model gpt2 \
    --conditions natural shuffled

say "probe"
run python -u scripts/04_probe.py --dataset long2 --model gpt2 \
    --layer-stride 2 --skip-mdl --max-train 8000 --max-test 4000 \
    --mlp-hidden 64

say "persistence"
run python -u scripts/10_persistence.py --dataset long2 --model gpt2 \
    --max-train 8000

say "MDL with regularisation sweep"
run python -u scripts/11_refresh_mdl.py --dataset long2 --model gpt2 \
    --max-train 8000

say "geometry at layer 11"
run python -u scripts/05_geometry.py --dataset long2 --model gpt2 \
    --layer 11 --max-samples 8000 --n-trajectories 6

say "figures"
run python -u scripts/08_figures.py --dataset long2 --model gpt2

say "report"
run python -u scripts/09_report.py --dataset long2 --model gpt2

say "LENGTH CONTROL COMPLETE"
