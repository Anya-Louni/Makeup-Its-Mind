#!/usr/bin/env bash
# Fourth model: Qwen/Qwen3-1.7B-Base, roughly 3x the largest model so far.
#
# Feng and Steinhardt find binding ID vectors in every sufficiently large model
# of the Pythia and LLaMA families. Our three models run 124M to 600M and the
# region specific component is inert in two of them. This arm asks whether that
# is a scale effect.
#
# Two things get measured here, not one. Whether the region specific component
# gains a causal effect, and whether the dominant region effect in the read out
# shrinks as the model grows.
set -u
cd "$(dirname "$0")/.." || exit 1
LOG=results/qwen17_scale.log
: > "$LOG"
say() { echo "=== $* ===" | tee -a "$LOG"; }
run() { echo "--- $*" >> "$LOG"; "$@" >> "$LOG" 2>&1; echo "exit=$?" >> "$LOG"; }

MODEL=Qwen/Qwen3-1.7B-Base

# wait for the mechanism runs to release the cores
until ls results/readout_mechanism_pilot_Qwen__Qwen3-0.6B-Base.json >/dev/null 2>&1; do
    sleep 30
done
say "mechanism runs finished, starting the scale arm"

say "vocabulary check"
run python -u scripts/01_vocab_and_benchmark.py --models "$MODEL" --vocab-only \
    --out results/vocab_check_qwen17.json

say "extract pilot activations"
run python -u scripts/03_extract.py --dataset pilot --model "$MODEL" \
    --conditions natural shuffled

say "probe"
run python -u scripts/04_probe.py --dataset pilot --model "$MODEL" \
    --layer-stride 3 --skip-mdl --max-train 6000 --max-test 3000 --mlp-hidden 64

say "significance against every baseline"
run python -u scripts/17_significance.py --dataset pilot --model "$MODEL" \
    --seeds 1 --max-train 6000 --n-boot 4000

say "binding by distance"
run python -u scripts/15_binding_by_distance.py --dataset pilot --model "$MODEL" \
    --n-boot 1000

say "read out validity"
run python -u scripts/23_readout_validity.py --dataset pilot --model "$MODEL" \
    --per-cell 60

say "read out mechanism, does the dominant region effect shrink with scale"
run python -u scripts/24_readout_mechanism.py --dataset pilot --model "$MODEL" \
    --trials 220

say "direction decomposition"
run python -u scripts/22_decompose_direction.py --dataset pilot --model "$MODEL" \
    --entity lips --other-entity eyes --attribute finish --trials 100

say "report"
run python -u scripts/09_report.py --dataset pilot --model "$MODEL"

say "SCALE ARM COMPLETE"
