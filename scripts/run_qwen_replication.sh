#!/usr/bin/env bash
# Third model: Qwen/Qwen3-0.6B-Base, 28 layers, hidden 1024.
#
# Two models established that the pattern of which variables beat a lexical
# baseline does not transfer. A third from a different family and a much more
# recent training mix tests whether the one result that did replicate, the
# insensitivity of the causal effect to entity identity, holds more widely.
#
# The decomposition experiment runs here too, so the entity-specific component
# is tested on all three models rather than on gpt2 alone.
set -u
cd "$(dirname "$0")/.." || exit 1
LOG=results/qwen_replication.log
: > "$LOG"
say() { echo "=== $* ===" | tee -a "$LOG"; }
run() { echo "--- $*" >> "$LOG"; "$@" >> "$LOG" 2>&1; echo "exit=$?" >> "$LOG"; }

MODEL=Qwen/Qwen3-0.6B-Base

# wait for the gpt2 decomposition to free the CPU
until ls results/decompose_pilot_gpt2_lips.finish_L*.json >/dev/null 2>&1; do
    sleep 20
done
say "gpt2 decomposition finished, starting Qwen3"

say "vocabulary check"
run python -u scripts/01_vocab_and_benchmark.py --models "$MODEL" --vocab-only \
    --out results/vocab_check_qwen.json

say "extract pilot activations"
run python -u scripts/03_extract.py --dataset pilot --model "$MODEL" \
    --conditions natural shuffled

say "probe"
run python -u scripts/04_probe.py --dataset pilot --model "$MODEL" \
    --layer-stride 2 --skip-mdl --max-train 6000 --max-test 3000 --mlp-hidden 64

say "significance against every baseline"
run python -u scripts/17_significance.py --dataset pilot --model "$MODEL" \
    --seeds 1 --max-train 6000 --n-boot 4000

say "binding by distance"
run python -u scripts/15_binding_by_distance.py --dataset pilot --model "$MODEL" \
    --n-boot 1000

say "direction decomposition, shared against differential"
run python -u scripts/22_decompose_direction.py --dataset pilot --model "$MODEL" \
    --entity lips --other-entity eyes --attribute finish --trials 100

say "intervention with random and wrong-region controls"
run python -u scripts/16_intervene_v2.py --dataset pilot --model "$MODEL" \
    --entity lips --other-entity eyes --attribute finish \
    --trials-per-bin 60 --alphas 0 2 8

say "report"
run python -u scripts/09_report.py --dataset pilot --model "$MODEL"

say "QWEN REPLICATION COMPLETE"
