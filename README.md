# Do LLMs hold an implicit world model of a makeup routine?

Testing whether a language model's internal activations track the true state of
several face regions. Colour, finish and coverage, tracked as a makeup
application narrative unfolds.

**Read the write-up: <https://anya-louni.github.io/Makeup-Its-Mind/>**

Most world-model probing work tracks a single structure (an Othello board) or a
single scalar. A makeup routine forces **several independent entities to be
tracked at once**, each with its own attributes, updated in an order-dependent
sequence. That makes it a test of binding. The question is which attribute
belongs to which region.

## Research question

As a model reads a makeup routine sentence by sentence, do its hidden states
linearly encode the true state of each face region, in a way that is

1. better than a non-contextual baseline **and** better than a bag-of-ngrams
   model over the same text,
2. causally load-bearing for the model's own downstream predictions, and
3. not an artifact of probe capacity?

## Layout

```
src/mwm/
  state.py      explicit state machine: 4 entities x 3 attributes
  lexicon.py    paraphrase banks + sentence skeletons, split-partitioned
  generate.py   narrative generation with exact per-position labels
  models.py     model loading, hidden-state extraction, scoring
  extract.py    activation extraction over a dataset
  probe.py      linear/MLP probes, control task, MDL, ngram baseline
  geometry.py   CAVs, subspace angles, dimensionality, projections
  intervene.py  activation patching + behavioural read-outs
scripts/
  01_vocab_and_benchmark.py   vocabulary sanity check + model benchmark
  02_generate_dataset.py      dataset generation
  03_extract.py               activation extraction
  04_probe.py                 probing + all baseline controls
  05_geometry.py              representation geometry
  06_intervene.py             causal intervention
  07_capacity_ablation.py     probe-capacity + MDL critique
  08_figures.py               figures, incl. animated 3D trajectory
  09_report.py                collect every result into one markdown report
  10_persistence.py           accuracy vs distance since last state change
  11_refresh_mdl.py           MDL with a regularisation sweep
  12_intervention_stats.py    bootstrap CIs + both p-value conventions
  13_add_shuffled.py          add the shuffled comparison to an existing run
  14_order_sensitivity.py     does the shuffled control agree with depth?
  15_binding_by_distance.py   binding margin stratified by distance
  16_intervene_v2.py          NEAR/FAR + random + wrong-direction controls
  17_significance.py          cluster-bootstrap gap vs every baseline
  18_binding_map.py           animated face diagram: true state vs probe
  19_multiple_comparisons.py  Benjamini-Hochberg across the whole test grid
  20_crossmodel_figures.py    cross-model comparison figures
  21_dual_binding_map.py      two-region face diagram, several models at once
  22_decompose_direction.py   shared vs region-specific steering, orthogonality
  23_readout_validity.py      does the read-out answer about the named region?
  24_readout_mechanism.py     what predicts the read-out if the region does not
  25_build_paper.py           assemble docs/index.html from its three sources
```

## Running it

```bash
python scripts/01_vocab_and_benchmark.py
python scripts/02_generate_dataset.py --name pilot --entities lips eyes --attributes color finish --n 1000
python scripts/03_extract.py  --dataset pilot --model gpt2
python scripts/04_probe.py    --dataset pilot --model gpt2
python scripts/05_geometry.py --dataset pilot --model gpt2
python scripts/06_intervene.py --dataset pilot --model gpt2 --attribute finish
python scripts/07_capacity_ablation.py --dataset pilot --model gpt2
python scripts/08_figures.py  --dataset pilot --model gpt2
```

Scale to the full setting with
`python scripts/02_generate_dataset.py --name full --n 4000` (4 entities,
3 attributes) and the same commands with `--dataset full`.

## Design decisions that make the result defensible

**Paraphrase variation.** Every attribute value has ten distinct realisations,
most of which avoid the literal label word. `matte` appears as *"a
chalk-flat surface"*, *"a dead-flat look that caught no light"*, *"a fully
dried-down, non-wet look"*. A probe that succeeds by spotting the word `matte`
cannot succeed here.

**No surface overlap between splits.** Phrase banks *and* sentence skeletons
are partitioned 6/2/2 by index. A test narrative is built entirely from
wordings the probe has never seen. This tests generalisation to unseen
phrasings, not memorisation of templates.

**Distractor sentences.** Some sentences mention a colour without applying it
(*"She held a deep crimson up to the light and shook her head"*). A
bag-of-words model is actively misled by these; a state tracker is not.

**Entity-agnostic colour wording.** The same colour phrases are used for every
region, so "which region is this colour on" cannot be read off the wording.

**Order-sensitive positions are reported separately.** Roughly a quarter of
positions are ones where a value was overridden earlier in the narrative, so
the first mention is the *wrong* answer. Accuracy on that subset is the
sharpest evidence of state tracking rather than keyword retrieval.

**Every control from the spec, plus two more.** Static embedding baseline,
shuffled-order condition, Hewitt & Liang control task and selectivity,
prequential MDL, and probe-capacity ablation. We add a TF-IDF bag-of-ngrams
baseline on the raw prefix text and the override stratification above.

**Causal intervention with three controls.** The intervention adds a concept
direction to the residual stream and reads out the model's own preference over
the state words. Three controls run alongside, and the third is the one that
mattered:

1. the other face region's read-out, measured in the same forward pass;
2. a **random direction** at matched norm. It moves nothing at g = 2;
3. the **other region's concept direction** at matched norm. It works as well
   as the correct one. See `docs/CORRECTIONS.md` #10 and #14.

Steering strength matters. At g = 8 even a random direction of matched norm
reaches significance, so g = 2 is the interpretable dose and all causal claims
use it. The gain is written g to keep it apart from the significance level.

## What we found

Full numbers in `docs/FINDINGS.md`. Every revised claim is in
`docs/CORRECTIONS.md`, twenty four entries.

A concept direction built by differencing class means holds two parts. One is
shared by every entity that takes the value. The other is specific to one
entity. We separate them and steer with each alone, across four models.

| model | shared effect | region specific difference | p |
|---|---|---|---|
| gpt2-124M | +0.183 | +0.016 [+0.009, +0.023] | <0.0001 |
| pythia-410m | +0.190 | 0.000 [-0.006, +0.005] | 0.85 |
| Qwen3-0.6B-Base | +0.317 | 0.000 [-0.010, +0.010] | 0.91 |
| Qwen3-1.7B-Base | +0.556 | +0.007 [-0.008, +0.022] | 0.36 |

The shared value component triples in causal strength from 124M to 1.7B. A
random direction of matched norm does nothing anywhere.

The two parts are orthogonal only when the two class directions have equal
norms, since `s . r = (|d1|^2 - |d2|^2) / 4`. They do not, quite, so we also
steer `r` with `s` projected out. The leak is negative in every model, so it
suppresses the difference rather than inflating it.

| model | raw `r` | `r` with `s` removed | p |
|---|---|---|---|
| gpt2-124M | +0.016 | +0.013 [+0.007, +0.020] | <0.0001 |
| pythia-410m | -0.000 | +0.002 [-0.004, +0.007] | 0.58 |
| Qwen3-0.6B | -0.000 | +0.002 [-0.008, +0.013] | 0.65 |
| Qwen3-1.7B | +0.007 | +0.024 [+0.005, +0.042] | 0.015 |

The gpt2 effect survives the projection, so it is not an artifact of the norm
mismatch. At 1.7B the projection turns a null into a dose responsive effect,
since the leak had been cancelling it. Neither 1.7B result survives
Benjamini-Hochberg across the forty test grid, and the cleaned component raises
both read outs rather than separating them, so we report it as a small uneven
push and not as binding.

The behavioural read out answers a question about one region using the other
region's value. Mean sensitivity to leakage ratio is 1.07 in the three smaller
models. At 1.7B one of the two read outs reaches 2.00, so it begins to bind.
Put through that read out, the region specific component raises the region it
should lower, by +0.110 against +0.134 on the region it was steered toward.

The representation improves with scale. Qwen3-1.7B is the first model where all
four state variables clear the lexical baseline, and its binding margins are the
highest measured.

## Hardware note

The reference machine is CPU-only (12 cores, ~15 GB RAM), which is why the
candidate set is 0.1–0.5 B models and why `gpt2` is the working default. See
`docs/model_benchmark.md`.
