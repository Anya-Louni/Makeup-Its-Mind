# Do LLMs hold an implicit world model of a makeup routine?

Testing whether a language model's internal activations track the true state of
several face regions — their colour, finish and coverage — as a makeup
application narrative unfolds.

Most world-model probing work tracks a single structure (an Othello board) or a
single scalar. A makeup routine forces **several independent entities to be
tracked at once**, each with its own attributes, updated in an order-dependent
sequence. That makes it a test of *binding* — which attribute belongs to which
region — not just of presence.

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
most of which never use the literal label word — `matte` appears as *"a
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
prequential MDL, and probe-capacity ablation — plus a TF-IDF bag-of-ngrams
baseline on the raw prefix text (a stronger lexical competitor than static
embeddings) and the override stratification above.

**Causal intervention with three controls.** The intervention adds a concept
direction to the residual stream and reads out the model's own preference over
the state words. Three controls run alongside, and the third is the one that
mattered:

1. the other face region's read-out, measured in the same forward pass;
2. a **random direction** at matched norm — does any push of this size move the
   read-out? (No, at alpha = 2.);
3. the **other region's concept direction** at matched norm — does the edit
   need to be *this* region's vector? (No — and that is why we do not claim the
   intervention demonstrates binding. See `docs/CORRECTIONS.md` #10.)

Steering strength matters: at alpha = 8 even random directions become
significant, so alpha = 2 is the interpretable dose and all causal claims use
it.

## What we found

Full numbers in `docs/RESULTS_*.md`; every claim that was revised along the way
is in `docs/CORRECTIONS.md` (eleven entries, including four of our own
overclaims caught by controls we added later).

**The canonical write-up is `docs/FINDINGS.md`** — abstract, central claim, and
the cross-model table of what replicates. Summary below.

### The headline, stated as narrowly as the evidence supports

> In small language models, an attribute's *value* is linearly decodable,
> causally load-bearing, and **shared across entities**; the *binding* of that
> value to a particular entity is linearly decodable and decays with narrative
> distance, but we find no evidence the model uses it. Which specific variables
> clear a lexical baseline is a property of the model, not of the task — no
> target is significant in the same direction in both models tested.

### Supported

- **A real causal effect on the attribute *value*, in both models.** The
  concept direction shifts the model's own log-odds of the steered value —
  gpt2 +0.055 [+0.041, +0.070], pythia +0.147 [+0.123, +0.171] at alpha = 2 —
  while a random direction of matched norm does nothing (gpt2 +0.014 n.s.,
  pythia -0.009 n.s.). It is **not** entity-specific: the other region's
  direction works as well or better, in both models.
- **Binding is probe-recoverable, in both models** (+0.04 to +0.12 at short
  distance), and on the 4-entity gpt2 set it **decays with distance**: +0.113
  at distance 0 to -0.124 by distance >= 5. 33/36 region pairs survive
  Benjamini-Hochberg correction across all 180 tests at distance 0, 29/36 at
  distance >= 5. Past ~5 sentences the representation still carries "this value
  occurred somewhere" but has lost which region.
- **Entity count, not narrative length, breaks the colour code.** A control
  dataset with 2 entities at 4-entity narrative length keeps colour
  selectivity (+0.119, best layer 2, contextual); 4 entities at the same
  length collapses it (+0.016, best layer 0, embeddings).
- **The signal is linear.** Flat across a 64x range of probe capacity on every
  target tested.
- **Four independent measures agree** on which targets use the transformer:
  best-layer depth, prequential MDL (r = +0.92 with depth), shuffled-order
  sensitivity, and selectivity.
- **`eyes.finish` beats every lexical baseline in gpt2**: +0.097
  [+0.048, +0.148], two-sided p < 0.0001. It does **not** replicate in pythia
  (+0.014, p = 0.639) — see the next section.

### Does not replicate across models

The pattern of which state variables beat a lexical baseline is model-specific.
No target is significant in the same direction in both:

| target | gpt2 gap | p2 | pythia gap | p2 |
|---|---|---|---|---|
| `eyes.color` | +0.062 | **0.008** | +0.031 | 0.232 |
| `eyes.finish` | +0.097 | **0.000** | +0.014 | 0.639 |
| `lips.color` | -0.044 | 0.126 | +0.064 | **0.028** |
| `lips.finish` | -0.058 | **0.029** | +0.024 | 0.415 |

The two models also place their state code very differently: gpt2 across
embeddings and middle layers, pythia almost entirely at the output end.

### Not supported, and reported as such

- **The intervention does not demonstrate *binding*, in either model.** The
  other region's finish direction moves the read-out as well as the correct one
  (gpt2 +0.149 vs +0.148; pythia +0.195 vs +0.147). The geometry predicts this:
  those direction vectors have cosine +0.67 to +0.95. The experiment shows a
  usable attribute-value code, not a bound one.
- **Colour is not causally load-bearing.** A pre-registered prediction that
  colour would show a causal effect at short distance failed: no effect at
  either distance, for any arm.
- **Most targets do not beat a lexical baseline.** 8 of 12 full-dataset
  targets peak at the embedding layer; `lips.finish` is *significantly worse*
  than the static-embedding baseline (two-sided p = 0.0285).
- **MDL favours the static baseline throughout**, for a documented reason we
  cannot remove: prequential coding runs on a pool where wordings are shared.
- **`lips.coverage` intervention reversed** (-0.367 on the target). Diagnosed:
  the `bare` class sits at mean narrative position 3.65 against 7.5-7.7 for
  light/full, so its difference-of-means direction encodes "early in the text"
  rather than "no product". `bare` has been removed from the coverage read-out
  (matching the treatment of `none` for finish) and the old number is
  withdrawn. See `docs/CORRECTIONS.md` #13.

### Conventions

p-values are **two-sided** everywhere in the reports; one-sided values are kept
in the JSON as `p_one_sided_greater`. Confidence intervals on anything measured
over narrative positions use a **cluster bootstrap resampling whole
narratives**, since positions within a narrative are not independent.

## Hardware note

The reference machine is CPU-only (12 cores, ~15 GB RAM), which is why the
candidate set is 0.1–0.5 B models and why `gpt2` is the working default. See
`docs/model_benchmark.md`.
