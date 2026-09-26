# Corrections log

Every claim in this project that was stated and then revised, with what caused
the error and what fixed it. Kept because the sequence is part of the evidence:
several of these were caught only because a result looked too clean or too
convenient, and checking rather than accepting changed the conclusion.

Ordered roughly as they happened. The final, canonical statement of what the
project shows is `FINDINGS.md`; this file is the audit trail behind it.

Four of the thirteen entries below are our own overclaims, caught by controls
added after the fact: #7 (binding vs entity count), #10 (the intervention does
not show binding), #11 (colour is not causally load-bearing), #12 (nothing
replicates across models). Three more are errors in our own statistics or
tooling: #3, #5, #13.

---

## 1. Probe regularisation was not the bottleneck (checked, not assumed)

**Suspected**: the fixed `C = 1.0` was costing accuracy, understating the
model.

**Checked**: swept `C` on `eyes.color` @ L11 with a proper validation split.
Validation selects `C = 0.01`, giving test 0.350 against 0.348 at the default.

**Outcome**: no change warranted. Recorded in
`results/regularization_check.md`. The striking number from that sweep was
incidental: train accuracy 0.897 vs test 0.348, which is the split discipline
working. Train and test share no wordings, so the probe memorises phrasings on
train and must fall back on phrasing-invariant structure at test.

---

## 2. Control task: selectivity was inflated by construction

**Claimed**: Hewitt & Liang selectivity of +0.087 to +0.197, reported as
evidence against the probe-capacity objection.

**Problem**: in Hewitt & Liang, train and test share word types, so a
memorising probe transfers and the gap is meaningful. Here train and test share
**no narratives and no wordings**, so control-task *test* accuracy is at chance
no matter how capable the probe is. The gap was guaranteed, not earned.

**Fix**: report control-task **train** accuracy as the capacity measure
(it is 0.57–0.79, i.e. The probe genuinely can absorb arbitrary narrative-level
labels), and lean on MDL and the capacity ablation instead. The test-side
number is retained only for comparability with the literature.

---

## 3. MDL sub-1.0 compression: wrong explanation first

**Claimed**: the colour targets' compression of 0.75x–0.77x (worse than a
uniform code) was an artifact of a non-converged optimiser, corrected to 1.08x.

**Wrong**: that comparison was linear-probe 0.75x against MLP-8 1.08x, both
computed with identical settings in the same process. Convergence played no
part.

**Actual cause**: regularisation. Prequential coding charges `-log2 p` on every
block; the earliest blocks hold a handful of examples against 768 features, and
an unregularised logistic regression is confidently wrong there. MDL punishes
confident errors specifically. The MLP scored better largely because early
stopping keeps it timid.

**Fix**: `mdl_online_codelength` takes `C`, and `11_refresh_mdl.py` sweeps it.
With the sweep every target compresses above the uniform code
(1.24x–1.44x on the pilot). Previous values are preserved in the result files
under `mdl_superseded` with the reason.

**Residual issue**: on the full dataset the sweep's argmax sits at the grid
boundary (`C = 0.003`), so those numbers are lower bounds on compression, not
converged optima.

---

## 4. "Unfair pooling explains the lip results". Hypothesis rejected

**Claimed**: the static-embedding baseline beat the hidden state on
`lips.color` because the baseline pools over the whole prefix while the hidden
state is a single token, which makes the comparison unfair.

**Tested**: stored both read-out conventions from the same forward pass and
re-probed with mean pooling.

**Outcome**: hypothesis rejected. Pooling raises the lip numbers
(`lips.color` 0.308 → 0.344, `lips.finish` 0.359 → 0.419) but moves the best
layer to **layer 0**, which *is* the embedding baseline. It never overtakes it
(0.352 / 0.415). The transformer adds essentially nothing over a bag of
embeddings for the lip variables. The negative result is real.

Two consistency checks fell out and both passed: pooled-L0 accuracy tracks the
independently computed static-embedding baseline to within 0.008 on every
target, and pooled-L0 gives *identical* natural and shuffled accuracy
(0.343 vs 0.343), as an order-free representation must.

---

## 5. A biased test statistic of my own making

**Used**: `d_target - |d_other|` as the intervention specificity statistic.

**Problem**: `E[|noise|] > 0`, so at small effect sizes the other region's
noise alone drives the statistic negative. It reported "specificity fails" at
alpha 0.5–4.0 purely as an artifact.

**Fix**: signed paired difference `d_target - d_other`. With that, every alpha
is significant (p < 0.0001 for `lips.finish`, p ~ 0.002 for `lips.color`).

---

## 6. "The shuffled-order control is weak". Wrong reading of an average

**Claimed**: shuffling sentence order barely changes accuracy, so the control
cannot carry the temporal-structure claim.

**Problem**: that was an average over a mixed population. Eight of twelve
full-dataset targets peak at the **embedding layer**, which cannot encode order
at all, so they have no temporal structure to destroy. Pooling them with the
contextual targets hid the effect.

**Fix**: condition on best-layer depth (`14_order_sensitivity.py`).

| group | n | mean shuffled delta |
|---|---|---|
| best layer > 0 (contextual) | 4 | **-0.036** |
| best layer == 0 (embeddings) | 8 | -0.006 |

`corr(selectivity, shuffled delta) = -0.609`,
`corr(best layer, selectivity) = +0.625`. The control works; it was correctly
reporting that most targets are not contextual.

---

## 7. "Binding degrades with the number of entities". Withdrawn

**Claimed**, on seeing the 4-entity aggregate: colour selectivity ~0,
cross-entity transfer equal to within-entity, therefore the probe is
entity-agnostic and binding collapses as entities are added.

**Problem**: two variables changed at once (2 → 4 entities *and* 6.5 → 13
sentences), and the aggregate is dominated by positions far from the last
update, where the attribute is not decodable at all and both probes sit at
majority. They agree because both are uninformative, not because the code is
entity-agnostic.

**Fix**: stratify by distance since the variable last changed
(`15_binding_by_distance.py`). Averaged over 36 region pairs:

| distance | within - majority | cross - majority | binding margin |
|---|---|---|---|
| 0 | +0.120 | +0.008 | **+0.113** |
| 1 | +0.059 | +0.007 | **+0.052** |
| 2 | +0.077 | +0.010 | **+0.066** |
| 3-4 | +0.057 | +0.010 | **+0.047** |
| 5+ | -0.056 | +0.068 | **-0.124** |

**Revised claim**: gpt2 *does* bind attributes to
the correct region among four competitors wherever the attribute is decodable.
(Note: "bind" here means *probe-recoverable*. Correction #10 later showed the
causal evidence does not support entity-specific binding, so this claim is
correlational only.)
Binding decays with **distance**, and past ~5 sentences the representation
still carries "this value occurred somewhere" while having lost which region it
applied to. The shared presence code outlives the region-specific binding
code. This agrees independently with the concept-direction geometry
(difference-of-means CAVs aligned across regions at +0.62 to +0.87; the
discriminative directions pointing apart at -0.04 to -0.24).

A residual entity-count effect may still exist. At matched distance the pilot's
colour margins (+0.119/+0.159/+0.113 at d = 0/1/2) exceed the full set's
(+0.113/+0.052/+0.066). But it is a difference of degree, tested directly by
the length control (2 entities at 4-entity narrative length), not the collapse
originally claimed.

### Amendment: the length control reinstates the entity-count claim

The control dataset `long2` holds narrative length at the 4-entity value
(10-16 sentences) while using only 2 entities. Three outcomes were written into
the script header before it ran; this is outcome 1.

| dataset | entities | sentences | mean colour selectivity | colour best layers |
|---|---|---|---|---|
| pilot | 2 | ~6.5 | +0.097 | 11, 12 |
| **long2** | **2** | **~13** | **+0.119** | **2, 2** |
| full | 4 | ~13 | **+0.016** | 0, 0, 0, 0 |

Length is not the cause: 2 entities at 13 sentences does slightly *better* than
2 entities at 6.5 sentences. Entity count is: at matched length, 2 -> 4
entities takes colour selectivity from +0.119 to +0.016, and moves the best
layer from 2 (contextual) to 0 (embeddings).

So the original claim was right in substance and wrong in evidence. It was
withdrawn correctly at the time. The aggregate it rested on was confounded by
distance, and the distance-stratified analysis showed colour binding is
recoverable at 4 entities at short range. The refined statement is:

> Colour binding survives at 4 entities as a *probe-recoverable* signal at
> short distance, but the *contextual* colour code. The part the transformer
> layers contribute over the embeddings. Collapses with entity count, not
> with narrative length.

---

## 8. `lips.finish` non-linearity did not replicate

**Noted** in the pilot capacity ablation: MLP-512 reached 0.434 against a
linear 0.359, with selectivity doubling, and was flagged as possible genuine
non-linear structure worth investigating rather than dismissing.

**Outcome**: on the 4-entity set the same target is flat across a 64x capacity
range (0.372 / 0.377 / 0.371 / 0.373 / 0.373). We read the pilot value as
noise. Recorded
because it was flagged as provisional at the time rather than claimed.

---

## 9. MDL written off too quickly

**Claimed**: MDL "cannot adjudicate the claim this project is making", because
it favours the static-embedding baseline on every target.

**Partly wrong**: that is true for the *absolute* comparison against a lexical
baseline, and for the reason documented (prequential coding runs on the
training split, where wordings are shared, so a lexical code can memorise them).
But *relative* comparisons within MDL are highly informative:
`corr(best layer, MDL compression) = +0.921`,
`corr(selectivity, MDL compression) = +0.729`. It cleanly separates contextual
targets (mean 1.271x) from embedding-layer targets (mean 1.061x).

**Fix**: MDL is reported as a discriminator between targets, not as an
adjudicator of the hidden-state-vs-baseline question.

---

## 10. The intervention does not demonstrate binding (wrong-direction control)

**Claimed**: the activation-patching result "tests the binding question
causally". The edit moves the target region's read-out while the other
region's stays flat, so the representation is bound to a region.

**Missing control**: the design varied *which region is queried* but never
*which region's direction is injected*. A second design (`16_intervene_v2.py`)
added two arms at matched norm: a random direction, and the **other region's**
concept direction for the same attribute.

**Result** for `lips.finish`, edit L1, read L12, n = 80 per bin:

| arm | alpha = 2 | alpha = 8 |
|---|---|---|
| concept (this region) | **+0.055** [+0.041, +0.070] | +0.148 [+0.096, +0.201] |
| random, matched norm | +0.014 [-0.003, +0.030] n.s. | +0.067 [+0.001, +0.132] |
| other region's direction | **+0.040** [+0.027, +0.052] | **+0.149** [+0.104, +0.196] |

**What survives**: the concept direction beats a random direction of the same
magnitude at alpha = 2 (+0.055 vs +0.014, n.s.). There is a real,
direction-specific causal effect on the finish read-out.

**What does not**: the *other region's* finish direction is as effective as the
correct one (+0.149 vs +0.148 at alpha = 8). The effective direction is not
entity-specific, so the experiment demonstrates a usable **attribute-value**
representation, not a bound one.

This should have been predicted from geometry already in hand: the
difference-of-means directions for `eyes.finish=dewy` and `lips.finish=dewy`
have cosine +0.67 to +0.95. They are nearly the same vector, so they
interchange causally.

**Two secondary findings from the same run**:

- alpha = 8 is off-distribution: even random directions reach significance
  there. Alpha = 2 is the interpretable dose, and all reported claims should
  use it.
- The finish effect does **not** decay with distance (+0.148 near vs +0.160
  far), unlike the probe binding margin, which is further evidence that what
  the intervention moves is the attribute value rather than the binding.

---

## 11. Colour is decodable but not causally load-bearing

**Pre-registered prediction** (written into `run_*`/script headers before the
run): colour should show a causal effect NEAR (distance <= 1) and none FAR
(>= 5), which would explain the null in the all-distances run as dilution.

**Result**: no effect at either distance.

| bin | arm | alpha = 8 |
|---|---|---|
| near | concept | +0.022 [-0.010, +0.054] n.s. |
| near | random | -0.024 [-0.062, +0.015] n.s. |
| near | other region's direction | +0.032 [+0.001, +0.065] |
| far | concept | +0.008 [-0.028, +0.044] n.s. |

The prediction failed. Colour binding is recoverable by a probe (margin +0.113
at distance 0, 33/36 pairs significant) but steering the colour direction does
not change the model's behaviour. The correlational and causal results
dissociate, and colour lands on the wrong side of that line.

---

## 12. The probe-beats-baseline pattern does not replicate across models

**Claimed**, from gpt2 alone: face-region state is linearly decodable above
every lexical baseline for the eye variables (`eyes.finish` +0.097,
two-sided p < 0.0001) but not the lip variables, and that asymmetry was treated
as a property of the task.

**Tested**: the whole pilot pipeline on `EleutherAI/pythia-410m`, same data,
same splits, same cluster-bootstrap test.

| target | gpt2 gap | p2 | pythia gap | p2 | verdict |
|---|---|---|---|---|---|
| `eyes.color` | +0.062 | 0.0075 | +0.031 | 0.2315 | gpt2 only |
| `eyes.finish` | +0.097 | 0.0000 | +0.014 | 0.6390 | gpt2 only |
| `lips.color` | -0.044 | 0.1260 | +0.064 | 0.0280 | pythia only |
| `lips.finish` | -0.058 | 0.0285 | +0.024 | 0.4150 | opposite sign |

**Zero of four targets are significant in the same direction in both models.**
The most robust single result in the project (`eyes.finish`, p < 0.0001 in
gpt2) is p = 0.64 in pythia. `lips.finish` is significantly *negative* in gpt2
and positive in pythia.

**What this means**: the *specific pattern* of which state variables beat a
lexical baseline is a property of the model, not of the domain. Any claim of
the form "face-region state is decodable above baseline" must be stated
per-model, and the gpt2 lip-variable failure. Reported earlier as a real
negative result about the task. Is gpt2-specific.

**What partially survives**: all four pythia gaps are positive, against gpt2's
mixed signs, so the weaker claim that contextual representations carry *some*
state information above a lexical baseline is consistent across both models. A
sign test on 4/4 gives p ~ 0.125, which is suggestive and not evidence.

This is the strongest argument in the project for why single-model probing
results should not be generalised, and it was only visible because a second
model was run.

---

## 13. The `lips.coverage` intervention reversal was a confounded null value

**Observed**: steering `lips.coverage` moved the behavioural read-out strongly
in the *wrong* direction (-0.367 on the target at alpha = 8), while every other
intervention moved it the right way or not at all.

**Not a sign error.** Class indices, direction construction and norms were all
checked and are correct; the probe read-out moves the right way at every alpha
(steered 0.307 -> 0.373, true 0.473 -> 0.247).

**Actual cause**, from breaking the effect down by (true -> steered) pair:
the reversal is carried entirely by the two cells that steer *towards* `bare`
(`full -> bare` -0.955, `light -> bare` -1.012). And `bare` is confounded with
narrative position:

| coverage value | n | mean position |
|---|---|---|
| **bare** | 8551 | **3.65** |
| light | 7333 | 7.52 |
| full | 7634 | 7.74 |

The difference-of-means direction for `bare` therefore encodes "early in the
narrative" as much as "no product", and adding it pushes the model towards
beginning-of-text rather than towards the intended state.

**Why `finish` was unaffected**: `FINISH_READOUT` lists only matte / dewy /
satin, excluding the null value `none` -- which has exactly the same confound
(mean position 3.65). `COVERAGE_READOUT` included `bare`. The inconsistency was
in our own read-out specification, not in the model.

**Second defect in the same target**: the `light <-> full` direction has norm
1.05 against 11.9-12.9 for directions involving `bare`. Those two classes are
nearly coincident in activation space, so an alpha that is reasonable for the
others is a negligible edit for them. Coverage is a poor intervention target
for this reason independent of the confound.

**Fix**: `bare` removed from `COVERAGE_READOUT`, matching the treatment of
`none` for finish. Any coverage intervention number predating this fix is
withdrawn.

**General lesson for the design**: a difference-of-means concept direction is
only as clean as the class balance behind it. Where one class is
systematically located at a different point in the narrative, the direction
picks that up, and the intervention tests position rather than state.

---

## 14. The wrong region control cannot support the claim we built on it

**Claimed**: injecting one region's concept direction moves another region's
read out by the same amount, therefore the causal effect is not entity
specific.

**Problem**: that test has two readings and cannot separate them. Either the
model carries no usable binding, or our direction captured the shared value
component and left the entity specific component out. A difference of means is
built to separate two label groups, so it recovers what the groups differ on,
which is the value. Anything entity specific sits in the residual.

**Fix**: decompose the direction and steer with each part alone
(`22_decompose_direction.py`). For entities E1 and E2 and a value pair,

    shared        s = (d1 + d2) / 2
    differential  r = (d1 - d2) / 2      so that  d1 = s + r

Both parts rescaled to the norm of d1. Predictions written before the run: the
shared part raises both read outs equally, and the differential part separates
them if the model uses the binding.

**Result**, alpha 2, n = 100 per model:

| model | cos(d1, d2) | differential share | shared effect | differential difference | p |
|---|---|---|---|---|---|
| gpt2-124M | 0.848 | 28.4% | +0.183 | +0.016 [+0.009, +0.023] | <0.0001 |
| pythia-410m | 0.743 | 37.7% | +0.190 | -0.000 [-0.006, +0.005] | 0.85 |
| Qwen3-0.6B-Base | 0.740 | 35.5% | +0.317 | -0.000 [-0.010, +0.010] | 0.91 |

The shared part is causally effective in all three. The entity specific part
works in gpt2 at about a tenth the strength and does nothing in the other two.
A random direction of matched norm does nothing anywhere.

**What changed**: the wrong region result is now explained by the ratio between
the two components rather than by the absence of a binding representation. The
claim was restated as the entity specific component recovered by a difference
of means is not required for the read out we tested.

---

## 15. No target replicates across models

**Claimed**, from gpt2 and pythia: no target is significant in the same
direction in both models, therefore the pattern of which variables beat a
lexical baseline is a property of the checkpoint.

**Problem**: the conclusion rested on one comparison with one second model.

**Result** with Qwen3-0.6B-Base added:

| target | gpt2 | pythia | Qwen |
|---|---|---|---|
| `eyes.color` | +0.062 * | +0.031 | +0.111 * |
| `eyes.finish` | +0.097 * | +0.014 | +0.120 * |
| `lips.color` | -0.044 | +0.064 * | -0.031 |
| `lips.finish` | -0.058 * | +0.024 | -0.048 |

gpt2 and Qwen agree on the sign of all four targets and both reach significance
on the two eye variables. Pythia is the outlier. The earlier conclusion came
from the choice of second model.

**What stands**: the direction of a probing result still varies by checkpoint,
and a two model comparison can mislead in either direction.

---

## 16. The behavioural read out does not bind, so it cannot test binding

**Claimed**: the region specific component of a concept direction produces no
causal effect in pythia and Qwen, therefore the entity specific component is
not required for the attribute read out.

**Missing check**: the measurement itself was never validated. Unaided
behavioural accuracy is near chance, so a zero could be a floor effect.

**Test** (`23_readout_validity.py`), with no steering at all. Take held out
positions where the two regions carry different finishes, compare the model's
log odds across all four assignments of two values to two regions, and split
the result into two contrasts. Sensitivity changes the value of the region we
asked about. Leakage changes the other region's value.

| model | read out | sensitivity | leakage | ratio |
|---|---|---|---|---|
| gpt2-124M | lips | +0.201 | +0.316 | 0.64 |
| gpt2-124M | eyes | +0.358 | +0.228 | 1.57 |
| pythia-410m | lips | +0.385 | +0.544 | 0.71 |
| pythia-410m | eyes | +0.551 | +0.380 | 1.45 |
| Qwen3-0.6B | lips | +0.530 | +0.400 | 1.33 |
| Qwen3-0.6B | eyes | +0.379 | +0.500 | 0.76 |

Mean ratio 1.07. Sensitivity is significant everywhere, so the instrument
registers changes and the null is not a floor effect. It answers a question
about one region using the other region's value just as readily.

**What changed**: a steering experiment read through this instrument cannot
separate a model that fails to bind from an instrument that fails to bind. The
claim above is withdrawn. The probe based binding margin stands, because the
probe is trained per region and does bind.

---

## 17. Missing the closest prior work, and two wrong dataset descriptions

**Missed**: Feng and Steinhardt, *How do Language Models Bind Entities in
Context?* (ICLR 2024). They identify binding ID vectors attached to entities and
attributes, established through causal interventions, occupying a continuous
subspace where distance tracks discernability. This is the nearest neighbour to
our shared and differential decomposition and we cited only their 2025
propositional probes paper.

Two consequences. Our novelty claim for the decomposition shrinks to the narrow
point that a difference of means fails to isolate the entity part. And their
mechanism appears in *every sufficiently large* model of the Pythia and LLaMA
families, so a null at 124M to 600M parameters is consistent with sitting below
that threshold rather than evidence against the mechanism.

**Missed**: Li, Nye and Andreas, *Implicit Representations of Meaning in Neural
Language Models* (ACL 2021), which probes entity state in Alchemy and TextWorld
and is the earliest close relative of this domain.

**Wrong**: we described Kim and Schuster (ACL 2023) as evaluating on ProPara and
Recipes. Their data is a synthetic boxes and objects setup with operations that
move, remove and add objects. Corrected.

---

## 18. The leakage mean hid a per model structure, and recency explains one case of three

**Observed**: the sensitivity to leakage ratio averages 1.07 across three models
and six read outs, which reads as the read out mixing regions at random.

**It is not random.** Grouping by which region's value moves the read out,
rather than by which region the question named:

| model | lips value drives | eyes value drives | dominant |
|---|---|---|---|
| gpt2-124M | +0.219 | +0.288 | eyes |
| pythia-410m | +0.446 | +0.550 | eyes |
| Qwen3-0.6B | +0.524 | +0.432 | lips |

In each model one region drives both read outs at nearly the same strength.

**Mechanism test** (`24_readout_mechanism.py`). For every trial we know the
action history, so each position can be labelled by the value of the region
acted on last and the value of the region acted on more often, then compared
against the two fixed regions.

| model | region acted on last | region acted on more often | beats the dominant region? |
|---|---|---|---|
| gpt2-124M | +0.180 | +0.108 | no |
| pythia-410m | +0.449 | +0.300 | no |
| Qwen3-0.6B | +0.543 | **+0.576** | yes |

Qwen tracks the region acted on more often better than it tracks either fixed
region, which is a salience account. gpt2 and pythia track the eyes value
whichever region the question named, and neither recency nor frequency of
action explains it. We report a fixed per model preference with no mechanism
behind it.

**Limit**: we tested recency of state changes. Recency of mentions was not
tested, and the distractor sentences name values without applying them.

---

## 19. The framing put the failure on the tool

**Claimed**: the behavioural read out does not bind, so it cannot test binding.

**Better**: the read out is the model's own next token distribution, so the
leakage is a fact about model behaviour rather than about our tooling. Paired
with the probe result, which recovers binding at a margin of +0.04 to +0.15 in
all three models, the statement is that binding is linearly decodable in the
residual stream and the output does not use it. Present but unused is a sharper
claim than an instrument failure, and it fits Feng and Steinhardt's scale
threshold: their binding ID vectors appear only in sufficiently large models,
and ours run 124M to 600M.

---

## 20. The one positive result inherited the same caveat

**Claimed**: gpt2 shows a region specific causal effect of +0.016, p below
0.0001, with a dose response.

**Problem**: that effect was measured through the same read out that correction
18 shows does not bind. A differential could come from lexical content carried
in the region specific component rather than from a binding the model uses.

**Fix**: reported as a small effect of unclear origin. Attributing it would
need a read out that binds. We kept the caveat on the result that came out our
way as well as on the nulls.

---

## 21. Dataset generation was not reproducible

**Claimed** in `REPRODUCE.md`: `scripts/02_generate_dataset.py` is deterministic
given `--seed`.

**False.** The per split generator was seeded with
`hash((seed, split, condition))`. Python randomises string hashing per process,
so the same `--seed` produced different narratives on every run. Two processes
asked for the same seed return different draws:

    process A   0.23775159794830125
    process B   0.97599746632254070

**Why it did not corrupt the reported results.** Each dataset was generated once
and the files reused for every model, so all four arms ran on identical
narratives. The bug affects anyone trying to regenerate the data, including us
on a second machine.

**Fix**: seed from `zlib.crc32` of the key string, which is stable across
processes and platforms. The pilot narratives are now committed rather than
gitignored, so a replication runs on byte identical data instead of its own
draw. Fingerprints:

    data/pilot/natural.jsonl    1000 narratives   sha256 2cc9a1c85019ad94
    data/pilot/shuffled.jsonl   1000 narratives   sha256 180476f26a17be60
