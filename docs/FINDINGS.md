# Findings

The canonical statement of what this project shows. Written after the
cross-model replication, and deliberately narrower than the question we set out
to answer. `CORRECTIONS.md` records the thirteen claims that were revised on
the way here, four of which were our own overclaims caught by controls we added
later.

---

## Abstract

We ask whether small language models maintain an implicit world model of a
multi-entity scene: as a makeup routine is described sentence by sentence, do
the activations track the colour, finish and coverage of four face regions, and
is each attribute bound to the right region? We build a synthetic domain with
exact per-position labels, an order-destructive state machine, distractor
sentences that mention values without applying them, and train/test splits that
share **no surface forms at all**, so a probe must generalise to unseen
phrasings of the same state rather than memorise templates.

Across two models (gpt2-124M, pythia-410m) we find a **dissociation between
what is decodable and what is used**. A linear probe recovers which region an
attribute belongs to — the binding margin is +0.07 to +0.12 shortly after a
value is set, holds in both models, and survives FDR correction across 180
tests (33/36 region pairs at distance 0). But activation patching shows the
model's behaviour is driven by an **attribute-value** direction that is *not*
entity-specific: injecting another region's direction moves the read-out as
much as the correct region's, or more — across four tests spanning two models
and two attributes (+0.195 vs +0.147 and +0.069 vs +0.032 in pythia, +0.149 vs
+0.148 in gpt2) — while a random direction of matched norm does nothing. A
pre-registered prediction that colour would show a causal effect at short
distance failed in gpt2 at every distance and every arm, though colour *is*
causally effective in pythia: which attributes are load-bearing is itself
model-specific.

Two further results bound the claim. First, binding **decays with narrative
distance**: past roughly five sentences the within-region signal falls below
majority while the cross-region read rises above it, so the representation
still carries "this value occurred somewhere" after it has lost *where*.
Second, a controlled comparison isolating entity count from narrative length
shows the *contextual* colour code collapses when four regions compete rather
than two (selectivity +0.119 to +0.016 at matched length), with the best layer
falling from 2 to 0.

Most importantly, **the pattern of which state variables beat a lexical
baseline does not replicate across the two models**: no target is significant
in the same direction in both, and our single most robust gpt2 result
(`eyes.finish`, two-sided p < 0.0001) is p = 0.64 in pythia. We therefore
report per-model rather than claiming a property of the domain, and we take
this as the project's clearest methodological result: single-model probing
findings of this kind should not be generalised.

---

## The central claim, stated as narrowly as the evidence supports

> In small language models, an attribute's *value* is linearly decodable,
> causally load-bearing, and shared across entities; the *binding* of that
> value to a particular entity is linearly decodable and decays with narrative
> distance, but we find no evidence that the model uses it. Which specific
> variables clear a lexical baseline is a property of the model, not of the
> task.

---

## Centerpiece: what replicates across models

Same dataset, same splits, same tests. ✅ = holds in both, ❌ = fails in both
(which is still a replicated result), ⚠️ = differs between models, — = not
tested in that model.

| # | Claim | gpt2-124M | pythia-410m | Replicates? |
|---|---|---|---|---|
| 1 | Probe beats the static-embedding baseline | 2 of 4 targets (p₂ = 0.008, <0.0001) | 1 of 4 targets (p₂ = 0.028) | ⚠️ **no target in the same direction** |
| 2 | Binding is probe-recoverable at short distance | +0.073 / +0.096 / +0.105 (d = 0/1/2) | +0.036 / +0.104 / +0.115 | ✅ |
| 3 | Causal effect on attribute **value** (concept ≫ random) | +0.055 vs +0.014 n.s. | +0.147 vs −0.009 n.s. | ✅ |
| 4 | Causal effect is **entity-specific** (concept > other region's direction) | finish +0.148 vs **+0.149** | finish +0.147 vs **+0.195**; colour +0.032 vs **+0.069** | ❌ **fails in both models, both attributes** |
| 5 | Colour is causally load-bearing | **no** effect at any distance or arm | **yes** — concept +0.032 [+0.017, +0.045] vs random -0.010 n.s. | ⚠️ **differs** |
| 6 | Where the state code lives | embeddings + middle layers | almost entirely the output end | ⚠️ differs sharply |
| 7 | Signal is linear, not probe capacity | flat over 64× capacity | — | — |
| 8 | Binding decays with distance; sign flips past d ≥ 5 | +0.113 → −0.124 (4-entity set) | — (pilot too short) | — |
| 9 | Entity count, not length, collapses the contextual colour code | +0.119 → +0.016 at matched length | — | — |

Rows 7, 8 and 9 were run on gpt2 only; they are stated as gpt2 findings, not as
properties of small language models. Row 8 needs a 4-entity dataset, which the
pythia replication did not have time to extract.

### How to read the table

The two rows that matter most point in opposite directions.

**Row 3 is the strongest positive result.** In both models, adding the concept
direction moves the model's own log-odds of the steered value, while a random
direction of the same magnitude does not. That is a real mechanism, not a
correlation.

**Row 4 is why we do not claim it demonstrates binding.** Across four
independent tests — two models x two attributes — the *other* region's
direction works at least as well as the correct one, and for pythia's colour it
works **2.2x better** (+0.069 against +0.032). The geometry predicts this and we
should have checked it sooner: difference-of-means directions for the same
value on two different regions have cosine +0.67 to +0.95, i.e. they are nearly
the same vector.

**Row 5 is a second instance of row 1's lesson.** Colour steering does nothing
in gpt2 at any distance or arm, and is a clean significant effect in pythia
(concept +0.032, random -0.010 n.s.). Whether a given attribute is causally
load-bearing is, again, a property of the model.

**Row 1 is why every claim here is stated per-model.** The specific pattern of
which variables beat a lexical baseline simply does not transfer.

---

## What generalises, what does not

**Generalises across both models**

- Attribute values have a linearly decodable, causally effective direction
  (concept beats a matched-norm random direction in every test run).
- That direction is **shared between entities** — four independent tests, two
  models x two attributes — so the causal evidence supports a value code rather
  than a bound one.
- Binding is recoverable by a probe at short distance (+0.04 to +0.12).
- Small positive gaps over lexical baselines are the norm; large ones are not.

**Does not generalise**

- Which targets beat the baseline, and by how much.
- Which attributes are causally load-bearing (colour: inert in gpt2,
  significant in pythia).
- Where in the network the state code sits (gpt2: layers 1–12 of 12 scattered,
  with 8 of 12 targets at the embedding layer on the 4-entity set;
  pythia: layers 20–24 of 24).

**Established on gpt2 only**

- Binding decays with distance and inverts past ~5 sentences.
- Entity count rather than narrative length collapses the contextual colour
  code.
- The signal is linear rather than probe capacity.

---

## Statistical conventions

- p-values are **two-sided** throughout; one-sided values are retained in the
  JSON as `p_one_sided_greater`.
- Intervals over narrative positions use a **cluster bootstrap resampling whole
  narratives**, because positions inside a narrative share wordings and state.
- The binding table is corrected for multiple comparisons with
  **Benjamini-Hochberg** across all 180 tests; 33/36 pairs at distance 0 and
  29/36 at distance ≥5 survive at FDR 0.05.
- Causal claims use **α = 2**. At α = 8 even random directions of matched norm
  reach significance, i.e. the edit is off-distribution.
- MDL is reported over a **sweep of probe regularisation**; a single arbitrary
  `C` yields compression anywhere from 0.75× to 1.44× on identical data.

## Honest limits

- Two models, both small. Kim & Schuster (ACL 2023) find entity tracking in
  procedural text needs code-pretrained GPT-3.5-class models, so this is well
  below the regime where strong results should be expected.
- Synthetic text. The paraphrase-disjoint split makes it a real generalisation
  test, but it is not natural prose.
- The behavioural read-out is weak at this scale (0.32 unintervened for
  `lips.finish` in gpt2), so causal claims rest on paired log-odds changes
  rather than the model's own accuracy.
- Rows 5, 7, 8, 9 of the centerpiece are single-model results, and row 1 is
  precisely the reason not to assume they transfer.
