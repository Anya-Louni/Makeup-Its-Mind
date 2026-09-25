# Design and methods

## 1. Why this domain

Existing world-model probing work tracks a single structure — an Othello board,
a chess position, a scalar such as temperature. A makeup routine forces several
independent entities to be tracked at once, each carrying its own attributes,
updated in an order-dependent sequence. That turns the question from "is the
state recoverable" into "is each attribute *bound to the right entity*", which
is the harder and more interesting claim.

## 2. The world

Four entities — `lips`, `eyes`, `cheeks`, `skin`. Three attributes each:

| attribute | values |
|---|---|
| colour | bare, nude, pink, red, brown |
| finish | none, matte, dewy, satin |
| coverage | bare, light, full |

The palette is deliberately identical across entities. If lips could only be
red and eyes only brown, "which region is red" would be answerable from the
colour alone and the binding question would evaporate.

Nine action kinds drive the state machine: `APPLY` (sets colour, finish and
coverage at once), `BLOT` / `GLOSS` / `SOFTEN` (set finish), `BUILD` / `SHEER`
(change coverage), `REMOVE` (reset to bare), plus `NOOP` and `DISTRACTOR`,
which change nothing.

Because the narratives are generated *from* the state machine, the label at
every sentence position is exact and free. There is no annotation noise.

## 3. Making the task un-gameable

Four separate defences, because a probing result is only as strong as the
cheapest strategy that beats it.

**Paraphrase variation.** Every attribute value has ten realisations and most
avoid the literal label word entirely. `matte` appears as *"a chalk-flat
surface"*, *"a dead-flat look that caught no light"*, *"a fully dried-down,
non-wet look"*. Keyword spotting cannot carry the task.

**Zero surface overlap between splits.** Phrase banks *and* sentence skeletons
are partitioned 6/2/2 by index. A test narrative is assembled entirely from
wordings the probe has never seen. This is stricter than the usual
narrative-level split: it tests generalisation to unseen phrasings of the same
state, not memorisation of templates.

**Distractor sentences (~18% of positions).** These mention a colour without
applying it — *"She held a deep crimson up to the light and shook her head"*,
*"A tube of a soft pink rolled to the edge of the counter, unused"*. A
bag-of-words model is actively misled; a state tracker is not.

**Entity-agnostic colour wording.** The same colour phrases serve every region,
so binding cannot be read off the colour phrasing.

## 4. Baselines and controls

| control | what it rules out |
|---|---|
| majority class | trivial |
| static embedding (model's own embedding layer, mean over prefix) | that no contextual computation is needed |
| **TF-IDF bag of 1–2-grams over the raw prefix text** | that a purely lexical model does just as well — a stronger competitor than static embeddings, and not in the original spec |
| shuffled narrative order | reliance on plausible routine ordering |
| **override stratification** | keyword retrieval: on ~25% of positions the value was overridden earlier, so the first mention is the *wrong* answer |
| Hewitt & Liang control task | probe capacity |
| prequential MDL (Voita & Titov) | probe capacity, accounting for the cost of the probe itself |
| capacity ablation (linear → MLP 8…512) | that non-linear gains are structure rather than parameters |
| cross-entity probe transfer | that the probe reads "some region is red" rather than "the lips are red" |

### Read-out convention: last token vs. mean pooling

The standard convention is to read the residual stream at the **last token** of
each sentence. But the non-contextual baseline required by the spec pools the
model's embeddings over the *whole prefix*. That is not a like-for-like
comparison: the baseline gets to aggregate every token, the hidden state gets
one position.

The pilot made this concrete. For `lips.color`, the pooled static-embedding
baseline (0.349) *beat* the last-token hidden state (0.308). Reporting only
that would have misattributed a pooling advantage to "the model has no world
model for lip colour".

So extraction stores both conventions from the same forward pass — `X.npy`
(last token) and `Xmean.npy` (mean over the prefix) — and `04_probe.py` takes
`--pooling {last,mean}`. The mean-pooled hidden state is the fair counterpart
to the pooled baseline; the last-token number is the one comparable to the
probing literature. Both are reported.

**Outcome: the hypothesis was wrong, and that is the useful part.** Pooling
does raise the lip numbers (`lips.color` 0.308 → 0.344, `lips.finish`
0.359 → 0.419), but in both cases the best layer moves to **layer 0** — which
is the mean of the model's own input embeddings, i.e. the static-embedding
baseline itself. It never overtakes that baseline (0.352 / 0.415). The
transformer layers add essentially nothing over a bag of embeddings for the
lip variables. The negative result is real, not a measurement artifact.

Two consistency checks fell out of this and both passed, which is worth
recording because they validate the harness rather than the hypothesis:

- pooled layer-0 accuracy tracks the independently computed static-embedding
  baseline to within 0.008 on every target;
- pooled layer 0 gives *identical* natural and shuffled-order accuracy
  (0.343 vs 0.343), exactly as a genuinely order-free representation must.

### MDL measures something different from our test accuracy

Two results forced this section.

First, regularisation dominates the MDL number. With `C = 1.0` the colour
targets appeared to compress *worse* than a uniform code (0.75x, 0.77x). That
is not a property of the representation: prequential coding charges -log2 p on
every block, the first blocks hold a handful of examples against 768 features,
and an unregularised probe is confidently wrong there. Sweeping C:

| target | C=0.003 | C=0.01 | C=0.1 | C=1.0 |
|---|---|---|---|---|
| `eyes.color` @L11 | 1.21x | **1.24x** | 1.11x | 0.75x |
| `eyes.finish` @L1 | 1.37x | **1.43x** | 1.40x | 1.06x |
| `lips.color` @L12 | 1.30x | **1.34x** | 1.17x | 0.77x |
| `lips.finish` @L1 | 1.38x | **1.44x** | 1.41x | 1.08x |

Every target compresses comfortably better than the uniform code once the
probe is regularised. Reporting a single arbitrary C would have produced a
spurious negative result.

Second, and more important: **the static-embedding baseline compresses
better than the hidden states on every target** (1.48x, 1.63x, 1.58x, 1.66x
against 1.24x, 1.43x, 1.34x, 1.44x) — even on the eye variables, where
held-out accuracy clearly favours the hidden states.

That is not a contradiction; the two metrics answer different questions here.
Prequential MDL trains and evaluates inside one homogeneous pool, so it is
computed on the training split, where wordings *are* shared. A strong lexical
code can memorise those wordings and pay very little to transmit the labels.
Our accuracy number is deliberately the opposite: it is measured on narratives
whose phrasings never appear in training, so it charges for phrasing-invariant
generalisation specifically.

So MDL as usually computed cannot adjudicate the claim this project is making.
It is reported because it is the standard critique and because the C-sweep is
worth having, but the load-bearing evidence is held-out-wording accuracy, the
capacity ablation, and the causal intervention.

### Binding emerges with depth

Repeating the geometry at layers 1 and 11 (the default run used the median of
the four best layers, which was nobody's best) shows the cleanest structural
result in the pilot. Mean principal angle between subspaces:

| pair | L1 | L11 |
|---|---|---|
| `eyes.color` \| `lips.color` (cross-entity) | 49.7 | **58.8** |
| `eyes.color` \| `lips.finish` (cross-entity) | 43.4 | **64.3** |
| `eyes.color` \| `eyes.finish` (within-entity) | 46.3 | 46.9 |
| `lips.color` \| `lips.finish` (within-entity) | 49.7 | 50.1 |

Cross-entity subspaces rotate apart with depth; within-entity ones do not
move. The difference-of-means cosine for the same colour on the two regions
falls over the same span (`red`: +0.87 at L1 to +0.63 at L11). Early layers
encode roughly "a colour word occurred"; deeper layers begin to separate
*which region it applies to*. That is the binding question answered
structurally, and it agrees with the causal intervention.

### A caveat we state rather than hide

The Hewitt & Liang control task assigns each narrative a random label drawn
from the real label marginal. In the original paper, train and test share word
types, so a memorising probe transfers and the selectivity gap is meaningful.
Here train and test share **no narratives and no wordings** by construction, so
control-task *test* accuracy sits at chance no matter what the probe is capable
of, and the raw selectivity gap flatters the result.

We therefore report control-task **train** accuracy as the capacity measure,
and lean on MDL and the capacity ablation for the "is this just a big probe"
critique. The test-side selectivity number is reported for comparability with
the literature, not treated as the load-bearing evidence.

## 5. Causal intervention

Probing is correlational. The intervention adds `alpha * d` to the residual
stream at layer `L`, where `d = mean(activations | target value) -
mean(activations | current value)`, computed on the **training split only** so
no trial narrative contributes to its own edit.

Two read-outs, reported separately:

- **Probe read-out.** A probe trained on clean data at a *later* layer reads
  the intervened hidden state. If an edit at `L` flips the state code at
  `L' > L`, the direction is part of what the model carries forward. The
  paired control is the other region's probe on the very same forward pass.
- **Behavioural read-out.** Append a cue and compare the model's own
  log-probabilities of the competing state words. This is the stronger claim,
  but it is only interpretable if the unintervened model reads the true state
  out above chance — so the alpha = 0 baseline is always reported, and every
  effect is a **paired** change from it. For `gpt2` the unintervened
  behavioural read-out is near chance, and the pilot showed a strong constant
  word-preference bias; raw flip rates would be meaningless, which is why the
  metric is the paired change in log-odds and the flip rate restricted to
  trials the model initially got right.

## 6. Model selection

Selection was benchmarked, not assumed. See `docs/model_benchmark.md`. The
reference machine is CPU-only (12 cores, ~15 GB RAM), which caps the candidate
set; `meta-llama/Llama-3.2-1B` and `google/gemma-2-2b` are gated on the Hub and
could not be evaluated without an access token.

The vocabulary check runs before anything else. Small base LMs are not
instruction-tuned, so asking them to define "matte" is unreliable evidence
either way; instead each term is scored on minimal pairs
(`"A matte finish reflects almost no ___"` → *light* vs *sound*). Terms that
scored zero on a single pair were re-tested with three independent pairs, which
resolved almost all of them — a single failed pair is usually a bad probe, not
a missing word. The one substantive finding: `pythia-410m` fails `matte`
(0.40), while `gpt2` handles it (0.80). Since finish is one of the two pilot
attributes, that counts against pythia despite its marginally higher overall
score.
