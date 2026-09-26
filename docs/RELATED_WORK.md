# Related work and what is actually new here

Three lines of work bound this project. Being precise about the boundary
matters: two of the three already do most of what a casual reader might think
this project introduces.

## 1. Emergent world representations in sequence models

Li et al. (ICLR 2023) trained a GPT model on legal Othello moves and recovered
the board state from the residual stream, establishing the template this
project follows: a synthetic domain with exact ground-truth state, probes on
hidden activations, and intervention to show the representation is used rather
than merely present. They recovered the board with **non-linear** probes (a
two-layer MLP); linear probes performed near chance.

Nanda et al. (BlackboxNLP 2023) then showed the representation *is* linear once
the target is reframed in player-relative terms (`MINE`/`YOURS`/`EMPTY`),
reaching >99% from layer four onward. That correction is methodologically
important for us: **whether a world model looks linear can depend on how the
target variable is parameterised, not on the model**.

*Difference here*: Othello is one board, a single global state object. This
domain has four independent entities each carrying three attributes, so the
question shifts from "is the state recoverable" to "is each attribute bound to
the right entity". The input is also English prose with paraphrase variation
rather than a closed move vocabulary, which introduces a lexical-baseline
problem Othello does not have: a bag of move tokens cannot reconstruct a board,
but a bag of words can go a long way on "what colour was mentioned".

That difference is not favourable to us, and it is the single biggest caveat in
our results: on 8 of 12 full-dataset targets the model's own embedding layer is
the best layer, and only `eyes.finish` beats the static-embedding baseline
robustly (gap +0.097, cluster-bootstrap CI [+0.048, +0.148]).

## 2. Binding and propositional structure

Feng, Russell and Steinhardt, *Monitoring Latent World States in Language
Models with Propositional Probes* (ICLR 2025), is the closest relative. They
decode lexical concepts at token positions, identify a **binding subspace** via
a Hessian-based method, and compose propositions such as `WorksAs(Greg, nurse)`.
Trained on simple templates, their probes generalise to short stories and to
Spanish, and they show models can keep faithful internal representations while
producing unfaithful outputs.

*Difference here*: their setting is a largely static context. Entities and
their properties are asserted, and the probe reads off the resulting
proposition set. This domain is **sequential and destructive**: an attribute is
set, later overwritten, blotted, built up, or removed, so the correct answer at
position *i* is a function of the whole prefix and the first mention is often
wrong (~23–27% of positions by construction). That lets us ask a question their
setup does not pose: **how long does a binding survive?**

Our main positive result is exactly that. Averaged over 36 region pairs, the
binding margin. (within-entity accuracy − its majority) − (cross-entity
accuracy − its majority). Is +0.113 at distance 0 and decays to **−0.124 by
distance ≥5**, where within-entity falls *below* majority while the cross-entity
read rises *above* it. Past roughly five sentences the representation still
carries "this value occurred somewhere" but has lost which region it applied
to. The shared presence code outlives the region-specific binding code.

We have not seen that dissociation characterised elsewhere, and it is
corroborated independently by the concept-direction geometry: difference-of-means
CAVs for the same value on different regions are strongly aligned (+0.62 to
+0.87, the presence component) while the discriminative directions point apart
(−0.04 to −0.24, the binding component).

## 3. Procedural / entity state tracking

Kim and Schuster, *Entity Tracking in Language Models* (ACL 2023), test whether
a model can infer an entity's final state from an initial description plus a
sequence of state-changing operations, using ProPara (488 science procedures)
and Recipes (866 cooking recipes). They find non-trivial but limited tracking,
and. They also find that only models pretrained on large amounts of code
(GPT-3.5) exhibit the ability at all.

*Difference here*: their evaluation is **behavioural** (question answering
about the final state); ours is **representational and causal** (probes on
activations plus activation patching). Their finding is also a direct warning
about our scale: if entity tracking in natural procedural text needs
code-pretrained GPT-3.5-class models, a 124M-parameter GPT-2 is far below the
regime where strong results should be expected. Our modest effect sizes are
consistent with theirs, and our behavioural read-out is indeed near chance
un-intervened (0.32 for `lips.finish`), which is why the causal claim rests on
the paired change in log-odds rather than on raw read-out accuracy.

## 4. Probing methodology

- Hewitt and Liang (EMNLP 2019), control tasks and selectivity. Used here, with
  a documented caveat: because our train/test splits share no wordings, control
  *test* accuracy is at chance by construction and the usual selectivity gap is
  inflated. We report control *train* accuracy as the capacity measure instead.
- Voita and Titov (EMNLP 2020), prequential MDL probing. Used here, with the
  finding that it is highly sensitive to probe regularisation (compression
  ranges 0.75×–1.44× on the same data purely as a function of `C`) and that it
  is computed on a pool where wordings are shared, so it cannot adjudicate a
  phrasing-invariance claim. It does discriminate contextual from
  embedding-layer targets well (r = +0.92 with best-layer depth).

## What is genuinely new

1. **Binding decay over narrative distance**, and the presence-outlives-binding
   dissociation, measured two independent ways (probe transfer and direction
   geometry).
2. **A multi-entity, multi-attribute, order-destructive text domain** with
   exact per-position labels, distractor sentences that mention values without
   applying them, and train/test splits that share *no surface forms*. A
   harder generalisation test than narrative-level splits.
3. **Convergence of four independent measures** (best-layer depth, prequential
   MDL, shuffled-order sensitivity, selectivity) on the same partition of
   targets, with r = +0.92 between depth and MDL compression. Nothing about
   that agreement was built in.

## What is extending, not inventing

Linear probing of emergent state (Othello lineage), binding as a probing
question (propositional probes), activation patching for causal validation
(standard practice), control tasks and MDL (standard critiques). The domain is
new; the machinery mostly is not.

## Honest limits

- One 124M model. The pythia-410m replication is the test of whether any of
  this is gpt2-specific.
- Synthetic text from templates. Paraphrase-disjoint splits make it a real
  generalisation test, but it is not natural prose.
- Most targets do not beat a lexical baseline. The positive results are
  concentrated in `finish` and `coverage`, and in the near-distance regime.
- The behavioural intervention read-out is weak at this scale; the strong
  causal evidence is the paired log-odds shift and the probe read-out, not the
  model's own text.

## Sources

- [Emergent World Representations: Exploring a Sequence Model Trained on a Synthetic Task](https://arxiv.org/abs/2210.13382) (Li et al., ICLR 2023); code: [likenneth/othello_world](https://github.com/likenneth/othello_world)
- [Emergent Linear Representations in World Models of Self-Supervised Sequence Models](https://arxiv.org/abs/2309.00941) (Nanda et al., BlackboxNLP 2023)
- [Monitoring Latent World States in Language Models with Propositional Probes](https://arxiv.org/html/2406.19501v1) (Feng, Russell, Steinhardt, ICLR 2025)
- [Entity Tracking in Language Models](https://aclanthology.org/2023.acl-long.213/) (Kim and Schuster, ACL 2023)
