# Results: `EleutherAI/pythia-410m` on the `pilot` dataset

> **Status note.** This file is auto-generated per dataset/model and
> reports that run in isolation. The canonical, cross-model statement of
> what the project shows is [`FINDINGS.md`](FINDINGS.md); claims that did
> not replicate across models, and the thirteen revisions made along the
> way, are recorded in [`CORRECTIONS.md`](CORRECTIONS.md). In particular:
> the causal intervention here demonstrates an attribute-**value** code,
> not entity-specific binding (see CORRECTIONS #10), and the pattern of
> which targets beat a lexical baseline is model-specific (#12).

## Dataset

- 2000 narratives, 13097 labelled positions (6.55 sentences each)
- action mix: APPLY 7439, DISTRACTOR 2364, NOOP 1131, SOFTEN 412, BLOT 405, GLOSS 380, BUILD 379, REMOVE 297, SHEER 290
- fraction of positions where the value was overridden earlier (first mention is the wrong answer): eyes.color 0.23, eyes.finish 0.27, lips.color 0.23, lips.finish 0.27

## Probing

All numbers are test-split accuracy on narratives built from wordings never seen in training.

| target | classes | majority | best layer | linear | MLP | static emb | bag-of-ngrams | control task (test) | control task (train) | selectivity | shuffled order |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `eyes.color` | 5 | 0.179 | 24 | **0.334** | 0.368 | 0.303 | 0.255 | 0.209 | 0.899 | 0.125 | 0.379 |
| `eyes.finish` | 4 | 0.261 | 24 | **0.385** | 0.405 | 0.371 | 0.260 | 0.237 | 0.875 | 0.148 | 0.374 |
| `lips.color` | 5 | 0.201 | 20 | **0.318** | 0.337 | 0.254 | 0.322 | 0.204 | 0.873 | 0.114 | 0.334 |
| `lips.finish` | 4 | 0.267 | 24 | **0.369** | 0.462 | 0.346 | 0.393 | 0.267 | 0.878 | 0.103 | 0.392 |

### Positions where the state was overridden

The subset where the first value mentioned for that region is *not* the answer. A keyword heuristic cannot score above chance here.

| target | overridden positions | accuracy there | accuracy elsewhere |
|---|---|---|---|
| `eyes.color` | 179 | 0.235 | 0.361 |
| `eyes.finish` | 252 | 0.357 | 0.397 |
| `lips.color` | 279 | 0.265 | 0.345 |
| `lips.finish` | 340 | 0.321 | 0.403 |

### Binding check

A probe trained on one region, read against the *other* region's label. Near the other region's majority rate means the regions are kept apart.

| probe trained on | read against | accuracy | that region's majority |
|---|---|---|---|
| `eyes.color` | `lips.color` | 0.254 | 0.201 |
| `eyes.finish` | `lips.finish` | 0.304 | 0.267 |
| `lips.color` | `eyes.color` | 0.262 | 0.179 |
| `lips.finish` | `eyes.finish` | 0.329 | 0.261 |

### MDL (prequential code length)

Lower is better; compression is relative to the uniform code.

| target | layer | code length (kbit) | compression | static emb compression |
|---|---|---|---|---|
| `eyes.color` | 0 | 9.0 | 0.98x | 1.45x |
| `eyes.color` | 24 | 6.7 | 1.31x | 1.45x |
| `eyes.finish` | 0 | 7.6 | 1.00x | 1.62x |
| `eyes.finish` | 24 | 5.6 | 1.35x | 1.62x |
| `lips.color` | 0 | 8.8 | 1.00x | 1.54x |
| `lips.color` | 20 | 6.5 | 1.35x | 1.54x |
| `lips.color` | 24 | 6.4 | 1.38x | 1.54x |
| `lips.finish` | 0 | 7.6 | 1.00x | 1.63x |
| `lips.finish` | 24 | 5.6 | 1.36x | 1.63x |

## Representation geometry

Layer 24, 839 held-out positions. PCA-3 explains 33.3% of variance.

### Same attribute value, different face region

Cosine between the two regions' concept directions. Two ways of computing the direction are shown because they disagree, and the disagreement is the finding: the difference-of-means CAV is dominated by the shared "this colour is present somewhere" component, while the discriminative probe weights project that shared component out and point the two regions apart.

| pair | CAV cosine | probe-weight cosine |
|---|---|---|
| eyes.color=bare | lips.color=bare | 0.677 | -0.179 |
| eyes.color=brown | lips.color=brown | 0.498 | -0.093 |
| eyes.color=nude | lips.color=nude | 0.519 | -0.165 |
| eyes.color=pink | lips.color=pink | 0.475 | -0.095 |
| eyes.color=red | lips.color=red | 0.621 | -0.064 |
| eyes.finish=dewy | lips.finish=dewy | 0.486 | -0.131 |
| eyes.finish=matte | lips.finish=matte | 0.576 | -0.153 |
| eyes.finish=none | lips.finish=none | 0.677 | -0.179 |
| eyes.finish=satin | lips.finish=satin | 0.636 | -0.160 |

### Subspace principal angles

| pair | mean angle (deg) |
|---|---|
| eyes.color | eyes.finish | 49.0 |
| eyes.color | lips.color | 56.16 |
| eyes.color | lips.finish | 63.82 |
| eyes.finish | lips.color | 64.3 |
| eyes.finish | lips.finish | 60.88 |
| lips.color | lips.finish | 51.74 |

### Dimensionality

| target | 1 | 2 | 3 | 5 | 8 | 16 | 32 | 64 | 128 | 256 |
|---|---|---|---|---|---|---|---|---|---|---|
| `eyes.color` | 0.333 | 0.321 | 0.329 | 0.361 | 0.425 | 0.484 | 0.520 | 0.607 | 0.595 | 0.667 |
| `eyes.finish` | 0.385 | 0.417 | 0.444 | 0.425 | 0.417 | 0.464 | 0.520 | 0.579 | 0.643 | 0.710 |
| `lips.color` | 0.246 | 0.238 | 0.238 | 0.274 | 0.302 | 0.381 | 0.468 | 0.512 | 0.571 | 0.615 |
| `lips.finish` | 0.226 | 0.238 | 0.246 | 0.313 | 0.306 | 0.429 | 0.512 | 0.583 | 0.627 | 0.690 |

## State persistence

Accuracy binned by how many sentences have passed since that variable last changed. A representation that only reflects the current sentence should collapse towards the majority rate as the last update recedes; one that maintains state should stay flat.

| target | layer | distance | n | accuracy | majority in bin |
|---|---|---|---|---|---|
| `eyes.color` | 24 | 0 | 198 | 0.288 | 0.222 |
| `eyes.color` | 24 | 1 | 253 | 0.423 | 0.150 |
| `eyes.color` | 24 | 2 | 169 | 0.331 | 0.172 |
| `eyes.color` | 24 | 3-4 | 166 | 0.277 | 0.199 |
| `eyes.color` | 24 | 5-7 | 53 | 0.264 | 0.113 |
| `eyes.finish` | 24 | 0 | 243 | 0.329 | 0.321 |
| `eyes.finish` | 24 | 1 | 275 | 0.411 | 0.233 |
| `eyes.finish` | 24 | 2 | 163 | 0.417 | 0.252 |
| `eyes.finish` | 24 | 3-4 | 130 | 0.462 | 0.262 |
| `eyes.finish` | 24 | 5-7 | 28 | 0.071 | 0.071 |
| `lips.color` | 20 | 0 | 230 | 0.291 | 0.074 |
| `lips.color` | 20 | 1 | 250 | 0.372 | 0.264 |
| `lips.color` | 20 | 2 | 164 | 0.311 | 0.238 |
| `lips.color` | 20 | 3-4 | 152 | 0.296 | 0.237 |
| `lips.color` | 20 | 5-7 | 43 | 0.256 | 0.256 |
| `lips.finish` | 24 | 0 | 269 | 0.390 | 0.320 |
| `lips.finish` | 24 | 1 | 246 | 0.402 | 0.256 |
| `lips.finish` | 24 | 2 | 156 | 0.327 | 0.256 |
| `lips.finish` | 24 | 3-4 | 138 | 0.341 | 0.225 |
| `lips.finish` | 24 | 5-7 | 30 | 0.267 | 0.133 |

## Causal intervention

### `lips.color` steered at layer 20 (120 held-out positions)

Mean hidden-state norm at the edited layer: 52.7. Probe read-out trained at layer 24 on clean data.

| alpha | probe: true | probe: steered | probe: steered (trials initially correct) | other region correct | other region changed | behaviour: true | behaviour: delta log-odds steered | behaviour: delta log-odds other |
|---|---|---|---|---|---|---|---|---|
| 0.0 | 0.200 | 0.300 | 0.000 | 0.233 | 0.000 | 0.300 | +0.000 | +0.000 |
| 0.5 | 0.108 | 0.400 | 0.292 | 0.225 | 0.133 | 0.300 | +0.066 | +0.001 |
| 1.0 | 0.042 | 0.517 | 0.583 | 0.225 | 0.208 | 0.300 | +0.132 | +0.002 |
| 2.0 | 0.000 | 0.775 | 0.792 | 0.208 | 0.317 | 0.300 | +0.261 | +0.002 |
| 4.0 | 0.000 | 0.992 | 1.000 | 0.192 | 0.508 | 0.292 | +0.515 | -0.012 |
| 8.0 | 0.000 | 1.000 | 1.000 | 0.183 | 0.608 | 0.292 | +0.997 | -0.066 |

### `lips.finish` steered at layer 24 (120 held-out positions)

Mean hidden-state norm at the edited layer: 105.7. Probe read-out trained at layer 24 on clean data.

| alpha | probe: true | probe: steered | probe: steered (trials initially correct) | other region correct | other region changed | behaviour: true | behaviour: delta log-odds steered | behaviour: delta log-odds other |
|---|---|---|---|---|---|---|---|---|
| 0.0 | 0.358 | 0.300 | 0.000 | 0.325 | 0.000 | 0.350 | +0.000 | +0.000 |
| 0.5 | 0.142 | 0.625 | 0.488 | 0.333 | 0.100 | 0.308 | +0.106 | -0.003 |
| 1.0 | 0.075 | 0.775 | 0.698 | 0.300 | 0.250 | 0.242 | +0.210 | -0.007 |
| 2.0 | 0.000 | 0.942 | 0.953 | 0.350 | 0.392 | 0.183 | +0.412 | -0.025 |
| 4.0 | 0.000 | 1.000 | 1.000 | 0.333 | 0.550 | 0.067 | +0.792 | -0.086 |
| 8.0 | 0.000 | 1.000 | 1.000 | 0.358 | 0.683 | 0.000 | +1.438 | -0.288 |


## Causal intervention: effect sizes and significance

Paired bootstrap over trials (10,000 resamples). The binding claim is the last column: the edit must move the target region's read-out more than the other region's. p-values are two-sided throughout this report; the one-sided values are kept in the result JSON as `p_one_sided_greater`.

### `lips.color` (edit L20 -> read L24, n=120)

| alpha | delta log-odds, target | delta log-odds, other region | target - other | p (two-sided) |
|---|---|---|---|---|
| 0.5 | +0.066 [+0.062, +0.070] | +0.001 [-0.009, +0.011] | +0.065 [+0.055, +0.074] | 0.0000 |
| 1.0 | +0.132 [+0.123, +0.141] | +0.002 [-0.018, +0.021] | +0.130 [+0.111, +0.149] | 0.0000 |
| 2.0 | +0.261 [+0.245, +0.278] | +0.002 [-0.037, +0.040] | +0.260 [+0.223, +0.297] | 0.0000 |
| 4.0 | +0.515 [+0.485, +0.547] | -0.012 [-0.089, +0.065] | +0.527 [+0.455, +0.598] | 0.0000 |
| 8.0 | +0.997 [+0.939, +1.058] | -0.066 [-0.221, +0.087] | +1.063 [+0.915, +1.210] | 0.0000 |

Probe read-out flips to the steered value on 1.000 [1.000, 1.000] of the 24 trials whose unintervened read-out was already correct. The small n is the limitation: the read-out probe sits downstream of the edit, so it is weaker than the best-layer probe.

### `lips.finish` (edit L24 -> read L24, n=120)

| alpha | delta log-odds, target | delta log-odds, other region | target - other | p (two-sided) |
|---|---|---|---|---|
| 0.5 | +0.106 [+0.098, +0.113] | -0.003 [-0.022, +0.015] | +0.109 [+0.091, +0.127] | 0.0000 |
| 1.0 | +0.210 [+0.195, +0.224] | -0.007 [-0.045, +0.030] | +0.217 [+0.181, +0.255] | 0.0000 |
| 2.0 | +0.412 [+0.384, +0.440] | -0.025 [-0.099, +0.047] | +0.437 [+0.365, +0.512] | 0.0000 |
| 4.0 | +0.792 [+0.736, +0.846] | -0.086 [-0.231, +0.056] | +0.878 [+0.734, +1.025] | 0.0000 |
| 8.0 | +1.438 [+1.335, +1.540] | -0.288 [-0.560, -0.020] | +1.727 [+1.453, +2.005] | 0.0000 |

Probe read-out flips to the steered value on 1.000 [1.000, 1.000] of the 43 trials whose unintervened read-out was already correct. The small n is the limitation: the read-out probe sits downstream of the edit, so it is weaker than the best-layer probe.

