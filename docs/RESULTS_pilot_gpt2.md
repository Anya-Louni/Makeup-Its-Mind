# Results: `gpt2` on the `pilot` dataset

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
| `eyes.color` | 5 | 0.179 | 11 | **0.348** | 0.354 | 0.287 | 0.255 | 0.261 | 0.766 | 0.087 | 0.314 |
| `eyes.finish` | 4 | 0.261 | 1 | **0.435** | 0.451 | 0.336 | 0.260 | 0.238 | 0.580 | 0.197 | 0.415 |
| `lips.color` | 5 | 0.201 | 12 | **0.308** | 0.318 | 0.349 | 0.322 | 0.201 | 0.786 | 0.106 | 0.329 |
| `lips.finish` | 4 | 0.267 | 1 | **0.359** | 0.435 | 0.414 | 0.393 | 0.266 | 0.573 | 0.093 | 0.316 |

### Positions where the state was overridden

The subset where the first value mentioned for that region is *not* the answer. A keyword heuristic cannot score above chance here.

| target | overridden positions | accuracy there | accuracy elsewhere |
|---|---|---|---|
| `eyes.color` | 179 | 0.302 | 0.361 |
| `eyes.finish` | 252 | 0.381 | 0.458 |
| `lips.color` | 279 | 0.247 | 0.338 |
| `lips.finish` | 340 | 0.359 | 0.359 |

### Binding check

A probe trained on one region, read against the *other* region's label. Near the other region's majority rate means the regions are kept apart.

| probe trained on | read against | accuracy | that region's majority |
|---|---|---|---|
| `eyes.color` | `lips.color` | 0.250 | 0.201 |
| `eyes.finish` | `lips.finish` | 0.344 | 0.267 |
| `lips.color` | `eyes.color` | 0.274 | 0.179 |
| `lips.finish` | `eyes.finish` | 0.325 | 0.261 |

### MDL (prequential code length)

Lower is better; compression is relative to the uniform code.

| target | layer | code length (kbit) | compression | static emb compression |
|---|---|---|---|---|
| `eyes.color` | 0 | 8.9 | 0.99x | 1.48x |
| `eyes.color` | 11 | 7.1 | 1.24x | 1.48x |
| `eyes.color` | 12 | 6.9 | 1.28x | 1.48x |
| `eyes.finish` | 0 | 7.4 | 1.02x | 1.63x |
| `eyes.finish` | 1 | 5.3 | 1.43x | 1.63x |
| `eyes.finish` | 12 | 5.8 | 1.30x | 1.63x |
| `lips.color` | 0 | 8.7 | 1.01x | 1.58x |
| `lips.color` | 12 | 6.6 | 1.34x | 1.58x |
| `lips.finish` | 0 | 7.4 | 1.03x | 1.66x |
| `lips.finish` | 1 | 5.3 | 1.44x | 1.66x |
| `lips.finish` | 12 | 5.9 | 1.28x | 1.66x |

## Representation geometry

Layer 6, 839 held-out positions. PCA-3 explains 40.9% of variance.

### Same attribute value, different face region

Cosine between the two regions' concept directions. Two ways of computing the direction are shown because they disagree, and the disagreement is the finding: the difference-of-means CAV is dominated by the shared "this colour is present somewhere" component, while the discriminative probe weights project that shared component out and point the two regions apart.

| pair | CAV cosine | probe-weight cosine |
|---|---|---|
| eyes.color=bare | lips.color=bare | 0.931 | -0.301 |
| eyes.color=brown | lips.color=brown | 0.646 | -0.036 |
| eyes.color=nude | lips.color=nude | 0.545 | -0.167 |
| eyes.color=pink | lips.color=pink | 0.702 | -0.041 |
| eyes.color=red | lips.color=red | 0.693 | -0.042 |
| eyes.finish=dewy | lips.finish=dewy | 0.808 | -0.237 |
| eyes.finish=matte | lips.finish=matte | 0.795 | -0.222 |
| eyes.finish=none | lips.finish=none | 0.931 | -0.301 |
| eyes.finish=satin | lips.finish=satin | 0.894 | -0.209 |

### Subspace principal angles

| pair | mean angle (deg) |
|---|---|
| eyes.color | eyes.finish | 47.95 |
| eyes.color | lips.color | 51.81 |
| eyes.color | lips.finish | 60.16 |
| eyes.finish | lips.color | 56.64 |
| eyes.finish | lips.finish | 42.73 |
| lips.color | lips.finish | 53.96 |

### Dimensionality

| target | 1 | 2 | 3 | 5 | 8 | 16 | 32 | 64 | 128 | 256 |
|---|---|---|---|---|---|---|---|---|---|---|
| `eyes.color` | 0.325 | 0.341 | 0.341 | 0.302 | 0.345 | 0.361 | 0.401 | 0.512 | 0.496 | 0.579 |
| `eyes.finish` | 0.333 | 0.349 | 0.325 | 0.444 | 0.480 | 0.528 | 0.591 | 0.587 | 0.595 | 0.651 |
| `lips.color` | 0.266 | 0.266 | 0.266 | 0.234 | 0.258 | 0.325 | 0.357 | 0.409 | 0.563 | 0.567 |
| `lips.finish` | 0.254 | 0.242 | 0.250 | 0.401 | 0.460 | 0.512 | 0.532 | 0.571 | 0.611 | 0.627 |

## State persistence

Accuracy binned by how many sentences have passed since that variable last changed. A representation that only reflects the current sentence should collapse towards the majority rate as the last update recedes; one that maintains state should stay flat.

| target | layer | distance | n | accuracy | majority in bin |
|---|---|---|---|---|---|
| `eyes.color` | 11 | 0 | 198 | 0.288 | 0.222 |
| `eyes.color` | 11 | 1 | 253 | 0.419 | 0.150 |
| `eyes.color` | 11 | 2 | 169 | 0.385 | 0.172 |
| `eyes.color` | 11 | 3-4 | 166 | 0.283 | 0.199 |
| `eyes.color` | 11 | 5-7 | 53 | 0.321 | 0.113 |
| `eyes.finish` | 1 | 0 | 243 | 0.444 | 0.321 |
| `eyes.finish` | 1 | 1 | 275 | 0.444 | 0.233 |
| `eyes.finish` | 1 | 2 | 163 | 0.442 | 0.252 |
| `eyes.finish` | 1 | 3-4 | 130 | 0.408 | 0.262 |
| `eyes.finish` | 1 | 5-7 | 28 | 0.357 | 0.071 |
| `lips.color` | 12 | 0 | 230 | 0.257 | 0.074 |
| `lips.color` | 12 | 1 | 250 | 0.352 | 0.264 |
| `lips.color` | 12 | 2 | 164 | 0.305 | 0.238 |
| `lips.color` | 12 | 3-4 | 152 | 0.316 | 0.237 |
| `lips.color` | 12 | 5-7 | 43 | 0.302 | 0.256 |
| `lips.finish` | 1 | 0 | 269 | 0.387 | 0.320 |
| `lips.finish` | 1 | 1 | 246 | 0.358 | 0.256 |
| `lips.finish` | 1 | 2 | 156 | 0.321 | 0.256 |
| `lips.finish` | 1 | 3-4 | 138 | 0.377 | 0.225 |
| `lips.finish` | 1 | 5-7 | 30 | 0.233 | 0.133 |

## Probe-capacity ablation

| target | probe | real | control | selectivity | MDL (kbit) |
|---|---|---|---|---|---|
| `eyes.color` | linear | 0.348 | 0.156 | +0.192 | 11.7 |
| `eyes.color` | MLP-8 | 0.316 | 0.155 | +0.161 | 8.1 |
| `eyes.color` | MLP-32 | 0.317 | 0.188 | +0.129 | 8.4 |
| `eyes.color` | MLP-128 | 0.338 | 0.195 | +0.143 | 8.0 |
| `eyes.color` | MLP-512 | 0.361 | 0.173 | +0.188 | 9.1 |
| `eyes.finish` | linear | 0.435 | 0.251 | +0.184 | 7.2 |
| `eyes.finish` | MLP-8 | 0.451 | 0.251 | +0.199 | 6.1 |
| `eyes.finish` | MLP-32 | 0.414 | 0.246 | +0.168 | 6.7 |
| `eyes.finish` | MLP-128 | 0.406 | 0.212 | +0.194 | 6.5 |
| `eyes.finish` | MLP-512 | 0.423 | 0.203 | +0.221 | 6.8 |
| `lips.color` | linear | 0.308 | 0.178 | +0.130 | 11.4 |
| `lips.color` | MLP-8 | 0.298 | 0.259 | +0.039 | 7.7 |
| `lips.color` | MLP-32 | 0.291 | 0.225 | +0.066 | 7.4 |
| `lips.color` | MLP-128 | 0.305 | 0.218 | +0.087 | 7.6 |
| `lips.color` | MLP-512 | 0.311 | 0.216 | +0.095 | 8.0 |
| `lips.finish` | linear | 0.359 | 0.281 | +0.077 | 7.0 |
| `lips.finish` | MLP-8 | 0.386 | 0.282 | +0.104 | 6.1 |
| `lips.finish` | MLP-32 | 0.400 | 0.257 | +0.143 | 6.0 |
| `lips.finish` | MLP-128 | 0.373 | 0.263 | +0.110 | 6.3 |
| `lips.finish` | MLP-512 | 0.434 | 0.275 | +0.159 | 7.1 |

## Causal intervention

### `lips.color` steered at layer 12 (120 held-out positions)

Mean hidden-state norm at the edited layer: 277.5. Probe read-out trained at layer 12 on clean data.

| alpha | probe: true | probe: steered | probe: steered (trials initially correct) | other region correct | other region changed | behaviour: true | behaviour: delta log-odds steered | behaviour: delta log-odds other |
|---|---|---|---|---|---|---|---|---|
| 0.0 | 0.208 | 0.217 | 0.000 | 0.250 | 0.000 | 0.300 | +0.000 | +0.000 |
| 0.5 | 0.208 | 0.217 | 0.000 | 0.250 | 0.008 | 0.300 | +0.003 | +0.001 |
| 1.0 | 0.192 | 0.242 | 0.080 | 0.250 | 0.008 | 0.300 | +0.005 | +0.002 |
| 2.0 | 0.175 | 0.267 | 0.080 | 0.267 | 0.025 | 0.300 | +0.011 | +0.004 |
| 4.0 | 0.133 | 0.300 | 0.120 | 0.275 | 0.033 | 0.300 | +0.022 | +0.008 |
| 8.0 | 0.100 | 0.392 | 0.280 | 0.250 | 0.058 | 0.300 | +0.043 | +0.016 |

### `lips.finish` steered at layer 1 (120 held-out positions)

Mean hidden-state norm at the edited layer: 47.9. Probe read-out trained at layer 12 on clean data.

| alpha | probe: true | probe: steered | probe: steered (trials initially correct) | other region correct | other region changed | behaviour: true | behaviour: delta log-odds steered | behaviour: delta log-odds other |
|---|---|---|---|---|---|---|---|---|
| 0.0 | 0.242 | 0.258 | 0.000 | 0.258 | 0.000 | 0.317 | +0.000 | +0.000 |
| 0.5 | 0.217 | 0.275 | 0.034 | 0.258 | 0.000 | 0.317 | +0.031 | -0.000 |
| 1.0 | 0.200 | 0.283 | 0.069 | 0.233 | 0.042 | 0.317 | +0.063 | +0.000 |
| 2.0 | 0.175 | 0.308 | 0.138 | 0.242 | 0.067 | 0.317 | +0.130 | +0.003 |
| 4.0 | 0.183 | 0.342 | 0.172 | 0.225 | 0.108 | 0.308 | +0.276 | +0.015 |
| 8.0 | 0.142 | 0.383 | 0.276 | 0.233 | 0.217 | 0.300 | +0.591 | +0.066 |


## Causal intervention: effect sizes and significance

Paired bootstrap over trials (10,000 resamples). The binding claim is the last column: the edit must move the target region's read-out more than the other region's.

### `lips.color` (edit L12 -> read L12, n=120)

| alpha | delta log-odds, target | delta log-odds, other region | target - other | bootstrap p |
|---|---|---|---|---|
| 0.5 | +0.003 [+0.002, +0.003] | +0.001 [+0.000, +0.002] | +0.002 [+0.001, +0.003] | 0.0024 |
| 1.0 | +0.005 [+0.004, +0.007] | +0.002 [+0.000, +0.004] | +0.003 [+0.001, +0.006] | 0.0018 |
| 2.0 | +0.011 [+0.008, +0.014] | +0.004 [+0.000, +0.008] | +0.007 [+0.002, +0.011] | 0.0018 |
| 4.0 | +0.022 [+0.016, +0.027] | +0.008 [+0.001, +0.015] | +0.013 [+0.005, +0.022] | 0.0018 |
| 8.0 | +0.043 [+0.031, +0.055] | +0.016 [+0.001, +0.031] | +0.027 [+0.009, +0.045] | 0.0020 |

Probe read-out flips to the steered value on 0.280 [0.120, 0.480] of the 25 trials whose unintervened read-out was already correct. The small n is the limitation: the read-out probe sits downstream of the edit, so it is weaker than the best-layer probe.

### `lips.finish` (edit L1 -> read L12, n=120)

| alpha | delta log-odds, target | delta log-odds, other region | target - other | bootstrap p |
|---|---|---|---|---|
| 0.5 | +0.031 [+0.027, +0.035] | -0.000 [-0.008, +0.007] | +0.031 [+0.023, +0.039] | 0.0000 |
| 1.0 | +0.063 [+0.055, +0.071] | +0.000 [-0.015, +0.015] | +0.062 [+0.046, +0.079] | 0.0000 |
| 2.0 | +0.130 [+0.115, +0.146] | +0.003 [-0.028, +0.032] | +0.127 [+0.095, +0.161] | 0.0000 |
| 4.0 | +0.276 [+0.248, +0.305] | +0.015 [-0.045, +0.074] | +0.261 [+0.196, +0.329] | 0.0000 |
| 8.0 | +0.591 [+0.541, +0.642] | +0.066 [-0.050, +0.178] | +0.525 [+0.400, +0.655] | 0.0000 |

Probe read-out flips to the steered value on 0.276 [0.103, 0.448] of the 29 trials whose unintervened read-out was already correct. The small n is the limitation: the read-out probe sits downstream of the edit, so it is weaker than the best-layer probe.

