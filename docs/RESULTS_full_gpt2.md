# Results: `gpt2` on the `full` dataset

> **Status note.** This file is auto-generated per dataset/model and
> reports that run in isolation. The canonical, cross-model statement of
> what the project shows is [`FINDINGS.md`](FINDINGS.md); claims that did
> not replicate across models, and the thirteen revisions made along the
> way, are recorded in [`CORRECTIONS.md`](CORRECTIONS.md). In particular:
> the causal intervention here demonstrates an attribute-**value** code,
> not entity-specific binding (see CORRECTIONS #10), and the pattern of
> which targets beat a lexical baseline is model-specific (#12).

## Dataset

- 3600 narratives, 47040 labelled positions (13.07 sentences each)
- action mix: APPLY 25498, DISTRACTOR 7704, NOOP 3648, BLOT 1948, SOFTEN 1888, BUILD 1839, GLOSS 1839, SHEER 1466, REMOVE 1210
- fraction of positions where the value was overridden earlier (first mention is the wrong answer): cheeks.coverage 0.19, cheeks.finish 0.26, eyes.coverage 0.19, eyes.finish 0.26, cheeks.color 0.20, eyes.color 0.21, lips.finish 0.25, skin.coverage 0.19, skin.finish 0.25, lips.color 0.20, lips.coverage 0.18, skin.color 0.20

## Probing

All numbers are test-split accuracy on narratives built from wordings never seen in training.

| target | classes | majority | best layer | linear | MLP | static emb | bag-of-ngrams | control task (test) | control task (train) | selectivity | shuffled order |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `cheeks.color` | 5 | 0.314 | 0 | **0.351** | 0.348 | 0.321 | 0.314 | 0.333 | 0.364 | 0.019 | 0.342 |
| `cheeks.coverage` | 3 | 0.350 | 0 | **0.475** | 0.473 | 0.356 | 0.314 | 0.360 | 0.440 | 0.115 | 0.463 |
| `cheeks.finish` | 4 | 0.314 | 0 | **0.394** | 0.377 | 0.329 | 0.314 | 0.298 | 0.392 | 0.096 | 0.368 |
| `eyes.color` | 5 | 0.317 | 0 | **0.357** | 0.356 | 0.367 | 0.317 | 0.358 | 0.407 | -0.002 | 0.362 |
| `eyes.coverage` | 3 | 0.382 | 12 | **0.498** | 0.483 | 0.519 | 0.318 | 0.342 | 0.616 | 0.156 | 0.478 |
| `eyes.finish` | 4 | 0.317 | 2 | **0.418** | 0.404 | 0.417 | 0.318 | 0.259 | 0.519 | 0.158 | 0.364 |
| `lips.color` | 5 | 0.306 | 0 | **0.345** | 0.347 | 0.376 | 0.307 | 0.335 | 0.393 | 0.011 | 0.350 |
| `lips.coverage` | 3 | 0.306 | 0 | **0.459** | 0.459 | 0.443 | 0.340 | 0.328 | 0.437 | 0.131 | 0.469 |
| `lips.finish` | 4 | 0.306 | 0 | **0.374** | 0.373 | 0.404 | 0.317 | 0.330 | 0.423 | 0.044 | 0.353 |
| `skin.color` | 5 | 0.346 | 0 | **0.359** | 0.361 | 0.346 | 0.348 | 0.322 | 0.358 | 0.037 | 0.359 |
| `skin.coverage` | 3 | 0.333 | 10 | **0.471** | 0.437 | 0.495 | 0.382 | 0.329 | 0.581 | 0.142 | 0.451 |
| `skin.finish` | 4 | 0.346 | 6 | **0.388** | 0.413 | 0.426 | 0.355 | 0.243 | 0.527 | 0.145 | 0.339 |

### Positions where the state was overridden

The subset where the first value mentioned for that region is *not* the answer. A keyword heuristic cannot score above chance here.

| target | overridden positions | accuracy there | accuracy elsewhere |
|---|---|---|---|
| `cheeks.color` | 642 | 0.227 | 0.382 |
| `cheeks.coverage` | 638 | 0.379 | 0.498 |
| `cheeks.finish` | 867 | 0.279 | 0.436 |
| `eyes.color` | 688 | 0.201 | 0.399 |
| `eyes.coverage` | 672 | 0.432 | 0.516 |
| `eyes.finish` | 878 | 0.317 | 0.455 |
| `lips.color` | 784 | 0.202 | 0.391 |
| `lips.coverage` | 632 | 0.397 | 0.474 |
| `lips.finish` | 875 | 0.241 | 0.424 |
| `skin.color` | 831 | 0.230 | 0.403 |
| `skin.coverage` | 749 | 0.439 | 0.480 |
| `skin.finish` | 1034 | 0.284 | 0.436 |

### Binding check

A probe trained on one region, read against the *other* region's label. Near the other region's majority rate means the regions are kept apart.

| probe trained on | read against | accuracy | that region's majority |
|---|---|---|---|
| `cheeks.color` | `eyes.color` | 0.347 | 0.317 |
| `cheeks.color` | `lips.color` | 0.344 | 0.306 |
| `cheeks.color` | `skin.color` | 0.353 | 0.346 |
| `cheeks.coverage` | `eyes.coverage` | 0.485 | 0.382 |
| `cheeks.coverage` | `lips.coverage` | 0.455 | 0.306 |
| `cheeks.coverage` | `skin.coverage` | 0.455 | 0.333 |
| `cheeks.finish` | `eyes.finish` | 0.408 | 0.317 |
| `cheeks.finish` | `lips.finish` | 0.378 | 0.306 |
| `cheeks.finish` | `skin.finish` | 0.391 | 0.346 |
| `eyes.color` | `cheeks.color` | 0.359 | 0.314 |
| `eyes.color` | `lips.color` | 0.339 | 0.306 |
| `eyes.color` | `skin.color` | 0.348 | 0.346 |
| `eyes.coverage` | `cheeks.coverage` | 0.401 | 0.350 |
| `eyes.coverage` | `lips.coverage` | 0.421 | 0.306 |
| `eyes.coverage` | `skin.coverage` | 0.388 | 0.333 |
| `eyes.finish` | `cheeks.finish` | 0.350 | 0.314 |
| `eyes.finish` | `lips.finish` | 0.356 | 0.306 |
| `eyes.finish` | `skin.finish` | 0.361 | 0.346 |
| `lips.color` | `cheeks.color` | 0.352 | 0.314 |
| `lips.color` | `eyes.color` | 0.359 | 0.317 |
| `lips.color` | `skin.color` | 0.343 | 0.346 |
| `lips.coverage` | `cheeks.coverage` | 0.471 | 0.350 |
| `lips.coverage` | `eyes.coverage` | 0.485 | 0.382 |
| `lips.coverage` | `skin.coverage` | 0.465 | 0.333 |
| `lips.finish` | `cheeks.finish` | 0.388 | 0.314 |
| `lips.finish` | `eyes.finish` | 0.386 | 0.317 |
| `lips.finish` | `skin.finish` | 0.390 | 0.346 |
| `skin.color` | `cheeks.color` | 0.352 | 0.314 |
| `skin.color` | `eyes.color` | 0.348 | 0.317 |
| `skin.color` | `lips.color` | 0.340 | 0.306 |
| `skin.coverage` | `cheeks.coverage` | 0.404 | 0.350 |
| `skin.coverage` | `eyes.coverage` | 0.424 | 0.382 |
| `skin.coverage` | `lips.coverage` | 0.408 | 0.306 |
| `skin.finish` | `cheeks.finish` | 0.315 | 0.314 |
| `skin.finish` | `eyes.finish` | 0.305 | 0.317 |
| `skin.finish` | `lips.finish` | 0.310 | 0.306 |

### MDL (prequential code length)

Lower is better; compression is relative to the uniform code.

| target | layer | code length (kbit) | compression | static emb compression |
|---|---|---|---|---|
| `cheeks.color` | 0 | 17.0 | 1.07x | 1.48x |
| `cheeks.color` | 12 | 14.1 | 1.29x | 1.48x |
| `cheeks.coverage` | 0 | 11.6 | 1.07x | 1.61x |
| `cheeks.coverage` | 12 | 9.4 | 1.32x | 1.61x |
| `cheeks.finish` | 0 | 14.8 | 1.05x | 1.54x |
| `cheeks.finish` | 12 | 12.5 | 1.25x | 1.54x |
| `eyes.color` | 0 | 17.3 | 1.05x | 1.47x |
| `eyes.color` | 12 | 14.2 | 1.28x | 1.47x |
| `eyes.coverage` | 0 | 11.7 | 1.06x | 1.59x |
| `eyes.coverage` | 12 | 9.1 | 1.35x | 1.59x |
| `eyes.finish` | 0 | 15.0 | 1.04x | 1.52x |
| `eyes.finish` | 2 | 12.5 | 1.25x | 1.52x |
| `eyes.finish` | 12 | 12.2 | 1.28x | 1.52x |
| `lips.color` | 0 | 17.1 | 1.06x | 1.54x |
| `lips.color` | 12 | 13.9 | 1.30x | 1.54x |
| `lips.coverage` | 0 | 11.6 | 1.06x | 1.62x |
| `lips.coverage` | 12 | 9.4 | 1.32x | 1.62x |
| `lips.finish` | 0 | 14.8 | 1.06x | 1.57x |
| `lips.finish` | 12 | 12.4 | 1.26x | 1.57x |
| `skin.color` | 0 | 17.1 | 1.06x | 1.51x |
| `skin.color` | 12 | 14.1 | 1.29x | 1.51x |
| `skin.coverage` | 0 | 11.5 | 1.08x | 1.68x |
| `skin.coverage` | 10 | 9.6 | 1.29x | 1.68x |
| `skin.coverage` | 12 | 9.2 | 1.35x | 1.68x |
| `skin.finish` | 0 | 14.8 | 1.06x | 1.58x |
| `skin.finish` | 6 | 13.1 | 1.19x | 1.58x |
| `skin.finish` | 12 | 12.4 | 1.26x | 1.58x |

## Binding, stratified by distance

The aggregate binding test (probe trained on one region, read against another) is dominated by positions far from the last update, where the attribute is not decodable at all and both probes sit at majority. Stratifying by distance since the variable last changed shows what the aggregate hides.

Binding margin = (within-entity accuracy - its majority) - (cross-entity accuracy - its majority). Positive means the probe knows *which region*, not merely that the value occurred.

| distance | pairs | within - majority | cross - majority | binding margin |
|---|---|---|---|---|
| 0 | 36 | +0.111 | +0.032 | **+0.079** |
| 1 | 36 | +0.122 | +0.063 | **+0.060** |
| 2 | 36 | +0.126 | +0.055 | **+0.071** |
| 3-4 | 36 | +0.103 | +0.041 | **+0.062** |
| 5+ | 36 | -0.005 | +0.075 | **-0.079** |

The sign flip at the largest distance is the substantive finding: within-entity accuracy falls below majority while the cross-entity read rises above it. Past roughly five sentences the representation still carries "this value occurred somewhere" but has lost which region it applied to. The shared presence code outlives the region-specific binding code -- which is what the concept-direction geometry independently says.


## Does the shuffled-order control work?

Averaged over all 12 targets, scrambling sentence order costs 1.6 accuracy points, which reads as "the model ignores order". That average pools targets whose best layer is the *embedding layer* -- which cannot encode order at all -- with targets that genuinely use the transformer.

| group | n | mean shuffled delta |
|---|---|---|
| best layer > 0 (contextual) | 4 | -0.0356 |
| best layer == 0 (embeddings) | 8 | -0.0063 |

- corr(best layer, shuffled delta) = -0.367
- corr(best layer, selectivity) = +0.625
- corr(selectivity, shuffled delta) = -0.609

Three independent measures -- best-layer depth, selectivity and order-sensitivity -- pick out the same targets. The control is not weak; it was correctly reporting that most targets have no temporal structure to destroy.


## State persistence

Accuracy binned by how many sentences have passed since that variable last changed. A representation that only reflects the current sentence should collapse towards the majority rate as the last update recedes; one that maintains state should stay flat.

| target | layer | distance | n | accuracy | majority in bin |
|---|---|---|---|---|---|
| `cheeks.color` | 0 | 0 | 392 | 0.176 | 0.048 |
| `cheeks.color` | 0 | 1 | 563 | 0.432 | 0.368 |
| `cheeks.color` | 0 | 2 | 469 | 0.422 | 0.377 |
| `cheeks.color` | 0 | 3-4 | 752 | 0.415 | 0.366 |
| `cheeks.color` | 0 | 5-7 | 689 | 0.332 | 0.340 |
| `cheeks.color` | 0 | 8+ | 369 | 0.230 | 0.279 |
| `cheeks.coverage` | 0 | 0 | 419 | 0.313 | 0.504 |
| `cheeks.coverage` | 0 | 1 | 582 | 0.570 | 0.337 |
| `cheeks.coverage` | 0 | 2 | 484 | 0.601 | 0.318 |
| `cheeks.coverage` | 0 | 3-4 | 741 | 0.524 | 0.317 |
| `cheeks.coverage` | 0 | 5-7 | 675 | 0.409 | 0.332 |
| `cheeks.coverage` | 0 | 8+ | 333 | 0.354 | 0.336 |
| `cheeks.finish` | 0 | 0 | 479 | 0.209 | 0.040 |
| `cheeks.finish` | 0 | 1 | 626 | 0.439 | 0.331 |
| `cheeks.finish` | 0 | 2 | 501 | 0.461 | 0.353 |
| `cheeks.finish` | 0 | 3-4 | 741 | 0.483 | 0.371 |
| `cheeks.finish` | 0 | 5-7 | 607 | 0.392 | 0.386 |
| `cheeks.finish` | 0 | 8+ | 280 | 0.257 | 0.368 |
| `eyes.color` | 0 | 0 | 411 | 0.158 | 0.044 |
| `eyes.color` | 0 | 1 | 583 | 0.422 | 0.372 |
| `eyes.color` | 0 | 2 | 493 | 0.467 | 0.369 |
| `eyes.color` | 0 | 3-4 | 753 | 0.448 | 0.367 |
| `eyes.color` | 0 | 5-7 | 649 | 0.319 | 0.350 |
| `eyes.color` | 0 | 8+ | 345 | 0.197 | 0.307 |
| `eyes.coverage` | 12 | 0 | 423 | 0.348 | 0.504 |
| `eyes.coverage` | 12 | 1 | 593 | 0.536 | 0.337 |
| `eyes.coverage` | 12 | 2 | 498 | 0.504 | 0.339 |
| `eyes.coverage` | 12 | 3-4 | 755 | 0.560 | 0.354 |
| `eyes.coverage` | 12 | 5-7 | 632 | 0.492 | 0.381 |
| `eyes.coverage` | 12 | 8+ | 333 | 0.483 | 0.432 |
| `eyes.finish` | 2 | 0 | 499 | 0.449 | 0.036 |
| `eyes.finish` | 2 | 1 | 627 | 0.456 | 0.346 |
| `eyes.finish` | 2 | 2 | 507 | 0.440 | 0.359 |
| `eyes.finish` | 2 | 3-4 | 735 | 0.423 | 0.376 |
| `eyes.finish` | 2 | 5-7 | 585 | 0.378 | 0.388 |
| `eyes.finish` | 2 | 8+ | 281 | 0.306 | 0.377 |
| `lips.color` | 0 | 0 | 405 | 0.165 | 0.044 |
| `lips.color` | 0 | 1 | 578 | 0.405 | 0.348 |
| `lips.color` | 0 | 2 | 481 | 0.439 | 0.360 |
| `lips.color` | 0 | 3-4 | 739 | 0.414 | 0.364 |
| `lips.color` | 0 | 5-7 | 677 | 0.331 | 0.340 |
| `lips.color` | 0 | 8+ | 354 | 0.209 | 0.277 |
| `lips.coverage` | 0 | 0 | 398 | 0.289 | 0.045 |
| `lips.coverage` | 0 | 1 | 573 | 0.503 | 0.351 |
| `lips.coverage` | 0 | 2 | 479 | 0.549 | 0.361 |
| `lips.coverage` | 0 | 3-4 | 727 | 0.530 | 0.370 |
| `lips.coverage` | 0 | 5-7 | 679 | 0.420 | 0.339 |
| `lips.coverage` | 0 | 8+ | 378 | 0.394 | 0.259 |
| `lips.finish` | 0 | 0 | 497 | 0.219 | 0.036 |
| `lips.finish` | 0 | 1 | 616 | 0.407 | 0.326 |
| `lips.finish` | 0 | 2 | 490 | 0.463 | 0.353 |
| `lips.finish` | 0 | 3-4 | 721 | 0.436 | 0.373 |
| `lips.finish` | 0 | 5-7 | 626 | 0.367 | 0.367 |
| `lips.finish` | 0 | 8+ | 284 | 0.282 | 0.345 |
| `skin.color` | 0 | 0 | 405 | 0.175 | 0.057 |
| `skin.color` | 0 | 1 | 565 | 0.439 | 0.375 |
| `skin.color` | 0 | 2 | 480 | 0.458 | 0.373 |
| `skin.color` | 0 | 3-4 | 739 | 0.433 | 0.384 |
| `skin.color` | 0 | 5-7 | 679 | 0.343 | 0.423 |
| `skin.color` | 0 | 8+ | 366 | 0.186 | 0.363 |
| `skin.coverage` | 10 | 0 | 406 | 0.466 | 0.475 |
| `skin.coverage` | 10 | 1 | 558 | 0.581 | 0.310 |
| `skin.coverage` | 10 | 2 | 480 | 0.527 | 0.308 |
| `skin.coverage` | 10 | 3-4 | 726 | 0.472 | 0.299 |
| `skin.coverage` | 10 | 5-7 | 686 | 0.405 | 0.299 |
| `skin.coverage` | 10 | 8+ | 378 | 0.360 | 0.370 |
| `skin.finish` | 6 | 0 | 487 | 0.246 | 0.047 |
| `skin.finish` | 6 | 1 | 613 | 0.423 | 0.346 |
| `skin.finish` | 6 | 2 | 492 | 0.419 | 0.364 |
| `skin.finish` | 6 | 3-4 | 724 | 0.427 | 0.392 |
| `skin.finish` | 6 | 5-7 | 629 | 0.420 | 0.456 |
| `skin.finish` | 6 | 8+ | 289 | 0.332 | 0.460 |

## Probe-capacity ablation

| target | probe | real | control | selectivity | MDL (kbit) |
|---|---|---|---|---|---|
| `eyes.color` | linear | 0.358 | 0.291 | +0.067 | 21.7 |
| `eyes.color` | MLP-8 | 0.357 | 0.308 | +0.048 | 13.7 |
| `eyes.color` | MLP-32 | 0.353 | 0.297 | +0.055 | 14.1 |
| `eyes.color` | MLP-128 | 0.346 | 0.294 | +0.052 | 14.8 |
| `eyes.color` | MLP-512 | 0.365 | 0.292 | +0.073 | 16.3 |
| `lips.color` | linear | 0.345 | 0.271 | +0.074 | 21.2 |
| `lips.color` | MLP-8 | 0.335 | 0.298 | +0.037 | 13.7 |
| `lips.color` | MLP-32 | 0.342 | 0.280 | +0.062 | 14.0 |
| `lips.color` | MLP-128 | 0.342 | 0.271 | +0.071 | 14.8 |
| `lips.color` | MLP-512 | 0.334 | 0.269 | +0.065 | 15.5 |
| `lips.coverage` | linear | 0.461 | 0.331 | +0.130 | 13.5 |
| `lips.coverage` | MLP-8 | 0.460 | 0.329 | +0.131 | 9.1 |
| `lips.coverage` | MLP-32 | 0.459 | 0.315 | +0.144 | 9.6 |
| `lips.coverage` | MLP-128 | 0.463 | 0.332 | +0.132 | 9.8 |
| `lips.coverage` | MLP-512 | 0.453 | 0.322 | +0.132 | 11.1 |
| `lips.finish` | linear | 0.372 | 0.328 | +0.045 | 18.0 |
| `lips.finish` | MLP-8 | 0.377 | 0.339 | +0.038 | 11.4 |
| `lips.finish` | MLP-32 | 0.371 | 0.316 | +0.055 | 12.2 |
| `lips.finish` | MLP-128 | 0.373 | 0.323 | +0.050 | 12.9 |
| `lips.finish` | MLP-512 | 0.373 | 0.328 | +0.045 | 14.0 |

## Causal intervention

### `cheeks.color` steered at layer 1 (150 held-out positions)

Mean hidden-state norm at the edited layer: 47.5. Probe read-out trained at layer 12 on clean data.

| alpha | probe: true | probe: steered | probe: steered (trials initially correct) | other region correct | other region changed | behaviour: true | behaviour: delta log-odds steered | behaviour: delta log-odds other |
|---|---|---|---|---|---|---|---|---|
| 0.0 | 0.167 | 0.200 | 0.000 | 0.193 | 0.000 | 0.360 | +0.000 | +0.000 |
| 0.5 | 0.160 | 0.200 | 0.000 | 0.200 | 0.007 | 0.360 | -0.002 | +0.001 |
| 1.0 | 0.147 | 0.207 | 0.000 | 0.200 | 0.007 | 0.360 | -0.005 | +0.001 |
| 2.0 | 0.147 | 0.207 | 0.000 | 0.200 | 0.007 | 0.360 | -0.010 | +0.002 |
| 4.0 | 0.147 | 0.207 | 0.000 | 0.200 | 0.053 | 0.360 | -0.019 | +0.003 |
| 8.0 | 0.167 | 0.227 | 0.000 | 0.207 | 0.093 | 0.360 | -0.032 | +0.006 |

### `lips.color` steered at layer 1 (150 held-out positions)

Mean hidden-state norm at the edited layer: 47.5. Probe read-out trained at layer 12 on clean data.

| alpha | probe: true | probe: steered | probe: steered (trials initially correct) | other region correct | other region changed | behaviour: true | behaviour: delta log-odds steered | behaviour: delta log-odds other |
|---|---|---|---|---|---|---|---|---|
| 0.0 | 0.273 | 0.233 | 0.000 | 0.127 | 0.000 | 0.147 | +0.000 | +0.000 |
| 0.5 | 0.273 | 0.233 | 0.000 | 0.127 | 0.007 | 0.147 | -0.000 | -0.000 |
| 1.0 | 0.267 | 0.227 | 0.000 | 0.127 | 0.007 | 0.147 | -0.001 | -0.001 |
| 2.0 | 0.273 | 0.240 | 0.000 | 0.127 | 0.007 | 0.147 | -0.002 | -0.002 |
| 4.0 | 0.260 | 0.240 | 0.000 | 0.140 | 0.067 | 0.147 | -0.002 | -0.003 |
| 8.0 | 0.247 | 0.260 | 0.073 | 0.107 | 0.147 | 0.147 | +0.024 | +0.019 |

### `lips.coverage` steered at layer 1 (150 held-out positions)

Mean hidden-state norm at the edited layer: 47.5. Probe read-out trained at layer 12 on clean data.

| alpha | probe: true | probe: steered | probe: steered (trials initially correct) | other region correct | other region changed | behaviour: true | behaviour: delta log-odds steered | behaviour: delta log-odds other |
|---|---|---|---|---|---|---|---|---|
| 0.0 | 0.473 | 0.307 | 0.000 | 0.440 | 0.000 | 0.380 | +0.000 | +0.000 |
| 0.5 | 0.453 | 0.327 | 0.028 | 0.460 | 0.053 | 0.407 | -0.020 | -0.017 |
| 1.0 | 0.420 | 0.320 | 0.085 | 0.467 | 0.100 | 0.420 | -0.041 | -0.044 |
| 2.0 | 0.380 | 0.340 | 0.141 | 0.440 | 0.140 | 0.413 | -0.106 | -0.106 |
| 4.0 | 0.373 | 0.313 | 0.099 | 0.407 | 0.173 | 0.427 | -0.146 | +0.021 |
| 8.0 | 0.247 | 0.373 | 0.268 | 0.353 | 0.187 | 0.440 | -0.367 | +0.128 |

### `lips.finish` steered at layer 1 (150 held-out positions)

Mean hidden-state norm at the edited layer: 47.5. Probe read-out trained at layer 12 on clean data.

| alpha | probe: true | probe: steered | probe: steered (trials initially correct) | other region correct | other region changed | behaviour: true | behaviour: delta log-odds steered | behaviour: delta log-odds other |
|---|---|---|---|---|---|---|---|---|
| 0.0 | 0.313 | 0.307 | 0.000 | 0.267 | 0.000 | 0.313 | +0.000 | +0.000 |
| 0.5 | 0.313 | 0.313 | 0.000 | 0.267 | 0.007 | 0.313 | +0.013 | -0.002 |
| 1.0 | 0.307 | 0.320 | 0.021 | 0.267 | 0.007 | 0.313 | +0.026 | -0.004 |
| 2.0 | 0.293 | 0.340 | 0.064 | 0.267 | 0.027 | 0.313 | +0.050 | -0.008 |
| 4.0 | 0.273 | 0.353 | 0.064 | 0.273 | 0.053 | 0.313 | +0.094 | -0.016 |
| 8.0 | 0.187 | 0.453 | 0.277 | 0.273 | 0.087 | 0.313 | +0.173 | -0.018 |

### `skin.finish` steered at layer 6 (150 held-out positions)

Mean hidden-state norm at the edited layer: 65.0. Probe read-out trained at layer 12 on clean data.

| alpha | probe: true | probe: steered | probe: steered (trials initially correct) | other region correct | other region changed | behaviour: true | behaviour: delta log-odds steered | behaviour: delta log-odds other |
|---|---|---|---|---|---|---|---|---|
| 0.0 | 0.140 | 0.247 | 0.000 | 0.333 | 0.000 | 0.387 | +0.000 | +0.000 |
| 0.5 | 0.133 | 0.267 | 0.048 | 0.333 | 0.000 | 0.393 | +0.003 | +0.000 |
| 1.0 | 0.127 | 0.287 | 0.048 | 0.340 | 0.007 | 0.393 | +0.007 | +0.001 |
| 2.0 | 0.120 | 0.327 | 0.095 | 0.340 | 0.040 | 0.393 | +0.016 | +0.004 |
| 4.0 | 0.060 | 0.380 | 0.238 | 0.353 | 0.073 | 0.393 | +0.041 | +0.014 |
| 8.0 | 0.013 | 0.560 | 0.714 | 0.340 | 0.160 | 0.453 | +0.094 | +0.043 |


## Causal intervention: effect sizes and significance

Paired bootstrap over trials (10,000 resamples). The binding claim is the last column: the edit must move the target region's read-out more than the other region's. p-values are two-sided throughout this report; the one-sided values are kept in the result JSON as `p_one_sided_greater`.

### `cheeks.color` (edit L1 -> read L12, n=150)

| alpha | delta log-odds, target | delta log-odds, other region | target - other | p (two-sided) |
|---|---|---|---|---|
| 0.5 | -0.002 [-0.003, -0.002] | +0.001 [-0.000, +0.001] | -0.003 [-0.004, -0.002] | 0.0000 |
| 1.0 | -0.005 [-0.006, -0.003] | +0.001 [-0.000, +0.003] | -0.006 [-0.008, -0.004] | 0.0000 |
| 2.0 | -0.010 [-0.012, -0.007] | +0.002 [-0.001, +0.005] | -0.011 [-0.016, -0.007] | 0.0000 |
| 4.0 | -0.019 [-0.026, -0.013] | +0.003 [-0.003, +0.010] | -0.022 [-0.032, -0.013] | 0.0000 |
| 8.0 | -0.032 [-0.048, -0.017] | +0.006 [-0.010, +0.022] | -0.038 [-0.061, -0.015] | 0.0010 |

Probe read-out flips to the steered value on 0.000 [0.000, 0.000] of the 25 trials whose unintervened read-out was already correct. The small n is the limitation: the read-out probe sits downstream of the edit, so it is weaker than the best-layer probe.

### `lips.color` (edit L1 -> read L12, n=150)

| alpha | delta log-odds, target | delta log-odds, other region | target - other | p (two-sided) |
|---|---|---|---|---|
| 0.5 | -0.000 [-0.002, +0.001] | -0.000 [-0.002, +0.001] | -0.000 [-0.002, +0.001] | 0.8140 |
| 1.0 | -0.001 [-0.004, +0.002] | -0.001 [-0.003, +0.002] | -0.000 [-0.003, +0.003] | 0.9094 |
| 2.0 | -0.002 [-0.007, +0.004] | -0.002 [-0.008, +0.003] | +0.000 [-0.005, +0.006] | 0.9242 |
| 4.0 | -0.002 [-0.015, +0.010] | -0.003 [-0.015, +0.008] | +0.001 [-0.012, +0.014] | 0.8862 |
| 8.0 | +0.024 [-0.004, +0.053] | +0.019 [-0.007, +0.045] | +0.006 [-0.030, +0.041] | 0.7558 |

Probe read-out flips to the steered value on 0.073 [0.000, 0.171] of the 41 trials whose unintervened read-out was already correct. The small n is the limitation: the read-out probe sits downstream of the edit, so it is weaker than the best-layer probe.

### `lips.coverage` (edit L1 -> read L12, n=150)

| alpha | delta log-odds, target | delta log-odds, other region | target - other | p (two-sided) |
|---|---|---|---|---|
| 0.5 | -0.020 [-0.036, -0.004] | -0.017 [-0.032, -0.001] | -0.002 [-0.025, +0.019] | 0.8292 |
| 1.0 | -0.041 [-0.082, +0.000] | -0.044 [-0.087, -0.001] | +0.002 [-0.056, +0.060] | 0.9394 |
| 2.0 | -0.106 [-0.201, -0.014] | -0.106 [-0.196, -0.018] | +0.000 [-0.141, +0.136] | 0.9930 |
| 4.0 | -0.146 [-0.230, -0.063] | +0.021 [-0.053, +0.100] | -0.167 [-0.293, -0.050] | 0.0050 |
| 8.0 | -0.367 [-0.471, -0.267] | +0.128 [+0.018, +0.236] | -0.495 [-0.657, -0.335] | 0.0000 |

Probe read-out flips to the steered value on 0.268 [0.169, 0.367] of the 71 trials whose unintervened read-out was already correct. The small n is the limitation: the read-out probe sits downstream of the edit, so it is weaker than the best-layer probe.

### `lips.finish` (edit L1 -> read L12, n=150)

| alpha | delta log-odds, target | delta log-odds, other region | target - other | p (two-sided) |
|---|---|---|---|---|
| 0.5 | +0.013 [+0.010, +0.016] | -0.002 [-0.005, +0.002] | +0.015 [+0.011, +0.019] | 0.0000 |
| 1.0 | +0.026 [+0.021, +0.031] | -0.004 [-0.011, +0.003] | +0.030 [+0.021, +0.038] | 0.0000 |
| 2.0 | +0.050 [+0.039, +0.060] | -0.008 [-0.022, +0.006] | +0.058 [+0.040, +0.076] | 0.0000 |
| 4.0 | +0.094 [+0.072, +0.116] | -0.016 [-0.045, +0.013] | +0.110 [+0.072, +0.148] | 0.0000 |
| 8.0 | +0.173 [+0.126, +0.219] | -0.018 [-0.081, +0.044] | +0.191 [+0.108, +0.276] | 0.0000 |

Probe read-out flips to the steered value on 0.277 [0.149, 0.404] of the 47 trials whose unintervened read-out was already correct. The small n is the limitation: the read-out probe sits downstream of the edit, so it is weaker than the best-layer probe.

### `skin.finish` (edit L6 -> read L12, n=150)

| alpha | delta log-odds, target | delta log-odds, other region | target - other | p (two-sided) |
|---|---|---|---|---|
| 0.5 | +0.003 [-0.006, +0.012] | +0.000 [-0.008, +0.009] | +0.003 [-0.009, +0.014] | 0.6836 |
| 1.0 | +0.007 [-0.011, +0.024] | +0.001 [-0.016, +0.019] | +0.005 [-0.018, +0.029] | 0.6710 |
| 2.0 | +0.016 [-0.020, +0.050] | +0.004 [-0.030, +0.040] | +0.012 [-0.036, +0.058] | 0.6366 |
| 4.0 | +0.041 [-0.031, +0.110] | +0.014 [-0.056, +0.085] | +0.027 [-0.067, +0.122] | 0.5770 |
| 8.0 | +0.094 [-0.048, +0.230] | +0.043 [-0.097, +0.185] | +0.051 [-0.138, +0.242] | 0.6112 |

Probe read-out flips to the steered value on 0.714 [0.524, 0.905] of the 21 trials whose unintervened read-out was already correct. The small n is the limitation: the read-out probe sits downstream of the edit, so it is weaker than the best-layer probe.

