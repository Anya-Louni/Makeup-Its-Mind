# Results: `gpt2` on the `long2` dataset

> **Status note.** This file is auto-generated per dataset/model and
> reports that run in isolation. The canonical, cross-model statement of
> what the project shows is [`FINDINGS.md`](FINDINGS.md); claims that did
> not replicate across models, and the thirteen revisions made along the
> way, are recorded in [`CORRECTIONS.md`](CORRECTIONS.md). In particular:
> the causal intervention here demonstrates an attribute-**value** code,
> not entity-specific binding (see CORRECTIONS #10), and the pattern of
> which targets beat a lexical baseline is model-specific (#12).

## Dataset

- 3600 narratives, 46873 labelled positions (13.02 sentences each)
- action mix: APPLY 21336, DISTRACTOR 9906, NOOP 4636, SOFTEN 2071, GLOSS 2057, BLOT 2056, BUILD 1832, REMOVE 1492, SHEER 1487
- fraction of positions where the value was overridden earlier (first mention is the wrong answer): lips.finish 0.41, lips.color 0.39, eyes.color 0.38, eyes.finish 0.43, eyes.coverage 0.32, lips.coverage 0.32

## Probing

All numbers are test-split accuracy on narratives built from wordings never seen in training.

| target | classes | majority | best layer | linear | MLP | static emb | bag-of-ngrams | control task (test) | control task (train) | selectivity | shuffled order |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `eyes.color` | 5 | 0.202 | 2 | **0.310** | 0.300 | 0.244 | 0.215 | 0.198 | 0.489 | 0.111 | 0.308 |
| `eyes.coverage` | 3 | 0.404 | 2 | **0.470** | 0.484 | 0.313 | 0.253 | 0.346 | 0.580 | 0.123 | 0.457 |
| `eyes.finish` | 4 | 0.264 | 8 | **0.357** | 0.357 | 0.287 | 0.222 | 0.269 | 0.497 | 0.089 | 0.373 |
| `lips.color` | 5 | 0.205 | 2 | **0.307** | 0.266 | 0.323 | 0.263 | 0.181 | 0.486 | 0.126 | 0.288 |
| `lips.coverage` | 3 | 0.392 | 2 | **0.486** | 0.491 | 0.455 | 0.451 | 0.374 | 0.576 | 0.112 | 0.448 |
| `lips.finish` | 4 | 0.272 | 2 | **0.382** | 0.374 | 0.387 | 0.350 | 0.248 | 0.503 | 0.134 | 0.320 |

### Positions where the state was overridden

The subset where the first value mentioned for that region is *not* the answer. A keyword heuristic cannot score above chance here.

| target | overridden positions | accuracy there | accuracy elsewhere |
|---|---|---|---|
| `eyes.color` | 1543 | 0.272 | 0.344 |
| `eyes.coverage` | 1162 | 0.466 | 0.472 |
| `eyes.finish` | 1552 | 0.336 | 0.376 |
| `lips.color` | 1457 | 0.286 | 0.324 |
| `lips.coverage` | 1166 | 0.454 | 0.504 |
| `lips.finish` | 1427 | 0.334 | 0.419 |

### Binding check

A probe trained on one region, read against the *other* region's label. Near the other region's majority rate means the regions are kept apart.

| probe trained on | read against | accuracy | that region's majority |
|---|---|---|---|
| `eyes.color` | `lips.color` | 0.263 | 0.205 |
| `eyes.coverage` | `lips.coverage` | 0.423 | 0.392 |
| `eyes.finish` | `lips.finish` | 0.316 | 0.272 |
| `lips.color` | `eyes.color` | 0.296 | 0.202 |
| `lips.coverage` | `eyes.coverage` | 0.486 | 0.404 |
| `lips.finish` | `eyes.finish` | 0.342 | 0.264 |

### MDL (prequential code length)

Lower is better; compression is relative to the uniform code.

| target | layer | code length (kbit) | compression | static emb compression |
|---|---|---|---|---|
| `eyes.color` | 0 | 18.2 | 1.00x | 1.35x |
| `eyes.color` | 2 | 15.7 | 1.16x | 1.35x |
| `eyes.color` | 12 | 15.0 | 1.21x | 1.35x |
| `eyes.coverage` | 0 | 11.7 | 1.06x | 1.57x |
| `eyes.coverage` | 2 | 8.9 | 1.40x | 1.57x |
| `eyes.coverage` | 12 | 9.3 | 1.33x | 1.57x |
| `eyes.finish` | 0 | 15.5 | 1.01x | 1.43x |
| `eyes.finish` | 8 | 13.1 | 1.19x | 1.43x |
| `eyes.finish` | 12 | 12.7 | 1.23x | 1.43x |
| `lips.color` | 0 | 18.1 | 1.00x | 1.38x |
| `lips.color` | 2 | 15.5 | 1.17x | 1.38x |
| `lips.color` | 12 | 15.0 | 1.21x | 1.38x |
| `lips.coverage` | 0 | 11.6 | 1.07x | 1.56x |
| `lips.coverage` | 2 | 8.5 | 1.46x | 1.56x |
| `lips.coverage` | 12 | 9.2 | 1.34x | 1.56x |
| `lips.finish` | 0 | 15.3 | 1.02x | 1.49x |
| `lips.finish` | 2 | 11.3 | 1.38x | 1.49x |
| `lips.finish` | 12 | 12.5 | 1.25x | 1.49x |

## State persistence

Accuracy binned by how many sentences have passed since that variable last changed. A representation that only reflects the current sentence should collapse towards the majority rate as the last update recedes; one that maintains state should stay flat.

| target | layer | distance | n | accuracy | majority in bin |
|---|---|---|---|---|---|
| `eyes.color` | 2 | 0 | 679 | 0.321 | 0.237 |
| `eyes.color` | 2 | 1 | 721 | 0.316 | 0.191 |
| `eyes.color` | 2 | 2 | 541 | 0.298 | 0.194 |
| `eyes.color` | 2 | 3-4 | 685 | 0.296 | 0.199 |
| `eyes.color` | 2 | 5-7 | 460 | 0.322 | 0.191 |
| `eyes.color` | 2 | 8+ | 161 | 0.298 | 0.168 |
| `eyes.coverage` | 2 | 0 | 648 | 0.528 | 0.452 |
| `eyes.coverage` | 2 | 1 | 700 | 0.487 | 0.387 |
| `eyes.coverage` | 2 | 2 | 540 | 0.463 | 0.385 |
| `eyes.coverage` | 2 | 3-4 | 705 | 0.445 | 0.389 |
| `eyes.coverage` | 2 | 5-7 | 476 | 0.429 | 0.405 |
| `eyes.coverage` | 2 | 8+ | 178 | 0.421 | 0.416 |
| `eyes.finish` | 8 | 0 | 822 | 0.335 | 0.302 |
| `eyes.finish` | 8 | 1 | 759 | 0.375 | 0.264 |
| `eyes.finish` | 8 | 2 | 537 | 0.400 | 0.264 |
| `eyes.finish` | 8 | 3-4 | 628 | 0.379 | 0.245 |
| `eyes.finish` | 8 | 5-7 | 400 | 0.302 | 0.242 |
| `eyes.finish` | 8 | 8+ | 101 | 0.257 | 0.158 |
| `lips.color` | 2 | 0 | 693 | 0.299 | 0.244 |
| `lips.color` | 2 | 1 | 698 | 0.354 | 0.195 |
| `lips.color` | 2 | 2 | 531 | 0.324 | 0.203 |
| `lips.color` | 2 | 3-4 | 680 | 0.293 | 0.191 |
| `lips.color` | 2 | 5-7 | 454 | 0.273 | 0.185 |
| `lips.color` | 2 | 8+ | 191 | 0.251 | 0.199 |
| `lips.coverage` | 2 | 0 | 664 | 0.545 | 0.455 |
| `lips.coverage` | 2 | 1 | 689 | 0.527 | 0.361 |
| `lips.coverage` | 2 | 2 | 522 | 0.452 | 0.366 |
| `lips.coverage` | 2 | 3-4 | 703 | 0.459 | 0.387 |
| `lips.coverage` | 2 | 5-7 | 483 | 0.462 | 0.408 |
| `lips.coverage` | 2 | 8+ | 186 | 0.376 | 0.339 |
| `lips.finish` | 2 | 0 | 888 | 0.352 | 0.307 |
| `lips.finish` | 2 | 1 | 758 | 0.418 | 0.263 |
| `lips.finish` | 2 | 2 | 532 | 0.383 | 0.256 |
| `lips.finish` | 2 | 3-4 | 591 | 0.376 | 0.262 |
| `lips.finish` | 2 | 5-7 | 352 | 0.409 | 0.264 |
| `lips.finish` | 2 | 8+ | 126 | 0.310 | 0.222 |
