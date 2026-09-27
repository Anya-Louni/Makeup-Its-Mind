# Results: `Qwen/Qwen3-1.7B-Base` on the `pilot` dataset

## Dataset

- 2000 narratives, 13097 labelled positions (6.55 sentences each)
- action mix: APPLY 7439, DISTRACTOR 2364, NOOP 1131, SOFTEN 412, BLOT 405, GLOSS 380, BUILD 379, REMOVE 297, SHEER 290
- fraction of positions where the value was overridden earlier (first mention is the wrong answer): eyes.color 0.23, eyes.finish 0.27, lips.color 0.23, lips.finish 0.27

## Probing

All numbers are test-split accuracy on narratives built from wordings never seen in training.

| target | classes | majority | best layer | linear | MLP | static emb | bag-of-ngrams | control task (test) | control task (train) | selectivity | shuffled order |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `eyes.color` | 5 | 0.179 | 28 | **0.400** | 0.408 | 0.329 | 0.255 | 0.179 | 0.999 | 0.222 | 0.428 |
| `eyes.finish` | 4 | 0.261 | 18 | **0.466** | 0.471 | 0.368 | 0.260 | 0.248 | 0.999 | 0.218 | 0.385 |
| `lips.color` | 5 | 0.201 | 28 | **0.369** | 0.398 | 0.308 | 0.322 | 0.205 | 0.999 | 0.164 | 0.352 |
| `lips.finish` | 4 | 0.267 | 27 | **0.504** | 0.507 | 0.406 | 0.393 | 0.297 | 0.999 | 0.207 | 0.455 |

### Positions where the state was overridden

The subset where the first value mentioned for that region is *not* the answer. A keyword heuristic cannot score above chance here.

| target | overridden positions | accuracy there | accuracy elsewhere |
|---|---|---|---|
| `eyes.color` | 179 | 0.380 | 0.406 |
| `eyes.finish` | 252 | 0.440 | 0.477 |
| `lips.color` | 279 | 0.326 | 0.391 |
| `lips.finish` | 340 | 0.482 | 0.519 |

### Binding check

A probe trained on one region, read against the *other* region's label. Near the other region's majority rate means the regions are kept apart.

| probe trained on | read against | accuracy | that region's majority |
|---|---|---|---|
| `eyes.color` | `lips.color` | 0.292 | 0.201 |
| `eyes.finish` | `lips.finish` | 0.334 | 0.267 |
| `lips.color` | `eyes.color` | 0.230 | 0.179 |
| `lips.finish` | `eyes.finish` | 0.348 | 0.261 |

## Binding, stratified by distance

The aggregate binding test (probe trained on one region, read against another) is dominated by positions far from the last update, where the attribute is not decodable at all and both probes sit at majority. Stratifying by distance since the variable last changed shows what the aggregate hides.

Binding margin = (within-entity accuracy - its majority) - (cross-entity accuracy - its majority). Positive means the probe knows *which region*, not merely that the value occurred.

| distance | pairs | within - majority | cross - majority | binding margin |
|---|---|---|---|---|
| 0 | 4 | +0.172 | +0.044 | **+0.129** |
| 1 | 4 | +0.256 | +0.068 | **+0.188** |
| 2 | 4 | +0.212 | +0.013 | **+0.199** |
| 3-4 | 4 | +0.178 | +0.079 | **+0.099** |
| 5+ | 3 | +0.182 | +0.113 | **+0.068** |

The sign flip at the largest distance is the substantive finding: within-entity accuracy falls below majority while the cross-entity read rises above it. Past roughly five sentences the representation still carries "this value occurred somewhere" but has lost which region it applied to. The shared presence code outlives the region-specific binding code -- which is what the concept-direction geometry independently says.

