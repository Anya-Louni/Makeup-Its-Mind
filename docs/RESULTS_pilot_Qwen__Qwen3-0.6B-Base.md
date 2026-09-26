# Results: `Qwen/Qwen3-0.6B-Base` on the `pilot` dataset

## Dataset

- 2000 narratives, 13097 labelled positions (6.55 sentences each)
- action mix: APPLY 7439, DISTRACTOR 2364, NOOP 1131, SOFTEN 412, BLOT 405, GLOSS 380, BUILD 379, REMOVE 297, SHEER 290
- fraction of positions where the value was overridden earlier (first mention is the wrong answer): eyes.color 0.23, eyes.finish 0.27, lips.color 0.23, lips.finish 0.27

## Probing

All numbers are test-split accuracy on narratives built from wordings never seen in training.

| target | classes | majority | best layer | linear | MLP | static emb | bag-of-ngrams | control task (test) | control task (train) | selectivity | shuffled order |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `eyes.color` | 5 | 0.179 | 26 | **0.395** | 0.411 | 0.284 | 0.255 | 0.234 | 0.944 | 0.161 | 0.348 |
| `eyes.finish` | 4 | 0.261 | 16 | **0.434** | 0.495 | 0.313 | 0.260 | 0.243 | 0.732 | 0.191 | 0.426 |
| `lips.color` | 5 | 0.201 | 22 | **0.319** | 0.335 | 0.350 | 0.322 | 0.195 | 0.876 | 0.124 | 0.292 |
| `lips.finish` | 4 | 0.267 | 18 | **0.405** | 0.449 | 0.453 | 0.393 | 0.271 | 0.750 | 0.135 | 0.375 |

### Positions where the state was overridden

The subset where the first value mentioned for that region is *not* the answer. A keyword heuristic cannot score above chance here.

| target | overridden positions | accuracy there | accuracy elsewhere |
|---|---|---|---|
| `eyes.color` | 179 | 0.313 | 0.417 |
| `eyes.finish` | 252 | 0.369 | 0.462 |
| `lips.color` | 279 | 0.355 | 0.302 |
| `lips.finish` | 340 | 0.400 | 0.409 |

### Binding check

A probe trained on one region, read against the *other* region's label. Near the other region's majority rate means the regions are kept apart.

| probe trained on | read against | accuracy | that region's majority |
|---|---|---|---|
| `eyes.color` | `lips.color` | 0.219 | 0.201 |
| `eyes.finish` | `lips.finish` | 0.303 | 0.267 |
| `lips.color` | `eyes.color` | 0.262 | 0.179 |
| `lips.finish` | `eyes.finish` | 0.340 | 0.261 |

## Binding, stratified by distance

The aggregate binding test (probe trained on one region, read against another) is dominated by positions far from the last update, where the attribute is not decodable at all and both probes sit at majority. Stratifying by distance since the variable last changed shows what the aggregate hides.

Binding margin = (within-entity accuracy - its majority) - (cross-entity accuracy - its majority). Positive means the probe knows *which region*, not merely that the value occurred.

| distance | pairs | within - majority | cross - majority | binding margin |
|---|---|---|---|---|
| 0 | 4 | +0.154 | +0.036 | **+0.119** |
| 1 | 4 | +0.194 | +0.053 | **+0.141** |
| 2 | 4 | +0.168 | +0.017 | **+0.151** |
| 3-4 | 4 | +0.115 | +0.004 | **+0.111** |
| 5+ | 3 | +0.086 | +0.107 | **-0.021** |

The sign flip at the largest distance is the substantive finding: within-entity accuracy falls below majority while the cross-entity read rises above it. Past roughly five sentences the representation still carries "this value occurred somewhere" but has lost which region it applied to. The shared presence code outlives the region-specific binding code -- which is what the concept-direction geometry independently says.

