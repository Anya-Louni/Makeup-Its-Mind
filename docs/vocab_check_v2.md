# Model benchmark

Host: CPU-only, 12 cores, float32.

| model | params (M) | layers | hidden | load (s) | RSS (MB) | tok/s | extract s/narr | vocab acc | status |
|---|---|---|---|---|---|---|---|---|---|
| `gpt2` | 124.4 | 12 | 768 | 7.1 | 0.0 | - | - | 0.820 | ok |
| `EleutherAI/pythia-410m` | 405.3 | 24 | 1024 | 4.8 | 0.0 | - | - | 0.840 | ok |

## Vocabulary sanity check (per term)

Fraction of minimal pairs where the model prefers the semantically correct completion. Chance = 0.5.

| term | `gpt2` | `EleutherAI/pythia-410m` |
|---|---|---|
| beige | 0.67 | 0.67 |
| blend | 1.00 | 1.00 |
| blot | 0.33 | 1.00 |
| blush | 1.00 | 1.00 |
| bronzer | 1.00 | 0.67 |
| brown | 1.00 | 1.00 |
| cheekbones | 0.67 | 0.67 |
| complexion | 0.67 | 1.00 |
| concealer | 1.00 | 1.00 |
| coverage | 1.00 | 1.00 |
| crimson | 1.00 | 1.00 |
| dewy | 0.80 | 0.80 |
| eyeshadow | 1.00 | 1.00 |
| lids | 0.67 | 1.00 |
| lipstick | 0.67 | 0.67 |
| matte | 0.80 | 0.40 |
| nude | 1.00 | 1.00 |
| opaque | 1.00 | 1.00 |
| pink | 1.00 | 1.00 |
| primer | 1.00 | 1.00 |
| red | 1.00 | 1.00 |
| satin | 1.00 | 1.00 |
| sheer | 1.00 | 1.00 |
