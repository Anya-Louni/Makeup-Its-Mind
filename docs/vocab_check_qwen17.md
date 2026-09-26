# Model benchmark

Host: CPU-only, 12 cores, float32.

| model | params (M) | layers | hidden | load (s) | RSS (MB) | tok/s | extract s/narr | vocab acc | status |
|---|---|---|---|---|---|---|---|---|---|
| `Qwen/Qwen3-1.7B-Base` | 1720.6 | 28 | 2048 | 765.3 | 0.0 | - | - | 0.940 | ok |

## Vocabulary sanity check (per term)

Fraction of minimal pairs where the model prefers the semantically correct completion. Chance = 0.5.

| term | `Qwen/Qwen3-1.7B-Base` |
|---|---|
| beige | 1.00 |
| blend | 1.00 |
| blot | 1.00 |
| blush | 1.00 |
| bronzer | 1.00 |
| brown | 1.00 |
| cheekbones | 0.67 |
| complexion | 0.67 |
| concealer | 1.00 |
| coverage | 1.00 |
| crimson | 1.00 |
| dewy | 1.00 |
| eyeshadow | 1.00 |
| lids | 1.00 |
| lipstick | 1.00 |
| matte | 0.80 |
| nude | 1.00 |
| opaque | 1.00 |
| pink | 1.00 |
| primer | 1.00 |
| red | 1.00 |
| satin | 1.00 |
| sheer | 1.00 |
