# Model benchmark

Host: CPU-only, 12 cores, float32.

| model | params (M) | layers | hidden | load (s) | RSS (MB) | tok/s | extract s/narr | vocab acc | status |
|---|---|---|---|---|---|---|---|---|---|
| `gpt2` | 124.4 | 12 | 768 | 5.0 | 0.0 | 355.4 | 0.211 | 0.759 | ok |
| `EleutherAI/pythia-410m` | 405.3 | 24 | 1024 | 64.1 | 0.0 | 124.9 | 0.566 | 0.793 | ok |
| `HuggingFaceTB/SmolLM2-360M` | 361.8 | 32 | 960 | 60.0 | 0.0 | 139.0 | 0.476 | 0.759 | ok |
| `Qwen/Qwen2.5-0.5B` | 494.0 | 24 | 896 | 126.6 | 0.0 | 124.9 | 0.551 | 0.793 | ok |
| `gpt2-medium` | 354.8 | 24 | 1024 | 143.7 | 0.0 | 126.0 | 0.538 | 0.793 | ok |
| `meta-llama/Llama-3.2-1B` | - | - | - | - | - | - | - | - | OSError: You are trying to access a gated repo.
Make sure to have access to it at https://huggingface.co/meta-llama/Llama-3.2-1B.
401 Client Error. (Request ID: Root=1-6a9d4a2f-2f05044c427e48d17a331298;9f292e12-fad4-45a5-aa93-6d3 |
| `google/gemma-2-2b` | - | - | - | - | - | - | - | - | OSError: You are trying to access a gated repo.
Make sure to have access to it at https://huggingface.co/google/gemma-2-2b.
401 Client Error. (Request ID: Root=1-6a9d4a2f-43ab5cb66fa6b7883f550e59;8ac4be26-47fa-413f-8d21-eeaf2590f |

## Vocabulary sanity check (per term)

Fraction of minimal pairs where the model prefers the semantically correct completion. Chance = 0.5.

| term | `gpt2` | `EleutherAI/pythia-410m` | `HuggingFaceTB/SmolLM2-360M` | `Qwen/Qwen2.5-0.5B` | `gpt2-medium` |
|---|---|---|---|---|---|
| beige | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| blend | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| blot | 0.00 | 1.00 | 0.00 | 1.00 | 1.00 |
| blush | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| bronzer | 1.00 | 0.00 | 0.00 | 1.00 | 0.00 |
| brown | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| cheekbones | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| complexion | 0.00 | 1.00 | 0.00 | 0.00 | 0.00 |
| concealer | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| coverage | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| crimson | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| dewy | 0.67 | 1.00 | 0.67 | 0.67 | 0.67 |
| eyeshadow | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| lids | 0.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| lipstick | 0.00 | 0.00 | 1.00 | 0.00 | 1.00 |
| matte | 1.00 | 0.33 | 1.00 | 1.00 | 0.67 |
| nude | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| opaque | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| pink | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| primer | 1.00 | 1.00 | 0.00 | 0.00 | 1.00 |
| red | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| satin | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| sheer | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
