# Findings

What this study shows, after three models. `CORRECTIONS.md` lists the
fourteen claims that were revised on the way here.

## Main result

A difference of means concept direction holds two parts. One part is shared
by every entity that takes the value. The other is specific to one entity.
Steering with each part alone separates them.

| model | cos(d_lips, d_eyes) | differential share of norm | shared effect | entity specific difference | p |
|---|---|---|---|---|---|
| gpt2-124M | 0.848 | 28.4% | +0.183 | +0.016 [+0.009, +0.023] | 0.0000 |
| pythia-410m | 0.743 | 37.7% | +0.190 | -0.000 [-0.006, +0.005] | 0.8476 |
| Qwen3-0.6B | 0.740 | 35.5% | +0.317 | -0.000 [-0.010, +0.010] | 0.9106 |

The shared part carries the causal effect in all three models. It raises both
regions' read outs by the same amount. The entity specific part moves the two
regions apart in gpt2 by +0.016 with a dose response. In pythia and Qwen it
moves them apart by 0.000. A random direction of matched norm does nothing in
any model.

The gpt2 effect is about a tenth the size of the shared effect while making up
28 per cent of the direction by norm. That ratio explains why injecting the
wrong region's direction works about as well as the right one.

## Central claim

> The entity specific component recovered by a difference of means is not
> required for the attribute read out we tested. The shared value component
> carries the causal effect. A probe recovers the binding and the margin
> decays with narrative distance.

## What replicates

| claim | gpt2 | pythia-410m | Qwen3-0.6B | verdict |
|---|---|---|---|---|
| Shared component is causally effective | +0.183 | +0.190 | +0.317 | 3 of 3 |
| Entity specific component is causally effective | +0.016, p<0.0001 | 0.000, p=0.85 | 0.000, p=0.91 | 1 of 3 |
| Random direction does nothing | yes | yes | yes | 3 of 3 |
| Binding margin positive at short distance | +0.073 | +0.036 | +0.119 | 3 of 3 |
| Eye variables beat the lexical baseline | yes | no | yes | 2 of 3 |
| State code sits deep in the network | no | yes | yes | 2 of 3 |

## Baseline gaps

Accuracy gap over the static embedding baseline. Cluster bootstrap over
narratives. Two sided p.

| target | gpt2 | p | pythia | p | Qwen | p |
|---|---|---|---|---|---|---|
| `eyes.color` | +0.062 | 0.0075 | +0.031 | 0.2315 | +0.111 | 0.0000 |
| `eyes.finish` | +0.097 | 0.0000 | +0.014 | 0.6390 | +0.120 | 0.0000 |
| `lips.color` | -0.044 | 0.1260 | +0.064 | 0.0280 | -0.031 | 0.3625 |
| `lips.finish` | -0.058 | 0.0285 | +0.024 | 0.4150 | -0.048 | 0.0980 |

gpt2 and Qwen agree on the sign of all four targets. pythia reverses the
pattern. An earlier version of this study used gpt2 and pythia only and
concluded that nothing replicated. The third model shows that conclusion came
from the choice of second model.

## Binding margin by distance

| model | 0 | 1 | 2 | 3-4 | 5+ |
|---|---|---|---|---|---|
| gpt2-124M | +0.073 | +0.096 | +0.105 | +0.064 | +0.049 |
| pythia-410m | +0.036 | +0.104 | +0.115 | +0.104 | +0.045 |
| Qwen3-0.6B | +0.119 | +0.141 | +0.151 | +0.111 | -0.021 |

The margin peaks two sentences after a value is set and falls after that. On
the four region dataset it turns negative past five sentences, where the
within region signal drops below majority and the cross region read rises
above it.

## Conventions

All p values are two sided. Intervals over narrative positions use a cluster
bootstrap that resamples whole narratives. Causal claims use alpha equal to 2,
because a random direction of matched norm reaches significance at alpha 8.
MDL is reported over a sweep of probe regularisation.

## Limitations

Three models at 124M, 410M and 600M parameters. The text is synthetic. The
models report the true state at roughly chance without intervention, so causal
claims rest on paired changes in log odds rather than model accuracy. The
decomposition uses alpha 2 and 4, and a larger entity specific effect above
that range would be unreadable.
