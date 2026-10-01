# Findings

Four models, 124M to 1.7B parameters, run on byte identical narratives.
`CORRECTIONS.md` lists the twenty three claims revised on the way here.

## Main result

A difference of means concept direction holds two parts. One is shared by
every entity that takes the value. The other is specific to one entity.
Steering with each part alone separates them.

| model | cos | region specific share | shared effect | region specific difference | p |
|---|---|---|---|---|---|
| gpt2-124M | 0.848 | 28.4% | +0.183 | +0.016 [+0.009, +0.023] | 0.0000 |
| pythia-410m | 0.743 | 37.7% | +0.190 | -0.000 [-0.006, +0.005] | 0.8476 |
| Qwen3-0.6B | 0.740 | 35.5% | +0.317 | -0.000 [-0.010, +0.010] | 0.9106 |
| Qwen3-1.7B | 0.724 | 37.2% | +0.556 | +0.007 [-0.008, +0.022] | 0.3598 |

The shared part carries the causal effect in every model and triples from
+0.183 to +0.556 across a fourteen fold parameter range. The region specific
part is significant only in the smallest model, where section 4.2 of the
write up argues the effect could be lexical. At 1.7B it is +0.007, p = 0.36.

Across the arm by model by gain grid of 38 difference tests,
Benjamini-Hochberg at 0.05 keeps four, all of them gpt2, on the region
specific arm and on its orthogonalised version at both gains.

## Is the decomposition clean?

`s` and `r` are orthogonal only when the two class directions have equal
norms, since `s . r = (|d1|^2 - |d2|^2) / 4`. They do not, quite, so some
value content leaks into `r`. The `orthogonal` arm steers `r` with `s`
projected out and cannot carry any of the value direction.

| model | \|d2\|/\|d1\| | cos(s, r) | raw `r` | `r` with `s` removed | p |
|---|---|---|---|---|---|
| gpt2-124M | 1.065 | -0.133 | +0.016 | +0.013 [+0.007, +0.020] | 0.0000 |
| pythia-410m | 1.106 | -0.160 | -0.000 | +0.002 [-0.004, +0.007] | 0.5818 |
| Qwen3-0.6B | 1.000 | -0.010 | -0.000 | +0.002 [-0.008, +0.013] | 0.6528 |
| Qwen3-1.7B | 1.007 est | -0.010 est | +0.007 | not run | |

Four fifths of the gpt2 effect survives the projection and the two nulls stay
null, so neither result is an artifact of the norm mismatch. The leak is
negative in all three measured models, so it lowers the read outs rather than
raising them. The 1.7B row is estimated from the stored geometry, since its
activations were produced on a hosted GPU and only the result files came
back. `colab_scale_arm.ipynb` produces the measured version.

## The region specific component through a read out that binds

At 1.7B the eyes read out reaches a selectivity ratio of 2.00, which makes it
the one instrument here with a demonstrated ability to tell the two regions
apart. Steering `r` toward the lips should lower the eyes read out. It raises
it, by +0.062 [+0.037, +0.088], close to its +0.069 push on the lips. The gap
between the two is +0.007. A read out that can register a region registers
none here.

## Central claim

> The value component of a steering direction strengthens with scale and the
> entity specific component does not appear. A probe recovers the binding and
> the model's output does not use it.

## What scale changes

| quantity | 124M | 410M | 600M | 1.7B |
|---|---|---|---|---|
| shared component causal effect | +0.183 | +0.190 | +0.317 | +0.556 |
| region specific causal effect | +0.016 | -0.000 | -0.000 | +0.007 |
| region specific, shared part removed | +0.013 | +0.002 | +0.002 | not run |
| targets clearing the lexical baseline | 2 of 4 | 1 of 4 | 2 of 4 | 4 of 4 |
| best read out selectivity ratio | 1.57 | 1.45 | 1.33 | 2.00 |
| peak binding margin | +0.105 | +0.115 | +0.151 | +0.199 |

Three things improve with size. The value code strengthens, the
representation improves until all four state variables clear the lexical
baseline at 1.7B, and one of the two read outs begins to bind, reaching a
selectivity ratio of 2.00 against roughly 1.07 in the smaller models. The
causal binding component does not appear anywhere in that range.

## Read out sensitivity against leakage

Sensitivity changes the value of the region the question named. Leakage
changes the other region's value. A read out that binds moves on the first
and not the second.

| model | read out | sensitivity | leakage | ratio |
|---|---|---|---|---|
| gpt2-124M | lips | +0.201 | +0.316 | 0.64 |
| gpt2-124M | eyes | +0.358 | +0.228 | 1.57 |
| pythia-410m | lips | +0.385 | +0.544 | 0.71 |
| pythia-410m | eyes | +0.551 | +0.380 | 1.45 |
| Qwen3-0.6B | lips | +0.530 | +0.400 | 1.33 |
| Qwen3-0.6B | eyes | +0.379 | +0.500 | 0.76 |
| Qwen3-1.7B | lips | +0.654 | +0.598 | 1.09 |
| Qwen3-1.7B | eyes | +0.699 | +0.350 | 2.00 |

## Baseline gaps

| target | gpt2-124M | pythia-410m | Qwen3-0.6B | Qwen3-1.7B |
|---|---|---|---|---|
| `eyes.color` | +0.062 * | +0.031 | +0.111 * | +0.072 * |
| `eyes.finish` | +0.097 * | +0.014 | +0.120 * | +0.098 * |
| `lips.color` | -0.044 | +0.064 * | -0.031 | +0.062 * |
| `lips.finish` | -0.058 * | +0.024 | -0.048 | +0.098 * |

Qwen3-1.7B is the first model to clear the baseline on all four, and it also
breaks the agreement between gpt2 and Qwen3-0.6B by taking both lip targets
positive where those two put them negative. No target is significant in all
four models.

## Binding margin by distance

| model | 0 | 1 | 2 | 3-4 | 5+ |
|---|---|---|---|---|---|
| gpt2-124M | +0.073 | +0.096 | +0.105 | +0.064 | +0.049 |
| pythia-410m | +0.036 | +0.104 | +0.115 | +0.104 | +0.045 |
| Qwen3-0.6B | +0.119 | +0.141 | +0.151 | +0.111 | -0.021 |
| Qwen3-1.7B | +0.129 | +0.188 | +0.199 | +0.099 | +0.068 |

## Conventions

All p values are two sided. Intervals over narrative positions use a cluster
bootstrap that resamples whole narratives. Causal claims use a steering gain
of g = 2, a multiple of the full direction's norm, written as g to keep it
apart from the significance level. At g = 8 a random direction of matched
norm also reaches significance. MDL is reported over a sweep of probe
regularisation.

## Limitations

Four models at 124M to 1.7B. Feng and Steinhardt report binding ID vectors in
every sufficiently large model of the Pythia and LLaMA families without naming
a cutoff we could verify, so a null across this range is consistent with the
mechanism emerging at a larger scale and is not evidence against it. The text is synthetic. The models report the true state at
roughly chance without intervention, so causal claims rest on paired changes
in log odds. The decomposition uses g = 2 and g = 4.
