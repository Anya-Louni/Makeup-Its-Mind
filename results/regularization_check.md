# Probe regularisation check (eyes.color, gpt2, layer 11, pilot)

Asked whether the fixed `C = 1.0` in the linear probe was costing accuracy.
n_train = 3879, n_val = 831, n_test = 839.

| C | train | val | test |
|---|---|---|---|
| 0.001 | 0.568 | 0.337 | 0.336 |
| 0.003 | 0.629 | 0.337 | 0.343 |
| 0.01  | 0.694 | 0.338 | 0.350 |
| 0.03  | 0.750 | 0.331 | 0.365 |
| 0.1   | 0.810 | 0.314 | 0.364 |
| 1.0   | 0.897 | 0.323 | 0.348 |

Conclusion: selecting C on the validation split would pick 0.01, giving test
0.350 against 0.348 at the default. Regularisation strength is not the
limiting factor here, so the reported numbers stand without per-layer tuning.

The striking number is the train/test gap at C = 1.0: 0.897 vs 0.348. That is
the split discipline doing its job -- train and test share no wordings at all,
so the probe can memorise phrasings on train and still has to fall back on
whatever is genuinely phrasing-invariant at test time. The ~0.35 is that
phrasing-invariant part.
