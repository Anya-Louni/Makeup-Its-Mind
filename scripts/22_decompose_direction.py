"""Does the entity-specific component of the direction do anything causally?

The wrong-region result has two readings and they are not equivalent.

  (a) the model has no usable binding representation
  (b) the model has one, and our difference-of-means direction mostly captured
      the shared value component, so both directions carry the same "dewy"
      vector and of course they interchange

Reading (b) is the stronger objection and the earlier design could not rule it
out. This experiment separates the two components explicitly and steers with
each one on its own.

For a value pair (src -> dst) and two entities E1, E2, take

    d1 = mean(X | E1 = dst) - mean(X | E1 = src)
    d2 = mean(X | E2 = dst) - mean(X | E2 = src)

and decompose exactly:

    shared        s = (d1 + d2) / 2
    differential  r = (d1 - d2) / 2          so that  d1 = s + r,  d2 = s - r

`s` is the component both entities agree on, i.e. the value. `r` is the
component that distinguishes them, i.e. everything entity-specific that a
difference of means can see. All arms are rescaled to the norm of d1 so the
comparison is about direction and not magnitude.

The split is only clean when s and r are orthogonal, and

    s . r = (|d1|^2 - |d2|^2) / 4

so they are orthogonal exactly when the two class directions have equal norms.
A norm mismatch leaks shared value content into r, and after r is rescaled to
|d1| that leak is amplified. An effect seen on the `differential` arm could
then be the value component in disguise. The `orthogonal` arm removes the
ambiguity: it is r with s projected out,

    r_perp = r - (r . s_hat) s_hat

rescaled to |d1| like the others. cos(s, r) and |d2| / |d1| are recorded per
trial so the size of the leak is reported rather than assumed.

Predictions, written before the run:

  shared arm        raises the read-out for BOTH entities by similar amounts,
                    so the signed difference D = delta(E1) - delta(E2) is ~0.
  differential arm  if the entity-specific component is causally used, it
                    raises E1 and lowers E2, so D > 0.
                    If it is not used, neither read-out moves and D ~ 0.
  full arm          contains s, so it raises both. This reproduces the earlier
                    result and is included as a check on the harness.
  orthogonal arm    r with the shared component projected out. This is the
                    strict test of the entity-specific claim, since it cannot
                    carry any of the value direction.
  random arm        nothing.

D for the differential arm is the quantity the objection turns on. A clearly
positive D means the model does carry a causally effective binding component
and our earlier direction simply failed to isolate it. A D indistinguishable
from zero, with the shared arm working, means the entity-specific component
recovered by a difference of means is not causally necessary for this read-out.

    python scripts/22_decompose_direction.py --dataset pilot --model gpt2 \
        --entity lips --other-entity eyes --attribute finish
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mwm import intervene as IV  # noqa: E402
from mwm import models as M  # noqa: E402
from mwm import probe as P  # noqa: E402
from mwm.extract import load_extracted  # noqa: E402


def boot(x, n_boot=10000, seed=0):
    x = np.asarray(x, float)
    if len(x) == 0:
        return {"mean": float("nan"), "ci95": [float("nan")] * 2,
                "p_two_sided": float("nan")}
    rng = np.random.default_rng(seed)
    dr = x[rng.integers(0, len(x), (n_boot, len(x)))].mean(axis=1)
    out = {"mean": float(x.mean()),
           "ci95": [float(np.quantile(dr, .025)), float(np.quantile(dr, .975))]}
    out.update(P.bootstrap_pvalues(dr))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="pilot")
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--entity", default="lips")
    ap.add_argument("--other-entity", default="eyes")
    ap.add_argument("--attribute", default="finish")
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--alphas", nargs="*", type=float, default=[2.0, 4.0])
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    tag = args.model.replace("/", "__")
    t1 = f"{args.entity}.{args.attribute}"
    t2 = f"{args.other_entity}.{args.attribute}"
    X, E, meta, info = load_extracted(
        ROOT / "activations" / f"{args.dataset}__{tag}__natural")
    pr = json.loads((ROOT / "results" / f"probe_{args.dataset}_{tag}.json")
                    .read_text(encoding="utf-8"))

    layer = args.layer
    if layer is None:
        # edit mid-network: deep enough to be a representation, far enough from
        # the unembedding that the edit is not just a logit nudge
        layer = max(1, info["layers"][-1] // 2)
    li = info["layers"].index(layer)
    print(f"edit at hidden_states[{layer}] of {info['layers'][-1]}", flush=True)

    y1, c1 = P.encode_labels(P.target_vector(meta, t1))
    y2, c2 = P.encode_labels(P.target_vector(meta, t2))
    spec = IV.READOUTS[args.attribute]
    values = [v for v in spec.options if v in c1 and v in c2]
    if len(values) < 2:
        raise SystemExit("need at least two shared values")

    tr = np.flatnonzero(P.split_mask(meta, "train", min_position=1))
    rng = np.random.default_rng(args.seed)
    if len(tr) > 8000:
        tr = rng.choice(tr, 8000, False)
    A = np.asarray(X[tr, li], dtype=np.float32)

    def dmean(y, classes, src, dst):
        a = A[y[tr] == classes.index(dst)].mean(axis=0)
        b = A[y[tr] == classes.index(src)].mean(axis=0)
        return a - b

    cand = [m for m in meta if m["split"] == "test" and m["position"] >= 1
            and m["labels"][t1] in values and m["labels"][t2] in values]
    rng.shuffle(cand)
    cand = cand[:args.trials]
    print(f"{len(cand)} trials, values {values}", flush=True)

    geom = defaultdict(list)
    rows = []
    with M.loaded_model(args.model) as obj:
        fast = IV.first_tokens_distinct(obj, spec)
        score = IV.readout_scores_fast if fast else IV.readout_scores

        for n, m in enumerate(cand):
            text = m["text"]
            src = m["labels"][t1]
            dst = str(rng.choice([v for v in values if v != src]))

            d1 = dmean(y1, c1, src, dst)
            d2 = dmean(y2, c2, src, dst)
            s = (d1 + d2) / 2.0
            r = (d1 - d2) / 2.0
            n1 = float(np.linalg.norm(d1))
            n2 = float(np.linalg.norm(d2))
            ns, nr = float(np.linalg.norm(s)), float(np.linalg.norm(r))
            geom["frac_differential"].append(nr / n1)
            geom["cos_d1_d2"].append(float(d1 @ d2 / (n1 * n2)))
            geom["cos_s_r"].append(float(s @ r / (ns * nr)))
            geom["norm_ratio_d2_d1"].append(n2 / n1)

            # r with the shared component projected out, so the arm carries no
            # part of the value direction whatever the two norms do.
            r_perp = r - (r @ s) / (ns * ns) * s

            def unit(v):
                nv = np.linalg.norm(v)
                return torch.from_numpy((v / nv * n1).astype(np.float32))
            g = rng.standard_normal(d1.shape[0])
            arms = {"full": unit(d1), "shared": unit(s),
                    "differential": unit(r), "orthogonal": unit(r_perp),
                    "random": unit(g)}

            base1 = score(obj, text, args.entity, spec)
            base2 = score(obj, text, args.other_entity, spec)
            b1, b2 = IV.logit_gap(base1, dst), IV.logit_gap(base2, dst)

            for arm, d in arms.items():
                for alpha in args.alphas:
                    hk = IV.ResidualAdd(d, alpha, positions=None)
                    s1 = score(obj, text, args.entity, spec,
                               hook_layer=layer - 1, hook=hk)
                    s2 = score(obj, text, args.other_entity, spec,
                               hook_layer=layer - 1,
                               hook=IV.ResidualAdd(d, alpha))
                    rows.append({
                        "trial": n, "arm": arm, "alpha": alpha,
                        "src": src, "dst": dst,
                        "d_target": IV.logit_gap(s1, dst) - b1,
                        "d_other": IV.logit_gap(s2, dst) - b2,
                    })
            if (n + 1) % 20 == 0:
                print(f"  {n + 1}/{len(cand)}", flush=True)

    summary = {}
    for arm in ("full", "shared", "differential", "orthogonal", "random"):
        for alpha in args.alphas:
            rs = [x for x in rows if x["arm"] == arm and x["alpha"] == alpha]
            if not rs:
                continue
            dt = np.array([x["d_target"] for x in rs])
            do = np.array([x["d_other"] for x in rs])
            summary[f"{arm}|{alpha}"] = {
                "n": len(rs),
                "delta_target": boot(dt),
                "delta_other": boot(do),
                "difference": boot(dt - do),
            }

    out = {"meta": {"model": args.model, "dataset": args.dataset,
                    "target": t1, "other": t2, "layer": layer,
                    "alphas": args.alphas, "n_trials": len(cand),
                    "mean_frac_differential": float(np.mean(geom["frac_differential"])),
                    "mean_cos_d1_d2": float(np.mean(geom["cos_d1_d2"])),
                    "mean_cos_s_r": float(np.mean(geom["cos_s_r"])),
                    "max_abs_cos_s_r": float(np.max(np.abs(geom["cos_s_r"]))),
                    "mean_norm_ratio_d2_d1": float(np.mean(geom["norm_ratio_d2_d1"]))},
           "summary": summary, "rows": rows}
    path = (ROOT / "results" /
            f"decompose_{args.dataset}_{tag}_{t1}_L{layer}.json")
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(f"\ncos(d1, d2) = {out['meta']['mean_cos_d1_d2']:.3f};  "
          f"differential component is {100 * out['meta']['mean_frac_differential']:.1f}% "
          f"of the full direction by norm")
    _m = out["meta"]
    print(f"cos(s, r) = {_m['mean_cos_s_r']:+.3f} "
          f"(max |cos| {_m['max_abs_cos_s_r']:.3f}), "
          f"|d2|/|d1| = {_m['mean_norm_ratio_d2_d1']:.3f}. "
          f"s and r are orthogonal only when that ratio is 1.\n")
    print(f"{'arm':>14}{'a':>5}{'n':>5}"
          f"{'delta target':>26}{'delta other':>26}{'difference':>26}{'p2':>9}")
    for k, v in summary.items():
        arm, a = k.split("|")
        f = lambda d: f"{d['mean']:>+7.3f} [{d['ci95'][0]:+.3f},{d['ci95'][1]:+.3f}]"
        print(f"{arm:>14}{a:>5}{v['n']:>5}  {f(v['delta_target']):>24}"
              f"  {f(v['delta_other']):>24}  {f(v['difference']):>24}"
              f"{v['difference']['p_two_sided']:>9.4f}")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
