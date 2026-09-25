"""Causal intervention, second design.

Three things the first design did not do.

1. Distance stratification. The binding analysis showed the region-specific
   code decays: binding margin +0.113 at distance 0, -0.124 by distance 5+.
   So a single intervention averaged over all positions confounds the regime
   where the state is held with the regime where it is already gone. This
   samples matched trial sets NEAR (distance 0-1) and FAR (distance 5+) and
   reports them separately. The prediction, if the probe result is real, is a
   large effect NEAR and a small or absent one FAR.

2. A random-direction control. The cross-entity control shows the edit does
   not smear onto another region, but it does not show that the *direction*
   matters: a large enough push in any direction might move the read-out.
   Each trial is therefore also run with a random unit vector scaled to the
   same norm as the concept direction, same layer, same alpha. If the random
   arm moves the read-out as much as the concept arm, the causal claim fails.

3. A "wrong-direction" arm: the concept direction for the *other* region's
   attribute, same norm. Stronger than random -- it is a real feature
   direction, just bound to the wrong entity.

    python scripts/16_intervene_v2.py --dataset full --model gpt2 \
        --entity lips --other-entity eyes --attribute finish
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from importlib import import_module
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from mwm import intervene as IV  # noqa: E402
from mwm import models as M  # noqa: E402
from mwm import probe as P  # noqa: E402
from mwm.extract import load_extracted  # noqa: E402

distance_since_change = import_module("10_persistence").distance_since_change


def boot_ci(x, n_boot=10000, seed=0):
    x = np.asarray(x, float)
    if len(x) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    m = x[rng.integers(0, len(x), (n_boot, len(x)))].mean(axis=1)
    return float(x.mean()), float(np.quantile(m, .025)), float(np.quantile(m, .975))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="full")
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--entity", default="lips")
    ap.add_argument("--other-entity", default="eyes")
    ap.add_argument("--attribute", default="finish")
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--read-layer", type=int, default=None)
    ap.add_argument("--alphas", nargs="*", type=float, default=[0.0, 2.0, 8.0])
    ap.add_argument("--trials-per-bin", type=int, default=80)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    tag = args.model.replace("/", "__")
    target = f"{args.entity}.{args.attribute}"
    other = f"{args.other_entity}.{args.attribute}"
    X, E, meta, info = load_extracted(
        ROOT / "activations" / f"{args.dataset}__{tag}__natural")
    pr = json.loads((ROOT / "results" / f"probe_{args.dataset}_{tag}.json")
                    .read_text(encoding="utf-8"))

    layer = args.layer if args.layer is not None else pr["targets"][target]["best_layer"]
    layer = max(layer, info["layers"][1])
    li = info["layers"].index(layer)
    read_layer = args.read_layer if args.read_layer is not None else info["layers"][-1]
    if read_layer <= layer:
        read_layer = info["layers"][-1]
    ri = info["layers"].index(read_layer)

    y_t, classes = P.encode_labels(P.target_vector(meta, target))
    y_o, classes_o = P.encode_labels(P.target_vector(meta, other))
    spec = IV.READOUTS[args.attribute]
    values = [v for v in spec.options if v in classes]

    tr = np.flatnonzero(P.split_mask(meta, "train", min_position=1))
    rng = np.random.default_rng(args.seed)
    if len(tr) > 8000:
        tr = rng.choice(tr, 8000, False)
    Xtr = np.asarray(X[tr, li], dtype=np.float32)

    dirs, dirs_other = {}, {}
    for src in values:
        for dst in values:
            if src == dst:
                continue
            dirs[(src, dst)] = IV.direction_from_activations(
                Xtr, y_t[tr], classes.index(src), classes.index(dst))
            if src in classes_o and dst in classes_o:
                dirs_other[(src, dst)] = IV.direction_from_activations(
                    Xtr, y_o[tr], classes_o.index(src), classes_o.index(dst))

    def read_probe(rows, yy, idx):
        A = np.asarray(X[rows, idx], dtype=np.float32)
        sc = StandardScaler().fit(A)
        clf = P.make_probe("linear")
        clf.fit(sc.transform(A), yy[rows])
        return sc, clf

    sc_t, clf_t = read_probe(tr, y_t, ri)
    sc_o, clf_o = read_probe(tr, y_o, ri)
    del Xtr

    dist = distance_since_change(meta, target)
    bins = {"near": lambda d: d <= 1, "far": lambda d: d >= 5}
    cand = {}
    for name, fn in bins.items():
        pool = [m for m in meta
                if m["split"] == "test" and m["position"] >= 1
                and m["labels"][target] in values
                and m["labels"][other] in values
                and fn(dist[m["row"]])]
        rng.shuffle(pool)
        cand[name] = pool[:args.trials_per_bin]
        print(f"{name}: {len(cand[name])} trials (pool {len(pool)})", flush=True)

    rows = []
    with M.loaded_model(args.model) as obj:
        fast = IV.first_tokens_distinct(obj, spec)
        score = IV.readout_scores_fast if fast else IV.readout_scores
        for bin_name, pool in cand.items():
            for n, m in enumerate(pool):
                text = m["text"]
                true_v = m["labels"][target]
                dst = str(rng.choice([v for v in values if v != true_v]))
                d_concept = dirs[(true_v, dst)]
                norm = float(d_concept.norm())

                g = torch.from_numpy(
                    rng.standard_normal(d_concept.shape[0]).astype(np.float32))
                d_random = g / g.norm() * norm
                d_wrong = dirs_other.get((true_v, dst))
                if d_wrong is not None and float(d_wrong.norm()) > 0:
                    d_wrong = d_wrong / d_wrong.norm() * norm

                arms = {"concept": d_concept, "random": d_random}
                if d_wrong is not None:
                    arms["other_entity_direction"] = d_wrong

                for arm, d in arms.items():
                    for alpha in args.alphas:
                        hook = (None if alpha == 0.0 else
                                IV.ResidualAdd(d, alpha, positions=None))
                        h = IV.hidden_under_hook(obj, text, read_layer,
                                                 layer - 1, hook).numpy()[None]
                        s_t = score(obj, text, args.entity, spec,
                                    hook_layer=layer - 1, hook=hook)
                        s_o = score(obj, text, args.other_entity, spec,
                                    hook_layer=layer - 1,
                                    hook=(None if alpha == 0.0 else
                                          IV.ResidualAdd(d, alpha)))
                        rows.append({
                            "bin": bin_name, "trial": n, "arm": arm,
                            "alpha": alpha, "true": true_v, "steered_to": dst,
                            "other_true": m["labels"][other],
                            "distance": int(dist[m["row"]]),
                            "probe_pred": classes[int(clf_t.predict(
                                sc_t.transform(h))[0])],
                            "probe_pred_other": classes_o[int(clf_o.predict(
                                sc_o.transform(h))[0])],
                            "beh_gap_steered": IV.logit_gap(s_t, dst),
                            "beh_gap_true": IV.logit_gap(s_t, true_v),
                            "beh_gap_other_true": IV.logit_gap(s_o,
                                                               m["labels"][other]),
                        })
                if (n + 1) % 20 == 0:
                    print(f"  {bin_name} {n + 1}/{len(pool)}", flush=True)

    summary = summarise(rows, args.alphas)
    out = {"meta": {"model": args.model, "dataset": args.dataset,
                    "target": target, "other": other, "edit_layer": layer,
                    "read_layer": read_layer, "alphas": args.alphas,
                    "trials_per_bin": args.trials_per_bin},
           "summary": summary, "rows": rows}
    path = (ROOT / "results" /
            f"interventionv2_{args.dataset}_{tag}_{target}_L{layer}.json")
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("\n" + render(summary))
    print(f"\nwrote {path}")


def summarise(rows, alphas):
    base = {(r["bin"], r["arm"], r["trial"]): r
            for r in rows if r["alpha"] == 0.0}
    groups = defaultdict(list)
    for r in rows:
        groups[(r["bin"], r["arm"], r["alpha"])].append(r)

    out = {}
    for (b, arm, a), rs in groups.items():
        d_t = np.array([r["beh_gap_steered"]
                        - base[(b, arm, r["trial"])]["beh_gap_steered"]
                        for r in rs])
        d_o = np.array([r["beh_gap_other_true"]
                        - base[(b, arm, r["trial"])]["beh_gap_other_true"]
                        for r in rs])
        m, lo, hi = boot_ci(d_t)
        mo, loo, hio = boot_ci(d_o)
        out[f"{b}|{arm}|{a}"] = {
            "n": len(rs),
            "delta_logodds_steered": {"mean": m, "ci95": [lo, hi]},
            "delta_logodds_other": {"mean": mo, "ci95": [loo, hio]},
            "probe_rate_steered": float(np.mean(
                [r["probe_pred"] == r["steered_to"] for r in rs])),
            "probe_acc_true": float(np.mean(
                [r["probe_pred"] == r["true"] for r in rs])),
        }
    return out


def render(summary) -> str:
    lines = [f"{'bin':>5}{'arm':>24}{'alpha':>7}{'n':>5}"
             f"{'d log-odds steered':>28}{'probe->steered':>16}"]
    for k in sorted(summary, key=lambda k: (k.split('|')[0], k.split('|')[1],
                                            float(k.split('|')[2]))):
        b, arm, a = k.split("|")
        s = summary[k]
        d = s["delta_logodds_steered"]
        lines.append(f"{b:>5}{arm:>24}{a:>7}{s['n']:>5}"
                     f"   {d['mean']:>+8.3f} [{d['ci95'][0]:+.3f},"
                     f"{d['ci95'][1]:+.3f}]{s['probe_rate_steered']:>16.3f}")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
