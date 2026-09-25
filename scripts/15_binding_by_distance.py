"""Is the attribute bound to the right region, at distances where it is decodable?

The aggregate binding test trains a probe on one region and reads it against
another. On the full dataset that gave cross-entity accuracy equal to
within-entity accuracy, which looks like a flat failure to bind.

But the aggregate is dominated by positions far from the last update, where the
attribute is not decodable *at all* -- both probes are near majority there, so
of course they agree. The question only has content where the state is actually
recoverable.

This stratifies both the within-entity and the cross-entity read by distance
since the variable last changed, and reports the binding margin

    (within-entity accuracy - majority) - (cross-entity accuracy - majority)

at each distance. A representation that binds should show a positive margin
wherever the attribute is decodable; one that only encodes "this value occurred
somewhere" should show ~0 everywhere.

    python scripts/15_binding_by_distance.py --dataset full --model gpt2
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sklearn.preprocessing import StandardScaler  # noqa: E402

from mwm import probe as P  # noqa: E402
from mwm.extract import load_extracted  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from importlib import import_module  # noqa: E402

_persist = import_module("10_persistence")
distance_since_change = _persist.distance_since_change


def margin_cluster_bootstrap(pred, y_true, y_other, maj, groups,
                             n_boot: int = 2000, seed: int = 0) -> dict:
    """CI on the binding margin, resampling whole narratives.

    Positions inside a narrative share its wordings and its state, so they are
    not independent draws. An interval built by treating them -- or the 36
    region pairs, which are computed over the same narratives -- as independent
    is far too narrow. The narrative is the unit that is exchangeable here.
    """
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(groups, return_inverse=True)
    idx_by_group = [np.flatnonzero(inv == g) for g in range(len(uniq))]

    def margin(rows):
        w = np.mean(pred[rows] == y_true[rows]) - np.mean(y_true[rows] == maj)
        c = np.mean(pred[rows] == y_other[rows]) - np.mean(y_other[rows] == maj)
        return w - c

    draws = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        draws[b] = margin(np.concatenate([idx_by_group[g] for g in pick]))
    lo, hi = np.quantile(draws, [0.025, 0.975])
    out = {"ci95": [float(lo), float(hi)], "n_narratives": int(len(uniq))}
    out.update(P.bootstrap_pvalues(draws))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="full")
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--attributes", nargs="*",
                    default=["color", "finish", "coverage"])
    ap.add_argument("--max-train", type=int, default=8000)
    ap.add_argument("--bins", nargs="*", type=int, default=[0, 1, 2, 3, 5])
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()

    tag = args.model.replace("/", "__")
    X, E, meta, info = load_extracted(
        ROOT / "activations" / f"{args.dataset}__{tag}__natural")
    pr = json.loads((ROOT / "results" / f"probe_{args.dataset}_{tag}.json")
                    .read_text(encoding="utf-8"))
    targets = sorted(pr["targets"])

    tr = np.flatnonzero(P.split_mask(meta, "train", min_position=1))
    te = np.flatnonzero(P.split_mask(meta, "test", min_position=1))
    if len(tr) > args.max_train:
        tr = np.random.default_rng(0).choice(tr, args.max_train, False)

    out = {"meta": {"dataset": args.dataset, "model": args.model}, "pairs": {}}

    for attr in args.attributes:
        group = [t for t in targets if t.endswith("." + attr)]
        for t in group:
            layer = pr["targets"][t]["best_layer"]
            li = info["layers"].index(layer)
            y, classes = P.encode_labels(P.target_vector(meta, t))
            Xtr = np.asarray(X[tr, li], dtype=np.float32)
            sc = StandardScaler().fit(Xtr)
            clf = P.make_probe("linear")
            clf.fit(sc.transform(Xtr), y[tr])
            pred = clf.predict(sc.transform(
                np.asarray(X[te, li], dtype=np.float32)))
            maj = np.bincount(y[tr]).argmax()

            dist = distance_since_change(meta, t)[te]
            groups = np.array([meta[i]["narrative_id"] for i in te])
            for other in group:
                if other == t:
                    continue
                yo, classes_o = P.encode_labels(P.target_vector(meta, other))
                if classes_o != classes:
                    continue
                rows = []
                edges = args.bins + [10 ** 9]
                for lo, hi in zip(edges[:-1], edges[1:]):
                    m = (dist >= lo) & (dist < hi)
                    if m.sum() < 30:
                        continue
                    own = float(np.mean(pred[m] == y[te][m]))
                    own_maj = float(np.mean(y[te][m] == maj))
                    cross = float(np.mean(pred[m] == yo[te][m]))
                    cross_maj = float(np.mean(yo[te][m] == maj))
                    ci = margin_cluster_bootstrap(
                        pred[m], y[te][m], yo[te][m], maj, groups[m],
                        n_boot=args.n_boot)
                    rows.append({
                        "distance_min": lo,
                        "distance_max": None if hi > 10 ** 8 else hi - 1,
                        "n": int(m.sum()),
                        "n_narratives": ci["n_narratives"],
                        "within_acc": own, "within_majority": own_maj,
                        "cross_acc": cross, "cross_majority": cross_maj,
                        "binding_margin": (own - own_maj) - (cross - cross_maj),
                        "binding_margin_ci95": ci["ci95"],
                        "p_two_sided": ci["p_two_sided"],
                    })
                out["pairs"][f"{t} -> {other}"] = {"layer": layer, "bins": rows}

    path = ROOT / "results" / f"binding_by_distance_{args.dataset}_{tag}.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(f"{'probe -> read against':<34}{'dist':>6}{'n':>7}"
          f"{'within-maj':>12}{'cross-maj':>11}{'margin':>9}"
          f"{'95% CI (cluster)':>22}{'p2':>8}")
    for k, v in out["pairs"].items():
        for b in v["bins"]:
            d = (f"{b['distance_min']}+" if b["distance_max"] is None
                 else (str(b["distance_min"])
                       if b["distance_max"] == b["distance_min"]
                       else f"{b['distance_min']}-{b['distance_max']}"))
            ci = b["binding_margin_ci95"]
            print(f"{k:<34}{d:>6}{b['n']:>7}"
                  f"{b['within_acc'] - b['within_majority']:>+12.3f}"
                  f"{b['cross_acc'] - b['cross_majority']:>+11.3f}"
                  f"{b['binding_margin']:>+9.3f}"
                  f"   [{ci[0]:+.3f},{ci[1]:+.3f}]{b['p_two_sided']:>8.4f}")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
