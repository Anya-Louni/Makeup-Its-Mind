"""State persistence: does the code survive intervening sentences?

The shuffled-order control turned out to be weak in the pilot -- accuracy
barely moves when the routine is scrambled, which tells us the model is not
leaning on plausible routine ordering, but says nothing about whether it
*maintains* state.

This is the sharper test, and it needs no new activations. For every position
we compute how many sentences have passed since that entity-attribute last
changed, then bin accuracy by that distance. A model that merely reflects the
current sentence collapses to chance as soon as the last update scrolls out of
the immediate context; a model that carries state stays flat.

The companion split is by whether the intervening sentences mentioned a
*different* value for the same attribute (a distractor or an update to the
other region) -- the case where a "most recent colour word" heuristic is
actively wrong.

    python scripts/10_persistence.py --dataset pilot --model gpt2
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sklearn.preprocessing import StandardScaler  # noqa: E402

from mwm import probe as P  # noqa: E402
from mwm.extract import load_extracted, load_pooled  # noqa: E402


def distance_since_change(meta: list[dict], target: str) -> np.ndarray:
    """Sentences since this variable last took a different value."""
    out = np.zeros(len(meta), dtype=int)
    by_narr: dict[str, list[dict]] = defaultdict(list)
    for m in meta:
        by_narr[m["narrative_id"]].append(m)
    for rows in by_narr.values():
        rows.sort(key=lambda m: m["position"])
        last_change = 0
        prev = None
        for m in rows:
            v = m["labels"][target]
            if prev is not None and v != prev:
                last_change = m["position"]
            prev = v
            out[m["row"]] = m["position"] - last_change
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="pilot")
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--pooling", choices=["last", "mean"], default="last")
    ap.add_argument("--max-train", type=int, default=6000)
    ap.add_argument("--bins", nargs="*", type=int, default=[0, 1, 2, 3, 5, 8])
    args = ap.parse_args()

    tag = args.model.replace("/", "__")
    d = ROOT / "activations" / f"{args.dataset}__{tag}__natural"
    X, E, meta, info = load_extracted(d)
    if args.pooling == "mean":
        Xp = load_pooled(d)
        if Xp is None:
            raise SystemExit("no Xmean.npy; re-run scripts/03_extract.py")
        X = Xp

    probe_path = ROOT / "results" / f"probe_{args.dataset}_{tag}.json"
    best = {}
    if probe_path.exists():
        pr = json.loads(probe_path.read_text(encoding="utf-8"))
        best = {t: pr["targets"][t]["best_layer"] for t in pr["targets"]}

    targets = sorted(meta[0]["labels"].keys())
    tr = np.flatnonzero(P.split_mask(meta, "train", min_position=1))
    te = np.flatnonzero(P.split_mask(meta, "test", min_position=1))
    if len(tr) > args.max_train:
        tr = np.random.default_rng(0).choice(tr, args.max_train, False)

    out = {"meta": {"model": args.model, "dataset": args.dataset,
                    "pooling": args.pooling}, "targets": {}}

    for t in targets:
        layer = best.get(t, info["layers"][len(info["layers"]) // 2])
        li = info["layers"].index(layer)
        y, classes = P.encode_labels(P.target_vector(meta, t))
        Xtr = np.asarray(X[tr, li], dtype=np.float32)
        sc = StandardScaler().fit(Xtr)
        clf = P.make_probe("linear")
        clf.fit(sc.transform(Xtr), y[tr])
        Xte = np.asarray(X[te, li], dtype=np.float32)
        pred = clf.predict(sc.transform(Xte))
        correct = (pred == y[te])

        dist = distance_since_change(meta, t)[te]
        maj_cls = np.bincount(y[tr]).argmax()

        rows = []
        edges = args.bins + [10 ** 9]
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (dist >= lo) & (dist < hi)
            if m.sum() < 15:
                continue
            rows.append({
                "distance_min": lo,
                "distance_max": None if hi > 10 ** 8 else hi - 1,
                "n": int(m.sum()),
                "accuracy": float(correct[m].mean()),
                "majority_here": float(np.mean(y[te][m] == maj_cls)),
            })
        out["targets"][t] = {"layer": layer, "classes": classes, "bins": rows}
        print(f"\n--- {t} @ L{layer}")
        for r in rows:
            hi = "+" if r["distance_max"] is None else f"-{r['distance_max']}"
            print(f"  {r['distance_min']}{hi:<4} sentences since change: "
                  f"acc {r['accuracy']:.3f}  (majority {r['majority_here']:.3f}, "
                  f"n={r['n']})")

    suffix = "" if args.pooling == "last" else f"_{args.pooling}"
    path = ROOT / "results" / f"persistence_{args.dataset}_{tag}{suffix}.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
