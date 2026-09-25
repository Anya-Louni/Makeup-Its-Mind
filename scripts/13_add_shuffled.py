"""Add the shuffled-order comparison to an existing probe result file.

The full-dataset probing run started before the shuffled condition had finished
extracting, so it ran with --skip-shuffled. Re-running the whole layer sweep to
recover one number per target would cost an hour; the shuffled comparison only
needs a fit at each target's already-chosen best layer, which costs minutes.

    python scripts/13_add_shuffled.py --dataset full --model gpt2
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mwm import probe as P  # noqa: E402
from mwm.extract import load_extracted, load_pooled  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="full")
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--pooling", choices=["last", "mean"], default="last")
    ap.add_argument("--max-train", type=int, default=8000)
    ap.add_argument("--max-test", type=int, default=4000)
    args = ap.parse_args()

    tag = args.model.replace("/", "__")
    suffix = "" if args.pooling == "last" else f"_{args.pooling}"
    path = ROOT / "results" / f"probe_{args.dataset}_{tag}{suffix}.json"
    if not path.exists():
        raise SystemExit(f"no such file: {path}")
    pr = json.loads(path.read_text(encoding="utf-8"))

    d = ROOT / "activations" / f"{args.dataset}__{tag}__shuffled"
    Xs, Es, Ms, info = load_extracted(d)
    if args.pooling == "mean":
        Xp = load_pooled(d)
        if Xp is None:
            raise SystemExit("shuffled extraction has no Xmean.npy")
        Xs = Xp

    mp = pr["meta"]["min_position"]
    rng = np.random.default_rng(0)

    def sub(mask, cap):
        idx = np.flatnonzero(mask)
        return idx if len(idx) <= cap else rng.choice(idx, cap, False)

    tr = sub(P.split_mask(Ms, "train", min_position=mp), args.max_train)
    te = sub(P.split_mask(Ms, "test", min_position=mp), args.max_test)
    print(f"shuffled: train={len(tr)} test={len(te)}", flush=True)

    for t in sorted(pr["targets"]):
        R = pr["targets"][t]
        classes = R["classes"]
        lut = {c: i for i, c in enumerate(classes)}
        vals = P.target_vector(Ms, t)
        unseen = set(vals.tolist()) - set(lut)
        if unseen:
            print(f"  {t}: skipped, shuffled has unseen classes {unseen}")
            continue
        y = np.array([lut[v] for v in vals])
        layer = R["best_layer"]
        li = info["layers"].index(layer)
        res = P.fit_eval(np.asarray(Xs[tr, li], dtype=np.float32), y[tr],
                         np.asarray(Xs[te, li], dtype=np.float32), y[te],
                         "linear")
        R["shuffled"] = {str(layer): res.as_dict()}
        nat = R["best_linear_accuracy"]
        print(f"  {t:<18} L{layer:<3} shuffled {res.accuracy:.3f}  "
              f"natural {nat:.3f}  delta {res.accuracy - nat:+.3f}", flush=True)

    pr["meta"]["shuffled_added_separately"] = True
    path.write_text(json.dumps(pr, indent=2), encoding="utf-8")
    print(f"\nrewrote {path}")


if __name__ == "__main__":
    main()
