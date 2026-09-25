"""Phase 5: probe-capacity ablation and the information-theoretic critique.

The objection to any probing result is "a strong enough probe can fit
anything".  Three answers, all computed here at a fixed layer:

  1. Sweep MLP capacity from 8 to 512 hidden units on the *real* task and on
     the Hewitt & Liang *control* task.  If accuracy on the real task rises
     while control accuracy also rises, the gain is capacity, not structure.
     Selectivity (real - control) at each capacity is the quantity to read.
  2. Prequential MDL at each capacity: a probe that only memorises pays for
     itself in code length, so codelength should *not* keep falling.
  3. The same two numbers for the non-contextual embedding baseline, so the
     comparison is like-for-like.

    python scripts/07_capacity_ablation.py --dataset pilot --model gpt2
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
from mwm.extract import load_extracted  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="pilot")
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--targets", nargs="*", default=None)
    ap.add_argument("--capacities", nargs="*", type=int,
                    default=[8, 16, 32, 64, 128, 256, 512])
    ap.add_argument("--max-train", type=int, default=6000)
    ap.add_argument("--max-test", type=int, default=3000)
    args = ap.parse_args()

    tag = args.model.replace("/", "__")
    X, E, meta, info = load_extracted(
        ROOT / "activations" / f"{args.dataset}__{tag}__natural")
    layers = info["layers"]
    targets = args.targets or sorted(meta[0]["labels"].keys())

    probe_path = ROOT / "results" / f"probe_{args.dataset}_{tag}.json"
    best = {}
    if probe_path.exists():
        pr = json.loads(probe_path.read_text(encoding="utf-8"))
        best = {t: pr["targets"][t]["best_layer"] for t in pr["targets"]}

    tr = _sub(P.split_mask(meta, "train", min_position=1), args.max_train)
    te = _sub(P.split_mask(meta, "test", min_position=1), args.max_test)

    out = {"meta": {"model": args.model, "dataset": args.dataset,
                    "capacities": args.capacities,
                    "n_train": len(tr), "n_test": len(te)},
           "targets": {}}

    for t in targets:
        layer = args.layer if args.layer is not None else best.get(
            t, layers[len(layers) // 2])
        li = layers.index(layer)
        y_all = P.target_vector(meta, t)
        y, classes = P.encode_labels(y_all)
        yc = P.control_labels(meta, y_all, P.split_mask(meta, "train"),
                              seed=abs(hash(t)) % 2**31)
        Xtr = np.asarray(X[tr, li], dtype=np.float32)
        Xte = np.asarray(X[te, li], dtype=np.float32)
        Etr = np.asarray(E[tr], dtype=np.float32)
        Ete = np.asarray(E[te], dtype=np.float32)

        print(f"\n--- {t} @ layer {layer} ({len(classes)} classes)", flush=True)
        rows = []
        lin = P.fit_eval(Xtr, y[tr], Xte, y[te], "linear")
        lin_c = P.fit_eval(Xtr, yc[tr], Xte, yc[te], "linear")
        lin_e = P.fit_eval(Etr, y[tr], Ete, y[te], "linear")
        lin_e_c = P.fit_eval(Etr, yc[tr], Ete, yc[te], "linear")
        rows.append({"capacity": 0, "probe": "linear",
                     "real": lin.accuracy, "control": lin_c.accuracy,
                     "selectivity": lin.accuracy - lin_c.accuracy,
                     "real_embed": lin_e.accuracy,
                     "control_embed": lin_e_c.accuracy,
                     "selectivity_embed": lin_e.accuracy - lin_e_c.accuracy,
                     "mdl": P.mdl_online_codelength(Xtr, y[tr], len(classes),
                                                    kind="linear")})
        print(f"  linear      real {lin.accuracy:.3f}  ctrl {lin_c.accuracy:.3f}"
              f"  sel {lin.accuracy - lin_c.accuracy:+.3f}"
              f"  | embed real {lin_e.accuracy:.3f}", flush=True)

        for h in args.capacities:
            r = P.fit_eval(Xtr, y[tr], Xte, y[te], "mlp", hidden=h)
            c = P.fit_eval(Xtr, yc[tr], Xte, yc[te], "mlp", hidden=h)
            re_ = P.fit_eval(Etr, y[tr], Ete, y[te], "mlp", hidden=h)
            ce_ = P.fit_eval(Etr, yc[tr], Ete, yc[te], "mlp", hidden=h)
            mdl = P.mdl_online_codelength(Xtr, y[tr], len(classes),
                                          kind="mlp", hidden=h)
            rows.append({"capacity": h, "probe": "mlp",
                         "real": r.accuracy, "control": c.accuracy,
                         "selectivity": r.accuracy - c.accuracy,
                         "real_embed": re_.accuracy,
                         "control_embed": ce_.accuracy,
                         "selectivity_embed": re_.accuracy - ce_.accuracy,
                         "mdl": mdl})
            print(f"  mlp h={h:<4d} real {r.accuracy:.3f}  ctrl {c.accuracy:.3f}"
                  f"  sel {r.accuracy - c.accuracy:+.3f}"
                  f"  MDL {mdl['codelength_kbits']:.1f}kb "
                  f"({mdl['compression']:.2f}x)", flush=True)

        out["targets"][t] = {"layer": layer, "classes": classes, "rows": rows}

    path = ROOT / "results" / f"capacity_{args.dataset}_{tag}.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {path}")


def _sub(mask: np.ndarray, cap: int, seed: int = 0) -> np.ndarray:
    idx = np.flatnonzero(mask)
    if len(idx) <= cap:
        return idx
    return np.random.default_rng(seed).choice(idx, cap, replace=False)


if __name__ == "__main__":
    main()
