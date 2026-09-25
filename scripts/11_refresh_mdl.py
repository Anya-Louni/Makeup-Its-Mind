"""Recompute the MDL numbers in an existing probe result file.

Why this exists: the pilot reported linear-probe compression ratios below 1.0
for the colour targets (0.75x, 0.77x) -- apparently saying the hidden states
make the labels *more* expensive to transmit than a uniform code. The capacity
ablation then reported 1.08x-1.18x for MLP probes on the same layer and target.

The explanation is not convergence (both were computed with the same settings
in the same process). It is regularisation: prequential coding charges
-log2 p(y) on every block, the earliest blocks hold a handful of examples
against 768 features, and an unregularised logistic regression is confidently
wrong on them. Confident errors are exactly what MDL punishes. The MLP looks
better mostly because early stopping keeps it timid.

So this recomputes the linear MDL at several regularisation strengths and
records all of them, rather than letting a single arbitrary C stand in for
"how compressible is this representation". Previous values are preserved under
`mdl_superseded`.

    python scripts/11_refresh_mdl.py --dataset pilot --model gpt2
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sklearn.exceptions import ConvergenceWarning  # noqa: E402

from mwm import probe as P  # noqa: E402
from mwm.extract import load_extracted, load_pooled  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="pilot")
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--pooling", choices=["last", "mean"], default="last")
    ap.add_argument("--max-train", type=int, default=6000)
    ap.add_argument("--Cs", nargs="*", type=float,
                    default=[0.003, 0.01, 0.1, 1.0])
    args = ap.parse_args()

    tag = args.model.replace("/", "__")
    suffix = "" if args.pooling == "last" else f"_{args.pooling}"
    path = ROOT / "results" / f"probe_{args.dataset}_{tag}{suffix}.json"
    if not path.exists():
        raise SystemExit(f"no such file: {path}")
    pr = json.loads(path.read_text(encoding="utf-8"))

    d = ROOT / "activations" / f"{args.dataset}__{tag}__natural"
    X, E, meta, info = load_extracted(d)
    if args.pooling == "mean":
        X = load_pooled(d)

    tr = np.flatnonzero(P.split_mask(meta, "train",
                                     min_position=pr["meta"]["min_position"]))
    if len(tr) > args.max_train:
        tr = np.random.default_rng(0).choice(tr, args.max_train, False)
    Etr = np.asarray(E[tr], dtype=np.float32)

    n_converge_warnings = 0
    for t in sorted(pr["targets"]):
        R = pr["targets"][t]
        y, classes = P.encode_labels(P.target_vector(meta, t))
        if "mdl" in R:
            R["mdl_superseded"] = R["mdl"]
            R["mdl_superseded_reason"] = (
                "computed with a single unregularised probe (C=1.0), which is "
                "confidently wrong on the small early blocks of the "
                "prequential code; superseded by a sweep over C")
        R["mdl"] = {}
        R["mdl_by_C"] = {}
        layers = [int(k) for k in R["layers"]]
        for layer in sorted({layers[0], R["best_layer"], layers[-1]}):
            Xl = np.asarray(X[tr, info["layers"].index(layer)],
                            dtype=np.float32)
            per_C = {}
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always", ConvergenceWarning)
                for C in args.Cs:
                    per_C[str(C)] = P.mdl_online_codelength(
                        Xl, y[tr], len(classes), C=C)
                n_converge_warnings += sum(
                    issubclass(x.category, ConvergenceWarning) for x in w)
            R["mdl_by_C"][str(layer)] = per_C
            # headline number: the best code length over the sweep, which is
            # the honest answer to "can this representation compress the labels"
            best_C = min(per_C, key=lambda c: per_C[c]["codelength_kbits"])
            R["mdl"][str(layer)] = dict(per_C[best_C], selected_C=float(best_C))
        R["mdl_static_embedding_superseded"] = R.get("mdl_static_embedding")
        emb_by_C = {str(C): P.mdl_online_codelength(Etr, y[tr], len(classes),
                                                    C=C) for C in args.Cs}
        best_C = min(emb_by_C, key=lambda c: emb_by_C[c]["codelength_kbits"])
        R["mdl_static_embedding"] = dict(emb_by_C[best_C],
                                         selected_C=float(best_C))
        b = R["mdl"][str(R["best_layer"])]
        old = R.get("mdl_superseded", {}).get(str(R["best_layer"]), {})
        sweep = R["mdl_by_C"][str(R["best_layer"])]
        print(f"{t:<14} L{R['best_layer']:<3} best {b['compression']:.2f}x "
              f"@C={b['selected_C']} (was {old.get('compression', float('nan')):.2f}x)"
              f"  static-emb {R['mdl_static_embedding']['compression']:.2f}x",
              flush=True)
        print("               sweep: " + "  ".join(
            f"C={c}:{v['compression']:.2f}x" for c, v in sweep.items()),
              flush=True)

    pr["meta"]["mdl_recomputed_with_max_iter"] = 1000
    pr["meta"]["mdl_convergence_warnings"] = n_converge_warnings
    path.write_text(json.dumps(pr, indent=2), encoding="utf-8")
    print(f"\nconvergence warnings during recompute: {n_converge_warnings}")
    print(f"rewrote {path}")


if __name__ == "__main__":
    main()
