"""Does the shuffled-order control agree with the other measures?

Taken as a flat average the shuffled control looks useless: scrambling sentence
order costs about 1.4 accuracy points across the twelve full-dataset targets,
which reads as "the model ignores order". That average is misleading, because
it pools targets whose best layer is the *embedding layer* -- which cannot
encode order at all -- with targets that genuinely use the transformer.

Conditioning on that fixes it. This script reports the shuffled delta split by
whether the best layer is contextual, and correlates it against best-layer
depth and selectivity. If those three independent measures pick out the same
targets, the control is working; it was simply reporting that most targets have
no temporal structure to destroy.

    python scripts/14_order_sensitivity.py --dataset full --model gpt2
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def pearson(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    da, db = a - a.mean(), b - b.mean()
    denom = np.sqrt((da ** 2).sum() * (db ** 2).sum())
    return float((da * db).sum() / denom) if denom else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="full")
    ap.add_argument("--model", default="gpt2")
    args = ap.parse_args()
    tag = f"{args.dataset}_{args.model.replace('/', '__')}"
    path = ROOT / "results" / f"probe_{tag}.json"
    pr = json.loads(path.read_text(encoding="utf-8"))

    rows = []
    for t in sorted(pr["targets"]):
        R = pr["targets"][t]
        bl = R["best_layer"]
        sh = R.get("shuffled", {}).get(str(bl))
        if not sh:
            continue
        rows.append({
            "target": t,
            "best_layer": bl,
            "natural": R["best_linear_accuracy"],
            "shuffled": sh["accuracy"],
            "delta": sh["accuracy"] - R["best_linear_accuracy"],
            "selectivity": R["selectivity"],
            "contextual": bl > 0,
        })
    if not rows:
        raise SystemExit("no shuffled results in this probe file")

    L = [r["best_layer"] for r in rows]
    D = [r["delta"] for r in rows]
    S = [r["selectivity"] for r in rows]
    ctx = [r["delta"] for r in rows if r["contextual"]]
    emb = [r["delta"] for r in rows if not r["contextual"]]

    out = {
        "dataset": args.dataset, "model": args.model, "n_targets": len(rows),
        "rows": rows,
        "mean_delta_all": float(np.mean(D)),
        "mean_delta_contextual": float(np.mean(ctx)) if ctx else None,
        "mean_delta_embedding_layer": float(np.mean(emb)) if emb else None,
        "n_contextual": len(ctx), "n_embedding_layer": len(emb),
        "corr_bestlayer_delta": pearson(L, D),
        "corr_bestlayer_selectivity": pearson(L, S),
        "corr_selectivity_delta": pearson(S, D),
    }

    print(f"{'target':<18}{'bestL':>6}{'natural':>9}{'shuffled':>10}"
          f"{'delta':>9}{'select':>9}")
    for r in sorted(rows, key=lambda r: -r["best_layer"]):
        print(f"{r['target']:<18}{r['best_layer']:>6}{r['natural']:>9.3f}"
              f"{r['shuffled']:>10.3f}{r['delta']:>+9.3f}"
              f"{r['selectivity']:>+9.3f}")
    print(f"\nmean delta, all targets            {out['mean_delta_all']:+.4f}")
    print(f"mean delta, best layer  > 0 (n={len(ctx)})  "
          f"{out['mean_delta_contextual']:+.4f}")
    print(f"mean delta, best layer == 0 (n={len(emb)})  "
          f"{out['mean_delta_embedding_layer']:+.4f}")
    print(f"\ncorr(best layer, shuffled delta)  {out['corr_bestlayer_delta']:+.3f}")
    print(f"corr(best layer, selectivity)     {out['corr_bestlayer_selectivity']:+.3f}")
    print(f"corr(selectivity, shuffled delta) {out['corr_selectivity_delta']:+.3f}")

    p = ROOT / "results" / f"order_sensitivity_{tag}.json"
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
