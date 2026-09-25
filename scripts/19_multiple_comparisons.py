"""Benjamini-Hochberg correction for the binding-by-distance tests.

The binding table reports "33 of 36 region pairs significant at p < 0.05" for
distance 0. With 36 tests per distance bin (180 in total) some of those are
expected to be false positives, so the raw count overstates the result.

Two families are corrected here and both are reported:

  per bin   36 tests -- the count quoted in the text for a single distance
  overall  180 tests -- every test in the table

FDR is the right correction rather than Bonferroni: the tests are positively
dependent (all 36 pairs are computed over the same narratives from overlapping
probes), which is exactly the regime where Benjamini-Hochberg is valid and
Bonferroni is needlessly conservative.

    python scripts/19_multiple_comparisons.py --dataset full --model gpt2
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


def benjamini_hochberg(pvals, alpha: float = 0.05):
    """Return (rejected mask, q-values) under BH FDR control."""
    p = np.asarray(pvals, float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * n / (np.arange(n) + 1)
    # enforce monotonicity from the largest p downward
    q = np.minimum.accumulate(q[::-1])[::-1]
    q_full = np.empty(n)
    q_full[order] = np.minimum(q, 1.0)
    return q_full <= alpha, q_full


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="full")
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args()

    tag = f"{args.dataset}_{args.model.replace('/', '__')}"
    src = ROOT / "results" / f"binding_by_distance_{tag}.json"
    data = json.loads(src.read_text(encoding="utf-8"))

    rows = []
    for pair, v in data["pairs"].items():
        for b in v["bins"]:
            d = (f"{b['distance_min']}+" if b["distance_max"] is None
                 else (str(b["distance_min"])
                       if b["distance_max"] == b["distance_min"]
                       else f"{b['distance_min']}-{b['distance_max']}"))
            rows.append({"pair": pair, "bin": d,
                         "margin": b["binding_margin"],
                         "p": b.get("p_two_sided", float("nan"))})

    ok = [r for r in rows if np.isfinite(r["p"])]
    if not ok:
        raise SystemExit("no two-sided p-values in the file; re-run "
                         "scripts/15_binding_by_distance.py")

    # ---- family 1: all tests together
    rej_all, q_all = benjamini_hochberg([r["p"] for r in ok], args.alpha)
    for r, rj, q in zip(ok, rej_all, q_all):
        r["q_overall"], r["sig_overall"] = float(q), bool(rj)

    # ---- family 2: within each distance bin
    by_bin = defaultdict(list)
    for r in ok:
        by_bin[r["bin"]].append(r)
    for d, rs in by_bin.items():
        rej, q = benjamini_hochberg([r["p"] for r in rs], args.alpha)
        for r, rj, qq in zip(rs, rej, q):
            r["q_bin"], r["sig_bin"] = float(qq), bool(rj)

    def key(d):
        return int(d.replace("+", "").split("-")[0])

    print(f"Benjamini-Hochberg, alpha = {args.alpha}\n")
    print(f"{'bin':>6}{'tests':>7}{'mean margin':>13}{'raw p<.05':>11}"
          f"{'BH within bin':>15}{'BH overall':>12}")
    summary = {}
    for d in sorted(by_bin, key=key):
        rs = by_bin[d]
        raw = sum(r["p"] < args.alpha for r in rs)
        sb = sum(r["sig_bin"] for r in rs)
        so = sum(r["sig_overall"] for r in rs)
        m = float(np.mean([r["margin"] for r in rs]))
        print(f"{d:>6}{len(rs):>7}{m:>+13.3f}{raw:>8}/{len(rs)}"
              f"{sb:>12}/{len(rs)}{so:>9}/{len(rs)}")
        summary[d] = {"n_tests": len(rs), "mean_margin": m, "raw_sig": raw,
                      "bh_within_bin_sig": sb, "bh_overall_sig": so}

    out = {"meta": {"dataset": args.dataset, "model": args.model,
                    "alpha": args.alpha, "n_tests_total": len(ok),
                    "method": "Benjamini-Hochberg FDR"},
           "by_bin": summary, "tests": ok}
    path = ROOT / "results" / f"multiple_comparisons_{tag}.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\ntotal tests corrected: {len(ok)}")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
