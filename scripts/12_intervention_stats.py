"""Significance and effect size for the intervention results.

The raw summary reports point estimates. Two of them rest on small n --
`probe_flip_rate_on_initially_correct` is computed over only the trials whose
unintervened read-out was already correct, which in the pilot is 29 of 120 --
so a bare percentage is not enough to make a causal claim.

This recomputes, from the stored per-trial rows:

  * a paired bootstrap CI on the change in log-odds of the steered value
  * the same for the *other* face region, which must stay near zero
  * a specificity ratio (target effect / other-region effect)
  * a paired bootstrap test that the target effect exceeds the other-region
    effect (signed, not absolute -- see the note in `analyse`), which is the
    binding claim

    python scripts/12_intervention_stats.py --dataset pilot --model gpt2
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

from mwm import probe as P  # noqa: E402

RES = ROOT / "results"


def boot_ci(x: np.ndarray, n_boot: int = 10000, seed: int = 0,
            alpha: float = 0.05) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    means = x[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(x.mean()), float(lo), float(hi)


def analyse(path: Path) -> dict:
    iv = json.loads(path.read_text(encoding="utf-8"))
    rows = iv["rows"]
    by_alpha = defaultdict(list)
    for r in rows:
        by_alpha[r["alpha"]].append(r)
    base = {r["trial"]: r for r in by_alpha[0.0]}

    out = {"meta": iv["meta"], "alphas": {}}
    for a in sorted(by_alpha):
        rs = sorted(by_alpha[a], key=lambda r: r["trial"])
        d_t = np.array([r["beh_gap_steered"] - base[r["trial"]]["beh_gap_steered"]
                        for r in rs])
        d_o = np.array([r["beh_gap_other_true"]
                        - base[r["trial"]]["beh_gap_other_true"] for r in rs])
        # Signed, paired. An earlier version used d_t - |d_o|, which is
        # biased against the target: E[|noise|] > 0, so at small effect
        # sizes the other region's noise alone made the statistic
        # negative. The binding claim is simply that the edit moves the
        # target region more than it moves the other one.
        diff = d_t - d_o

        m_t, lo_t, hi_t = boot_ci(d_t)
        m_o, lo_o, hi_o = boot_ci(d_o)
        m_d, lo_d, hi_d = boot_ci(diff)

        # Both p-value conventions, labelled. See P.bootstrap_pvalues.
        rng = np.random.default_rng(1)
        idx = rng.integers(0, len(diff), size=(10000, len(diff)))
        draws = diff[idx].mean(axis=1)
        pv = P.bootstrap_pvalues(draws)

        flips = np.array([r["probe_pred"] == r["steered_to"] for r in rs
                          if base[r["trial"]]["probe_pred"]
                          == base[r["trial"]]["true"]], dtype=float)
        f_m, f_lo, f_hi = (boot_ci(flips) if len(flips) else
                           (float("nan"),) * 3)

        out["alphas"][str(a)] = {
            "n": len(rs),
            "delta_logodds_target": {"mean": m_t, "ci95": [lo_t, hi_t]},
            "delta_logodds_other": {"mean": m_o, "ci95": [lo_o, hi_o]},
            "target_minus_other_signed": dict(
                {"mean": m_d, "ci95": [lo_d, hi_d]}, **pv),
            "specificity_ratio": (abs(m_t / m_o) if m_o not in (0.0,)
                                  else float("inf")),
            "probe_flip_initially_correct": {
                "n": int(len(flips)), "mean": f_m, "ci95": [f_lo, f_hi]},
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="pilot")
    ap.add_argument("--model", default="gpt2")
    args = ap.parse_args()
    tag = f"{args.dataset}_{args.model.replace('/', '__')}"

    all_out = {}
    for p in sorted(RES.glob(f"intervention_{tag}_*.json")):
        if p.name.endswith("_stats.json"):
            continue
        res = analyse(p)
        key = res["meta"]["target"]
        all_out[key] = res
        print(f"\n=== {key}  (edit L{res['meta']['edit_layer']} -> "
              f"read L{res['meta']['read_layer']}, n={res['meta']['n_trials']})")
        print(f"{'alpha':>6}  {'d log-odds target':>26}  "
              f"{'d log-odds other':>26}  {'target - other':>22}  {'p2':>8}")
        for a, s in res["alphas"].items():
            t, o, d = (s["delta_logodds_target"], s["delta_logodds_other"],
                       s["target_minus_other_signed"])
            print(f"{a:>6}  {t['mean']:>+8.3f} [{t['ci95'][0]:+.3f},{t['ci95'][1]:+.3f}]"
                  f"  {o['mean']:>+8.3f} [{o['ci95'][0]:+.3f},{o['ci95'][1]:+.3f}]"
                  f"  {d['mean']:>+8.3f} [{d['ci95'][0]:+.3f},{d['ci95'][1]:+.3f}]"
                  f"  {d['p_two_sided']:>8.4f}")
        f = res["alphas"][str(max(float(a) for a in res["alphas"]))][
            "probe_flip_initially_correct"]
        print(f"  probe flip rate at max alpha (n={f['n']}): "
              f"{f['mean']:.3f} [{f['ci95'][0]:.3f}, {f['ci95'][1]:.3f}]")

    path = RES / f"intervention_stats_{tag}.json"
    path.write_text(json.dumps(all_out, indent=2), encoding="utf-8")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
