"""Can the read out tell the two regions apart at all?

A null from steering means nothing if the measurement cannot register a
difference in the first place. Unaided behavioural accuracy here is near
chance, so a zero for the region specific component could be a floor effect
rather than a result.

This checks the read out without steering anything. Take held out positions
where the two regions carry different finishes and compare the model's log
odds under four assignments:

    lips dewy  / eyes matte
    lips matte / eyes dewy
    lips dewy  / eyes dewy
    lips matte / eyes matte

Two contrasts follow, both on the lips read out asking about dewy:

  sensitivity   change the LIPS value, hold the eyes value fixed.
                A read out that tracks the lips moves a lot here.
  leakage       change the EYES value, hold the lips value fixed.
                A read out that binds correctly moves very little here.

The ratio of the two is the measurement's own binding selectivity. If
sensitivity is near zero the instrument is blind and every steering null in
this study is uninterpretable. If sensitivity is large and leakage is near
zero the read out binds, and a steering null is a real result. If both are
large the read out responds to the value wherever it appears, which is the
same failure the decomposition found in the representation.

    python scripts/23_readout_validity.py --dataset pilot --model gpt2
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

from mwm import intervene as IV  # noqa: E402
from mwm import models as M  # noqa: E402
from mwm import probe as P  # noqa: E402
from mwm.extract import load_extracted  # noqa: E402


def boot(x, n_boot=10000, seed=0):
    x = np.asarray(x, float)
    if len(x) < 2:
        return {"mean": float("nan"), "ci95": [float("nan")] * 2, "n": len(x)}
    rng = np.random.default_rng(seed)
    dr = x[rng.integers(0, len(x), (n_boot, len(x)))].mean(axis=1)
    out = {"mean": float(x.mean()),
           "ci95": [float(np.quantile(dr, .025)), float(np.quantile(dr, .975))],
           "n": len(x)}
    out.update(P.bootstrap_pvalues(dr))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="pilot")
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--entity", default="lips")
    ap.add_argument("--other-entity", default="eyes")
    ap.add_argument("--attribute", default="finish")
    ap.add_argument("--per-cell", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    tag = args.model.replace("/", "__")
    t1 = f"{args.entity}.{args.attribute}"
    t2 = f"{args.other_entity}.{args.attribute}"
    _, _, meta, _ = load_extracted(
        ROOT / "activations" / f"{args.dataset}__{tag}__natural")
    spec = IV.READOUTS[args.attribute]
    vals = list(spec.options)

    # two values give the cleanest 2x2
    A, B = vals[0], vals[1]
    cells = {(A, B): [], (B, A): [], (A, A): [], (B, B): []}
    rng = np.random.default_rng(args.seed)
    pool = [m for m in meta if m["split"] == "test" and m["position"] >= 1]
    rng.shuffle(pool)
    for m in pool:
        k = (m["labels"][t1], m["labels"][t2])
        if k in cells and len(cells[k]) < args.per_cell:
            cells[k].append(m)
    for k, v in cells.items():
        print(f"  cell lips={k[0]:<6} eyes={k[1]:<6} n={len(v)}", flush=True)
    if min(len(v) for v in cells.values()) < 10:
        raise SystemExit("not enough positions in one cell")

    gaps = defaultdict(list)
    with M.loaded_model(args.model) as obj:
        fast = IV.first_tokens_distinct(obj, spec)
        score = IV.readout_scores_fast if fast else IV.readout_scores
        for k, rows in cells.items():
            for m in rows:
                s1 = score(obj, m["text"], args.entity, spec)
                s2 = score(obj, m["text"], args.other_entity, spec)
                # log odds that the read out prefers A over B
                gaps[("lips", k)].append(s1[A] - s1[B])
                gaps[("eyes", k)].append(s2[A] - s2[B])
            print(f"  scored {k}", flush=True)

    def diff(read, k_hi, k_lo):
        n = min(len(gaps[(read, k_hi)]), len(gaps[(read, k_lo)]))
        return (np.array(gaps[(read, k_hi)][:n])
                - np.array(gaps[(read, k_lo)][:n]))

    # lips read out: change lips value, eyes held at B then at A
    sens = np.concatenate([diff("lips", (A, B), (B, B)),
                           diff("lips", (A, A), (B, A))])
    # lips read out: change eyes value, lips held at A then at B
    leak = np.concatenate([diff("lips", (A, A), (A, B)),
                           diff("lips", (B, A), (B, B))])
    # same two contrasts for the eyes read out, roles swapped
    sens_e = np.concatenate([diff("eyes", (A, A), (A, B)),
                             diff("eyes", (B, A), (B, B))])
    leak_e = np.concatenate([diff("eyes", (A, B), (B, B)),
                             diff("eyes", (A, A), (B, A))])

    S, L = boot(sens), boot(leak)
    Se, Le = boot(sens_e), boot(leak_e)
    ratio = (abs(S["mean"]) / abs(L["mean"])) if L["mean"] else float("inf")
    ratio_e = (abs(Se["mean"]) / abs(Le["mean"])) if Le["mean"] else float("inf")

    out = {"meta": {"model": args.model, "dataset": args.dataset,
                    "target": t1, "other": t2, "values": [A, B],
                    "per_cell": args.per_cell},
           "lips_readout": {"sensitivity": S, "leakage": L,
                            "selectivity_ratio": ratio},
           "eyes_readout": {"sensitivity": Se, "leakage": Le,
                            "selectivity_ratio": ratio_e}}
    path = ROOT / "results" / f"readout_validity_{args.dataset}_{tag}.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(f"\n{args.model}  values {A} against {B}\n")
    for name, d in (("lips read out", out["lips_readout"]),
                    ("eyes read out", out["eyes_readout"])):
        s, l = d["sensitivity"], d["leakage"]
        print(f"{name}")
        print(f"  sensitivity  {s['mean']:+.3f} [{s['ci95'][0]:+.3f},"
              f"{s['ci95'][1]:+.3f}]  p={s['p_two_sided']:.4f}  n={s['n']}")
        print(f"  leakage      {l['mean']:+.3f} [{l['ci95'][0]:+.3f},"
              f"{l['ci95'][1]:+.3f}]  p={l['p_two_sided']:.4f}")
        print(f"  ratio        {d['selectivity_ratio']:.2f}\n")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
