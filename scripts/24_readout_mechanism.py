"""Which region's value does the read out actually report?

The validity check found that both read outs move about equally when either
region's value changes. Averaged, the sensitivity to leakage ratio is 1.07,
which reads as noise. It is not noise. In each model one region drives both
read outs at nearly the same strength, and which region it is varies by model.

This asks what selects that region. For every trial position we know the full
action history, so we can label each position by

    v_own     the value carried by the region we asked about
    v_other   the value carried by the other region
    v_last    the value carried by whichever region was acted on most recently
    v_more    the value carried by whichever region was acted on more often

and measure how much the read out moves when each of those flips. If v_last
explains more of the read out than v_own does, the model is answering from
recency rather than from the region named in the question.

    python scripts/24_readout_mechanism.py --dataset pilot --model gpt2
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mwm import intervene as IV  # noqa: E402
from mwm import models as M  # noqa: E402
from mwm import probe as P  # noqa: E402
from mwm.extract import load_extracted  # noqa: E402
from mwm.generate import read_jsonl  # noqa: E402


def boot(x, n_boot=8000, seed=0):
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
    ap.add_argument("--trials", type=int, default=220)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    tag = args.model.replace("/", "__")
    E1, E2 = args.entity, args.other_entity
    t1, t2 = f"{E1}.{args.attribute}", f"{E2}.{args.attribute}"
    _, _, meta, _ = load_extracted(
        ROOT / "activations" / f"{args.dataset}__{tag}__natural")
    spec = IV.READOUTS[args.attribute]
    A, B = list(spec.options)[:2]

    # action history, so we can say which region was touched last
    recs = {r["id"]: r for r in
            read_jsonl(ROOT / "data" / args.dataset / "natural.jsonl")}

    def history(m):
        r = recs.get(m["narrative_id"])
        if r is None:
            return None, None
        ents = [a.get("entity") for a in r["actions"][:m["position"] + 1]]
        touched = [e for e in ents if e in (E1, E2)]
        if not touched:
            return None, None
        counts = Counter(touched)
        more = E1 if counts[E1] > counts[E2] else (E2 if counts[E2] > counts[E1] else None)
        return touched[-1], more

    rng = np.random.default_rng(args.seed)
    pool = [m for m in meta if m["split"] == "test" and m["position"] >= 1
            and m["labels"][t1] in (A, B) and m["labels"][t2] in (A, B)]
    rng.shuffle(pool)
    rows = []
    with M.loaded_model(args.model) as obj:
        fast = IV.first_tokens_distinct(obj, spec)
        score = IV.readout_scores_fast if fast else IV.readout_scores
        for m in pool:
            if len(rows) >= args.trials:
                break
            last, more = history(m)
            if last is None:
                continue
            s1 = score(obj, m["text"], E1, spec)
            s2 = score(obj, m["text"], E2, spec)
            v1, v2 = m["labels"][t1], m["labels"][t2]
            rows.append({
                "gap_e1": s1[A] - s1[B], "gap_e2": s2[A] - s2[B],
                "v_e1": v1, "v_e2": v2,
                "v_last": v1 if last == E1 else v2,
                "v_more": (None if more is None else (v1 if more == E1 else v2)),
                "last": last, "more": more,
            })
            if len(rows) % 40 == 0:
                print(f"  {len(rows)}/{args.trials}", flush=True)

    def effect(gapkey, valkey):
        """How far the read out moves when this label flips from A to B."""
        hi = [r[gapkey] for r in rows if r.get(valkey) == A]
        lo = [r[gapkey] for r in rows if r.get(valkey) == B]
        n = min(len(hi), len(lo))
        if n < 10:
            return None
        return boot(np.array(hi[:n]) - np.array(lo[:n]))

    out = {"meta": {"model": args.model, "dataset": args.dataset,
                    "values": [A, B], "n": len(rows),
                    "last_region_counts": dict(Counter(r["last"] for r in rows))},
           "effects": {}}
    labels = [("v_e1", f"value of {E1}"), ("v_e2", f"value of {E2}"),
              ("v_last", "value of the region acted on last"),
              ("v_more", "value of the region acted on more often")]
    for gapkey, who in (("gap_e1", E1), ("gap_e2", E2)):
        for vk, desc in labels:
            e = effect(gapkey, vk)
            if e:
                out["effects"][f"{who}|{vk}"] = dict(e,description=desc)

    path = ROOT / "results" / f"readout_mechanism_{args.dataset}_{tag}.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n{args.model}   n={len(rows)}   values {A} against {B}")
    print(f"region acted on last: {out['meta']['last_region_counts']}\n")
    print(f"{'read out':<8}{'flipping':<38}{'effect on the read out':>26}")
    for gapkey, who in (("gap_e1", E1), ("gap_e2", E2)):
        for vk, desc in labels:
            k = f"{who}|{vk}"
            if k not in out["effects"]:
                continue
            e = out["effects"][k]
            star = " *" if not (e["ci95"][0] <= 0 <= e["ci95"][1]) else "  "
            print(f"{who:<8}{desc:<38}"
                  f"{e['mean']:>+9.3f} [{e['ci95'][0]:+.3f},{e['ci95'][1]:+.3f}]{star}")
        print()
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
