"""Phase 4: causal intervention.

Adds a concept direction to the residual stream mid-narrative and measures
whether the state the model carries forward actually changes.

Two read-outs, because they answer different questions and a small base LM
may only support one of them:

  probe read-out (always interpretable)
      A probe trained at a *later* layer on clean data reads the intervened
      hidden state.  If the edit at layer L propagates and flips the state
      code at layer L' > L, the direction is part of the state representation
      the model actually carries forward -- not just a direction we can find.
      The paired control is the other face region's probe on the very same
      forward pass: a bound representation moves one and leaves the other.

  behavioural read-out (stronger claim, often uninformative for small models)
      Append a cue and compare the model's log-probability of the competing
      state words.  This is only meaningful if the *unintervened* model reads
      the true state out above chance, so the alpha=0 baseline is reported and
      every effect is measured as a paired change from it.

    python scripts/06_intervene.py --dataset pilot --model gpt2 \
        --attribute finish --entity lips --trials 150
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mwm import intervene as IV  # noqa: E402
from mwm import models as M  # noqa: E402
from mwm import probe as P  # noqa: E402
from mwm.extract import load_extracted  # noqa: E402


def train_readout_probe(X, rows, y, li):
    Xtr = np.asarray(X[rows, li], dtype=np.float32)
    sc = StandardScaler().fit(Xtr)
    clf = P.make_probe("linear")
    clf.fit(sc.transform(Xtr), y[rows])
    return sc, clf


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="pilot")
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--entity", default="lips")
    ap.add_argument("--other-entity", default="eyes")
    ap.add_argument("--attribute", default="finish")
    ap.add_argument("--layer", type=int, default=None,
                    help="residual-stream index to edit; default = best probe layer")
    ap.add_argument("--read-layer", type=int, default=None,
                    help="layer the read-out probe is trained on; default = last")
    ap.add_argument("--alphas", nargs="*", type=float,
                    default=[0.0, 0.5, 1.0, 2.0, 4.0, 8.0])
    ap.add_argument("--trials", type=int, default=120)
    ap.add_argument("--max-probe-train", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    tag = args.model.replace("/", "__")
    target = f"{args.entity}.{args.attribute}"
    other = f"{args.other_entity}.{args.attribute}"
    X, E, meta, info = load_extracted(
        ROOT / "activations" / f"{args.dataset}__{tag}__natural")
    layers = info["layers"]

    layer = args.layer
    probe_path = ROOT / "results" / f"probe_{args.dataset}_{tag}.json"
    if layer is None and probe_path.exists():
        layer = json.loads(probe_path.read_text(encoding="utf-8")
                           )["targets"][target]["best_layer"]
    if layer is None:
        layer = layers[len(layers) // 2]
    layer = max(layer, layers[1])
    li = layers.index(layer)
    read_layer = args.read_layer if args.read_layer is not None else layers[-1]
    if read_layer <= layer:
        read_layer = layers[-1]
    ri = layers.index(read_layer)
    print(f"edit hidden_states[{layer}] (block {layer - 1}); "
          f"probe read-out at hidden_states[{read_layer}]", flush=True)

    y_t, classes = P.encode_labels(P.target_vector(meta, target))
    y_o, classes_o = P.encode_labels(P.target_vector(meta, other))
    spec = IV.READOUTS[args.attribute]
    values = [v for v in spec.options if v in classes]

    tr = np.flatnonzero(P.split_mask(meta, "train", min_position=1))
    if len(tr) > args.max_probe_train:
        tr = np.random.default_rng(0).choice(tr, args.max_probe_train, False)

    # directions come from the training split only, never from trial narratives
    Xtr = np.asarray(X[tr, li], dtype=np.float32)
    dirs = {}
    for src in values:
        for dst in values:
            if src != dst:
                dirs[(src, dst)] = IV.direction_from_activations(
                    Xtr, y_t[tr], classes.index(src), classes.index(dst))
    mean_norm = float(np.linalg.norm(Xtr, axis=1).mean())
    del Xtr

    sc_t, clf_t = train_readout_probe(X, tr, y_t, ri)
    sc_o, clf_o = train_readout_probe(X, tr, y_o, ri)
    print(f"read-out probes trained at L{read_layer} on {len(tr)} rows; "
          f"mean |h| at L{layer} = {mean_norm:.1f}", flush=True)

    rng = np.random.default_rng(args.seed)
    cand = [m for m in meta
            if m["split"] == "test" and m["position"] >= 1
            and m["labels"][target] in values
            and m["labels"][other] in values]
    rng.shuffle(cand)
    cand = cand[:args.trials]
    print(f"{len(cand)} trial positions", flush=True)

    rows = []
    with M.loaded_model(args.model) as obj:
        fast = IV.first_tokens_distinct(obj, spec)
        score = IV.readout_scores_fast if fast else IV.readout_scores
        print(f"behavioural read-out: "
              f"{'first-token forced choice' if fast else 'full logprob'}",
              flush=True)

        for n, m in enumerate(cand):
            text = m["text"]
            true_v = m["labels"][target]
            other_v = m["labels"][other]
            dst = str(rng.choice([v for v in values if v != true_v]))
            d = dirs[(true_v, dst)]

            for alpha in args.alphas:
                hook = (None if alpha == 0.0
                        else IV.ResidualAdd(d, alpha, positions=None))
                h = IV.hidden_under_hook(obj, text, read_layer, layer - 1,
                                         hook).numpy()[None, :]
                p_t = classes[int(clf_t.predict(sc_t.transform(h))[0])]
                p_o = classes_o[int(clf_o.predict(sc_o.transform(h))[0])]

                s_t = score(obj, text, args.entity, spec,
                            hook_layer=layer - 1, hook=hook)
                s_o = score(obj, text, args.other_entity, spec,
                            hook_layer=layer - 1,
                            hook=(None if alpha == 0.0
                                  else IV.ResidualAdd(d, alpha)))
                rows.append({
                    "trial": n,
                    "narrative_id": m["narrative_id"],
                    "position": m["position"],
                    "alpha": alpha,
                    "true": true_v,
                    "steered_to": dst,
                    "other_true": other_v,
                    # probe read-out
                    "probe_pred": p_t,
                    "probe_pred_other": p_o,
                    # behavioural read-out
                    "beh_pred": IV.argmax_option(s_t),
                    "beh_gap_true": IV.logit_gap(s_t, true_v),
                    "beh_gap_steered": IV.logit_gap(s_t, dst),
                    "beh_pred_other": IV.argmax_option(s_o),
                    "beh_gap_other_true": IV.logit_gap(s_o, other_v),
                })
            if (n + 1) % 20 == 0:
                print(f"  {n + 1}/{len(cand)}", flush=True)

    summary = summarise(rows, args.alphas)
    out = {"meta": {"model": args.model, "dataset": args.dataset,
                    "target": target, "other": other, "edit_layer": layer,
                    "read_layer": read_layer, "alphas": args.alphas,
                    "n_trials": len(cand), "mean_hidden_norm": mean_norm,
                    "direction_norms": {f"{k[0]}->{k[1]}": float(v.norm())
                                        for k, v in dirs.items()}},
           "summary": summary, "rows": rows}
    path = (ROOT / "results" /
            f"intervention_{args.dataset}_{tag}_{target}_L{layer}.json")
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"\nwrote {path}")


def summarise(rows: list[dict], alphas: list[float]) -> dict:
    by_alpha = defaultdict(list)
    for r in rows:
        by_alpha[r["alpha"]].append(r)
    base = {r["trial"]: r for r in by_alpha[0.0]}

    out = {}
    for a in alphas:
        rs = by_alpha[a]
        if not rs:
            continue
        # trials whose *unintervened* read-out was already correct: the only
        # ones where "did the edit change the answer" is a meaningful question
        clean = [r for r in rs
                 if base[r["trial"]]["probe_pred"] == base[r["trial"]]["true"]]
        clean_beh = [r for r in rs
                     if base[r["trial"]]["beh_pred"] == base[r["trial"]]["true"]]
        out[str(a)] = {
            "n": len(rs),
            # --- probe read-out ---
            "probe_acc_true": float(np.mean([r["probe_pred"] == r["true"]
                                             for r in rs])),
            "probe_rate_steered": float(np.mean([r["probe_pred"] == r["steered_to"]
                                                 for r in rs])),
            "probe_flip_rate_on_initially_correct": (
                float(np.mean([r["probe_pred"] == r["steered_to"]
                               for r in clean])) if clean else None),
            "n_initially_correct": len(clean),
            # paired control: the other region read from the same forward pass
            "probe_other_acc": float(np.mean([r["probe_pred_other"] == r["other_true"]
                                              for r in rs])),
            "probe_other_changed_vs_baseline": float(np.mean(
                [r["probe_pred_other"] != base[r["trial"]]["probe_pred_other"]
                 for r in rs])),
            # --- behavioural read-out ---
            "beh_acc_true": float(np.mean([r["beh_pred"] == r["true"]
                                           for r in rs])),
            "beh_rate_steered": float(np.mean([r["beh_pred"] == r["steered_to"]
                                               for r in rs])),
            "beh_rate_steered_minus_baseline": float(
                np.mean([r["beh_pred"] == r["steered_to"] for r in rs])
                - np.mean([base[r["trial"]]["beh_pred"] == r["steered_to"]
                           for r in rs])),
            "beh_flip_rate_on_initially_correct": (
                float(np.mean([r["beh_pred"] == r["steered_to"]
                               for r in clean_beh])) if clean_beh else None),
            # paired change in the log-odds the steered value wins
            "beh_mean_delta_gap_steered": float(np.mean(
                [r["beh_gap_steered"] - base[r["trial"]]["beh_gap_steered"]
                 for r in rs])),
            "beh_mean_delta_gap_other": float(np.mean(
                [r["beh_gap_other_true"] - base[r["trial"]]["beh_gap_other_true"]
                 for r in rs])),
        }
    return out


if __name__ == "__main__":
    main()
