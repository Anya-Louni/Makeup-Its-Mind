"""Significance testing for the probe-vs-baseline gap.

Most of this project's claims reduce to "the hidden state beats baseline X".
So far each of those is a single point estimate from a single fit, which is not
enough to publish. Two sources of variance have to be handled, and they are
different:

test-set variance
    Positions are *not* independent: ~13 of them come from the same narrative
    and share its wordings and its state. A naive bootstrap over positions
    badly understates the interval. This resamples whole narratives (cluster
    bootstrap), which is the right unit of independence.

training variance
    A probe fit on a different training subsample gives a different answer.
    Repeating over several training subsamples gives an honest spread rather
    than one lucky fit.

Reports, for each target, the gap against every baseline with a cluster
bootstrap 95% CI and BOTH p-value conventions, labelled:

    p_one_sided_greater = P(gap <= 0)   -- small means the hidden state wins
    p_two_sided         = 2*min(...)    -- small means they differ either way

Both are given because a one-sided p quoted for a difference that came out the
wrong way reads as 0.94 or 0.99, which looks like a result and is not one. The
two-sided value is the default a reviewer expects.

    python scripts/17_significance.py --dataset full --model gpt2
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

from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from mwm import probe as P  # noqa: E402
from mwm.extract import load_extracted  # noqa: E402


def cluster_bootstrap(correct_a: np.ndarray, correct_b: np.ndarray,
                      groups: np.ndarray, n_boot: int = 4000,
                      seed: int = 0) -> dict:
    """Bootstrap the difference in accuracy, resampling whole narratives."""
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(groups, return_inverse=True)
    idx_by_group = [np.flatnonzero(inv == g) for g in range(len(uniq))]
    obs = float(correct_a.mean() - correct_b.mean())

    diffs = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        rows = np.concatenate([idx_by_group[g] for g in pick])
        diffs[b] = correct_a[rows].mean() - correct_b[rows].mean()
    lo, hi = np.quantile(diffs, [0.025, 0.975])
    out = {"gap": obs, "ci95": [float(lo), float(hi)],
           "n_narratives": int(len(uniq))}
    out.update(P.bootstrap_pvalues(diffs))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="full")
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--seeds", type=int, default=3,
                    help="training subsamples per target")
    ap.add_argument("--max-train", type=int, default=6000)
    ap.add_argument("--max-test", type=int, default=4000)
    ap.add_argument("--targets", nargs="*", default=None)
    ap.add_argument("--n-boot", type=int, default=4000)
    args = ap.parse_args()

    tag = args.model.replace("/", "__")
    X, E, meta, info = load_extracted(
        ROOT / "activations" / f"{args.dataset}__{tag}__natural")
    pr = json.loads((ROOT / "results" / f"probe_{args.dataset}_{tag}.json")
                    .read_text(encoding="utf-8"))
    targets = args.targets or sorted(pr["targets"])

    tr_all = np.flatnonzero(P.split_mask(meta, "train", min_position=1))
    te = np.flatnonzero(P.split_mask(meta, "test", min_position=1))
    if len(te) > args.max_test:
        te = np.random.default_rng(7).choice(te, args.max_test, False)
    groups = np.array([meta[i]["narrative_id"] for i in te])
    te_text = [meta[i]["text"] for i in te]

    out = {"meta": {"dataset": args.dataset, "model": args.model,
                    "seeds": args.seeds, "n_test": len(te),
                    "n_test_narratives": int(len(set(groups)))},
           "targets": {}}

    for t in targets:
        layer = pr["targets"][t]["best_layer"]
        li = info["layers"].index(layer)
        y, classes = P.encode_labels(P.target_vector(meta, t))
        maj_cls = None
        per_seed = defaultdict(list)
        corr = {}

        for s in range(args.seeds):
            rng = np.random.default_rng(100 + s)
            tr = (rng.choice(tr_all, args.max_train, False)
                  if len(tr_all) > args.max_train else tr_all)
            tr_text = [meta[i]["text"] for i in tr]
            maj_cls = np.bincount(y[tr]).argmax()

            def fit(F_tr, F_te):
                sc = StandardScaler().fit(F_tr)
                clf = LogisticRegression(max_iter=1000, n_jobs=-1)
                clf.fit(sc.transform(F_tr), y[tr])
                return clf.predict(sc.transform(F_te)) == y[te]

            c_hid = fit(np.asarray(X[tr, li], dtype=np.float32),
                        np.asarray(X[te, li], dtype=np.float32))
            c_emb = fit(np.asarray(E[tr], dtype=np.float32),
                        np.asarray(E[te], dtype=np.float32))

            vec = TfidfVectorizer(ngram_range=(1, 2), max_features=30000,
                                  sublinear_tf=True)
            Ttr = vec.fit_transform(tr_text)
            clf = LogisticRegression(max_iter=1000, n_jobs=-1).fit(Ttr, y[tr])
            c_tfidf = clf.predict(vec.transform(te_text)) == y[te]

            per_seed["hidden"].append(float(c_hid.mean()))
            per_seed["static_embedding"].append(float(c_emb.mean()))
            per_seed["ngram_tfidf"].append(float(c_tfidf.mean()))
            if s == 0:
                corr = {"hidden": c_hid, "static_embedding": c_emb,
                        "ngram_tfidf": c_tfidf}

        c_maj = (y[te] == maj_cls)
        corr["majority"] = c_maj
        per_seed["majority"] = [float(c_maj.mean())]

        R = {"layer": layer, "n_classes": len(classes),
             "accuracy_by_seed": {k: v for k, v in per_seed.items()},
             "accuracy_mean": {k: float(np.mean(v)) for k, v in per_seed.items()},
             "accuracy_sd_over_seeds": {
                 k: (float(np.std(v, ddof=1)) if len(v) > 1 else 0.0)
                 for k, v in per_seed.items()},
             "gaps": {}}
        for base in ("static_embedding", "ngram_tfidf", "majority"):
            R["gaps"][base] = cluster_bootstrap(
                corr["hidden"], corr[base], groups, n_boot=args.n_boot)
        out["targets"][t] = R

        g = R["gaps"]["static_embedding"]
        sd = R["accuracy_sd_over_seeds"]["hidden"]
        print(f"{t:<18} L{layer:<3} hidden {R['accuracy_mean']['hidden']:.3f}"
              f" (sd {sd:.3f})  vs static-emb "
              f"{R['accuracy_mean']['static_embedding']:.3f}  gap "
              f"{g['gap']:+.3f} [{g['ci95'][0]:+.3f},{g['ci95'][1]:+.3f}] "
              f"p2={g['p_two_sided']:.4f} p1={g['p_one_sided_greater']:.4f}",
              flush=True)

    path = ROOT / "results" / f"significance_{args.dataset}_{tag}.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
