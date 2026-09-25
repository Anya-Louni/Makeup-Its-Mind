"""Phase 2: full probing pipeline with all baseline controls.

For every state variable (e.g. `lips.color`) this produces, on the held-out
test split whose wordings were never seen in training:

  * linear probe accuracy at every layer
  * MLP probe accuracy at every layer (is the code linear?)
  * static-embedding baseline (non-contextual, model's own embedding space)
  * bag-of-ngrams TF-IDF baseline on the raw prefix text
  * majority-class baseline
  * shuffled-narrative-order condition, same pipeline
  * Hewitt & Liang control task + selectivity
  * MDL prequential code length per layer
  * accuracy split by order-sensitive vs. first-mention-correct positions
  * cross-entity binding confusion (lips probe read against eye labels)

    python scripts/04_probe.py --dataset pilot --model gpt2
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sklearn.preprocessing import StandardScaler  # noqa: E402

from mwm import probe as P  # noqa: E402
from mwm.extract import load_extracted, load_pooled  # noqa: E402


def subsample(mask: np.ndarray, cap: int | None, seed: int = 0) -> np.ndarray:
    idx = np.flatnonzero(mask)
    if cap is None or len(idx) <= cap:
        return idx
    return np.random.default_rng(seed).choice(idx, cap, replace=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="pilot")
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--max-train", type=int, default=6000)
    ap.add_argument("--max-test", type=int, default=3000)
    ap.add_argument("--mlp-hidden", type=int, default=64)
    ap.add_argument("--skip-mlp", action="store_true")
    ap.add_argument("--pooling", choices=["last", "mean"], default="last",
                    help="last token of the sentence, or mean over the "
                         "whole prefix (matches the pooled baseline)")
    ap.add_argument("--layer-stride", type=int, default=1,
                    help="probe every Nth layer (first and last always kept)")
    ap.add_argument("--skip-mdl", action="store_true")
    ap.add_argument("--skip-shuffled", action="store_true",
                    help="ignore the shuffled condition (e.g. while it "
                         "is still being extracted)")
    ap.add_argument("--min-position", type=int, default=1,
                    help="drop position 0 (always the all-bare initial state)")
    args = ap.parse_args()

    tag = args.model.replace("/", "__")
    base = ROOT / "activations"
    Xn, En, Mn, info = load_extracted(base / f"{args.dataset}__{tag}__natural")
    if args.pooling == "mean":
        Xp = load_pooled(base / f"{args.dataset}__{tag}__natural")
        if Xp is None:
            raise SystemExit("this extraction has no Xmean.npy; re-run "
                             "scripts/03_extract.py to store pooled states")
        Xn = Xp
    shuffled_dir = base / f"{args.dataset}__{tag}__shuffled"
    have_shuffled = shuffled_dir.exists() and not args.skip_shuffled
    if have_shuffled:
        Xs, Es, Ms, _ = load_extracted(shuffled_dir)
        if args.pooling == "mean":
            Xsp = load_pooled(shuffled_dir)
            if Xsp is not None:
                Xs = Xsp
            else:
                have_shuffled = False

    targets = sorted(Mn[0]["labels"].keys())
    layers = info["layers"]
    if args.layer_stride > 1:
        keep = set(layers[::args.layer_stride]) | {layers[0], layers[-1]}
        layers = [l for l in layers if l in keep]
    print(f"model={args.model} layers={layers} targets={targets} "
          f"positions={len(Mn)}", flush=True)

    tr = subsample(P.split_mask(Mn, "train", min_position=args.min_position),
                   args.max_train)
    te = subsample(P.split_mask(Mn, "test", min_position=args.min_position),
                   args.max_test)
    print(f"train={len(tr)} test={len(te)}", flush=True)

    results: dict = {"meta": {"model": args.model, "dataset": args.dataset,
                              "layers": layers, "n_train": len(tr),
                              "n_test": len(te),
                              "min_position": args.min_position,
                              "pooling": args.pooling},
                     "targets": {}}

    # cached per-target label encodings
    enc = {}
    for t in targets:
        y_all = P.target_vector(Mn, t)
        y, classes = P.encode_labels(y_all)
        enc[t] = (y, classes)

    t_start = time.time()
    for t in targets:
        y, classes = enc[t]
        R: dict = {"classes": classes, "layers": {}}
        print(f"\n--- {t}  ({len(classes)} classes)", flush=True)

        # ---------- baselines that do not depend on layer ----------
        Etr = np.asarray(En[tr], dtype=np.float32)
        Ete = np.asarray(En[te], dtype=np.float32)
        R["static_embedding"] = P.fit_eval(Etr, y[tr], Ete, y[te],
                                           "linear").as_dict()
        R["ngram_tfidf"] = P.ngram_baseline(
            [Mn[i]["text"] for i in tr], y[tr],
            [Mn[i]["text"] for i in te], y[te]).as_dict()
        print(f"  static-emb  {R['static_embedding']['accuracy']:.3f} | "
              f"tfidf {R['ngram_tfidf']['accuracy']:.3f} | "
              f"majority {R['static_embedding']['majority']:.3f}", flush=True)

        # ---------- per-layer ----------
        best_layer, best_acc = None, -1.0
        for li, layer in enumerate(layers):
            ai = info["layers"].index(layer)
            Xtr = np.asarray(Xn[tr, ai], dtype=np.float32)
            Xte = np.asarray(Xn[te, ai], dtype=np.float32)
            lin = P.fit_eval(Xtr, y[tr], Xte, y[te], "linear")
            entry = {"linear": lin.as_dict()}
            if not args.skip_mlp:
                entry["mlp"] = P.fit_eval(Xtr, y[tr], Xte, y[te], "mlp",
                                          hidden=args.mlp_hidden).as_dict()
            R["layers"][str(layer)] = entry
            if lin.accuracy > best_acc:
                best_acc, best_layer = lin.accuracy, layer
            msg = f"  L{layer:<2d} linear {lin.accuracy:.3f}"
            if "mlp" in entry:
                msg += f"  mlp {entry['mlp']['accuracy']:.3f}"
            print(msg, flush=True)

        R["best_layer"] = best_layer
        R["best_linear_accuracy"] = best_acc
        bi = info["layers"].index(best_layer)
        Xtr = np.asarray(Xn[tr, bi], dtype=np.float32)
        Xte = np.asarray(Xn[te, bi], dtype=np.float32)

        # ---------- control task + selectivity ----------
        yc = P.control_labels(Mn, P.target_vector(Mn, t),
                              P.split_mask(Mn, "train"), seed=abs(hash(t)) % 2**31)
        ctrl = P.fit_eval(Xtr, yc[tr], Xte, yc[te], "linear")
        R["control_task"] = ctrl.as_dict()
        R["selectivity"] = best_acc - ctrl.accuracy
        if not args.skip_mlp:
            ctrl_mlp = P.fit_eval(Xtr, yc[tr], Xte, yc[te], "mlp",
                                  hidden=args.mlp_hidden)
            R["control_task_mlp"] = ctrl_mlp.as_dict()
            R["selectivity_mlp"] = (
                R["layers"][str(best_layer)]["mlp"]["accuracy"]
                - ctrl_mlp.accuracy)
        print(f"  control {ctrl.accuracy:.3f} -> selectivity "
              f"{R['selectivity']:.3f}  (best layer {best_layer})", flush=True)

        # ---------- MDL ----------
        if not args.skip_mdl:
            R["mdl"] = {}
            for layer in (layers[0], best_layer, layers[-1]):
                Xl = np.asarray(Xn[tr, info["layers"].index(layer)],
                                dtype=np.float32)
                R["mdl"][str(layer)] = P.mdl_online_codelength(
                    Xl, y[tr], len(classes))
            R["mdl_static_embedding"] = P.mdl_online_codelength(
                Etr, y[tr], len(classes))
            b = R["mdl"][str(best_layer)]
            print(f"  MDL best-layer {b['codelength_kbits']:.1f} kbit "
                  f"(compression {b['compression']:.2f}x) vs static-emb "
                  f"{R['mdl_static_embedding']['compression']:.2f}x", flush=True)

        # ---------- order-sensitive stratification ----------
        os_flags = np.array([m["order_sensitive"][t] for m in Mn])
        clf = P.make_probe("linear")
        sc = StandardScaler().fit(Xtr)
        clf.fit(sc.transform(Xtr), y[tr])
        pred = clf.predict(sc.transform(Xte))
        m_os = os_flags[te]
        R["order_sensitive_split"] = {
            "acc_order_sensitive": float(np.mean(pred[m_os] == y[te][m_os]))
            if m_os.any() else None,
            "n_order_sensitive": int(m_os.sum()),
            "acc_first_mention_ok": float(np.mean(pred[~m_os] == y[te][~m_os]))
            if (~m_os).any() else None,
            "n_first_mention_ok": int((~m_os).sum()),
        }
        oss = R["order_sensitive_split"]
        print(f"  order-sensitive positions {oss['acc_order_sensitive']} "
              f"(n={oss['n_order_sensitive']}) vs rest "
              f"{oss['acc_first_mention_ok']}", flush=True)

        # ---------- binding: read this probe against the other entity ----------
        entity, attr = t.split(".")
        others = [o for o in targets
                  if o.endswith("." + attr) and o != t]
        R["binding"] = {}
        for o in others:
            yo, _ = enc[o]
            same_classes = classes == enc[o][1]
            if not same_classes:
                continue
            R["binding"][o] = {
                "acc_reading_other_entity": float(
                    np.mean(pred == yo[te])),
                "majority_other": float(
                    np.mean(yo[te] == np.bincount(yo[tr]).argmax())),
            }
        if R["binding"]:
            print("  binding (this probe read against other entity): " +
                  ", ".join(f"{k}={v['acc_reading_other_entity']:.3f}"
                            for k, v in R["binding"].items()), flush=True)

        # ---------- shuffled-order condition ----------
        if have_shuffled:
            ys_all = P.target_vector(Ms, t)
            lut = {c: i for i, c in enumerate(classes)}
            ys = np.array([lut[v] for v in ys_all])
            strr = subsample(P.split_mask(Ms, "train",
                                          min_position=args.min_position),
                             args.max_train)
            ste = subsample(P.split_mask(Ms, "test",
                                         min_position=args.min_position),
                            args.max_test)
            R["shuffled"] = {}
            for layer in (best_layer,):
                li = info["layers"].index(layer)
                res = P.fit_eval(np.asarray(Xs[strr, li], dtype=np.float32),
                                 ys[strr],
                                 np.asarray(Xs[ste, li], dtype=np.float32),
                                 ys[ste], "linear")
                R["shuffled"][str(layer)] = res.as_dict()
            print(f"  shuffled-order @L{best_layer} "
                  f"{R['shuffled'][str(best_layer)]['accuracy']:.3f} "
                  f"(natural {best_acc:.3f})", flush=True)

        results["targets"][t] = R

    results["meta"]["runtime_seconds"] = round(time.time() - t_start, 1)
    suffix = "" if args.pooling == "last" else f"_{args.pooling}"
    out = ROOT / "results" / f"probe_{args.dataset}_{tag}{suffix}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
