"""Phase 3: representation geometry.

    python scripts/05_geometry.py --dataset pilot --model gpt2
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mwm import geometry as G  # noqa: E402
from mwm import probe as P  # noqa: E402
from mwm.extract import load_extracted  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="pilot")
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--layer", type=int, default=None,
                    help="default: best layer from the probe results")
    ap.add_argument("--max-samples", type=int, default=8000)
    ap.add_argument("--n-trajectories", type=int, default=12)
    args = ap.parse_args()

    tag = args.model.replace("/", "__")
    X, E, meta, info = load_extracted(
        ROOT / "activations" / f"{args.dataset}__{tag}__natural")
    layers = info["layers"]
    targets = sorted(meta[0]["labels"].keys())

    probe_path = ROOT / "results" / f"probe_{args.dataset}_{tag}.json"
    best_layers = {}
    if probe_path.exists():
        pr = json.loads(probe_path.read_text(encoding="utf-8"))
        best_layers = {t: pr["targets"][t]["best_layer"] for t in pr["targets"]}

    layer = args.layer
    if layer is None:
        layer = (int(np.median(list(best_layers.values())))
                 if best_layers else layers[len(layers) // 2])
    li = layers.index(layer)
    print(f"geometry at layer {layer} (of {layers})")

    mask = P.split_mask(meta, "test", min_position=1)
    idx = np.flatnonzero(mask)
    if len(idx) > args.max_samples:
        idx = np.random.default_rng(0).choice(idx, args.max_samples, False)
    Xl = np.asarray(X[idx, li], dtype=np.float32)

    labels, classes = {}, {}
    for t in targets:
        y_all = P.target_vector(meta, t)
        y, cls = P.encode_labels(y_all)
        labels[t] = y[idx]
        classes[t] = cls

    out: dict = {"model": args.model, "dataset": args.dataset, "layer": layer,
                 "n_samples": int(len(idx))}

    # ---------------- concept directions and their overlap ----------------
    bank_cav = G.direction_bank(Xl, labels, classes, method="cav")
    bank_lr = G.direction_bank(Xl, labels, classes, method="probe")
    keys, Mcav = G.similarity_matrix(bank_cav)
    _, Mlr = G.similarity_matrix(bank_lr)
    out["direction_keys"] = keys
    out["cosine_cav"] = Mcav.round(4).tolist()
    out["cosine_probe"] = Mlr.round(4).tolist()
    out["cav_vs_probe_agreement"] = {
        k: round(G.cosine(bank_cav[k], bank_lr[k]), 4) for k in keys}

    # same attribute value, different face region -> the binding question
    same_value_cross_entity = {}
    for k1 in keys:
        t1, v1 = k1.split("=")
        e1, a1 = t1.split(".")
        for k2 in keys:
            if k2 <= k1:
                continue
            t2, v2 = k2.split("=")
            e2, a2 = t2.split(".")
            if a1 == a2 and v1 == v2 and e1 != e2:
                same_value_cross_entity[f"{k1} | {k2}"] = {
                    "cav": round(G.cosine(bank_cav[k1], bank_cav[k2]), 4),
                    "probe": round(G.cosine(bank_lr[k1], bank_lr[k2]), 4),
                }
    out["same_value_cross_entity_cosine"] = same_value_cross_entity

    # ---------------- subspace principal angles ----------------
    angles = {}
    for i, t1 in enumerate(targets):
        for t2 in targets[i + 1:]:
            A = G.target_subspace(Xl, labels[t1], classes[t1])
            B = G.target_subspace(Xl, labels[t2], classes[t2])
            th = G.subspace_principal_angles(A, B)
            angles[f"{t1} | {t2}"] = {
                "principal_angles_deg": np.degrees(th).round(2).tolist(),
                "mean_deg": round(float(np.degrees(th).mean()), 2),
            }
    out["subspace_principal_angles"] = angles

    # ---------------- intrinsic dimensionality ----------------
    out["dimensionality"] = {
        t: G.dimensionality_curve(Xl, labels[t]) for t in targets}

    # ---------------- projection + trajectories for animation --------------
    project, pca = G.fit_projection(Xl, 3)
    Z = project(Xl)
    out["pca_explained_variance"] = pca.explained_variance_ratio_.round(4).tolist()

    nids: list[str] = []
    for m in meta:
        if m["split"] == "test" and m["narrative_id"] not in nids:
            nids.append(m["narrative_id"])
        if len(nids) >= args.n_trajectories:
            break
    traj = []
    for nid in nids:
        rows = [m for m in meta if m["narrative_id"] == nid]
        rows.sort(key=lambda m: m["position"])
        H = np.asarray(X[[m["row"] for m in rows], li], dtype=np.float32)
        traj.append({
            "narrative_id": nid,
            "coords": project(H).round(4).tolist(),
            "sentences": [m["text"].split(". ")[-1] for m in rows],
            "labels": [m["labels"] for m in rows],
            "action_kinds": [m["action_kind"] for m in rows],
        })

    res_dir = ROOT / "results"
    res_dir.mkdir(exist_ok=True)
    # an explicitly requested layer gets its own filename so several layers
    # can be compared without overwriting each other
    sfx = "" if args.layer is None else f"_L{layer}"
    (res_dir / f"geometry_{args.dataset}_{tag}{sfx}.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")
    np.savez_compressed(
        res_dir / f"geometry_{args.dataset}_{tag}{sfx}.npz",
        Z=Z.astype(np.float32),
        **{f"y_{t}": labels[t] for t in targets},
        directions=np.stack([bank_cav[k] for k in keys]).astype(np.float32),
        direction_keys=np.array(keys),
    )
    (res_dir / f"trajectories_{args.dataset}_{tag}{sfx}.json").write_text(
        json.dumps(traj, indent=2), encoding="utf-8")

    print(json.dumps({
        "layer": layer,
        "pca_explained_variance": out["pca_explained_variance"],
        "same_value_cross_entity_cosine": same_value_cross_entity,
        "subspace_mean_angles_deg": {k: v["mean_deg"]
                                     for k, v in angles.items()},
        "dimensionality_first_target": out["dimensionality"][targets[0]],
    }, indent=2))
    print(f"\nwrote {res_dir}")


if __name__ == "__main__":
    main()
