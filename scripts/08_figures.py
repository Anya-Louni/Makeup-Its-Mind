"""Phase 6: figures.

Static matplotlib panels for the write-up plus an animated interactive 3D
trajectory in plotly.

    python scripts/08_figures.py --dataset pilot --model gpt2
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

FIG = ROOT / "figures"
RES = ROOT / "results"

# swatch colours for the palette, so the colour attribute reads visually
SWATCH = {"bare": "#d9d3cc", "nude": "#d9b8a0", "pink": "#e58fa8",
          "red": "#c2183c", "brown": "#7a4a2b",
          "none": "#d9d3cc", "matte": "#8a8f98", "dewy": "#4bb3d4",
          "satin": "#b58fd6", "light": "#cbb7e0", "full": "#5b3f8c"}


def load(name: str):
    p = RES / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


# --------------------------------------------------------------- fig 1 + 2
def fig_layer_curves(pr: dict, tag: str) -> None:
    targets = sorted(pr["targets"])
    n = len(targets)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 3.8), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, t in zip(axes, targets):
        R = pr["targets"][t]
        layers = sorted(int(k) for k in R["layers"])
        lin = [R["layers"][str(l)]["linear"]["accuracy"] for l in layers]
        ax.plot(layers, lin, "-o", ms=4, lw=2, color="#c2183c",
                label="linear probe")
        if "mlp" in R["layers"][str(layers[0])]:
            mlp = [R["layers"][str(l)]["mlp"]["accuracy"] for l in layers]
            ax.plot(layers, mlp, "-s", ms=3.5, lw=1.6, color="#7a4a2b",
                    alpha=0.85, label="MLP probe")
        ax.axhline(R["static_embedding"]["accuracy"], ls="--", lw=1.4,
                   color="#4bb3d4", label="static embedding")
        ax.axhline(R["ngram_tfidf"]["accuracy"], ls="-.", lw=1.4,
                   color="#5b3f8c", label="bag-of-ngrams")
        ax.axhline(R["static_embedding"]["majority"], ls=":", lw=1.4,
                   color="#888", label="majority class")
        if "control_task" in R:
            ax.axhline(R["control_task"]["accuracy"], ls="--", lw=1.2,
                       color="#e58fa8", label="control task")
        ax.set_title(t, fontsize=11)
        ax.set_xlabel("layer")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("test accuracy (unseen wordings)")
    axes[-1].legend(fontsize=7.5, loc="lower right")
    fig.suptitle(f"Linear decodability of face-region state by layer  "
                 f"({pr['meta']['model']})", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG / f"layer_accuracy_{tag}.png", dpi=170)
    plt.close(fig)


def fig_controls(pr: dict, tag: str) -> None:
    targets = sorted(pr["targets"])
    keys = ["best_linear", "shuffled", "static_embedding", "ngram_tfidf",
            "control_task", "majority"]
    labels = ["hidden state\n(best layer)", "shuffled\norder",
              "static\nembedding", "bag-of-\nngrams", "control\ntask",
              "majority"]
    colors = ["#c2183c", "#7a4a2b", "#4bb3d4", "#5b3f8c", "#e58fa8", "#999"]
    fig, ax = plt.subplots(figsize=(1.9 * len(targets) + 4, 4.0))
    w = 0.8 / len(keys)
    xs = np.arange(len(targets))
    for i, (k, lab, c) in enumerate(zip(keys, labels, colors)):
        vals = []
        for t in targets:
            R = pr["targets"][t]
            if k == "best_linear":
                v = R["best_linear_accuracy"]
            elif k == "shuffled":
                v = (R.get("shuffled", {})
                     .get(str(R["best_layer"]), {}).get("accuracy", np.nan))
            elif k == "majority":
                v = R["static_embedding"]["majority"]
            else:
                v = R.get(k, {}).get("accuracy", np.nan)
            vals.append(v)
        ax.bar(xs + i * w - 0.4 + w / 2, vals, w, label=lab, color=c)
    ax.set_xticks(xs)
    ax.set_xticklabels(targets)
    ax.set_ylabel("test accuracy")
    ax.set_title("Probe vs. every control", fontsize=12)
    ax.legend(fontsize=7.5, ncol=3)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG / f"controls_{tag}.png", dpi=170)
    plt.close(fig)


def fig_order_sensitive(pr: dict, tag: str) -> None:
    targets = sorted(pr["targets"])
    a = [pr["targets"][t]["order_sensitive_split"]["acc_order_sensitive"]
         for t in targets]
    b = [pr["targets"][t]["order_sensitive_split"]["acc_first_mention_ok"]
         for t in targets]
    fig, ax = plt.subplots(figsize=(1.6 * len(targets) + 3, 3.8))
    xs = np.arange(len(targets))
    ax.bar(xs - 0.2, b, 0.4, label="first mention is still correct",
           color="#4bb3d4")
    ax.bar(xs + 0.2, a, 0.4, label="value was overridden earlier",
           color="#c2183c")
    ax.set_xticks(xs)
    ax.set_xticklabels(targets)
    ax.set_ylabel("test accuracy")
    ax.set_title("Positions where the state was overridden\n"
                 "(a keyword heuristic cannot answer these)", fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG / f"order_sensitive_{tag}.png", dpi=170)
    plt.close(fig)


# ------------------------------------------------------------------ fig 3
def fig_capacity(cap: dict, tag: str) -> None:
    targets = sorted(cap["targets"])
    fig, axes = plt.subplots(2, len(targets), figsize=(3.8 * len(targets), 6.4),
                             squeeze=False)
    for j, t in enumerate(targets):
        rows = cap["targets"][t]["rows"]
        xs = [r["capacity"] if r["capacity"] else 1 for r in rows]
        ax = axes[0][j]
        ax.plot(xs, [r["real"] for r in rows], "-o", color="#c2183c",
                label="real labels")
        ax.plot(xs, [r["control"] for r in rows], "-s", color="#e58fa8",
                label="control task")
        ax.plot(xs, [r["selectivity"] for r in rows], "--^", color="#5b3f8c",
                label="selectivity")
        ax.set_xscale("log", base=2)
        ax.set_title(f"{t} @ L{cap['targets'][t]['layer']}", fontsize=10)
        ax.set_xlabel("MLP hidden units (1 = linear)")
        ax.grid(alpha=0.25)
        if j == 0:
            ax.set_ylabel("accuracy")
            ax.legend(fontsize=7.5)
        ax = axes[1][j]
        ax.plot(xs, [r["mdl"]["codelength_kbits"] for r in rows], "-o",
                color="#7a4a2b")
        ax.axhline(rows[0]["mdl"]["uniform_kbits"], ls=":", color="#888",
                   label="uniform code")
        ax.set_xscale("log", base=2)
        ax.set_xlabel("MLP hidden units")
        ax.grid(alpha=0.25)
        if j == 0:
            ax.set_ylabel("MDL code length (kbits)")
            ax.legend(fontsize=7.5)
    fig.suptitle("Probe capacity ablation: does more capacity buy real signal?",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG / f"capacity_{tag}.png", dpi=170)
    plt.close(fig)


# ------------------------------------------------------------------ fig 4-5
def fig_geometry(geo: dict, tag: str) -> None:
    keys = geo["direction_keys"]
    M = np.array(geo["cosine_cav"])
    fig, ax = plt.subplots(figsize=(0.42 * len(keys) + 3.4,
                                    0.42 * len(keys) + 2.8))
    im = ax.imshow(M, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(keys, rotation=90, fontsize=7)
    ax.set_yticks(range(len(keys)))
    ax.set_yticklabels(keys, fontsize=7)
    ax.set_title(f"Concept-direction cosine similarity (layer {geo['layer']})\n"
                 "off-diagonal blocks near 0 = regions kept separate",
                 fontsize=10)
    fig.colorbar(im, shrink=0.8)
    fig.tight_layout()
    fig.savefig(FIG / f"direction_cosine_{tag}.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    for t, rows in geo["dimensionality"].items():
        ax.plot([r["n_components"] for r in rows],
                [r["accuracy"] for r in rows], "-o", ms=4, label=t)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("PCA components retained")
    ax.set_ylabel("linear probe accuracy")
    ax.set_title(f"How many dimensions carry the state? (layer {geo['layer']})",
                 fontsize=11)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / f"dimensionality_{tag}.png", dpi=170)
    plt.close(fig)


def fig_scatter3d(tag: str, dataset: str, model_tag: str) -> None:
    npz_path = RES / f"geometry_{dataset}_{model_tag}.npz"
    traj_path = RES / f"trajectories_{dataset}_{model_tag}.json"
    if not npz_path.exists():
        return
    import plotly.graph_objects as go

    d = np.load(npz_path, allow_pickle=True)
    Z = d["Z"]
    ykeys = [k for k in d.files if k.startswith("y_")]
    geo = load(f"geometry_{dataset}_{model_tag}.json")
    classes_of = {}
    for k in ykeys:
        t = k[2:]
        classes_of[t] = sorted({c.split("=")[1] for c in geo["direction_keys"]
                                if c.startswith(t + "=")})

    for k in ykeys:
        t = k[2:]
        y = d[k]
        cls = classes_of[t]
        fig = go.Figure()
        for ci, cname in enumerate(cls):
            m = y == ci
            if not m.any():
                continue
            fig.add_trace(go.Scatter3d(
                x=Z[m, 0], y=Z[m, 1], z=Z[m, 2], mode="markers",
                name=f"{t} = {cname}",
                marker=dict(size=2.4, color=SWATCH.get(cname, "#333"),
                            opacity=0.75)))
        fig.update_layout(title=f"Hidden states coloured by true {t} "
                                f"(layer {geo['layer']}, PCA-3)",
                          template="plotly_white", height=720)
        fig.write_html(FIG / f"scatter3d_{t}_{tag}.html",
                       include_plotlyjs="cdn")

    if traj_path.exists():
        traj = json.loads(traj_path.read_text(encoding="utf-8"))
        animate_trajectory(traj, Z, geo, tag)


def animate_trajectory(traj: list[dict], Z: np.ndarray, geo: dict,
                       tag: str) -> None:
    import plotly.graph_objects as go

    tr = traj[0]
    C = np.array(tr["coords"])
    frames = []
    for i in range(1, len(C) + 1):
        frames.append(go.Frame(
            name=str(i),
            traces=[1, 2],          # leave the background cloud (trace 0) alone
            data=[go.Scatter3d(x=C[:i, 0], y=C[:i, 1], z=C[:i, 2],
                               mode="lines+markers",
                               line=dict(width=6, color="#c2183c"),
                               marker=dict(size=5, color="#c2183c")),
                  go.Scatter3d(x=[C[i - 1, 0]], y=[C[i - 1, 1]],
                               z=[C[i - 1, 2]], mode="markers",
                               marker=dict(size=11, color="#5b3f8c"))],
            layout=go.Layout(
                annotations=[],
                title=f"{tr['sentences'][i - 1][:110]}<br>"
                      f"<sub>{tr['labels'][i - 1]}</sub>")))

    fig = go.Figure(
        data=[go.Scatter3d(x=Z[::7, 0], y=Z[::7, 1], z=Z[::7, 2],
                           mode="markers", name="all test states",
                           marker=dict(size=1.6, color="#cccccc",
                                       opacity=0.45)),
              frames[0].data[0], frames[0].data[1]],
        frames=frames)
    fig.update_layout(
        template="plotly_white", height=760,
        title="One narrative moving through representation space",
        updatemenus=[dict(type="buttons", showactive=False,
                          buttons=[dict(label="play", method="animate",
                                        args=[None, dict(frame=dict(
                                            duration=900, redraw=True),
                                            fromcurrent=True)]),
                                   dict(label="pause", method="animate",
                                        args=[[None], dict(frame=dict(
                                            duration=0, redraw=False),
                                            mode="immediate")])])],
        sliders=[dict(steps=[dict(method="animate", label=str(i),
                                  args=[[str(i)], dict(mode="immediate",
                                                       frame=dict(duration=0,
                                                                  redraw=True))])
                             for i in range(1, len(C) + 1)])])
    fig.write_html(FIG / f"trajectory3d_{tag}.html", include_plotlyjs="cdn")


# ------------------------------------------------------------------ fig 6
def fig_intervention(iv: dict, tag: str, name: str) -> None:
    s = iv["summary"]
    m = iv["meta"]
    alphas = sorted(float(a) for a in s)

    def g(key):
        return [s[str(a)].get(key) for a in alphas]

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2))
    ax = axes[0]
    ax.plot(alphas, g("probe_rate_steered"), "-o", color="#c2183c",
            label="probe reads the steered value")
    ax.plot(alphas, g("probe_acc_true"), "-s", color="#4bb3d4",
            label="probe reads the true value")
    ax.plot(alphas, g("probe_other_acc"), "--^", color="#7a4a2b",
            label="other region still correct (control)")
    ax.plot(alphas, g("probe_other_changed_vs_baseline"), ":v", color="#999",
            label="other region changed at all (control)")
    ax.set_xlabel("steering strength alpha")
    ax.set_ylabel("rate over trials")
    ax.set_title(f"Probe read-out at L{m.get('read_layer', '?')}", fontsize=10)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7.5)

    ax = axes[1]
    ax.plot(alphas, g("beh_mean_delta_gap_steered"), "-o", color="#c2183c",
            label="steered value (target region)")
    ax.plot(alphas, g("beh_mean_delta_gap_other"), "--^", color="#7a4a2b",
            label="true value (other region, control)")
    ax.axhline(0, color="#444", lw=0.8)
    ax.set_xlabel("steering strength alpha")
    ax.set_ylabel("paired change in log-odds vs. alpha = 0")
    base = s[str(alphas[0])]["beh_acc_true"]
    ax.set_title(f"Behavioural read-out\n(unintervened accuracy {base:.2f})",
                 fontsize=10)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7.5)

    fig.suptitle(f"Activation patching at layer {m.get('edit_layer', m.get('layer'))}: "
                 f"{m['target']}  (n = {m['n_trials']} held-out positions)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / f"intervention_{name}_{tag}.png", dpi=170)
    plt.close(fig)


def fig_persistence(per: dict, tag: str) -> None:
    targets = sorted(per["targets"])
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    colors = ["#c2183c", "#4bb3d4", "#7a4a2b", "#5b3f8c", "#e58fa8", "#5a8f3c"]
    for t, c in zip(targets, colors):
        rows = per["targets"][t]["bins"]
        xs = [r["distance_min"] for r in rows]
        ax.plot(xs, [r["accuracy"] for r in rows], "-o", ms=4, color=c,
                label=f"{t} (L{per['targets'][t]['layer']})")
        ax.plot(xs, [r["majority_here"] for r in rows], ":", lw=1.1, color=c,
                alpha=0.65)
    ax.set_xlabel("sentences since this variable last changed")
    ax.set_ylabel("test accuracy")
    ax.set_title("Is the state maintained, or only echoed?\n"
                 "solid = probe, dotted = majority rate within the same bin",
                 fontsize=10.5)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7.5)
    fig.tight_layout()
    fig.savefig(FIG / f"persistence_{tag}.png", dpi=170)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="pilot")
    ap.add_argument("--model", default="gpt2")
    args = ap.parse_args()
    FIG.mkdir(exist_ok=True)
    mt = args.model.replace("/", "__")
    tag = f"{args.dataset}_{mt}"

    pr = load(f"probe_{tag}.json")
    if pr:
        fig_layer_curves(pr, tag)
        fig_controls(pr, tag)
        fig_order_sensitive(pr, tag)
        print("wrote layer / controls / order-sensitive figures")
    cap = load(f"capacity_{tag}.json")
    if cap:
        fig_capacity(cap, tag)
        print("wrote capacity ablation figure")
    per = load(f"persistence_{tag}.json")
    if per:
        fig_persistence(per, tag)
        print("wrote persistence figure")
    geo = load(f"geometry_{tag}.json")
    if geo:
        fig_geometry(geo, tag)
        fig_scatter3d(tag, args.dataset, mt)
        print("wrote geometry figures + 3D html")
    for p in sorted(RES.glob(f"intervention_{tag}_*.json")):
        iv = json.loads(p.read_text(encoding="utf-8"))
        fig_intervention(iv, tag, iv["meta"]["target"].replace(".", "-"))
        print(f"wrote intervention figure for {p.name}")
    print(f"\nfigures in {FIG}")


if __name__ == "__main__":
    main()
