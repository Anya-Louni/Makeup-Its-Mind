"""Cross-model comparison figures: gpt2 vs pythia-410m.

Two panels the paper needs:

1. The replication bar chart -- gap over the static-embedding baseline on the
   same four targets, with cluster-bootstrap intervals, so the reader can see
   at a glance that nothing replicates in the same direction.
2. The layer-depth chart -- where each model's state code peaks. gpt2 is
   scattered over early and middle layers; pythia sits at the very end. That
   difference is a large part of why the two behave differently.

    python scripts/20_crossmodel_figures.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
RES, FIG = ROOT / "results", ROOT / "figures"

GPT2 = "#c2183c"
PYTHIA = "#2f6f9f"
GREY = "#9a9a9a"


def load(name):
    p = RES / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def fig_replication():
    g = load("significance_pilot_gpt2.json")
    p = load("significance_pilot_EleutherAI__pythia-410m.json")
    if not (g and p):
        print("missing significance files")
        return
    targets = [t for t in sorted(g["targets"]) if t in p["targets"]]

    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    x = np.arange(len(targets))
    w = 0.36
    for i, (res, colour, label) in enumerate(
            ((g, GPT2, "gpt2 (124M)"), (p, PYTHIA, "pythia-410m"))):
        vals, lo, hi, sig = [], [], [], []
        for t in targets:
            gp = res["targets"][t]["gaps"]["static_embedding"]
            vals.append(gp["gap"])
            lo.append(gp["gap"] - gp["ci95"][0])
            hi.append(gp["ci95"][1] - gp["gap"])
            sig.append(gp["p_two_sided"] < 0.05)
        pos = x + (i - 0.5) * w
        # matplotlib takes a scalar alpha, so encode significance in the
        # per-bar face colour instead
        faces = [colour if s else colour + "66" for s in sig]
        ax.bar(pos, vals, w, color=faces, label=label,
               edgecolor="black", linewidth=0.6)
        ax.errorbar(pos, vals, yerr=[lo, hi], fmt="none", ecolor="#333",
                    elinewidth=1.1, capsize=3)
        # place the marker clear of the error bar, not on top of it
        for xp, v, hh, ll, s in zip(pos, vals, hi, lo, sig):
            if s:
                top = v + hh if v >= 0 else v - ll
                ax.text(xp, top + (0.006 if v >= 0 else -0.016), "*",
                        ha="center", va="bottom" if v >= 0 else "top",
                        fontsize=14, color="#111")

    ax.axhline(0, color="#333", lw=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels([t.replace(".", "\n") for t in targets])
    ax.set_ylabel("accuracy gap over static-embedding baseline")
    ax.set_title("Does the probe beat a lexical baseline? Two models, same data\n"
                 "solid bars = significant (two-sided p < 0.05, cluster "
                 "bootstrap over narratives); faded = not",
                 fontsize=10.5)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(axis="y", alpha=0.25)
    ax.text(0.5, -0.20,
            "No target is significant in the same direction in both models.",
            transform=ax.transAxes, ha="center", fontsize=9.5,
            style="italic", color="#444")
    fig.tight_layout()
    fig.savefig(FIG / "crossmodel_replication.png", dpi=180,
                bbox_inches="tight")
    plt.close(fig)
    print("wrote crossmodel_replication.png")


def fig_layer_depth():
    g = load("probe_pilot_gpt2.json")
    p = load("probe_pilot_EleutherAI__pythia-410m.json")
    gf = load("probe_full_gpt2.json")
    if not (g and p):
        print("missing probe files")
        return

    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    series = [
        ("gpt2 (12 layers)", g, 12, GPT2, "o"),
        ("pythia-410m (24 layers)", p, 24, PYTHIA, "s"),
    ]
    if gf:
        series.append(("gpt2, 4-entity set", gf, 12, GREY, "^"))

    for row, (label, res, n_layers, colour, marker) in enumerate(series):
        fracs, names = [], []
        for t in sorted(res["targets"]):
            fracs.append(res["targets"][t]["best_layer"] / n_layers)
            names.append(t)
        jitter = (np.random.default_rng(0).random(len(fracs)) - 0.5) * 0.16
        ax.scatter(fracs, np.full(len(fracs), row) + jitter, s=70,
                   color=colour, marker=marker, edgecolor="black",
                   linewidth=0.5, zorder=3, label=label)

    ax.set_yticks(range(len(series)))
    ax.set_yticklabels([lab for lab, *_ in series], fontsize=9.5)
    ax.set_ylim(-0.75, len(series) - 0.25)
    ax.set_xlim(-0.04, 1.05)
    ax.set_xlabel("best layer, as a fraction of network depth "
                  "(0 = embeddings, 1 = final layer)")
    ax.set_title("Where does each model's state code live?", fontsize=11.5)
    for xv, lab in ((0.0, "embeddings"), (0.5, "middle"), (1.0, "output")):
        ax.axvline(xv, color="#bbb", lw=0.9, ls=":")
        ax.text(xv, len(series) - 0.45, lab, ha="center", fontsize=8.5,
                color="#666")
    ax.grid(axis="x", alpha=0.2)
    ax.text(0.5, -0.62,
            "gpt2 spreads across embeddings and middle layers; pythia sits "
            "almost entirely at the output end.\n"
            "On the 4-entity set gpt2 collapses to the embedding layer for "
            "8 of 12 targets.",
            transform=ax.transData, ha="center", fontsize=9,
            style="italic", color="#444")
    fig.tight_layout()
    fig.savefig(FIG / "crossmodel_layer_depth.png", dpi=180,
                bbox_inches="tight")
    plt.close(fig)
    print("wrote crossmodel_layer_depth.png")


if __name__ == "__main__":
    FIG.mkdir(exist_ok=True)
    fig_replication()
    fig_layer_depth()
