"""Side-by-side face map: true state vs gpt2's read-out vs pythia's read-out.

One narrative plays sentence by sentence. Three faces are drawn: the true state
of each region, what a linear probe recovers from gpt2, and what the same probe
recovers from pythia-410m. Regions are filled with the actual palette colour;
the outline shows agreement with truth.

This is the figure a non-specialist can read: where the two models agree with
the truth and with each other, and where they diverge.

    python scripts/21_dual_binding_map.py --dataset pilot --narrative 0
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sklearn.preprocessing import StandardScaler  # noqa: E402

from mwm import probe as P  # noqa: E402
from mwm.extract import load_extracted  # noqa: E402

SWATCH = {"bare": "#efe7de", "nude": "#dcb79c", "pink": "#e88fa6",
          "red": "#c01c3f", "brown": "#79482a"}
INK = "#2f2a26"
OK = "#1f8a4c"
BAD = "#c01c3f"
PANELS = [("TRUE STATE", 0.0), ("gpt2", 1.10), ("pythia-410m", 2.20)]

# schematic face: (cx, cy, w, h)
REGIONS = {
    "skin":   (0.50, 0.52, 0.72, 0.88),
    "eyes":   (0.50, 0.68, 0.46, 0.11),
    "cheeks": (0.50, 0.48, 0.54, 0.10),
    "lips":   (0.50, 0.28, 0.22, 0.085),
}


def ellipse(cx, cy, w, h, n=72):
    t = np.linspace(0, 2 * np.pi, n)
    return cx + w / 2 * np.cos(t), cy + h / 2 * np.sin(t)


def build_probes(dataset, model, targets, max_train=6000):
    tag = model.replace("/", "__")
    X, E, meta, info = load_extracted(
        ROOT / "activations" / f"{dataset}__{tag}__natural")
    pr = json.loads((ROOT / "results" / f"probe_{dataset}_{tag}.json")
                    .read_text(encoding="utf-8"))
    tr = np.flatnonzero(P.split_mask(meta, "train", min_position=1))
    if len(tr) > max_train:
        tr = np.random.default_rng(0).choice(tr, max_train, False)
    out = {}
    for t in targets:
        layer = pr["targets"][t]["best_layer"]
        li = info["layers"].index(layer)
        y, classes = P.encode_labels(P.target_vector(meta, t))
        A = np.asarray(X[tr, li], dtype=np.float32)
        sc = StandardScaler().fit(A)
        clf = P.make_probe("linear")
        clf.fit(sc.transform(A), y[tr])
        out[t] = (sc, clf, classes, li)
    return out, X, meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="pilot")
    ap.add_argument("--model-a", default="gpt2")
    ap.add_argument("--model-b", default="EleutherAI/pythia-410m")
    ap.add_argument("--narrative", type=int, default=0)
    args = ap.parse_args()

    import plotly.graph_objects as go

    pa = json.loads((ROOT / "results" /
                     f"probe_{args.dataset}_{args.model_a.replace('/', '__')}.json")
                    .read_text(encoding="utf-8"))
    targets = sorted(pa["targets"])
    entities = sorted({t.split(".")[0] for t in targets})

    print(f"training probes for {args.model_a} ...", flush=True)
    PA, XA, meta = build_probes(args.dataset, args.model_a, targets)
    print(f"training probes for {args.model_b} ...", flush=True)
    PB, XB, _ = build_probes(args.dataset, args.model_b, targets)

    nids = []
    for m in meta:
        if m["split"] == "test" and m["narrative_id"] not in nids:
            nids.append(m["narrative_id"])
    nid = nids[args.narrative]
    rows = sorted([m for m in meta if m["narrative_id"] == nid],
                  key=lambda m: m["position"])

    def read(probes, Xsrc, row):
        s = {}
        for t, (sc, clf, classes, li) in probes.items():
            h = np.asarray(Xsrc[row, li], dtype=np.float32)[None]
            s[t] = classes[int(clf.predict(sc.transform(h))[0])]
        return s

    states = []
    for m in rows:
        states.append({
            "true": {t: m["labels"][t] for t in targets},
            "a": read(PA, XA, m["row"]),
            "b": read(PB, XB, m["row"]),
        })

    def traces(i):
        st = states[i]
        out = []
        for (label, dx), key in zip(PANELS, ("true", "a", "b")):
            for ent in entities:
                cx, cy, w, h = REGIONS[ent]
                col_key = f"{ent}.color"
                colour = st[key].get(col_key, "bare")
                fill = SWATCH.get(colour, "#efe7de")
                cov = st[key].get(f"{ent}.coverage", "light")
                opacity = {"bare": 0.22, "light": 0.60,
                           "full": 0.95}.get(cov, 0.6)
                matches = (key == "true" or
                           st[key].get(col_key) == st["true"].get(col_key))
                edge = INK if key == "true" else (OK if matches else BAD)
                x, yy = ellipse(cx + dx, cy, w, h)
                out.append(go.Scatter(
                    x=x, y=yy, fill="toself", mode="lines",
                    line=dict(color=edge, width=1.5 if key == "true" else 2.6),
                    fillcolor=fill, opacity=opacity, showlegend=False,
                    hovertemplate=(
                        f"<b>{ent}</b> — {label}<br>"
                        f"colour {st[key].get(col_key)}<br>"
                        f"finish {st[key].get(f'{ent}.finish')}<br>"
                        f"coverage {st[key].get(f'{ent}.coverage')}"
                        f"<extra></extra>")))
        return out

    n = len(rows)

    def title(i):
        st = states[i]
        agree_a = sum(st["a"][t] == st["true"][t] for t in targets)
        agree_b = sum(st["b"][t] == st["true"][t] for t in targets)
        sent = rows[i]["text"].split(". ")[-1]
        return (f"<b>Sentence {i + 1} of {n}</b>  —  {sent[:130]}"
                f"<br><sub>correct readings: gpt2 {agree_a}/{len(targets)}"
                f" &nbsp;·&nbsp; pythia {agree_b}/{len(targets)}"
                f" &nbsp;·&nbsp; green outline = colour matches truth,"
                f" red = mismatch &nbsp;·&nbsp; fill opacity = coverage</sub>")

    frames = [go.Frame(name=str(i), data=traces(i),
                       layout=go.Layout(title=title(i))) for i in range(n)]
    fig = go.Figure(data=traces(0), frames=frames)

    for label, dx in PANELS:
        fig.add_annotation(x=0.5 + dx, y=1.03, text=f"<b>{label}</b>",
                           showarrow=False, font=dict(size=14, color=INK))
    for ent, (cx, cy, w, h) in REGIONS.items():
        for _, dx in PANELS:
            fig.add_annotation(x=cx + dx, y=cy, text=ent, showarrow=False,
                               font=dict(size=9, color="#5a5148"))

    fig.update_layout(
        template="plotly_white", height=560, width=1180,
        margin=dict(l=20, r=20, t=110, b=70),
        paper_bgcolor="#ffffff", plot_bgcolor="#ffffff",
        xaxis=dict(visible=False, range=[-0.05, 3.15], scaleanchor="y"),
        yaxis=dict(visible=False, range=[0.0, 1.10]),
        title=dict(text=title(0), x=0.01, xanchor="left",
                   font=dict(size=13)),
        updatemenus=[dict(type="buttons", showactive=False, x=0.01, y=-0.05,
                          direction="right", pad=dict(t=4),
                          buttons=[
                              dict(label="▶ play", method="animate",
                                   args=[None, dict(frame=dict(duration=1600,
                                                               redraw=True),
                                                    fromcurrent=True)]),
                              dict(label="❚❚ pause", method="animate",
                                   args=[[None], dict(frame=dict(duration=0,
                                                                 redraw=False),
                                                      mode="immediate")])])],
        sliders=[dict(active=0, y=-0.02, x=0.16, len=0.82, steps=[
            dict(method="animate", label=str(i + 1),
                 args=[[str(i)], dict(mode="immediate",
                                      frame=dict(duration=0, redraw=True))])
            for i in range(n)])])

    out = ROOT / "figures" / f"dual_binding_map_{args.dataset}.html"
    fig.write_html(out, include_plotlyjs="cdn")
    print(f"\nwrote {out}")

    aa = np.mean([[states[i]["a"][t] == states[i]["true"][t] for t in targets]
                  for i in range(n)])
    bb = np.mean([[states[i]["b"][t] == states[i]["true"][t] for t in targets]
                  for i in range(n)])
    ab = np.mean([[states[i]["a"][t] == states[i]["b"][t] for t in targets]
                  for i in range(n)])
    print(f"gpt2 agrees with truth  {aa:.1%}")
    print(f"pythia agrees with truth {bb:.1%}")
    print(f"the two models agree with each other {ab:.1%}")


if __name__ == "__main__":
    main()
