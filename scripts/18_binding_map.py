"""Face-region binding map: an animated view of what the probe recovers.

A narrative plays sentence by sentence. For each frame the figure shows, per
face region, the TRUE state next to the state the linear probe reads out of the
hidden activations at that position. Regions are drawn as a simple face
diagram, filled with the actual palette colour and hatched by finish, so a
reader can see binding succeed or fail without reading a table.

Produces an interactive HTML (plotly) with a play control and a slider.

    python scripts/18_binding_map.py --dataset full --model gpt2
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

SWATCH = {"bare": "#e8e0d8", "nude": "#d9b8a0", "pink": "#e58fa8",
          "red": "#c2183c", "brown": "#7a4a2b"}
FINISH_MARK = {"none": "", "matte": "flat", "dewy": "shine", "satin": "silk"}

# schematic face layout: (x, y, width, height) per region
REGIONS = {
    "skin":   (0.50, 0.50, 0.78, 0.94),
    "eyes":   (0.50, 0.66, 0.52, 0.14),
    "cheeks": (0.50, 0.46, 0.60, 0.12),
    "lips":   (0.50, 0.26, 0.26, 0.10),
}


def ellipse(cx, cy, w, h, n=64):
    t = np.linspace(0, 2 * np.pi, n)
    return cx + w / 2 * np.cos(t), cy + h / 2 * np.sin(t)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="full")
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--narrative", type=int, default=0,
                    help="index among held-out narratives")
    ap.add_argument("--max-train", type=int, default=8000)
    args = ap.parse_args()

    import plotly.graph_objects as go

    tag = args.model.replace("/", "__")
    X, E, meta, info = load_extracted(
        ROOT / "activations" / f"{args.dataset}__{tag}__natural")
    pr = json.loads((ROOT / "results" / f"probe_{args.dataset}_{tag}.json")
                    .read_text(encoding="utf-8"))
    targets = sorted(pr["targets"])
    entities = sorted({t.split(".")[0] for t in targets})

    tr = np.flatnonzero(P.split_mask(meta, "train", min_position=1))
    if len(tr) > args.max_train:
        tr = np.random.default_rng(0).choice(tr, args.max_train, False)

    probes = {}
    for t in targets:
        layer = pr["targets"][t]["best_layer"]
        li = info["layers"].index(layer)
        y, classes = P.encode_labels(P.target_vector(meta, t))
        A = np.asarray(X[tr, li], dtype=np.float32)
        sc = StandardScaler().fit(A)
        clf = P.make_probe("linear")
        clf.fit(sc.transform(A), y[tr])
        probes[t] = (sc, clf, classes, li)
        print(f"  probe ready: {t} @L{layer}", flush=True)

    nids = []
    for m in meta:
        if m["split"] == "test" and m["narrative_id"] not in nids:
            nids.append(m["narrative_id"])
    nid = nids[args.narrative]
    rows = sorted([m for m in meta if m["narrative_id"] == nid],
                  key=lambda m: m["position"])

    pred_state, true_state = [], []
    for m in rows:
        p, tt = {}, {}
        for t in targets:
            sc, clf, classes, li = probes[t]
            h = np.asarray(X[m["row"], li], dtype=np.float32)[None]
            p[t] = classes[int(clf.predict(sc.transform(h))[0])]
            tt[t] = m["labels"][t]
        pred_state.append(p)
        true_state.append(tt)

    def frame_traces(i):
        traces = []
        for panel, state in (("true", true_state[i]), ("pred", pred_state[i])):
            dx = 0.0 if panel == "true" else 1.15
            for ent in entities:
                cx, cy, w, h = REGIONS[ent]
                col = SWATCH.get(state.get(f"{ent}.color", "bare"), "#e8e0d8")
                fin = state.get(f"{ent}.finish", "none")
                cov = state.get(f"{ent}.coverage", "bare")
                x, yy = ellipse(cx + dx, cy, w, h)
                opacity = {"bare": 0.18, "light": 0.55, "full": 0.95}.get(cov, 0.18)
                agree = (true_state[i].get(f"{ent}.color")
                         == pred_state[i].get(f"{ent}.color"))
                traces.append(go.Scatter(
                    x=x, y=yy, fill="toself", mode="lines",
                    line=dict(color="#2b2b2b" if panel == "true" else
                              ("#2b7a2b" if agree else "#c2183c"),
                              width=1.6 if panel == "true" else 2.4),
                    fillcolor=col, opacity=opacity, showlegend=False,
                    hovertemplate=(f"<b>{ent}</b> ({panel})<br>"
                                   f"colour {state.get(f'{ent}.color')}<br>"
                                   f"finish {fin} ({FINISH_MARK.get(fin,'')})<br>"
                                   f"coverage {cov}<extra></extra>")))
        return traces

    n = len(rows)
    frames = []
    for i in range(n):
        frames.append(go.Frame(
            name=str(i), data=frame_traces(i),
            layout=go.Layout(title=(
                f"<b>{i + 1}/{n}</b>  {rows[i]['text'].split('. ')[-1][:120]}"
                f"<br><sub>left = true state &nbsp;|&nbsp; right = linear probe "
                f"read-out &nbsp;|&nbsp; green outline = colour matches, red = "
                f"mismatch</sub>"))))

    fig = go.Figure(data=frame_traces(0), frames=frames)
    for dx, lab in ((0.0, "TRUE STATE"), (1.15, "PROBE READ-OUT")):
        fig.add_annotation(x=0.5 + dx, y=1.02, text=f"<b>{lab}</b>",
                           showarrow=False, font=dict(size=13))
    for ent, (cx, cy, w, h) in REGIONS.items():
        for dx in (0.0, 1.15):
            fig.add_annotation(x=cx + dx, y=cy, text=ent, showarrow=False,
                               font=dict(size=10, color="#333"))
    fig.update_layout(
        template="plotly_white", height=620, width=1040,
        xaxis=dict(visible=False, range=[0, 2.15], scaleanchor="y"),
        yaxis=dict(visible=False, range=[0, 1.08]),
        title=frames[0].layout.title,
        updatemenus=[dict(type="buttons", showactive=False, x=0.02, y=-0.06,
                          direction="right",
                          buttons=[
                              dict(label="play", method="animate",
                                   args=[None, dict(frame=dict(duration=1400,
                                                               redraw=True),
                                                    fromcurrent=True)]),
                              dict(label="pause", method="animate",
                                   args=[[None], dict(frame=dict(duration=0,
                                                                 redraw=False),
                                                      mode="immediate")])])],
        sliders=[dict(active=0, y=-0.02, steps=[
            dict(method="animate", label=str(i + 1),
                 args=[[str(i)], dict(mode="immediate",
                                      frame=dict(duration=0, redraw=True))])
            for i in range(n)])])

    out = ROOT / "figures" / f"binding_map_{args.dataset}_{tag}.html"
    out.parent.mkdir(exist_ok=True)
    fig.write_html(out, include_plotlyjs="cdn")
    print(f"\nwrote {out}")

    agree = np.mean([[true_state[i][t] == pred_state[i][t] for t in targets]
                     for i in range(n)])
    print(f"probe agrees with truth on {agree:.1%} of "
          f"{n * len(targets)} region-attribute readings in this narrative")


if __name__ == "__main__":
    main()
