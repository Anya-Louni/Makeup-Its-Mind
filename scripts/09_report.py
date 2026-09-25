"""Collect every result file into one markdown report.

    python scripts/09_report.py --dataset pilot --model gpt2
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
RES = ROOT / "results"


def load(name: str):
    p = RES / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def fmt(x, nd=3):
    if x is None:
        return "-"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="pilot")
    ap.add_argument("--model", default="gpt2")
    args = ap.parse_args()
    mt = args.model.replace("/", "__")
    tag = f"{args.dataset}_{mt}"
    L: list[str] = []

    L += [f"# Results: `{args.model}` on the `{args.dataset}` dataset", ""]

    stats_p = ROOT / "data" / args.dataset / "stats.json"
    if stats_p.exists():
        st = json.loads(stats_p.read_text(encoding="utf-8"))
        L += ["## Dataset", "",
              f"- {st['n_narratives']} narratives, {st['n_positions']} labelled "
              f"positions ({st['sentences_per_narrative']} sentences each)",
              f"- action mix: " + ", ".join(
                  f"{k} {v}" for k, v in
                  list(st["action_kind_distribution"].items())),
              "- fraction of positions where the value was overridden earlier "
              "(first mention is the wrong answer): " + ", ".join(
                  f"{k} {v:.2f}" for k, v in st["order_sensitive_rate"].items()),
              ""]

    pr = load(f"probe_{tag}.json")
    if pr:
        L += ["## Probing", "",
              "All numbers are test-split accuracy on narratives built from "
              "wordings never seen in training.", "",
              "| target | classes | majority | best layer | linear | MLP | "
              "static emb | bag-of-ngrams | control task (test) | "
              "control task (train) | selectivity | shuffled order |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for t in sorted(pr["targets"]):
            R = pr["targets"][t]
            bl = str(R["best_layer"])
            mlp = R["layers"][bl].get("mlp", {}).get("accuracy")
            sh = R.get("shuffled", {}).get(bl, {}).get("accuracy")
            L.append(
                f"| `{t}` | {len(R['classes'])} | "
                f"{fmt(R['static_embedding']['majority'])} | {bl} | "
                f"**{fmt(R['best_linear_accuracy'])}** | {fmt(mlp)} | "
                f"{fmt(R['static_embedding']['accuracy'])} | "
                f"{fmt(R['ngram_tfidf']['accuracy'])} | "
                f"{fmt(R['control_task']['accuracy'])} | "
                f"{fmt(R['control_task'].get('train_accuracy'))} | "
                f"{fmt(R['selectivity'])} | {fmt(sh)} |")

        L += ["", "### Positions where the state was overridden", "",
              "The subset where the first value mentioned for that region is "
              "*not* the answer. A keyword heuristic cannot score above chance "
              "here.", "",
              "| target | overridden positions | accuracy there | "
              "accuracy elsewhere |", "|---|---|---|---|"]
        for t in sorted(pr["targets"]):
            o = pr["targets"][t]["order_sensitive_split"]
            L.append(f"| `{t}` | {o['n_order_sensitive']} | "
                     f"{fmt(o['acc_order_sensitive'])} | "
                     f"{fmt(o['acc_first_mention_ok'])} |")

        binding = {t: pr["targets"][t].get("binding", {})
                   for t in sorted(pr["targets"])}
        if any(binding.values()):
            L += ["", "### Binding check", "",
                  "A probe trained on one region, read against the *other* "
                  "region's label. Near the other region's majority rate means "
                  "the regions are kept apart.", "",
                  "| probe trained on | read against | accuracy | that "
                  "region's majority |", "|---|---|---|---|"]
            for t, b in binding.items():
                for o, v in b.items():
                    L.append(f"| `{t}` | `{o}` | "
                             f"{fmt(v['acc_reading_other_entity'])} | "
                             f"{fmt(v['majority_other'])} |")

        if any("mdl" in pr["targets"][t] for t in pr["targets"]):
            L += ["", "### MDL (prequential code length)", "",
                  "Lower is better; compression is relative to the uniform "
                  "code.", "",
                  "| target | layer | code length (kbit) | compression | "
                  "static emb compression |", "|---|---|---|---|---|"]
            for t in sorted(pr["targets"]):
                R = pr["targets"][t]
                if "mdl" not in R:
                    continue
                for layer, m in R["mdl"].items():
                    L.append(f"| `{t}` | {layer} | "
                             f"{m['codelength_kbits']:.1f} | "
                             f"{m['compression']:.2f}x | "
                             f"{R['mdl_static_embedding']['compression']:.2f}x |")

    geo = load(f"geometry_{tag}.json")
    if geo:
        L += ["", "## Representation geometry", "",
              f"Layer {geo['layer']}, {geo['n_samples']} held-out positions. "
              f"PCA-3 explains "
              f"{100 * sum(geo['pca_explained_variance']):.1f}% of variance.",
              "", "### Same attribute value, different face region", "",
              "Cosine between the two regions' concept directions. Two ways "
              "of computing the direction are shown because they disagree, and "
              "the disagreement is the finding: the difference-of-means CAV is "
              "dominated by the shared \"this colour is present somewhere\" "
              "component, while the discriminative probe weights project that "
              "shared component out and point the two regions apart.", "",
              "| pair | CAV cosine | probe-weight cosine |", "|---|---|---|"]
        for k, v in geo["same_value_cross_entity_cosine"].items():
            L.append(f"| {k} | {v['cav']:.3f} | {v['probe']:.3f} |")
        L += ["", "### Subspace principal angles", "",
              "| pair | mean angle (deg) |", "|---|---|"]
        for k, v in geo["subspace_principal_angles"].items():
            L.append(f"| {k} | {v['mean_deg']} |")
        L += ["", "### Dimensionality", "",
              "| target | " + " | ".join(
                  str(r["n_components"])
                  for r in list(geo["dimensionality"].values())[0]) + " |",
              "|---" * (len(list(geo["dimensionality"].values())[0]) + 1) + "|"]
        for t, rows in geo["dimensionality"].items():
            L.append(f"| `{t}` | " + " | ".join(
                f"{r['accuracy']:.3f}" for r in rows) + " |")

    bbd = load(f"binding_by_distance_{tag}.json")
    if bbd:
        import numpy as _np
        L += ["", "## Binding, stratified by distance", "",
              "The aggregate binding test (probe trained on one region, read "
              "against another) is dominated by positions far from the last "
              "update, where the attribute is not decodable at all and both "
              "probes sit at majority. Stratifying by distance since the "
              "variable last changed shows what the aggregate hides.", "",
              "Binding margin = (within-entity accuracy - its majority) "
              "- (cross-entity accuracy - its majority). Positive means the "
              "probe knows *which region*, not merely that the value occurred.",
              "",
              "| distance | pairs | within - majority | cross - majority | "
              "binding margin |", "|---|---|---|---|---|"]
        agg: dict = {}
        for v in bbd["pairs"].values():
            for b in v["bins"]:
                d = (f"{b['distance_min']}+" if b["distance_max"] is None
                     else (str(b["distance_min"])
                           if b["distance_max"] == b["distance_min"]
                           else f"{b['distance_min']}-{b['distance_max']}"))
                agg.setdefault(d, []).append(b)
        def _k(d):
            return int(d.replace("+", "").split("-")[0])
        for d in sorted(agg, key=_k):
            bs = agg[d]
            w = _np.mean([b["within_acc"] - b["within_majority"] for b in bs])
            c = _np.mean([b["cross_acc"] - b["cross_majority"] for b in bs])
            m = _np.mean([b["binding_margin"] for b in bs])
            L.append(f"| {d} | {len(bs)} | {w:+.3f} | {c:+.3f} | "
                     f"**{m:+.3f}** |")
        L += ["", "The sign flip at the largest distance is the substantive "
                  "finding: within-entity accuracy falls below majority while "
                  "the cross-entity read rises above it. Past roughly five "
                  "sentences the representation still carries \"this value "
                  "occurred somewhere\" but has lost which region it applied "
                  "to. The shared presence code outlives the region-specific "
                  "binding code -- which is what the concept-direction "
                  "geometry independently says.", ""]

    ordr = load(f"order_sensitivity_{tag}.json")
    if ordr and ordr.get("mean_delta_contextual") is not None:
        L += ["", "## Does the shuffled-order control work?", "",
              f"Averaged over all {ordr['n_targets']} targets, scrambling "
              f"sentence order costs {abs(ordr['mean_delta_all']) * 100:.1f} "
              "accuracy points, which reads as \"the model ignores order\". "
              "That average pools targets whose best layer is the *embedding "
              "layer* -- which cannot encode order at all -- with targets that "
              "genuinely use the transformer.", "",
              f"| group | n | mean shuffled delta |", "|---|---|---|",
              f"| best layer > 0 (contextual) | {ordr['n_contextual']} | "
              f"{ordr['mean_delta_contextual']:+.4f} |",
              f"| best layer == 0 (embeddings) | {ordr['n_embedding_layer']} | "
              f"{ordr['mean_delta_embedding_layer']:+.4f} |", "",
              f"- corr(best layer, shuffled delta) = "
              f"{ordr['corr_bestlayer_delta']:+.3f}",
              f"- corr(best layer, selectivity) = "
              f"{ordr['corr_bestlayer_selectivity']:+.3f}",
              f"- corr(selectivity, shuffled delta) = "
              f"{ordr['corr_selectivity_delta']:+.3f}", "",
              "Three independent measures -- best-layer depth, selectivity and "
              "order-sensitivity -- pick out the same targets. The control is "
              "not weak; it was correctly reporting that most targets have no "
              "temporal structure to destroy.", ""]

    per = load(f"persistence_{tag}.json")
    if per:
        L += ["", "## State persistence", "",
              "Accuracy binned by how many sentences have passed since that "
              "variable last changed. A representation that only reflects the "
              "current sentence should collapse towards the majority rate as "
              "the last update recedes; one that maintains state should stay "
              "flat.", "",
              "| target | layer | distance | n | accuracy | majority in bin |",
              "|---|---|---|---|---|---|"]
        for t in sorted(per["targets"]):
            R = per["targets"][t]
            for b in R["bins"]:
                d = (f"{b['distance_min']}+" if b["distance_max"] is None
                     else (str(b["distance_min"]) if b["distance_max"] ==
                           b["distance_min"] else
                           f"{b['distance_min']}-{b['distance_max']}"))
                L.append(f"| `{t}` | {R['layer']} | {d} | {b['n']} | "
                         f"{b['accuracy']:.3f} | {b['majority_here']:.3f} |")

    cap = load(f"capacity_{tag}.json")
    if cap:
        L += ["", "## Probe-capacity ablation", "",
              "| target | probe | real | control | selectivity | MDL (kbit) |",
              "|---|---|---|---|---|---|"]
        for t in sorted(cap["targets"]):
            for r in cap["targets"][t]["rows"]:
                name = "linear" if r["capacity"] == 0 else f"MLP-{r['capacity']}"
                L.append(f"| `{t}` | {name} | {r['real']:.3f} | "
                         f"{r['control']:.3f} | {r['selectivity']:+.3f} | "
                         f"{r['mdl']['codelength_kbits']:.1f} |")

    ivs = sorted(RES.glob(f"intervention_{tag}_*.json"))
    if ivs:
        L += ["", "## Causal intervention", ""]
        for p in ivs:
            iv = json.loads(p.read_text(encoding="utf-8"))
            m = iv["meta"]
            L += [f"### `{m['target']}` steered at layer "
                  f"{m.get('edit_layer', m.get('layer'))} "
                  f"({m['n_trials']} held-out positions)", "",
                  f"Mean hidden-state norm at the edited layer: "
                  f"{m['mean_hidden_norm']:.1f}. Probe read-out trained at "
                  f"layer {m.get('read_layer', '?')} on clean data.", "",
                  "| alpha | probe: true | probe: steered | probe: steered "
                  "(trials initially correct) | other region correct | other "
                  "region changed | behaviour: true | behaviour: delta "
                  "log-odds steered | behaviour: delta log-odds other |",
                  "|---|---|---|---|---|---|---|---|---|"]
            for a, s in iv["summary"].items():
                L.append(
                    f"| {a} | {s['probe_acc_true']:.3f} | "
                    f"{s['probe_rate_steered']:.3f} | "
                    f"{fmt(s['probe_flip_rate_on_initially_correct'])} | "
                    f"{s['probe_other_acc']:.3f} | "
                    f"{s['probe_other_changed_vs_baseline']:.3f} | "
                    f"{s['beh_acc_true']:.3f} | "
                    f"{s['beh_mean_delta_gap_steered']:+.3f} | "
                    f"{s['beh_mean_delta_gap_other']:+.3f} |")
            L.append("")

    st = load(f"intervention_stats_{tag}.json")
    if st:
        L += ["", "## Causal intervention: effect sizes and significance", "",
              "Paired bootstrap over trials (10,000 resamples). The binding "
              "claim is the last column: the edit must move the target "
              "region's read-out more than the other region's. p-values are "
              "two-sided throughout this report; the one-sided values are kept "
              "in the result JSON as `p_one_sided_greater`.", ""]
        for target, res in st.items():
            m = res["meta"]
            L += [f"### `{target}` (edit L{m['edit_layer']} -> read "
                  f"L{m['read_layer']}, n={m['n_trials']})", "",
                  "| alpha | delta log-odds, target | delta log-odds, other "
                  "region | target - other | p (two-sided) |",
                  "|---|---|---|---|---|"]
            for a, v in res["alphas"].items():
                if float(a) == 0.0:
                    continue
                t, o, dd = (v["delta_logodds_target"],
                            v["delta_logodds_other"],
                            v["target_minus_other_signed"])
                L.append(
                    f"| {a} | {t['mean']:+.3f} "
                    f"[{t['ci95'][0]:+.3f}, {t['ci95'][1]:+.3f}] | "
                    f"{o['mean']:+.3f} "
                    f"[{o['ci95'][0]:+.3f}, {o['ci95'][1]:+.3f}] | "
                    f"{dd['mean']:+.3f} "
                    f"[{dd['ci95'][0]:+.3f}, {dd['ci95'][1]:+.3f}] | "
                    f"{dd.get('p_two_sided', dd.get('bootstrap_p_le_0', float('nan'))):.4f} |")
            f = res["alphas"][str(max(float(a) for a in res["alphas"]))][
                "probe_flip_initially_correct"]
            L += ["", f"Probe read-out flips to the steered value on "
                      f"{f['mean']:.3f} [{f['ci95'][0]:.3f}, "
                      f"{f['ci95'][1]:.3f}] of the {f['n']} trials whose "
                      f"unintervened read-out was already correct. The small "
                      f"n is the limitation: the read-out probe sits "
                      f"downstream of the edit, so it is weaker than the "
                      f"best-layer probe.", ""]

    out = ROOT / "docs" / f"RESULTS_{tag}.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
