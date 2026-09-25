"""Phase 1, steps 1-2: vocabulary sanity check + model benchmark.

Why a contrastive test instead of "ask the model to define matte": the
candidates are 0.1-2 B *base* LMs with no instruction tuning, so free-form
definitions are unreliable evidence either way.  What we actually need to know
is whether the term carries the right semantics in the model's distribution.
So for each domain term we score minimal pairs

    logP("A matte finish reflects almost no light")
    logP("A matte finish reflects almost no sound")

and count how often the model prefers the semantically correct completion.
Chance is 50%.  A term the model cannot separate is a confound: low probe
accuracy on it would mean "does not know the word", not "no world model".

Run:  python scripts/01_vocab_and_benchmark.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

from mwm import models as M  # noqa: E402

# ---------------------------------------------------------------- vocab test
# (term, prefix, correct continuation, foil continuation)
VOCAB_PROBES: list[tuple[str, str, str, str]] = [
    # --- finish ---
    ("matte", "A matte lipstick finish reflects almost no", " light", " sound"),
    ("matte", "Her lipstick was completely matte, so it looked", " flat", " wet"),
    ("matte", "To make her skin matte she blotted away the", " shine", " colour"),
    ("dewy", "Her skin looked dewy, as though it were slightly", " damp", " dusty"),
    ("dewy", "A dewy finish catches the", " light", " dust"),
    ("dewy", "She wanted a dewy glow, so she avoided heavy", " powder", " water"),
    ("satin", "A satin finish sits between matte and", " glossy", " purple"),
    ("satin", "Satin lipstick has a soft", " sheen", " smell"),
    # --- colour ---
    ("nude", "A nude lipstick is close to the colour of her own", " skin", " hair"),
    ("nude", "She chose a nude shade because she wanted something", " subtle", " loud"),
    ("pink", "The blush was a soft pink, a little like a", " rose", " lemon"),
    ("red", "She wore a bold red lipstick, the colour of a ripe", " cherry", " lime"),
    ("brown", "The eyeshadow was a warm brown, like", " chocolate", " snow"),
    ("crimson", "Crimson is a deep shade of", " red", " green"),
    ("beige", "Beige is a pale", " neutral", " blue"),
    # --- coverage ---
    ("sheer", "A sheer wash of colour lets the skin underneath still", " show", " itch"),
    ("opaque", "Full opaque coverage means nothing underneath is", " visible", " painted"),
    ("coverage", "She built up the coverage until the colour was completely",
     " solid", " sheer"),
    # --- actions ---
    ("blot", "She blotted her lips with a tissue to remove the", " shine", " colour"),
    ("blend", "She blended the edges so there was no harsh", " line", " smell"),
    ("blush", "Blush is applied to the", " cheeks", " elbows"),
    ("concealer", "Concealer is used to cover", " blemishes", " furniture"),
    ("eyeshadow", "Eyeshadow is applied to the", " eyelids", " lips"),
    ("lipstick", "Lipstick is applied to the", " lips", " eyelids"),
    ("bronzer", "Bronzer makes the skin look", " tanned", " paler"),
    ("primer", "Primer is applied before", " foundation", " dinner"),
    # --- regions (binding depends on these being distinct) ---
    ("lids", "She swept shadow across her lids, just above her",
     " lashes", " ankles"),
    ("cheekbones", "She dusted colour along the tops of her",
     " cheekbones", " fingernails"),
    ("complexion", "Foundation evens out the tone of her",
     " complexion", " conversation"),
    # --- second pass: extra pairs for terms that scored 0 on a single probe.
    # One failed minimal pair can be a bad probe rather than a missing word,
    # so every weak term gets three independent pairs before it is judged.
    ("beige", "Beige is closest in colour to", " sand", " midnight"),
    ("beige", "She wore a beige coat in a soft neutral", " colour", " noise"),
    ("cheekbones", "Blush goes on the cheekbones, which are part of the",
     " face", " foot"),
    ("cheekbones", "She dusted highlighter along her cheekbones and the bridge "
     "of her", " nose", " boot"),
    ("complexion", "Her complexion was pale, meaning her", " skin", " voice"),
    ("complexion", "Foundation is matched to the tone of a person's",
     " skin", " furniture"),
    ("lids", "She swept shadow over her lids and then closed her",
     " eyes", " ears"),
    ("lids", "Eyeshadow sits on the lids, just below the", " brow", " knee"),
    ("lipstick", "She reapplied her lipstick while looking in a",
     " mirror", " hammer"),
    ("lipstick", "Lipstick left a mark on the rim of the", " glass", " tyre"),
    ("blot", "She blotted the excess away with a", " tissue", " hammer"),
    ("blot", "Blotting paper is used to absorb", " oil", " music"),
    ("bronzer", "Bronzer warms the face, mimicking the effect of the",
     " sun", " fridge"),
    ("bronzer", "She swept bronzer across her", " face", " kettle"),
    ("primer", "Primer is applied first so that foundation sits",
     " smoothly", " loudly"),
    ("primer", "Primer helps makeup", " last", " argue"),
    ("matte", "The opposite of a glossy finish is a", " matte", " loud"),
    ("matte", "Matte paint does not", " shine", " exist"),
    ("dewy", "A dewy complexion looks fresh and", " hydrated", " angry"),
    ("dewy", "Dewy skin looks slightly", " wet", " rough"),
    ("satin", "Satin fabric feels smooth and looks faintly",
     " shiny", " angry"),
]

BENCH_TEXT = (
    "She swept a deep crimson across her lips, building it to full opaque "
    "coverage with a chalk-flat surface. She pressed a tissue to her mouth "
    "until every trace of shine was gone. Working slowly, she brought a cocoa "
    "shade onto her eyelids as a sheer wash, finishing in a muted lustre. "
    "She stepped back and checked the mirror."
)


def rss_mb() -> float:
    try:
        import ctypes
        from ctypes import wintypes

        class PMC(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        c = PMC()
        c.cb = ctypes.sizeof(PMC)
        ctypes.windll.psapi.GetProcessMemoryInfo(
            ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(c), c.cb)
        return c.WorkingSetSize / 1e6
    except Exception:
        return float("nan")


class _SkipBench(Exception):
    pass


def benchmark(name: str, vocab_only: bool = False) -> dict:
    rec: dict = {"model": name}
    base_rss = rss_mb()
    try:
        obj = M.load(name)
    except Exception as exc:  # gated repo, OOM, missing arch...
        rec["status"] = "failed"
        rec["error"] = f"{type(exc).__name__}: {str(exc)[:220]}"
        return rec

    try:
        rec.update(
            status="ok",
            params_m=round(obj.n_params / 1e6, 1),
            n_layers=obj.n_layers,
            hidden_size=obj.hidden_size,
            load_seconds=round(obj.load_seconds, 1),
            rss_delta_mb=round(rss_mb() - base_rss, 1),
        )

        # --- throughput ---
        if not vocab_only:
            ids = obj.tokenizer(BENCH_TEXT, return_tensors="pt").input_ids
            n_tok = ids.shape[1]
            with torch.no_grad():
                obj.model(ids.to(M.device()))          # warm-up
                t0 = time.time()
                for _ in range(3):
                    obj.model(ids.to(M.device()))
                dt = (time.time() - t0) / 3
            rec["forward_seconds"] = round(dt, 3)
            rec["tokens_per_sec"] = round(n_tok / dt, 1)

            # --- clean hidden-state extraction? ---
            sents = [s.strip() + "." for s in BENCH_TEXT.split(". ") if s.strip()]
            hs = M.sentence_hidden_states(obj, sents)
            rec["hidden_states_ok"] = bool(
                hs.shape == (len(sents), obj.n_layers + 1, obj.hidden_size)
                and torch.isfinite(hs).all()
            )
            rec["hidden_shape"] = list(hs.shape)
            rec["extract_seconds_per_narrative"] = None

            t0 = time.time()
            for _ in range(3):
                M.sentence_hidden_states(obj, sents)
            rec["extract_seconds_per_narrative"] = round((time.time() - t0) / 3, 3)

        # --- vocabulary sanity check ---
        per_term: dict[str, list[bool]] = {}
        details = []
        for term, prefix, good, bad in VOCAB_PROBES:
            lg = M.sequence_logprob(obj, prefix, good)
            lb = M.sequence_logprob(obj, prefix, bad)
            ok = lg > lb
            per_term.setdefault(term, []).append(ok)
            details.append({"term": term, "prefix": prefix, "good": good,
                            "bad": bad, "logp_good": round(lg, 3),
                            "logp_bad": round(lb, 3), "correct": ok,
                            "margin": round(lg - lb, 3)})
        rec["vocab_accuracy"] = round(
            sum(d["correct"] for d in details) / len(details), 3)
        rec["vocab_by_term"] = {
            t: round(sum(v) / len(v), 3) for t, v in per_term.items()}
        rec["vocab_details"] = details
    finally:
        M.release(obj)
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=list(M.CANDIDATES))
    ap.add_argument("--vocab-only", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "results" / "model_benchmark.json"))
    args = ap.parse_args()

    results = []
    for name in args.models:
        print(f"\n=== {name} ===", flush=True)
        rec = benchmark(name, args.vocab_only)
        print(json.dumps({k: v for k, v in rec.items()
                          if k not in ("vocab_details", "vocab_by_term")},
                         indent=2), flush=True)
        results.append(rec)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")

    md = ROOT / "docs" / (out.stem + ".md")
    write_markdown(results, md)
    print(f"\nwrote {out}")


def write_markdown(results: list[dict], path: Path) -> None:
    lines = ["# Model benchmark", "",
             f"Host: CPU-only, {os.cpu_count()} cores, float32.", "",
             "| model | params (M) | layers | hidden | load (s) | RSS (MB) | "
             "tok/s | extract s/narr | vocab acc | status |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        if r.get("status") != "ok":
            lines.append(f"| `{r['model']}` | - | - | - | - | - | - | - | - | "
                         f"{r.get('error', 'failed')} |")
            continue
        lines.append(
            f"| `{r['model']}` | {r['params_m']} | {r['n_layers']} | "
            f"{r['hidden_size']} | {r['load_seconds']} | {r['rss_delta_mb']} | "
            f"{r.get('tokens_per_sec', '-')} | "
            f"{r.get('extract_seconds_per_narrative', '-')} | "
            f"{r['vocab_accuracy']:.3f} | ok |")

    lines += ["", "## Vocabulary sanity check (per term)", "",
              "Fraction of minimal pairs where the model prefers the "
              "semantically correct completion. Chance = 0.5.", ""]
    ok = [r for r in results if r.get("status") == "ok"]
    if ok:
        terms = sorted(ok[0]["vocab_by_term"])
        lines.append("| term | " + " | ".join(f"`{r['model']}`" for r in ok) + " |")
        lines.append("|---" * (len(ok) + 1) + "|")
        for t in terms:
            lines.append(f"| {t} | " + " | ".join(
                f"{r['vocab_by_term'].get(t, float('nan')):.2f}" for r in ok) + " |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
