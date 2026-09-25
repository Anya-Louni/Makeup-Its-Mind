"""Extract residual-stream activations for a generated dataset.

    python scripts/03_extract.py --dataset pilot --model gpt2
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mwm.extract import extract_dataset  # noqa: E402
from mwm.generate import read_jsonl  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="pilot")
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--conditions", nargs="*", default=["natural", "shuffled"])
    ap.add_argument("--layers", nargs="*", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None,
                    help="cap narratives per condition (debugging)")
    args = ap.parse_args()

    data_dir = ROOT / "data" / args.dataset
    tag = args.model.replace("/", "__")

    for cond in args.conditions:
        records = read_jsonl(data_dir / f"{cond}.jsonl")
        if args.limit:
            records = records[:args.limit]
        out = ROOT / "activations" / f"{args.dataset}__{tag}__{cond}"
        print(f"\n=== {cond}: {len(records)} narratives -> {out}", flush=True)
        t0 = time.time()
        info = extract_dataset(records, args.model, out, layers=args.layers)
        print(f"done in {time.time() - t0:.1f}s: {info}", flush=True)


if __name__ == "__main__":
    main()
