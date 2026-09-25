"""Generate a narrative dataset with exact per-position state labels.

Examples
--------
Pilot (2 entities, 2 attributes, small):
    python scripts/02_generate_dataset.py --name pilot \
        --entities lips eyes --attributes color finish --n 1000

Full (4 entities, 3 attributes):
    python scripts/02_generate_dataset.py --name full --n 4000
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mwm.generate import GenConfig, generate_dataset, write_jsonl  # noqa: E402
from mwm.state import ATTRIBUTES, ENTITIES  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="full")
    ap.add_argument("--entities", nargs="*", default=list(ENTITIES))
    ap.add_argument("--attributes", nargs="*", default=list(ATTRIBUTES))
    ap.add_argument("--n", type=int, default=4000, help="total narratives")
    ap.add_argument("--conditions", nargs="*", default=["natural", "shuffled"])
    ap.add_argument("--min-sentences", type=int, default=None,
                    help="override the entity-count-derived length")
    ap.add_argument("--max-sentences", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    n_per_split = {"train": int(args.n * 0.70),
                   "val": int(args.n * 0.15),
                   "test": args.n - int(args.n * 0.70) - int(args.n * 0.15)}

    out_dir = ROOT / "data" / args.name
    all_records: list[dict] = []
    for cond in args.conditions:
        cfg = GenConfig(entities=tuple(args.entities),
                        attributes=tuple(args.attributes),
                        condition=cond).scale_to_entities()
        # Length control: hold narrative length fixed while varying the number
        # of entities, so "more entities" and "longer text" can be separated.
        if args.min_sentences is not None:
            cfg.min_sentences = args.min_sentences
        if args.max_sentences is not None:
            cfg.max_sentences = args.max_sentences
        recs = generate_dataset(n_per_split, cfg, seed=args.seed)
        write_jsonl(recs, out_dir / f"{cond}.jsonl")
        all_records += recs
        print(f"{cond}: {len(recs)} narratives, "
              f"{sum(len(r['sentences']) for r in recs)} positions")

    stats = summarise(all_records)
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2),
                                        encoding="utf-8")
    print(json.dumps(stats, indent=2))
    print(f"\nwrote {out_dir}")


def summarise(records: list[dict]) -> dict:
    label_counts: dict[str, Counter] = {}
    kinds: Counter = Counter()
    os_counts: Counter = Counter()
    n_pos = 0
    for r in records:
        kinds.update(r["action_kinds"])
        for lab, flags in zip(r["labels"], r["order_sensitive"]):
            n_pos += 1
            for k, v in lab.items():
                label_counts.setdefault(k, Counter())[v] += 1
            for k, v in flags.items():
                if v:
                    os_counts[k] += 1
    return {
        "n_narratives": len(records),
        "n_positions": n_pos,
        "sentences_per_narrative": round(n_pos / max(len(records), 1), 2),
        "action_kind_distribution": dict(kinds.most_common()),
        "label_distribution": {k: dict(v.most_common())
                               for k, v in label_counts.items()},
        "order_sensitive_positions": dict(os_counts),
        "order_sensitive_rate": {k: round(v / n_pos, 3)
                                 for k, v in os_counts.items()},
    }


if __name__ == "__main__":
    main()
