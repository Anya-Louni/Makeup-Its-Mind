"""Assemble the write-up from its three sources.

The page is kept in three pieces so markup and chart code can be edited apart:

    results/_paper_head.html      markup and styles
    results/_paper_script.html    the chart code
    results/_paperdata_inline.json  every number the charts draw

The numbers live in their own file so a rerun of the pipeline can refresh them
without touching the chart code. This script inlines them into the `const D`
declaration at the top of the script block and writes the single file that gets
served.

Run it after any edit to the three sources:

    python scripts/25_build_paper.py
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HEAD = ROOT / "results" / "_paper_head.html"
SCRIPT = ROOT / "results" / "_paper_script.html"
DATA = ROOT / "results" / "_paperdata_inline.json"
OUT = ROOT / "docs" / "index.html"

DECL = re.compile(r"const D = \{.*?\};\n", re.S)


def build(out: Path = OUT) -> Path:
    head = HEAD.read_text(encoding="utf-8")
    script = SCRIPT.read_text(encoding="utf-8")
    data = json.dumps(json.loads(DATA.read_text(encoding="utf-8")),
                      separators=(",", ":"))

    script, n = DECL.subn("const D = " + data + ";\n", script, count=1)
    if n != 1:
        raise SystemExit(
            f"expected one `const D = {{...}};` declaration in {SCRIPT.name}, found {n}")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(head + script + "</html>\n", encoding="utf-8")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--check", action="store_true",
                    help="fail if the built page differs from the one on disk")
    args = ap.parse_args()

    if args.check:
        current = args.out.read_text(encoding="utf-8") if args.out.exists() else ""
        build(args.out)
        if args.out.read_text(encoding="utf-8") != current:
            raise SystemExit(f"{args.out} was stale and has been rebuilt")
        print(f"{args.out} is up to date")
        return

    p = build(args.out)
    print(f"wrote {p} ({p.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
