from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reports.active_opportunities import write_active_opportunity_reports


def main() -> int:
    parser = argparse.ArgumentParser(description="Build offline active opportunity JSON and Markdown reports.")
    parser.add_argument("--input", required=True, help="Sample/input JSON file. No live broker calls are made.")
    parser.add_argument("--json-out", default="reports/active_opportunities.json")
    parser.add_argument("--md-out", default="reports/active_opportunities.md")
    parser.add_argument("--top-n", type=int, default=5)
    args = parser.parse_args()

    source_path = Path(args.input)
    source = json.loads(source_path.read_text(encoding="utf-8"))
    report = write_active_opportunity_reports(
        source,
        json_path=args.json_out,
        markdown_path=args.md_out,
        top_n=args.top_n,
    )
    print(
        "active_opportunities built "
        f"eligible={report['eligible_count']} rejected={report['rejected_count']} "
        f"json={Path(args.json_out)} md={Path(args.md_out)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

