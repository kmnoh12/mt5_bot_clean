from __future__ import annotations

import argparse
from datetime import datetime, timezone


SECTIONS = ("Summary", "Changed files", "Verification", "Safety", "Remaining risks", "Next step")


def bullets(values: list[str], placeholder: str) -> str:
    if not values:
        return f"- {placeholder}"
    return "\n".join(f"- {value}" for value in values)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Print a concise Markdown final handoff report skeleton for Codex/Hermes reviews."
        )
    )
    parser.add_argument("--summary", action="append", default=[], help="Summary bullet. Repeatable.")
    parser.add_argument("--changed", action="append", default=[], help="Changed file bullet. Repeatable.")
    parser.add_argument("--verification", action="append", default=[], help="Verification bullet. Repeatable.")
    parser.add_argument("--safety", action="append", default=[], help="Safety note bullet. Repeatable.")
    parser.add_argument("--risk", action="append", default=[], help="Remaining risk bullet. Repeatable.")
    parser.add_argument("--next-step", action="append", default=[], help="Next step bullet. Repeatable.")
    parser.add_argument("--stamp", action="store_true", help="Include UTC generation timestamp.")
    args = parser.parse_args(argv)

    if args.stamp:
        print(f"Generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}\n")
    print("## Summary")
    print(bullets(args.summary, "What changed and why."))
    print("\n## Changed files")
    print(bullets(args.changed, "path/to/file.py - short purpose."))
    print("\n## Verification")
    print(bullets(args.verification, "Command run and result."))
    print("\n## Safety")
    print(bullets(args.safety, "Readonly/no live trading/config/credential impact."))
    print("\n## Remaining risks")
    print(bullets(args.risk, "Known limitation or residual risk."))
    print("\n## Next step")
    print(bullets(args.next_step, "Most useful follow-up."))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
