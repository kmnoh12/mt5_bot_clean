from __future__ import annotations

import argparse
from pathlib import Path

from core.control import DESIRED_STATE_RUN, RuntimeControlChannel, write_desired_state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Request RUN state for MT5 bot and re-enable watchdog relaunch.")
    parser.add_argument("--reason", default="external_run", help="Reason for run request")
    parser.add_argument("--source", default="external", help="Source label (telegram/ide/cli/etc)")
    parser.add_argument(
        "--control-path",
        default=str(Path(__file__).with_name("runtime_control.json")),
        help="Path to runtime_control.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    channel = RuntimeControlChannel(path=args.control_path)
    channel.request_resume()
    write_desired_state(
        DESIRED_STATE_RUN,
        source=str(args.source),
        reason=str(args.reason),
        metadata={"command": "control_run.py"},
    )
    print("RUN_REQUESTED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
