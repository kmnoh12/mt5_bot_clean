from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess

from core.control import DESIRED_STATE_STOP, RuntimeControlChannel, write_desired_state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Request intentional STOP for MT5 bot without force-kill.")
    parser.add_argument("--reason", default="external_stop", help="Reason for intentional stop request")
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
    channel.request_manual_halt(source=str(args.source), reason=str(args.reason))
    write_desired_state(
        DESIRED_STATE_STOP,
        source=str(args.source),
        reason=str(args.reason),
        metadata={"command": "control_stop.py"},
    )
    if os.name == "nt":
        for proc_name in ("terminal64.exe", "terminal.exe"):
            try:
                subprocess.run(
                    ["taskkill", "/F", "/IM", proc_name],
                    check=False,
                    capture_output=True,
                    text=True,
                )
            except Exception:
                pass
    print("STOP_REQUESTED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
