import os
import subprocess
import sys


USAGE = (
    "Usage: python telegram_command_bridge.py <command> [reason words...]\n"
    "Commands: /stop,/halt,/run,/resume,/start"
)


def main(argv: list[str]) -> int:
    if len(argv) == 1:
        print(USAGE)
        return 0

    command = argv[1].strip().lower()
    if command in ("-h", "--help"):
        print(USAGE)
        return 0

    reason = " ".join(argv[2:])
    base_dir = os.path.dirname(os.path.abspath(__file__))

    if command in ("/stop", "/halt"):
        target_script = os.path.join(base_dir, "control_stop.py")
        success_text = "STOP_REQUESTED"
    elif command in ("/run", "/resume", "/start"):
        target_script = os.path.join(base_dir, "control_run.py")
        success_text = "RUN_REQUESTED"
    else:
        print("UNSUPPORTED_COMMAND")
        return 2

    reason_arg = reason if reason else command
    result = subprocess.run(
        [sys.executable, target_script, "--source", "telegram", "--reason", reason_arg],
        capture_output=True,
        text=True,
    )

    if result.returncode == 0:
        child_stdout = result.stdout.strip()
        if child_stdout:
            print(child_stdout)
        else:
            print(success_text)

    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
