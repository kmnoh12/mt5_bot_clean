#!/usr/bin/env python3
from __future__ import annotations

import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple


BASE_DIR = Path(__file__).resolve().parent
RUNTIME_LOCK = BASE_DIR / "mt5_bot" / "runtime.lock"
WATCHDOG_LOG = BASE_DIR / "mt5_bot" / "watchdog_debug.log"
REINFORCEMENT_LOG = BASE_DIR / "memory" / "reinforcement_log.md"

POLL_SEC = 1.0
WAIT_SEC = 60


def now_ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def append_reinforcement(lines: List[str]) -> None:
    REINFORCEMENT_LOG.parent.mkdir(parents=True, exist_ok=True)
    with REINFORCEMENT_LOG.open("a", encoding="utf-8", errors="replace", newline="\n") as f:
        for line in lines:
            f.write(line.rstrip("\n") + "\n")


def read_pid(path: Path) -> Optional[int]:
    try:
        raw = path.read_text(encoding="utf-8", errors="replace").strip()
    except FileNotFoundError:
        return None
    except Exception:
        return None

    if not raw:
        return None

    try:
        pid = int(raw.splitlines()[0].strip())
    except Exception:
        return None

    return pid if pid > 0 else None


def run(cmd: List[str], timeout: int = 30) -> Tuple[int, str, str]:
    try:
        cp = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
        )
        return cp.returncode, (cp.stdout or "").strip(), (cp.stderr or "").strip()
    except subprocess.TimeoutExpired as e:
        out = (e.stdout or "") if isinstance(e.stdout, str) else ""
        err = (e.stderr or "") if isinstance(e.stderr, str) else ""
        return 124, out.strip(), (f"TimeoutExpired: {e}; {err}").strip()
    except Exception as e:
        return 1, "", f"{type(e).__name__}: {e}"


def pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False

    rc, out, _ = run(["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"], timeout=15)
    if rc != 0:
        return False
    if not out:
        return False
    if "No tasks are running" in out:
        return False

    # The CSV row contains the PID as a field; a substring check is sufficient here.
    return str(pid) in out


def kill_pid(pid: int) -> Tuple[int, str, str, str]:
    # Prefer taskkill to match watchdog log evidence.
    rc, out, err = run(["taskkill", "/PID", str(pid), "/F", "/T"], timeout=30)
    if rc == 0:
        return rc, out, err, "taskkill"

    # Fallback: os.kill
    try:
        os.kill(pid, 9)
        return 0, out, err, "os.kill"
    except Exception as e:
        merged = (err + f"; os.kill failed: {type(e).__name__}: {e}").strip("; ")
        return rc, out, merged, "taskkill+os.kill"


def wait_new_pid(old_pid: int, timeout_sec: int) -> Optional[int]:
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        pid = read_pid(RUNTIME_LOCK)
        # Return the first NEW PID observed in runtime.lock, regardless of whether it's alive yet.
        if pid and pid != old_pid:
            return pid
        time.sleep(POLL_SEC)
    return None


def read_appended(path: Path, offset: int) -> List[str]:
    try:
        data = path.read_bytes()
    except Exception:
        return []

    if offset < 0 or offset > len(data):
        offset = 0

    chunk = data[offset:]
    if not chunk:
        return []

    return chunk.decode("utf-8", errors="replace").splitlines()


def tail_lines(path: Path, n: int = 400) -> List[str]:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return []

    return lines[-n:] if len(lines) > n else lines


def evidence(lines: List[str]) -> Tuple[bool, bool]:
    has_taskkill = any("taskkill /F /T" in ln for ln in lines)
    has_worker = any("Worker launched" in ln for ln in lines)
    return has_taskkill, has_worker


def md_entry(result: str, details: List[str]) -> List[str]:
    lines: List[str] = []
    lines.append(f"\n### [{now_ts()}] Watchdog Restart Verification")
    lines.append(f"- Result: {result}")
    for d in details:
        lines.append(f"- {d}")
    return lines


def safe_append_md(result: str, details: List[str]) -> None:
    try:
        append_reinforcement(md_entry(result, details))
    except Exception:
        # Never block verification due to logging failure.
        pass


def main() -> int:
    mt5 = None
    mt5_initialized = False

    result = "Fail"
    details: List[str] = []

    old_pid: Optional[int] = None
    new_pid: Optional[int] = None

    kill_rc: Optional[int] = None
    kill_method: Optional[str] = None

    log_taskkill = False
    log_worker = False

    try:
        try:
            import MetaTrader5 as mt5  # type: ignore
        except Exception as e:
            details.append(f"mt5_import_failed: {type(e).__name__}: {e}")
            return 1

        if not mt5.initialize():
            details.append(f"mt5_initialize_failed: {mt5.last_error()}")
            return 1
        mt5_initialized = True

        pos_total = mt5.positions_total()
        if pos_total is None:
            details.append(f"positions_total_none: {mt5.last_error()}")
            return 1

        if int(pos_total) > 0:
            result = "Skipped"
            details.append("Skipped: Open positions")
            details.append(f"positions_total: {pos_total}")
            safe_append_md(result, details)
            print("Skipped: Open positions")
            return 0

        old_pid = read_pid(RUNTIME_LOCK)
        if not old_pid:
            details.append(f"runtime_lock_missing_or_invalid: {RUNTIME_LOCK}")
            return 1

        if not pid_alive(old_pid):
            details.append(f"runner_pid_not_alive: {old_pid}")
            return 1

        details.append(f"old_pid: {old_pid}")

        log_offset = 0
        try:
            log_offset = WATCHDOG_LOG.stat().st_size
        except Exception:
            log_offset = 0

        kill_rc, kill_out, kill_err, kill_method = kill_pid(old_pid)
        details.append(f"kill_method: {kill_method}")
        details.append(f"taskkill_rc: {kill_rc}")
        if kill_out:
            details.append(f"taskkill_stdout: {kill_out[:200]}")
        if kill_err:
            details.append(f"taskkill_stderr: {kill_err[:200]}")

        if kill_rc != 0:
            return 1

        new_pid = wait_new_pid(old_pid, WAIT_SEC)
        relaunch = bool(new_pid)
        details.append(f"relaunch_detected: {str(relaunch).lower()}")
        details.append(f"new_pid: {new_pid if new_pid else '<none>'}")
        if new_pid is not None:
            new_pid_alive = pid_alive(new_pid)
            details.append(f"new_pid_alive: {str(new_pid_alive).lower()}")

        appended = read_appended(WATCHDOG_LOG, log_offset)
        a_taskkill, a_worker = evidence(appended)
        log_taskkill = a_taskkill
        log_worker = a_worker
        source = "appended"

        if not (log_taskkill and log_worker):
            tail = tail_lines(WATCHDOG_LOG, n=600)
            t_taskkill, t_worker = evidence(tail)
            log_taskkill = log_taskkill or t_taskkill
            log_worker = log_worker or t_worker
            if t_taskkill or t_worker:
                source = "tail"

        details.append(f"log_taskkill_found: {str(log_taskkill).lower()}")
        details.append(f"log_worker_launched_found: {str(log_worker).lower()}")
        details.append(f"log_check_source: {source}")

        if relaunch and log_taskkill and log_worker:
            result = "Success"
            return 0

        result = "Fail"
        return 1
    finally:
        if result in ("Success", "Fail"):
            safe_append_md(result, details)
            print(
                f"{result}: old_pid={old_pid} new_pid={new_pid} rc={kill_rc} "
                f"log_taskkill={log_taskkill} log_worker={log_worker}"
            )

        if mt5 is not None and mt5_initialized:
            try:
                mt5.shutdown()
            except Exception:
                pass


if __name__ == "__main__":
    sys.exit(main())
