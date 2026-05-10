from __future__ import annotations

import ctypes
import ctypes.wintypes as wintypes
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
WATCHDOG_LOG = BASE_DIR / "mt5_bot" / "watchdog_debug.log"
RUNTIME_LOCK = BASE_DIR / "mt5_bot" / "runtime.lock"
REINFORCEMENT_LOG = BASE_DIR / "memory" / "reinforcement_log.md"


def utc_ts_seconds() -> str:
    # Seconds precision, UTC, ISO8601 with trailing Z.
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _exit(code: int) -> "None":
    raise SystemExit(code)


def append_lines_to_reinforcement_log(lines: list[str]) -> None:
    try:
        # Don't create directories; open must succeed as-is.
        with open(REINFORCEMENT_LOG, "a", encoding="utf-8", newline="\n") as f:
            for line in lines:
                if line.endswith("\n"):
                    f.write(line)
                else:
                    f.write(line + "\n")
    except Exception as e:
        print(f"FAIL log_open_error: {e}", file=sys.stdout)
        _exit(2)


def log_status_line(status: str, old_pid: str, new_pid: str, reason: str) -> None:
    line = f"{utc_ts_seconds()} | {status} | old_pid={old_pid} | new_pid={new_pid} | reason={reason}"
    append_lines_to_reinforcement_log([line])


def win_pid_exists(pid: int) -> bool:
    if pid <= 0:
        return False

    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    STILL_ACTIVE = 259
    ERROR_ACCESS_DENIED = 5

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

    OpenProcess = kernel32.OpenProcess
    OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    OpenProcess.restype = wintypes.HANDLE

    GetExitCodeProcess = kernel32.GetExitCodeProcess
    GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    GetExitCodeProcess.restype = wintypes.BOOL

    CloseHandle = kernel32.CloseHandle
    CloseHandle.argtypes = [wintypes.HANDLE]
    CloseHandle.restype = wintypes.BOOL

    handle = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not handle:
        err = ctypes.get_last_error()
        # If we don't have rights to query, treat as "exists" (taskkill will be the authority).
        return err == ERROR_ACCESS_DENIED

    try:
        code = wintypes.DWORD()
        if not GetExitCodeProcess(handle, ctypes.byref(code)):
            # If we can't query exit code, assume process exists (handle opened).
            return True
        return int(code.value) == STILL_ACTIVE
    finally:
        CloseHandle(handle)


def read_first_pid_from_lock() -> int:
    try:
        with open(RUNTIME_LOCK, "r", encoding="utf-8", errors="ignore") as f:
            for raw in f:
                s = raw.strip()
                if not s:
                    continue
                pid = int(s)
                return pid
    except FileNotFoundError:
        raise
    except Exception as e:
        raise RuntimeError(f"runtime_lock_read_error: {e}") from e
    raise RuntimeError("runtime_lock_empty")


def fail(reason: str, old_pid: str = "-", new_pid: str = "-", exit_code: int = 1) -> None:
    # Always attempt to log (unless logging itself fails).
    log_status_line("FAIL", old_pid, new_pid, reason)
    print(f"FAIL {reason}", file=sys.stdout)
    _exit(exit_code)


def main() -> None:
    t0 = time.time()
    _ = t0  # recorded per requirement

    try:
        watchdog_offset = WATCHDOG_LOG.stat().st_size
    except FileNotFoundError:
        fail("watchdog_log_missing", "-", "-", 1)
        return
    except Exception:
        fail("watchdog_log_stat_error", "-", "-", 1)
        return

    # Step 2: MT5 read-only connect + positions check.
    try:
        import MetaTrader5 as mt5  # required external import

        initialized = False
        try:
            if not mt5.initialize():
                raise RuntimeError("mt5_initialize_failed")
            initialized = True
            positions = mt5.positions_get()
        finally:
            if initialized:
                mt5.shutdown()
    except Exception:
        log_status_line("FAIL", "-", "-", "mt5_error")
        print("FAIL mt5_error", file=sys.stdout)
        _exit(1)
        return

    try:
        has_len = positions is not None and hasattr(positions, "__len__")
        if has_len and len(positions) > 0:
            append_lines_to_reinforcement_log(
                [
                    "Skipped: Open positions",
                    f"{utc_ts_seconds()} | SKIP | old_pid=- | new_pid=- | reason=open_positions",
                ]
            )
            print("SKIP Open positions", file=sys.stdout)
            _exit(0)
            return
    except Exception:
        fail("mt5_error", "-", "-", 1)
        return

    # Step 3: Read runner PID and validate.
    try:
        old_pid_int = read_first_pid_from_lock()
    except FileNotFoundError:
        fail("runtime_lock_missing", "-", "-", 1)
        return
    except Exception:
        fail("runtime_lock_bad_pid", "-", "-", 1)
        return

    if old_pid_int <= 0:
        fail("runtime_lock_bad_pid", "-", "-", 1)
        return
    if not win_pid_exists(old_pid_int):
        fail("old_pid_not_running", str(old_pid_int), "-", 1)
        return

    # Step 4: taskkill crash simulation.
    try:
        proc = subprocess.run(
            ["taskkill", "/PID", str(old_pid_int), "/F", "/T"],
            capture_output=True,
            text=True,
        )
    except Exception:
        fail("taskkill_failed", str(old_pid_int), "-", 1)
        return

    if proc.returncode != 0:
        fail("taskkill_failed", str(old_pid_int), "-", 1)
        return

    # Step 5: Poll for new PID in runtime.lock.
    new_pid_int: int | None = None
    deadline = time.time() + 60.0
    while time.time() < deadline:
        try:
            candidate = read_first_pid_from_lock()
        except Exception:
            time.sleep(0.5)
            continue

        if candidate and candidate != old_pid_int and candidate > 0 and win_pid_exists(candidate):
            new_pid_int = candidate
            break
        time.sleep(0.5)

    if new_pid_int is None:
        fail("restart_timeout", str(old_pid_int), "-", 1)
        return

    # Step 6: Verify watchdog log appended content contains both markers.
    try:
        with open(WATCHDOG_LOG, "rb") as f:
            f.seek(watchdog_offset)
            appended = f.read()
        appended_text = appended.decode("utf-8", errors="ignore")
    except Exception:
        fail("watchdog_read_error", str(old_pid_int), str(new_pid_int), 1)
        return

    if ("taskkill /F /T executed" not in appended_text) or ("Worker launched" not in appended_text):
        fail("watchdog_missing_markers", str(old_pid_int), str(new_pid_int), 1)
        return

    # Step 7/8: PASS log + stdout.
    log_status_line("PASS", str(old_pid_int), str(new_pid_int), "ok")
    print(f"PASS old_pid={old_pid_int} new_pid={new_pid_int}", file=sys.stdout)
    _exit(0)


if __name__ == "__main__":
    main()

