from __future__ import annotations

import atexit
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from core.control import (
    DESIRED_STATE_RUN,
    DESIRED_STATE_STOP,
    ensure_desired_state_file,
    load_desired_state,
)

SCRIPT_PATH = Path(__file__).resolve()
BOT_DIR = SCRIPT_PATH.parent
RUNTIME_DIR = BOT_DIR / "runtime"
HEARTBEAT_FILE = RUNTIME_DIR / "heartbeat.json"
LOCKDOWN_FILE = RUNTIME_DIR / "lockdown.flag"
ALERT_MARKER_FILE = RUNTIME_DIR / "alert.marker"
WATCHDOG_LOCK_FILE = RUNTIME_DIR / "watchdog.lock"
DESIRED_STATE_FILE = RUNTIME_DIR / "desired_state.json"
CONFIG_FILE = BOT_DIR / "config.yaml"
LOCK_FILE = BOT_DIR / "runtime.lock"
LOG_FILE = BOT_DIR / "watchdog_debug.log"
ENGINE_CMD = [sys.executable, "-u", str(BOT_DIR / "runner.py"), "--config", str(CONFIG_FILE)]

HEARTBEAT_TIMEOUT_SEC = 300
MAX_CRASHES_IN_WINDOW = 5
CRASH_WINDOW_SEC = 60 * 30
BACKOFF_BASE = 2
BACKOFF_MAX = 60
LOOP_SLEEP_SEC = 2
STARTUP_GRACE_SEC = 10
STARTUP_PROBE_SEC = 3
RECOVERY_SLEEP_SEC = 5
LOCK_CONFLICT_RETRY_SEC = 10
RESTART_CLEANUP_WAIT_SEC = 5
LOCK_CLEANUP_MAX_ATTEMPTS = 5
LOCK_CLEANUP_WAIT_SEC = 0.5
FAST_FAIL_WINDOW_SEC = 10
FAST_FAIL_BACKOFF_SEC = 30

LOGGER = logging.getLogger("watchdog")


def utc_ts() -> float:
    return datetime.now(timezone.utc).timestamp()


def read_desired_state_value() -> str:
    payload = load_desired_state(path=str(DESIRED_STATE_FILE))
    state = str(payload.get("state", DESIRED_STATE_RUN)).strip().upper()
    return state if state in {DESIRED_STATE_RUN, DESIRED_STATE_STOP} else DESIRED_STATE_RUN


def setup_logging() -> None:
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)

    LOGGER.setLevel(logging.INFO)
    LOGGER.handlers.clear()
    LOGGER.addHandler(file_handler)
    LOGGER.addHandler(stream_handler)
    LOGGER.propagate = False


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        try:
            import psutil  # type: ignore

            return bool(psutil.pid_exists(pid))
        except Exception:
            pass

        try:
            import ctypes

            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            handle = ctypes.windll.kernel32.OpenProcess(
                PROCESS_QUERY_LIMITED_INFORMATION, False, pid
            )
            if handle:
                ctypes.windll.kernel32.CloseHandle(handle)
                return True
            return ctypes.windll.kernel32.GetLastError() == 5
        except Exception:
            pass

    try:
        os.kill(pid, 0)
    except PermissionError:
        return True
    except (SystemError, ProcessLookupError, OSError):
        return False
    return True


def _read_lock_pid(lock_path: Path) -> Optional[int]:
    if not lock_path.exists():
        return None
    try:
        raw = lock_path.read_text(encoding="utf-8").strip()
    except Exception:
        return None
    if not raw:
        return None
    # Backward-compatible plain PID lock content.
    try:
        pid = int(raw)
        return pid if pid > 0 else None
    except Exception:
        pass
    # New JSON lock content from runner.
    try:
        payload = json.loads(raw)
        pid = int(payload.get("pid") or 0)
    except Exception:
        return None
    return pid if pid > 0 else None


def _cleanup_lock_file(lock_path: Path, owner_pid: Optional[int]) -> bool:
    for attempt in range(1, LOCK_CLEANUP_MAX_ATTEMPTS + 1):
        try:
            lock_path.unlink()
            LOGGER.info("Successfully removed lock file: %s", lock_path)
            return True
        except FileNotFoundError:
            LOGGER.info("Lock file already removed: %s", lock_path)
            return True
        except PermissionError as exc:
            if attempt >= LOCK_CLEANUP_MAX_ATTEMPTS:
                LOGGER.error(
                    "Failed removing lock file after %s attempts due to permission error: %s",
                    attempt,
                    exc,
                )
                return False
            if owner_pid and _pid_alive(owner_pid):
                LOGGER.warning(
                    "Permission denied removing %s (owner_pid=%s is alive) on attempt %s/%s. Force killing owner.",
                    lock_path,
                    owner_pid,
                    attempt,
                    LOCK_CLEANUP_MAX_ATTEMPTS,
                )
                if os.name == "nt":
                    _taskkill_process_tree(owner_pid, reason="cleanup_lock_file")
                else:
                    try:
                        os.kill(owner_pid, 9)
                    except Exception:
                        pass
            else:
                LOGGER.warning(
                    "Permission denied removing %s on attempt %s/%s: %s",
                    lock_path,
                    attempt,
                    LOCK_CLEANUP_MAX_ATTEMPTS,
                    exc,
                )
            # Windows stale handle fallback: kill only mt5 runner processes, never all python.exe.
            if os.name == "nt":
                _kill_rogue_runner_processes(exclude_pids={os.getpid(), owner_pid or 0})
            time.sleep(LOCK_CLEANUP_WAIT_SEC)
        except Exception as exc:
            if attempt >= LOCK_CLEANUP_MAX_ATTEMPTS:
                LOGGER.error(
                    "Failed removing lock file after %s attempts: %s",
                    attempt,
                    exc,
                )
                return False
            LOGGER.warning(
                "Failed removing %s on attempt %s/%s: %s",
                lock_path,
                attempt,
                LOCK_CLEANUP_MAX_ATTEMPTS,
                exc,
            )
            time.sleep(LOCK_CLEANUP_WAIT_SEC)

    return False


def _acquire_single_instance_lock(lock_path: Path, lock_name: str) -> Optional[int]:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    if lock_path.exists():
        owner_pid = _read_lock_pid(lock_path)
        owner_alive = bool(owner_pid and _pid_alive(owner_pid))
        if owner_alive:
            LOGGER.info(
                "%s lock already held by live pid=%s (%s). Exiting.",
                lock_name,
                owner_pid,
                lock_path,
            )
            return None
        if not _cleanup_lock_file(lock_path, owner_pid):
            return None
        LOGGER.info(
            "Removed stale %s lock: %s (owner_pid=%s)",
            lock_name,
            lock_path,
            owner_pid,
        )

    try:
        fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_RDWR)
    except FileExistsError:
        owner_pid = _read_lock_pid(lock_path)
        owner_alive = bool(owner_pid and _pid_alive(owner_pid))
        LOGGER.info(
            "%s lock conflict: %s owner_pid=%s owner_alive=%s",
            lock_name,
            lock_path,
            owner_pid,
            owner_alive,
        )
        return None
    except Exception:
        LOGGER.exception("Failed to acquire %s lock: %s", lock_name, lock_path)
        return None

    try:
        os.write(fd, str(os.getpid()).encode("utf-8"))
    except Exception:
        try:
            os.close(fd)
        except Exception:
            pass
        try:
            lock_path.unlink()
        except Exception:
            pass
        LOGGER.exception("Failed initializing %s lock file: %s", lock_name, lock_path)
        return None

    LOGGER.info("Acquired %s lock: %s pid=%s", lock_name, lock_path, os.getpid())
    return fd


def _release_single_instance_lock(lock_path: Path, fd: Optional[int]) -> None:
    try:
        if fd is not None:
            os.close(fd)
    except Exception:
        pass
    try:
        owner_pid = _read_lock_pid(lock_path)
        if owner_pid is None or owner_pid == os.getpid():
            lock_path.unlink()
    except Exception:
        pass


def cleanup_runtime_lock_on_start() -> None:
    if not LOCK_FILE.exists():
        return

    # Watchdog is the authority. If a lock exists before we launch a worker,
    # it is by definition stale or belonging to a rogue process.
    # We must clear it to prevent the worker from immediately exiting with code 4.

    owner_pid = _read_lock_pid(LOCK_FILE)
    if owner_pid:
        LOGGER.info(
            "Cleaning up runtime.lock (pid=%s) before worker launch: %s",
            owner_pid,
            LOCK_FILE,
        )
        # If it's alive, kill it.
        if _pid_alive(owner_pid):
            LOGGER.warning("Force killing rogue worker pid=%s found in lock file.", owner_pid)
            try:
                if os.name == "nt":
                    subprocess.run(["taskkill", "/PID", str(owner_pid), "/F", "/T"], check=False, capture_output=True)
                else:
                    os.kill(owner_pid, 9)
            except Exception:
                pass
            time.sleep(1.0)

    if not _cleanup_lock_file(LOCK_FILE, owner_pid):
        LOGGER.warning(
            "runtime.lock cleanup still failing after targeted kill. Running runner-only cleanup fallback."
        )
        if os.name == "nt":
            _kill_rogue_runner_processes(exclude_pids={os.getpid(), owner_pid or 0})
            time.sleep(LOCK_CLEANUP_WAIT_SEC)
        _cleanup_lock_file(LOCK_FILE, owner_pid)


def _kill_rogue_runner_processes(exclude_pids: set[int]) -> None:
    if os.name != "nt":
        return
    try:
        import psutil  # type: ignore
    except Exception:
        return

    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            pid = int(proc.info.get("pid") or 0)
            if pid <= 0 or pid in exclude_pids:
                continue
            cmdline = " ".join(proc.info.get("cmdline") or [])
            if "runner.py" not in cmdline:
                continue
            if str(BOT_DIR) not in cmdline:
                continue
            _taskkill_process_tree(pid, reason="cleanup_rogue_runner")
        except Exception:
            continue


def read_heartbeat_timestamp() -> Optional[float]:
    if not HEARTBEAT_FILE.exists():
        return None
    for attempt in range(3):
        try:
            raw = HEARTBEAT_FILE.read_text(encoding="utf-8")
        except OSError as exc:
            err_no = getattr(exc, "errno", None)
            win_error = getattr(exc, "winerror", None)
            benign_file_lock = (
                (isinstance(exc, PermissionError) and err_no == 13)
                or win_error == 32
            )
            log_fn = LOGGER.debug if benign_file_lock else LOGGER.warning
            message = (
                "Failed to read heartbeat file (benign lock): %s (attempt=%s): %s"
                if benign_file_lock
                else "Failed to read heartbeat file (OS Error): %s (attempt=%s): %s"
            )
            log_fn(
                message,
                HEARTBEAT_FILE,
                attempt + 1,
                exc,
            )
            if attempt < 2:
                time.sleep(0.2)
                continue
            return None
        except Exception:
            LOGGER.exception(
                "Failed to read heartbeat file: %s (attempt=%s)",
                HEARTBEAT_FILE,
                attempt + 1,
            )
            if attempt < 2:
                time.sleep(0.2)
                continue
            return None

        if not raw.strip():
            LOGGER.warning(
                "Heartbeat file is empty: %s (attempt=%s)",
                HEARTBEAT_FILE,
                attempt + 1,
            )
            if attempt < 2:
                time.sleep(0.2)
                continue
            return None

        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            LOGGER.warning(
                "Heartbeat JSON decode error in %s (attempt=%s): %s",
                HEARTBEAT_FILE,
                attempt + 1,
                exc,
            )
            if attempt < 2:
                time.sleep(0.2)
                continue
            return None
        except Exception:
            LOGGER.exception(
                "Failed to parse heartbeat JSON: %s (attempt=%s)",
                HEARTBEAT_FILE,
                attempt + 1,
            )
            if attempt < 2:
                time.sleep(0.2)
                continue
            return None

        if not isinstance(payload, dict):
            LOGGER.warning(
                "Heartbeat payload is not a JSON object: %s (type=%s)",
                HEARTBEAT_FILE,
                type(payload).__name__,
            )
            return None

        ts_raw = payload.get("ts")
        try:
            return float(ts_raw)
        except Exception:
            LOGGER.warning(
                "Heartbeat file has invalid ts field in %s: %r",
                HEARTBEAT_FILE,
                ts_raw,
            )
            return None

    return None


def write_lockdown_marker(reason: str) -> None:
    try:
        LOCKDOWN_FILE.parent.mkdir(parents=True, exist_ok=True)
        LOCKDOWN_FILE.write_text(
            f"LOCKDOWN at {datetime.now().isoformat()} | reason={reason}\n",
            encoding="utf-8",
        )
    except Exception:
        LOGGER.exception("Failed to write lockdown marker: %s", LOCKDOWN_FILE)


def send_telegram_alert(message: str) -> None:
    text = str(message or "").strip()
    if not text:
        return

    payload = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "source": "watchdog",
        "message": text,
    }
    alert_channel = (
        os.getenv("OPENCLAW_ALERT_CHANNEL")
        or os.getenv("OPENCLAW_NOTIFICATION_CHANNEL")
        or os.getenv("OPENCLAW_NOTIFY_CHANNEL")
    )
    if alert_channel:
        payload["channel"] = alert_channel

    try:
        ALERT_MARKER_FILE.parent.mkdir(parents=True, exist_ok=True)
        with ALERT_MARKER_FILE.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
    except Exception:
        LOGGER.exception("Failed to write alert marker: %s", ALERT_MARKER_FILE)


def terminate_worker(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return

    pid = proc.pid
    LOGGER.warning("Terminating worker process tree. pid=%s", pid)

    if os.name == "nt":
        _taskkill_process_tree(pid, reason="terminate_worker")
    else:
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except Exception:
            try:
                proc.kill()
            except Exception:
                LOGGER.exception("Failed to kill worker pid=%s", pid)


def _taskkill_process_tree(pid: int, reason: str) -> None:
    if pid <= 0:
        return
    try:
        result = subprocess.run(
            ["taskkill", "/PID", str(pid), "/F", "/T"],
            check=False,
            capture_output=True,
            text=True,
        )
        stdout = (result.stdout or "").strip()
        stderr = (result.stderr or "").strip()
        LOGGER.info(
            "taskkill /F /T executed for pid=%s reason=%s rc=%s stdout=%s stderr=%s",
            pid,
            reason,
            result.returncode,
            stdout[:200],
            stderr[:200],
        )
    except Exception:
        LOGGER.exception("taskkill /F /T failed for pid=%s reason=%s", pid, reason)


def _close_mt5_terminals() -> None:
    if os.name != "nt":
        return
    for proc_name in ("terminal64.exe", "terminal.exe"):
        try:
            result = subprocess.run(
                ["taskkill", "/F", "/IM", proc_name],
                check=False,
                capture_output=True,
                text=True,
            )
            LOGGER.info(
                "taskkill MT5 terminal proc=%s rc=%s stdout=%s stderr=%s",
                proc_name,
                result.returncode,
                (result.stdout or "").strip()[:200],
                (result.stderr or "").strip()[:200],
            )
        except Exception:
            LOGGER.exception("Failed to terminate MT5 terminal process: %s", proc_name)


def _pre_restart_cleanup(pid: Optional[int]) -> None:
    if pid is None or pid <= 0 or pid == os.getpid():
        return

    if os.name == "nt":
        # Always issue taskkill before restart to avoid orphaned child processes.
        # Retry with exponential backoff for Windows process cleanup
        for attempt in range(3):
            _taskkill_process_tree(pid, reason=f"pre-restart-{attempt+1}")
            deadline = time.time() + RESTART_CLEANUP_WAIT_SEC
            while time.time() < deadline:
                if not _pid_alive(pid):
                    return
                time.sleep(0.25)
            LOGGER.info("Retry attempt %s/3 for pid=%s...", attempt+1, pid)
        
        if _pid_alive(pid):
            LOGGER.warning("Process still appears alive after pre-restart cleanup (3 attempts). pid=%s", pid)
        return

    try:
        os.kill(pid, 15)
    except Exception:
        return


def launch_worker() -> Optional[subprocess.Popen]:
    flags = 0
    if os.name == "nt":
        flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)

    try:
        proc = subprocess.Popen(
            ENGINE_CMD,
            cwd=str(BOT_DIR),
            creationflags=flags,
            stdin=subprocess.DEVNULL,
        )
        LOGGER.info("Worker launched. pid=%s cmd=%s cwd=%s", proc.pid, ENGINE_CMD, BOT_DIR)
        deadline = time.time() + STARTUP_PROBE_SEC
        while time.time() < deadline:
            returncode = proc.poll()
            if returncode is not None:
                LOGGER.warning(
                    "Worker exited during startup probe. pid=%s code=%s probe_sec=%s",
                    proc.pid,
                    returncode,
                    STARTUP_PROBE_SEC,
                )
                if os.name == "nt":
                    _pre_restart_cleanup(proc.pid)
                return None
            time.sleep(0.2)
        return proc
    except Exception:
        LOGGER.exception("Failed to launch worker with cmd=%s", ENGINE_CMD)
        return None


def main() -> None:
    setup_logging()
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    ensure_desired_state_file(path=str(DESIRED_STATE_FILE))

    # Lock absolute working directory once so any relative file access stays stable.
    os.chdir(BOT_DIR)

    LOGGER.info("Starting watchdog with BOT_DIR=%s", BOT_DIR)
    LOGGER.info("Log file: %s", LOG_FILE)

    watchdog_lock_fd = _acquire_single_instance_lock(WATCHDOG_LOCK_FILE, "watchdog")
    if watchdog_lock_fd is None:
        LOGGER.info("Another watchdog instance is active. Exiting this process.")
        return
    atexit.register(_release_single_instance_lock, WATCHDOG_LOCK_FILE, watchdog_lock_fd)

    cleanup_runtime_lock_on_start()

    crash_times: list[float] = []
    proc: Optional[subprocess.Popen] = None
    proc_started_at_ts: Optional[float] = None
    pending_cleanup_pid: Optional[int] = None
    stop_hold_active = False
    stop_termination_attempted_pid: Optional[int] = None

    while True:
        try:
            desired_state = read_desired_state_value()
            if desired_state == DESIRED_STATE_STOP:
                if not stop_hold_active:
                    LOGGER.warning("Desired state is STOP. Holding worker launch/relaunch.")
                stop_hold_active = True
                if pending_cleanup_pid is not None:
                    pending_cleanup_pid = None
                if proc is not None and proc.poll() is None:
                    # Attempt worker termination once per pid while STOP is active.
                    if stop_termination_attempted_pid != proc.pid:
                        LOGGER.warning(
                            "Desired state STOP detected. Terminating running worker once. pid=%s",
                            proc.pid,
                        )
                        terminate_worker(proc)
                        # Runner can be force-killed before its normal finally-hook runs.
                        # Close MT5 terminal explicitly to guarantee STOP semantics.
                        _close_mt5_terminals()
                        stop_termination_attempted_pid = proc.pid
                if proc is not None and proc.poll() is not None:
                    proc = None
                time.sleep(LOOP_SLEEP_SEC)
                continue

            if stop_hold_active:
                LOGGER.info("Desired state switched to RUN. Watchdog relaunch is enabled.")
            stop_hold_active = False
            stop_termination_attempted_pid = None

            if proc is None or proc.poll() is not None:
                last_exit_was_lock_conflict = False
                last_exit_was_normal = False
                fast_fail = False
                if proc is not None:
                    pending_cleanup_pid = proc.pid
                    if proc_started_at_ts is not None:
                        life_sec = max(0.0, utc_ts() - proc_started_at_ts)
                        fast_fail = life_sec <= FAST_FAIL_WINDOW_SEC
                        if fast_fail:
                            LOGGER.warning(
                                "Worker exited within fast-fail window. pid=%s code=%s life_sec=%.2f window_sec=%s",
                                proc.pid,
                                proc.returncode,
                                life_sec,
                                FAST_FAIL_WINDOW_SEC,
                            )
                    if proc.returncode == 4:
                        last_exit_was_lock_conflict = True
                        owner_pid = _read_lock_pid(LOCK_FILE)
                        owner_alive = bool(owner_pid and _pid_alive(owner_pid))
                        LOGGER.info(
                            "Worker exited with code=4 (runtime lock conflict). lock=%s owner_pid=%s owner_alive=%s",
                            LOCK_FILE,
                            owner_pid,
                            owner_alive,
                        )
                        wait_sec = LOCK_CONFLICT_RETRY_SEC if owner_alive else RECOVERY_SLEEP_SEC
                        if fast_fail:
                            wait_sec = max(wait_sec, FAST_FAIL_BACKOFF_SEC)
                        LOGGER.info("Retrying worker launch after lock conflict in %ss.", wait_sec)
                        time.sleep(wait_sec)
                    elif proc.returncode == 0:
                        last_exit_was_normal = True
                        LOGGER.info("Worker exited normally with code=0")
                    else:
                        LOGGER.warning("Worker exited with code=%s", proc.returncode)
                        crash_times.append(utc_ts())

                now = utc_ts()
                crash_times = [t for t in crash_times if now - t <= CRASH_WINDOW_SEC]

                if (not last_exit_was_lock_conflict) and (not last_exit_was_normal) and len(crash_times) >= MAX_CRASHES_IN_WINDOW:
                    write_lockdown_marker("MAX_CRASHES_REACHED")
                    send_telegram_alert(
                        "LOCKDOWN_ERROR MAX_CRASHES_REACHED "
                        f"crashes={len(crash_times)} "
                        f"window_sec={CRASH_WINDOW_SEC}"
                    )
                    backoff = min(BACKOFF_MAX, BACKOFF_BASE ** min(len(crash_times), 6))
                else:
                    backoff = 0

                if backoff > 0:
                    LOGGER.error(
                        "Crash storm detected (%s crashes). Backing off for %ss.",
                        len(crash_times),
                        backoff,
                    )
                    time.sleep(backoff)
                elif fast_fail:
                    LOGGER.warning(
                        "Applying fast-fail backoff for %ss before relaunch.",
                        FAST_FAIL_BACKOFF_SEC,
                    )
                    time.sleep(FAST_FAIL_BACKOFF_SEC)

                if pending_cleanup_pid is not None:
                    _pre_restart_cleanup(pending_cleanup_pid)
                    pending_cleanup_pid = None

                # Ensure runtime lock is clean before launching new worker
                cleanup_runtime_lock_on_start()

                proc = launch_worker()
                if proc is None:
                    proc_started_at_ts = None
                    time.sleep(RECOVERY_SLEEP_SEC)
                    continue
                proc_started_at_ts = utc_ts()

                time.sleep(STARTUP_GRACE_SEC)
                continue

            heartbeat_ts = read_heartbeat_timestamp()
            if heartbeat_ts is not None and (utc_ts() - heartbeat_ts) > HEARTBEAT_TIMEOUT_SEC:
                LOGGER.error(
                    "Worker hang detected. Last heartbeat ts=%s timeout=%ss",
                    heartbeat_ts,
                    HEARTBEAT_TIMEOUT_SEC,
                )
                send_telegram_alert(
                    "Worker hang detected "
                    f"last_heartbeat_ts={heartbeat_ts} "
                    f"timeout_sec={HEARTBEAT_TIMEOUT_SEC}"
                )
                pending_cleanup_pid = proc.pid
                terminate_worker(proc)
                proc = None
                proc_started_at_ts = None
                time.sleep(RECOVERY_SLEEP_SEC)
                continue

            time.sleep(LOOP_SLEEP_SEC)

        except BaseException:
            # Never exit: always recover and continue watching.
            LOGGER.exception("Unhandled watchdog error. Recovering and continuing loop.")
            if proc is not None:
                try:
                    pending_cleanup_pid = proc.pid
                    terminate_worker(proc)
                except Exception:
                    LOGGER.exception("Additional error while terminating worker after loop exception.")
                proc = None
                proc_started_at_ts = None
            time.sleep(RECOVERY_SLEEP_SEC)


if __name__ == "__main__":
    main()
