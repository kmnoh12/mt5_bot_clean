from __future__ import annotations

import argparse
import atexit
import logging
import os
import json
import sys
from pathlib import Path
import subprocess
import time
from typing import Optional, Dict, Any, Tuple

from core.config import load_config
from core.control import DESIRED_STATE_STOP, load_desired_state
from core.runtime import TradingRuntime


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Professional MT5 quantitative bot runtime")
    parser.add_argument(
        "--config",
        default=str(Path(__file__).with_name("config.yaml")),
        help="Path to config.yaml",
    )
    parser.add_argument("--once", action="store_true", help="Run a single cycle and exit.")
    parser.add_argument(
        "--mode",
        choices=["live", "backtest"],
        default=None,
        help="Override general.mode from config.",
    )
    return parser.parse_args()


def setup_logging(level_name: str) -> None:
    level = getattr(logging, str(level_name).upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        # Prefer psutil when available for a stable Windows PID existence check.
        try:
            import psutil  # type: ignore

            return bool(psutil.pid_exists(pid))
        except Exception:
            pass

        # Fallback to WinAPI OpenProcess to avoid os.kill SystemError on Windows.
        try:
            import ctypes

            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            handle = ctypes.windll.kernel32.OpenProcess(
                PROCESS_QUERY_LIMITED_INFORMATION, False, pid
            )
            if handle:
                ctypes.windll.kernel32.CloseHandle(handle)
                return True
            return ctypes.windll.kernel32.GetLastError() == 5  # Access denied => alive.
        except Exception:
            pass

    try:
        os.kill(pid, 0)
    except PermissionError:
        return True
    except (SystemError, ProcessLookupError, OSError):
        return False
    return True


def _get_pid_start_time(pid: int) -> Optional[float]:
    if pid <= 0:
        return None
    try:
        import psutil  # type: ignore

        return float(psutil.Process(pid).create_time())
    except Exception:
        return None


def _wait_pid_exit(pid: int, timeout_sec: float = 3.0) -> None:
    if pid <= 0:
        return
    deadline = time.time() + max(0.1, float(timeout_sec))
    while time.time() < deadline:
        if not _pid_alive(pid):
            return
        time.sleep(0.2)


def _read_lock_owner(lock_path: Path) -> Tuple[Optional[int], Optional[float]]:
    if not lock_path.exists():
        return None, None
    try:
        raw = lock_path.read_text(encoding="utf-8").strip()
    except Exception:
        return None, None
    if not raw:
        return None, None
    # Backward-compatible: plain pid text.
    try:
        pid_plain = int(raw)
        return (pid_plain if pid_plain > 0 else None), None
    except Exception:
        pass
    # New format: JSON metadata.
    try:
        payload: Dict[str, Any] = json.loads(raw)
        pid = int(payload.get("pid") or 0)
        start_time = payload.get("start_time")
        start_time_f = float(start_time) if start_time is not None else None
        return (pid if pid > 0 else None), start_time_f
    except Exception:
        return None, None


def _cleanup_lock_file(lock_path: Path, owner_pid: Optional[int]) -> bool:
    for attempt in range(1, 6):
        try:
            lock_path.unlink()
            return True
        except FileNotFoundError:
            return True
        except PermissionError as exc:
            if attempt >= 5:
                return False
            if owner_pid and _pid_alive(owner_pid):
                try:
                    subprocess.run(
                        ["taskkill", "/PID", str(owner_pid), "/F", "/T"],
                        check=False,
                        capture_output=True,
                    )
                except Exception:
                    pass
                _wait_pid_exit(owner_pid, timeout_sec=3.0)
            # Last-resort cleanup for Windows stale file handles: target runner.py only.
            if os.name == "nt":
                _kill_rogue_runners(lock_path.parent, exclude_pids={os.getpid(), owner_pid or 0})
            time.sleep(0.5)
        except Exception:
            if attempt >= 5:
                return False
            time.sleep(0.5)
    return False


def _acquire_single_instance_lock(lock_path: Path) -> Optional[int]:
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    for _ in range(3):
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_RDWR)
        except FileExistsError:
            pid, locked_start_time = _read_lock_owner(lock_path)
            if isinstance(pid, int) and pid > 0:
                # If lock points to this same process id, treat it as stale and reclaim.
                if pid == os.getpid():
                    if not _cleanup_lock_file(lock_path, pid):
                        return None
                    continue
                try:
                    if _pid_alive(pid):
                        # Prevent false positives when PID was reused by another process.
                        if locked_start_time is not None:
                            current_start_time = _get_pid_start_time(pid)
                            if (
                                current_start_time is not None
                                and abs(current_start_time - locked_start_time) > 1.0
                            ):
                                if not _cleanup_lock_file(lock_path, pid):
                                    return None
                                continue
                        return None
                except Exception:
                    pass

            # Stale lock (or unreadable owner): remove and immediately retry acquisition.
            if not _cleanup_lock_file(lock_path, pid):
                return None
            continue
        except Exception:
            return None

        try:
            payload = {
                "pid": os.getpid(),
                "start_time": _get_pid_start_time(os.getpid()),
            }
            os.write(fd, json.dumps(payload, ensure_ascii=True).encode("utf-8"))
        except Exception:
            try:
                os.close(fd)
            except Exception:
                pass
            try:
                lock_path.unlink()
            except Exception:
                pass
            return None
        return fd

    # Force-takeover fallback: if lock is still present, kill owner/global python and retry once.
    owner_pid, _ = _read_lock_owner(lock_path)
    if owner_pid and _pid_alive(owner_pid):
        try:
            subprocess.run(
                ["taskkill", "/PID", str(owner_pid), "/F", "/T"],
                check=False,
                capture_output=True,
                text=True,
            )
        except Exception:
            pass
        _wait_pid_exit(owner_pid, timeout_sec=3.0)
    if os.name == "nt":
        _kill_rogue_runners(lock_path.parent, exclude_pids={os.getpid(), owner_pid or 0})
    time.sleep(0.5)
    _cleanup_lock_file(lock_path, owner_pid)
    try:
        fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_RDWR)
        payload = {
            "pid": os.getpid(),
            "start_time": _get_pid_start_time(os.getpid()),
        }
        os.write(fd, json.dumps(payload, ensure_ascii=True).encode("utf-8"))
        return fd
    except Exception:
        pass

    return None


def _kill_rogue_runners(bot_dir: Path, exclude_pids: set[int]) -> None:
    if os.name != "nt":
        return
    try:
        import psutil  # type: ignore
    except Exception:
        return

    for proc in psutil.process_iter(["pid", "cmdline"]):
        try:
            pid = int(proc.info.get("pid") or 0)
            if pid <= 0 or pid in exclude_pids:
                continue
            cmdline = " ".join(proc.info.get("cmdline") or [])
            if "runner.py" not in cmdline:
                continue
            if str(bot_dir) not in cmdline:
                continue
            subprocess.run(
                ["taskkill", "/PID", str(pid), "/F", "/T"],
                check=False,
                capture_output=True,
                text=True,
            )
        except Exception:
            continue


def _release_single_instance_lock(lock_path: Path, fd: Optional[int]) -> None:
    try:
        if fd is not None:
            os.close(fd)
    except Exception:
        pass
    try:
        if lock_path.exists():
            owner_pid, _ = _read_lock_owner(lock_path)
            owner_pid = int(owner_pid or 0)
            if owner_pid not in (0, os.getpid()):
                return
            lock_path.unlink()
    except Exception:
        pass


def _close_mt5_terminals() -> None:
    if os.name != "nt":
        return
    candidates = ["terminal64.exe", "terminal.exe"]
    for proc_name in candidates:
        try:
            subprocess.run(
                ["taskkill", "/F", "/IM", proc_name],
                capture_output=True,
                text=True,
                check=False,
            )
        except Exception:
            continue


def main() -> int:
    args = parse_args()
    config_path = Path(args.config).resolve()
    if not config_path.exists():
        print(f"Config file not found: {config_path}", file=sys.stderr)
        return 1

    lock_path = config_path.with_name("runtime.lock")
    lock_fd = _acquire_single_instance_lock(lock_path)
    if lock_fd is None:
        print(f"Another runner is already active (lock: {lock_path}).", file=sys.stderr)
        return 4
    atexit.register(_release_single_instance_lock, lock_path, lock_fd)

    try:
        config = load_config(config_path)
    except Exception as exc:
        print(f"Failed to load config: {exc}", file=sys.stderr)
        return 1

    if args.mode:
        config.setdefault("general", {})["mode"] = args.mode

    desired_state = load_desired_state()
    if desired_state.get("state") == DESIRED_STATE_STOP and os.environ.get("MT5_FORCE_RUN") != "1":
        print("Desired state is STOP. Skipping MT5 runtime startup. Set MT5_FORCE_RUN=1 to override.")
        return 0

    logging_cfg = config.get("logging", {}) or {}
    general_cfg = config.get("general", {}) or {}
    log_level = logging_cfg.get("level", general_cfg.get("log_level", "INFO"))
    setup_logging(str(log_level))
    logger = logging.getLogger("runner")
    logger.info(
        "Launching MT5 bot | mode=%s | dry_run=%s | config=%s",
        config.get("general", {}).get("mode"),
        config.get("general", {}).get("dry_run"),
        config_path,
    )

    try:
        runtime = TradingRuntime(config=config, config_path=str(config_path))
    except Exception as exc:
        logger.exception("Runtime initialization failed.")
        print(f"Runtime initialization failed: {exc}", file=sys.stderr)
        return 1

    mt5_cfg = config.get("mt5", {}) or {}
    close_on_exit = bool(mt5_cfg.get("close_terminal_on_exit", False))
    exit_code = 0
    try:
        exit_code = runtime.run(once=args.once)
    finally:
        runtime = None
        if close_on_exit:
            _close_mt5_terminals()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
