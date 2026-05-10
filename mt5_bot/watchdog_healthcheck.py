from __future__ import annotations

import argparse
import json
import os
import hashlib
import signal
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from core import control

LOG_LINES = 8
DEFAULT_NOTIFY_COOLDOWN_SEC = 15 * 60
HEALTH_ALERT_FILE = "runtime/health_alert.json"
HEALTH_ALERT_STATE_FILE = "runtime/watchdog_healthcheck.health_alert_state"


@dataclass
class CheckIssue:
    severity: str
    source: str
    message: str
    action: str = ""


@dataclass
class ProcessSnapshot:
    pid: int
    name: str = ""
    cmdline: str = ""
    match: str = ""


@dataclass
class HealthState:
    bot_dir: Path
    now: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    issues: List[CheckIssue] = field(default_factory=list)
    info: List[str] = field(default_factory=list)
    candidates: List[ProcessSnapshot] = field(default_factory=list)
    watchdog_pids: List[ProcessSnapshot] = field(default_factory=list)
    runner_pids: List[ProcessSnapshot] = field(default_factory=list)

    @property
    def has_blocker(self) -> bool:
        return any(issue.severity == "BLOCK" for issue in self.issues)

    @property
    def has_warning(self) -> bool:
        return any(issue.severity == "WARN" for issue in self.issues)

    @property
    def overall_severity(self) -> str:
        if self.has_blocker:
            return "BLOCK"
        if self.has_warning:
            return "WARN"
        return "OK"


def _read_json_payload(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    try:
        raw = path.read_text(encoding="utf-8").strip()
        if not raw:
            return {}
        return json.loads(raw)
    except Exception:
        return {}


def _write_json_payload(path: Path, payload: Dict[str, object]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass


def _should_emit_event(bot_dir: Path, state: "HealthState", cooldown_sec: int) -> bool:
    if cooldown_sec < 0:
        return True
    state_path = bot_dir / HEALTH_ALERT_STATE_FILE
    marker = _read_json_payload(state_path)
    now_ts = datetime.now(timezone.utc).timestamp()
    try:
        last_signature = str(marker.get("signature", ""))
    except Exception:
        last_signature = ""
    try:
        last_ts = float(marker.get("last_emitted_utc", 0.0))
    except Exception:
        last_ts = 0.0
    raw_signature = "|".join(
        f"{issue.severity}:{issue.source}:{issue.message}:{issue.action}"
        for issue in sorted(
            state.issues,
            key=lambda issue: (issue.severity, issue.source, issue.message, issue.action),
        )
    )
    if not raw_signature:
        raw_signature = "OK"
    current_signature = hashlib.md5(raw_signature.encode("utf-8")).hexdigest()
    if cooldown_sec > 0 and (now_ts - last_ts) < float(cooldown_sec):
        if last_signature == current_signature:
            return False

    _write_json_payload(
        state_path,
        {
            "last_emitted_utc": now_ts,
            "signature": current_signature,
            "severity": state.overall_severity,
        },
    )
    return True


def _emit_health_alert(
    bot_dir: Path,
    state: HealthState,
    cooldown_sec: int,
) -> None:
    if not state.issues:
        alert_path = bot_dir / HEALTH_ALERT_FILE
        if alert_path.exists():
            try:
                alert_path.unlink()
            except Exception:
                pass
        return
    if not _should_emit_event(bot_dir, state, cooldown_sec):
        return

    payload: Dict[str, object] = {
        "schema": "mt5.watchdog.health_alert.v1",
        "severity": state.overall_severity,
        "issued_at_utc": state.now.isoformat(),
        "bot_dir": str(bot_dir),
        "checks_total": len(state.issues),
        "block_count": len([issue for issue in state.issues if issue.severity == "BLOCK"]),
        "warn_count": len([issue for issue in state.issues if issue.severity == "WARN"]),
        "runtime": {
            "watchdog_pids": [item.pid for item in state.watchdog_pids],
            "runner_pids": [item.pid for item in state.runner_pids],
        },
        "issues": [
            {
                "severity": issue.severity,
                "source": issue.source,
                "message": issue.message,
                "action": issue.action,
            }
            for issue in state.issues
        ],
        "info": list(state.info),
    }

    try:
        path = bot_dir / HEALTH_ALERT_FILE
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception:
        pass


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        try:
            import psutil  # type: ignore

            return bool(psutil.pid_exists(int(pid)))
        except Exception:
            pass
        try:
            import ctypes

            handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, int(pid))
            if handle:
                ctypes.windll.kernel32.CloseHandle(handle)
                return True
            return ctypes.windll.kernel32.GetLastError() == 5
        except Exception:
            pass

    try:
        os.kill(int(pid), 0)
    except PermissionError:
        return True
    except Exception:
        return False
    return True


def _read_int_file(path: Path) -> Optional[int]:
    if not path.exists():
        return None
    try:
        raw = path.read_text(encoding="utf-8").strip()
    except Exception:
        return None
    if not raw:
        return None
    try:
        return int(raw)
    except Exception:
        return None


def _read_text(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return None


def _kill_process(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        try:
            result = subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(pid)],
                check=False,
                capture_output=True,
                text=True,
            )
            return result.returncode == 0
        except Exception:
            return False

    try:
        os.kill(pid, signal.SIGTERM)
        return True
    except Exception:
        return False


def _parse_processes() -> Sequence[ProcessSnapshot]:
    if os.name == "nt":
        try:
            import psutil  # type: ignore

            items: list[ProcessSnapshot] = []
            for proc in psutil.process_iter(["pid", "name", "cmdline", "create_time"]):
                try:
                    if not proc.is_running():
                        continue
                    cmdline_parts = proc.info.get("cmdline") or []
                    if not isinstance(cmdline_parts, list):
                        continue
                    cmdline = " ".join(str(v) for v in cmdline_parts)
                    if "runner.py" not in cmdline.lower() and "watchdog.py" not in cmdline.lower():
                        continue
                    items.append(
                        ProcessSnapshot(
                            pid=proc.info.get("pid") or 0,
                            name=str(proc.info.get("name") or ""),
                            cmdline=cmdline,
                            match="runner" if "runner.py" in cmdline.lower() else "watchdog",
                        )
                    )
                except Exception:
                    continue
            return items
        except Exception:
            return tuple()

    # Lightweight fallback for non-psutil environments.
    return tuple()


def _check_lock_file(
    path: Path,
    expected_script_hint: str,
    processes: Sequence[ProcessSnapshot],
    state: HealthState,
    fix: bool,
) -> Optional[int]:
    pid = _read_int_file(path)
    if not path.exists():
        state.info.append(f"{path.name}: no lock file")
        return None
    if pid is None:
        state.issues.append(
            CheckIssue(
                severity="WARN",
                source=path.name,
                message=f"{path} is not parseable or empty",
                action="remove malformed file manually or run --fix",
            )
        )
        if fix:
            try:
                path.unlink()
                state.info.append(f"{path.name}: removed malformed file")
            except Exception as exc:
                state.issues.append(
                    CheckIssue(
                        severity="WARN",
                        source=path.name,
                        message=f"Failed to remove malformed file {path}: {exc}",
                        action="manual remove with admin permission",
                    )
                )
        return None

    owner_alive = _pid_alive(pid)
    matched_owner = any(proc.pid == pid for proc in processes)
    if owner_alive:
        if not matched_owner:
            state.issues.append(
                CheckIssue(
                    severity="WARN",
                    source=path.name,
                    message=f"{path} owner_pid={pid} is alive but command does not match {expected_script_hint}.",
                    action="verify process ownership; run --fix to kill stale lock owner only if safe.",
                )
            )
        else:
            state.info.append(f"{path.name}: owner_pid={pid} alive")
    else:
        state.issues.append(
            CheckIssue(
                severity="BLOCK",
                source=path.name,
                message=f"{path} is stale (owner_pid={pid} is dead/not parseable).",
                action="remove stale lock before restart",
            )
        )
        if fix:
            try:
                path.unlink()
                state.info.append(f"{path.name}: stale lock removed")
                state.issues = [
                    x
                    for x in state.issues
                    if not (x.source == path.name and "is stale" in x.message)
                ]
                state.info.append(f"{path.name}: stale-blocker cleared by --fix")
            except Exception as exc:
                state.issues.append(
                    CheckIssue(
                        severity="BLOCK",
                        source=path.name,
                        message=f"{path} stale lock removal failed after --fix: {exc}",
                        action="manual cleanup while all bots are fully stopped",
                    )
                )
    return pid


def _check_desired_state(state: HealthState) -> None:
    desired_state_path = state.bot_dir / "runtime" / "desired_state.json"
    runtime_control_path = state.bot_dir / "runtime_control.json"
    lock_state = control.load_desired_state(path=str(desired_state_path))
    desired = str(lock_state.get("state", control.DESIRED_STATE_RUN)).upper()
    state.info.append(f"desired_state={desired}")

    control_state = control.RuntimeControlChannel(path=str(runtime_control_path)).load()
    manual_halt = bool(control_state.get("manual_halt"))
    intentional_stop = bool(control_state.get("intentional_stop_requested"))
    paused = bool(control_state.get("paused"))

    if manual_halt or intentional_stop or paused:
        state.issues.append(
            CheckIssue(
                severity="WARN",
                source="runtime_control",
                message="runtime_control has stop/halt flags active",
                action="clear via control_run.py/control_stop.py as intended",
            )
        )
    if desired == control.DESIRED_STATE_STOP:
        state.info.append("Desired state is STOP; startup should be held.")
    elif desired == control.DESIRED_STATE_RUN:
        state.info.append("Desired state is RUN; watchdog/worker should be active.")
    else:
        state.issues.append(
            CheckIssue(
                severity="WARN",
                source="desired_state",
                message=f"Invalid desired state: {desired}",
                action="set to RUN or STOP via control scripts",
            )
        )


def _group_processes(processes: Sequence[ProcessSnapshot], state: HealthState) -> None:
    for proc in processes:
        lower = proc.cmdline.lower()
        if "watchdog.py" in lower:
            state.watchdog_pids.append(proc)
        if "runner.py" in lower:
            state.runner_pids.append(proc)
        if proc.match:
            state.candidates.append(proc)


def run_checks(bot_dir: Path, fix: bool, verbose: bool) -> HealthState:
    bot_dir = bot_dir.resolve()
    state = HealthState(bot_dir=bot_dir)

    process_list = _parse_processes()
    _group_processes(process_list, state)

    _check_desired_state(state)

    rt_lock = bot_dir / "runtime.lock"
    wd_lock = bot_dir / "runtime" / "watchdog.lock"
    heartbeat = bot_dir / "runtime" / "heartbeat.json"
    lockdown = bot_dir / "runtime" / "lockdown.flag"
    _check_lock_file(rt_lock, "runner.py", process_list, state, fix)
    _check_lock_file(wd_lock, "watchdog.py", process_list, state, fix)

    if state.watchdog_pids:
        state.info.append(f"watchdog processes: {[p.pid for p in state.watchdog_pids]}")
    if state.runner_pids:
        state.info.append(f"runner processes: {[p.pid for p in state.runner_pids]}")

    if len(state.watchdog_pids) > 1:
        state.issues.append(
            CheckIssue(
                severity="BLOCK",
                source="process_scan",
                message=f"Multiple watchdog processes detected ({[p.pid for p in state.watchdog_pids]}).",
                action="terminate duplicates, keep one watchdog leader",
            )
        )
        if fix:
            for proc in sorted(state.watchdog_pids, key=lambda p: p.pid)[1:]:
                if _kill_process(proc.pid):
                    state.info.append(f"kill duplicate watchdog pid={proc.pid}")
                else:
                    state.issues.append(
                        CheckIssue(
                            severity="WARN",
                            source="process_scan",
                            message=f"Failed to kill duplicate watchdog pid={proc.pid}",
                            action="manual kill in task manager",
                        )
                    )

    if len(state.runner_pids) > 1:
        state.issues.append(
            CheckIssue(
                severity="BLOCK",
                source="process_scan",
                message=f"Multiple runner processes detected ({[p.pid for p in state.runner_pids]}).",
                action="keep one runner only",
            )
        )
        if fix:
            for proc in sorted(state.runner_pids, key=lambda p: p.pid)[1:]:
                if _kill_process(proc.pid):
                    state.info.append(f"kill duplicate runner pid={proc.pid}")
                else:
                    state.issues.append(
                        CheckIssue(
                            severity="WARN",
                            source="process_scan",
                            message=f"Failed to kill duplicate runner pid={proc.pid}",
                            action="manual kill in task manager",
                        )
                    )

    if lockdown.exists():
        state.issues.append(
            CheckIssue(
                severity="WARN",
                source="lockdown.flag",
                message="lockdown.flag exists from repeated crash state.",
                action="clear after inspection; usually auto-cleared by watchdog restart policy.",
            )
        )
        if verbose:
            state.info.append(f"lockdown content: {_read_text(lockdown) or ''}")

    # Heartbeat freshness check (if lock exists, should be updating frequently).
    heartbeat_text = _read_text(heartbeat)
    if heartbeat_text:
        if verbose:
            state.info.append(f"heartbeat: {heartbeat_text.strip()[:250]}")
    else:
        if rt_lock.exists():
            state.issues.append(
                CheckIssue(
                    severity="WARN",
                    source="heartbeat.json",
                    message="runtime.lock exists but heartbeat is missing/unreadable.",
                    action="inspect worker startup/IO errors",
                )
            )

    # Optional quick view of last watcher logs
    debug_log = bot_dir / "watchdog_debug.log"
    if verbose and debug_log.exists():
        lines = _read_text(debug_log)
        if lines:
            tail = "\n".join(lines.strip().splitlines()[-LOG_LINES:])
            state.info.append(f"watchdog_debug tail:\n{tail}")

    return state


def _print_report(state: HealthState) -> int:
    print(f"[watchdog_healthcheck] bot_dir={state.bot_dir}")
    print(f"[watchdog_healthcheck] time_utc={_now_iso()}")
    print(f"[watchdog_healthcheck] checks={len(state.issues)} warnings_block={len([i for i in state.issues if i.severity=='BLOCK'])}")
    for issue in state.issues:
        print(f"- {issue.severity} | {issue.source} | {issue.message}")
        if issue.action:
            print(f"  -> {issue.action}")
    for line in state.info:
        print(f"* {line}")

    if state.has_blocker:
        return 2
    if state.has_warning:
        return 1
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MT5 watchdog/runner 상태 점검")
    parser.add_argument(
        "--bot-dir",
        default=str(Path(__file__).resolve().parent),
        help="MT5 bot 경로",
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="정합성 오류(락/중복) 자동 복구 시도",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="상세 로그 출력",
    )
    parser.add_argument(
        "--notify",
        action="store_true",
        help="이슈 발견 시 runtime/health_alert.json에 이벤트 기록",
    )
    parser.add_argument(
        "--notify-cooldown-sec",
        type=int,
        default=DEFAULT_NOTIFY_COOLDOWN_SEC,
        help="health_alert 이벤트 발행 간격(초)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    bot_dir = Path(args.bot_dir)
    state = run_checks(bot_dir=bot_dir, fix=args.fix, verbose=args.verbose)
    if args.notify:
        _emit_health_alert(
            bot_dir=bot_dir,
            state=state,
            cooldown_sec=args.notify_cooldown_sec,
        )
    return _print_report(state)


if __name__ == "__main__":
    raise SystemExit(main())
