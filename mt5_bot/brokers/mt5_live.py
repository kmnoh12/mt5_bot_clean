from __future__ import annotations

import logging
import math
import os
import random
import re
import time
import json
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from brokers.base import BrokerGateway
from core.models import MarketTick, OrderIntent, OrderResult, Position, Side, SymbolConstraints

try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None

LOGGER = logging.getLogger(__name__)

class MT5LiveGateway(BrokerGateway):
    mode = "live"
    LIVE_TRADING_ENV_VAR = "MT5_ALLOW_LIVE_TRADING"
    LIVE_TRADING_ENV_TOKEN = "YES_I_ACCEPT_RISK"

    @staticmethod
    def _safe_int(value: Any, default: int) -> int:
        try:
            if isinstance(value, str):
                text = value.strip()
                if text == "":
                    return int(default)
                return int(float(text))
            return int(value)
        except Exception:
            return int(default)

    @staticmethod
    def _safe_float(value: Any, default: float) -> float:
        try:
            if isinstance(value, str):
                text = value.strip()
                if text == "":
                    return float(default)
                return float(text)
            return float(value)
        except Exception:
            return float(default)

    @staticmethod
    def _safe_int_code_set(raw_codes: Any) -> set:
        out: set = set()
        if raw_codes is None:
            return out
        if isinstance(raw_codes, str):
            items: List[Any] = [token for token in re.split(r"[,;\s]+", raw_codes) if token]
        elif isinstance(raw_codes, (list, tuple, set)):
            items = list(raw_codes)
        else:
            items = [raw_codes]
        for code in items:
            try:
                text = str(code).strip()
            except Exception:
                text = ""
            if not text:
                continue
            try:
                out.add(int(float(text)))
            except Exception:
                continue
        return out

    @staticmethod
    def _clamp_timeout_ms(value: Any, default: int, min_value: int = 1000, max_value: int = 300000) -> int:
        parsed = MT5LiveGateway._safe_int(value, default)
        return max(int(min_value), min(int(max_value), parsed))

    def __init__(self, config: Dict[str, Any], notifier: Optional[Any] = None) -> None:
        self.config = config
        raw_mt5_cfg = config.get("mt5", {})
        raw_general_cfg = config.get("general", {})
        raw_execution_cfg = config.get("execution", {})
        self.mt5_cfg = raw_mt5_cfg if isinstance(raw_mt5_cfg, dict) else {}
        self.general_cfg = raw_general_cfg if isinstance(raw_general_cfg, dict) else {}
        self.execution_cfg = raw_execution_cfg if isinstance(raw_execution_cfg, dict) else {}
        self.notifier = notifier
        self._watchdog_debug_path = Path(__file__).resolve().parents[1] / "watchdog_debug.log"
        self._refresh_order_gate_from_config()

        reconnect_raw = self.general_cfg.get("reconnect", {}) if isinstance(self.general_cfg, dict) else {}
        reconnect_cfg = reconnect_raw if isinstance(reconnect_raw, dict) else {}
        self.max_reconnect_attempts = max(1, self._safe_int(reconnect_cfg.get("max_attempts", 6), 6))
        self.reconnect_attempts_per_cycle = max(
            1,
            min(
                self.max_reconnect_attempts,
                self._safe_int(reconnect_cfg.get("attempts_per_cycle", 1), 1),
            ),
        )
        self.base_delay_seconds = max(0.1, self._safe_float(reconnect_cfg.get("base_delay_seconds", 1.0), 1.0))
        self.max_delay_seconds = max(
            self.base_delay_seconds,
            self._safe_float(reconnect_cfg.get("max_delay_seconds", 30.0), 30.0),
        )
        self.reconnect_cooldown_seconds = max(
            0.2,
            self._safe_float(reconnect_cfg.get("cooldown_seconds", 2.0), 2.0),
        )
        self.reconnect_jitter_seconds = max(0.0, self._safe_float(reconnect_cfg.get("jitter_seconds", 0.5), 0.5))
        self.max_ipc_failures_before_halt = max(
            3,
            self._safe_int(reconnect_cfg.get("max_ipc_failures_before_halt", 8), 8),
        )
        self.force_shutdown_on_reconnect = bool(reconnect_cfg.get("force_shutdown_on_reconnect", False))
        self.init_timeout_ms_default = self._clamp_timeout_ms(
            self.mt5_cfg.get("init_timeout_ms", 120000),
            default=120000,
            min_value=1000,
            max_value=300000,
        )
        self.connect_init_timeout_cap_ms = self._clamp_timeout_ms(
            reconnect_cfg.get(
                "init_timeout_cap_ms",
                self.mt5_cfg.get("connect_init_timeout_cap_ms", 30000),
            ),
            default=30000,
            min_value=1000,
            max_value=300000,
        )
        self.reconnect_init_timeout_ms = self._clamp_timeout_ms(
            self.mt5_cfg.get("reconnect_init_timeout_ms", 15000),
            default=15000,
            min_value=1000,
            max_value=self.connect_init_timeout_cap_ms,
        )
        self.reconnect_total_budget_seconds = max(
            1.0,
            self._safe_float(
                reconnect_cfg.get(
                    "total_budget_seconds",
                    reconnect_cfg.get("reconnect_budget_seconds", 90.0),
                ),
                90.0,
            ),
        )
        self.heartbeat_seconds = max(1.0, self._safe_float(self.general_cfg.get("heartbeat_seconds", 10), 10.0))

        ipc_codes = self.mt5_cfg.get("ipc_timeout_codes", [-10005])
        parsed_ipc = self._safe_int_code_set(ipc_codes)
        self.ipc_timeout_codes = {-10001, -10005} | parsed_ipc
        self.attach_only = bool(self.mt5_cfg.get("attach_only", True))
        self.allow_programmatic_login = bool(self.mt5_cfg.get("allow_programmatic_login", False))
        self.allow_terminal_launch = bool(self.mt5_cfg.get("allow_terminal_launch", False))

        expected_login_raw = self.mt5_cfg.get("login", 0)
        self.expected_login = max(0, self._safe_int(expected_login_raw, 0))
        self.expected_server = str(self.mt5_cfg.get("server", "") or "").strip()
        self.account_guard_enabled = bool(self.mt5_cfg.get("account_guard_enabled", True))
        self.require_trade_enabled = bool(self.mt5_cfg.get("require_trade_enabled", True))
        self.server_offset_seconds = self._safe_float(self.mt5_cfg.get("server_time_offset_hours", 0.0), 0.0) * 3600.0
        broker_request_raw = config.get("broker_request_guard", {})
        broker_request_cfg = broker_request_raw if isinstance(broker_request_raw, dict) else {}
        self.comment_ascii_only = bool(broker_request_cfg.get("comment_ascii_only", True))
        self.comment_max_len = max(
            1,
            self._safe_int(broker_request_cfg.get("comment_max_len", 24) or 24, 24),
        )
        self.retry_without_comment_on_invalid_comment = bool(
            broker_request_cfg.get("retry_without_comment_on_invalid_comment", True)
        )

        self.connected = False
        self._last_heartbeat_ts = 0.0
        self._next_reconnect_ts = 0.0
        self._fatal_reason: Optional[str] = None
        self._fatal_reported = False
        self._ipc_failure_count = 0
        self._ipc_threshold_reported = False
        self._api_trace: deque = deque(maxlen=80)

    def _refresh_order_gate_from_config(self) -> None:
        general_dry_run = bool(self.general_cfg.get("dry_run", True))
        execution_dry_run = bool(self.execution_cfg.get("dry_run", False))
        self.dry_run = bool(general_dry_run or execution_dry_run)
        live_enabled = self.execution_cfg.get("live_trading_enabled", self.mt5_cfg.get("live_trading_enabled", False))
        self.live_trading_enabled = bool(live_enabled)
        self.live_trading_env_confirmed = (
            os.environ.get(self.LIVE_TRADING_ENV_VAR) == self.LIVE_TRADING_ENV_TOKEN
        )
        self.orders_allowed = bool(
            not self.dry_run
            and self.live_trading_enabled
            and self.live_trading_env_confirmed
        )

        if self.dry_run:
            self.order_gate_reason = "dry_run enabled"
        elif not self.live_trading_enabled:
            self.order_gate_reason = "live_trading_enabled is false"
        elif not self.live_trading_env_confirmed:
            self.order_gate_reason = (
                f"{self.LIVE_TRADING_ENV_VAR} must be {self.LIVE_TRADING_ENV_TOKEN}"
            )
        else:
            self.order_gate_reason = ""

    def update_order_gate(self, config: Dict[str, Any]) -> None:
        self.config = config or {}
        raw_mt5_cfg = self.config.get("mt5", {})
        raw_general_cfg = self.config.get("general", {})
        raw_execution_cfg = self.config.get("execution", {})
        self.mt5_cfg = raw_mt5_cfg if isinstance(raw_mt5_cfg, dict) else {}
        self.general_cfg = raw_general_cfg if isinstance(raw_general_cfg, dict) else {}
        self.execution_cfg = raw_execution_cfg if isinstance(raw_execution_cfg, dict) else {}
        self._refresh_order_gate_from_config()

    def _blocked_order_result(self, operation: str, **fields: Any) -> Optional[OrderResult]:
        if self.orders_allowed:
            return None
        meta = {
            "operation": str(operation or ""),
            "orders_allowed": False,
            "dry_run": bool(self.dry_run),
            "live_trading_enabled": bool(self.live_trading_enabled),
            "env_var": self.LIVE_TRADING_ENV_VAR,
            "env_confirmed": bool(self.live_trading_env_confirmed),
            "reason": str(self.order_gate_reason or "live order gate closed"),
        }
        for key, value in fields.items():
            try:
                meta[str(key)] = value
            except Exception:
                meta[str(key)] = "<unserializable>"
        return OrderResult(
            ok=False,
            status="LIVE_TRADING_BLOCKED",
            message=f"{operation} blocked by live order gate: {meta['reason']}",
            raw={"live_order_gate": meta},
        )

    def _watchdog_debug(self, event: str, **fields: Any) -> None:
        payload = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "event": str(event),
            "fields": {},
        }
        for key, value in fields.items():
            try:
                payload["fields"][str(key)] = value
            except Exception:
                payload["fields"][str(key)] = "<unserializable>"
        try:
            self._watchdog_debug_path.parent.mkdir(parents=True, exist_ok=True)
            line = json.dumps(payload, ensure_ascii=False, default=str)
            with self._watchdog_debug_path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        except Exception:
            return

    def _notify_system(self, message: str) -> None:
        if self.notifier is None:
            return
        sender = getattr(self.notifier, "send_system", None)
        if callable(sender):
            try:
                sender(message)
            except Exception as exc:
                self._watchdog_debug("notifier_send_system_error", message=message, error=repr(exc))
            return
        # Backward compatibility with legacy notifier naming.
        sender = getattr(self.notifier, "send_info", None)
        if callable(sender):
            try:
                sender(message)
            except Exception as exc:
                self._watchdog_debug("notifier_send_info_error", message=message, error=repr(exc))

    def _notify_error(self, message: str) -> None:
        if self.notifier is None:
            return
        sender = getattr(self.notifier, "send_error", None)
        if callable(sender):
            try:
                sender(message)
            except Exception as exc:
                self._watchdog_debug("notifier_send_error_error", message=message, error=repr(exc))

    def fatal_error(self) -> Optional[str]:
        return self._fatal_reason

    def _trace_api(self, op: str, stage: str, **fields: Any) -> None:
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "op": str(op),
            "stage": str(stage),
            **fields,
        }
        self._api_trace.append(entry)
        LOGGER.debug("MT5 API TRACE op=%s stage=%s fields=%s", op, stage, fields)

    def _log_recent_trace(self, reason: str) -> None:
        if not self._api_trace:
            return
        tail = list(self._api_trace)[-15:]
        LOGGER.error("MT5 API trace tail before fatal (%s): %s", reason, tail)

    def _mark_fatal(self, reason: str) -> None:
        message = str(reason or "mt5_fatal_error").strip() or "mt5_fatal_error"
        if self._fatal_reason == message:
            return
        self._log_recent_trace(message)
        self._fatal_reason = message
        self.connected = False
        self._next_reconnect_ts = float("inf")
        LOGGER.error("MT5 fatal session guard triggered: %s", message)
        if not self._fatal_reported:
            self._fatal_reported = True
            self._notify_error(f"MT5 fatal session guard triggered: {message}")

    def _handle_ipc_failure(self, context: str, code: int, message: str) -> bool:
        self._ipc_failure_count += 1
        self.connected = False
        self._schedule_next_reconnect(attempt=self._ipc_failure_count)
        self._watchdog_debug(
            "ipc_failure",
            context=context,
            code=code,
            message=message,
            count=self._ipc_failure_count,
        )
        self._trace_api("ipc_failure", "recorded", context=context, code=code, message=message, count=self._ipc_failure_count)
        LOGGER.warning(
            "MT5 IPC instability detected context=%s code=%s count=%s message=%s",
            context,
            code,
            self._ipc_failure_count,
            message,
        )
        if self._ipc_failure_count >= self.max_ipc_failures_before_halt:
            if not self._ipc_threshold_reported:
                self._ipc_threshold_reported = True
                LOGGER.error(
                    "MT5 IPC failures reached threshold (%s). Continuing in degraded mode without process halt.",
                    self.max_ipc_failures_before_halt,
                )
                self._notify_error(
                    "MT5 IPC failures exceeded threshold; entering degraded retry mode (process kept alive)."
                )
        return True

    def _mark_session_healthy(self) -> None:
        if self._ipc_failure_count > 0:
            self._trace_api("session", "healthy_reset", previous_failures=self._ipc_failure_count)
        self._ipc_failure_count = 0
        self._ipc_threshold_reported = False

    @staticmethod
    def _int_or_none(value: Any) -> Optional[int]:
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _normalize_server(value: str) -> str:
        return str(value or "").strip().lower()

    def _validate_account_session(self, terminal: Any, account: Any, context: str) -> bool:
        if terminal is None or account is None:
            return False

        expected_login = self.expected_login if self.expected_login > 0 else None
        expected_server = self._normalize_server(self.expected_server)
        actual_login = self._int_or_none(getattr(account, "login", None))

        actual_server_raw = ""
        for candidate in (getattr(account, "server", None), getattr(terminal, "server", None)):
            text = str(candidate or "").strip()
            if text:
                actual_server_raw = text
                break
        actual_server = self._normalize_server(actual_server_raw)

        if self.account_guard_enabled and (expected_login is not None or bool(expected_server)):
            login_ok = True if expected_login is None else (actual_login == expected_login)
            server_ok = True if not expected_server else (actual_server == expected_server)
            if not login_ok or not server_ok:
                self._mark_fatal(
                    "account_mismatch "
                    f"context={context} expected_login={expected_login} actual_login={actual_login} "
                    f"expected_server='{self.expected_server}' actual_server='{actual_server_raw}'"
                )
                return False

        if self.require_trade_enabled:
            terminal_trade_allowed = getattr(terminal, "trade_allowed", True)
            account_trade_allowed = getattr(account, "trade_allowed", True)
            if terminal_trade_allowed is False or account_trade_allowed is False:
                self._mark_fatal(
                    "automated_trading_disabled "
                    f"context={context} terminal_trade_allowed={terminal_trade_allowed} "
                    f"account_trade_allowed={account_trade_allowed}"
                )
                return False

        return True

    def _schedule_next_reconnect(self, attempt: int = 1) -> None:
        exponent = max(0, int(attempt) - 1)
        delay = self.base_delay_seconds * (2**exponent)
        delay = min(self.max_delay_seconds, max(self.reconnect_cooldown_seconds, delay))
        if self.reconnect_jitter_seconds > 0:
            delay += random.uniform(0.0, self.reconnect_jitter_seconds)
        self._next_reconnect_ts = time.time() + delay
        self._trace_api("reconnect", "scheduled", attempt=attempt, delay_seconds=round(delay, 3))

    def _initialize_terminal_once(self, timeout_ms: Optional[int] = None) -> bool:
        if mt5 is None:
            return False

        init_timeout_ms = self._clamp_timeout_ms(
            timeout_ms if timeout_ms is not None else self.init_timeout_ms_default,
            default=self.init_timeout_ms_default,
            min_value=1000,
            max_value=self.connect_init_timeout_cap_ms,
        )

        self._trace_api("initialize", "start", timeout_ms=init_timeout_ms, attach_only=self.attach_only)
        try:
            attached = bool(mt5.initialize(timeout=init_timeout_ms))
        except Exception as exc:
            self._watchdog_debug("initialize_attach_exception", timeout_ms=init_timeout_ms, error=repr(exc))
            attached = False
        if attached:
            self._trace_api("initialize", "ok_attach")
            LOGGER.info("MT5 attached to running terminal session.")
            return True

        code, message = self._last_error()
        self._trace_api("initialize", "failed_attach", code=code, message=message)
        if self.attach_only and not self.allow_terminal_launch:
            return False

        # Optional fallback (disabled by default): path-based initialization can launch terminal.
        path = str(self.mt5_cfg.get("path", "")).strip()
        kwargs: Dict[str, Any] = {"timeout": init_timeout_ms}
        if path:
            kwargs["path"] = path
        self._trace_api("initialize", "fallback_start", kwargs=kwargs)
        try:
            ok = bool(mt5.initialize(**kwargs))
        except Exception as exc:
            self._watchdog_debug("initialize_fallback_exception", kwargs=kwargs, error=repr(exc))
            ok = False
        if ok:
            self._trace_api("initialize", "ok_fallback")
        else:
            code, message = self._last_error()
            self._trace_api("initialize", "failed_fallback", code=code, message=message)
        return ok

    def _login_once(self) -> bool:
        if self.attach_only or not self.allow_programmatic_login:
            self._trace_api("login", "skipped", attach_only=self.attach_only, allow_programmatic_login=self.allow_programmatic_login)
            return True

        login = self._safe_int(self.mt5_cfg.get("login", 0), 0)
        if login <= 0:
            return True
        password = str(self.mt5_cfg.get("password", "")).strip()
        server = str(self.mt5_cfg.get("server", "")).strip()
        self._trace_api("login", "start", login=login, server=server)
        try:
            ok = bool(mt5.login(login=login, password=password, server=server))
        except Exception as exc:
            self._watchdog_debug("login_exception", login=login, server=server, error=repr(exc))
            ok = False
        if ok:
            self._trace_api("login", "ok", login=login, server=server)
        else:
            code, message = self._last_error()
            self._trace_api("login", "failed", code=code, message=message, login=login, server=server)
        return ok

    def _last_error(self) -> Tuple[int, str]:
        if mt5 is None:
            return -1, "MT5 module missing"
        try:
            err = mt5.last_error()
        except Exception as exc:
            self._watchdog_debug("last_error_call_exception", error=repr(exc))
            return -1, f"last_error exception: {exc}"

        try:
            if isinstance(err, (list, tuple)) and len(err) >= 2:
                return self._safe_int(err[0], -1), str(err[1] or "")
            if isinstance(err, dict):
                return self._safe_int(err.get("code", -1), -1), str(err.get("message", "") or "")
            return -1, str(err or "")
        except Exception as exc:
            self._watchdog_debug("last_error_parse_exception", raw=repr(err), error=repr(exc))
            return -1, "last_error parse failure"

    @staticmethod
    def _is_invalid_comment_error(code: int, message: str) -> bool:
        return int(code) in {-2} and ("comment" in str(message or "").lower())

    def _sanitize_comment(self, value: Any) -> Tuple[str, bool]:
        original = str(value or "")
        text = original
        if self.comment_ascii_only:
            text = text.encode("ascii", errors="ignore").decode("ascii")
        text = re.sub(r"[^A-Za-z0-9 _-]+", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        if len(text) > self.comment_max_len:
            text = text[: self.comment_max_len]
        changed = text != original
        return text, changed

    def _order_send_with_comment_retry(
        self,
        request: Dict[str, Any],
        context: str,
    ) -> Tuple[Any, Dict[str, Any], Optional[int], Optional[str]]:
        req = dict(request or {})
        original_comment = str(req.get("comment", "") or "")
        sanitized_comment, changed = self._sanitize_comment(original_comment)
        req["comment"] = sanitized_comment

        meta = {
            "comment_original": original_comment,
            "comment_sanitized": sanitized_comment,
            "comment_sanitized_changed": bool(changed),
            "comment_retry_path": ["sanitized"],
            "comment_retry_attempts": [],
            "retried_without_comment": False,
            "retry_success": False,
        }

        attempt_no = 1
        res = mt5.order_send(req)
        if res is not None:
            meta["comment_retry_attempts"].append(
                {
                    "attempt": attempt_no,
                    "path": "sanitized",
                    "comment": sanitized_comment,
                    "comment_present": True,
                    "ok": True,
                }
            )
            return res, meta, None, None

        code, message = self._last_error()
        meta["comment_retry_attempts"].append(
            {
                "attempt": attempt_no,
                "path": "sanitized",
                "comment": sanitized_comment,
                "comment_present": True,
                "ok": False,
                "code": int(code) if code is not None else None,
                "message": str(message or ""),
            }
        )

        if not (
            self.retry_without_comment_on_invalid_comment
            and self._is_invalid_comment_error(code, message)
        ):
            return None, meta, code, message

        retry_requests: list[Tuple[str, Dict[str, Any]]] = []
        if sanitized_comment != "":
            retry_without_comment = dict(req)
            retry_without_comment["comment"] = ""
            retry_requests.append(("retry_without_comment", retry_without_comment))

        retry_without_key = dict(req)
        retry_without_key.pop("comment", None)
        if retry_without_key != req:
            retry_requests.append(("retry_remove_comment", retry_without_key))

        for attempt_no, (label, retry_req) in enumerate(retry_requests, start=2):
            meta["comment_retry_path"].append(label)
            retry_res = mt5.order_send(retry_req)
            if retry_res is not None:
                meta["comment_retry_attempts"].append(
                    {
                        "attempt": attempt_no,
                        "path": label,
                        "comment": retry_req.get("comment", ""),
                        "comment_present": "comment" in retry_req,
                        "ok": True,
                    }
                )
                meta["retried_without_comment"] = label != "sanitized"
                meta["retry_success"] = True
                if label == "retry_without_comment":
                    meta["comment_sanitized"] = ""
                return retry_res, meta, None, None

            retry_code, retry_message = self._last_error()
            meta["comment_retry_attempts"].append(
                {
                    "attempt": attempt_no,
                    "path": label,
                    "comment": retry_req.get("comment", ""),
                    "comment_present": "comment" in retry_req,
                    "ok": False,
                    "code": int(retry_code) if retry_code is not None else None,
                    "message": str(retry_message or ""),
                }
            )
            code, message = retry_code, retry_message

        return None, meta, code, message

    def _sleep_backoff(self, attempt: int, max_sleep_seconds: Optional[float] = None) -> None:
        exponent = max(0, attempt - 1)
        delay = min(self.max_delay_seconds, self.base_delay_seconds * (2**exponent))
        if max_sleep_seconds is not None:
            delay = min(delay, max(0.0, float(max_sleep_seconds)))
        if delay <= 0:
            return
        LOGGER.info("MT5 retry backoff: waiting %.1fs", delay)
        try:
            time.sleep(delay)
        except Exception as exc:
            self._watchdog_debug("sleep_backoff_exception", delay=delay, error=repr(exc))

    def connect(self) -> bool:
        if mt5 is None:
            LOGGER.error("MetaTrader5 package is not installed.")
            return False
        if self._fatal_reason:
            LOGGER.error("MT5 connect blocked by fatal session guard: %s", self._fatal_reason)
            return False
        return self._connect_with_attempt_limit(self.max_reconnect_attempts, init_timeout_ms=None)

    def _connect_with_attempt_limit(self, max_attempts: int, init_timeout_ms: Optional[int]) -> bool:
        attempts = max(1, self._safe_int(max_attempts, 1))
        budget_seconds = max(1.0, self._safe_float(self.reconnect_total_budget_seconds, 90.0))
        deadline = time.monotonic() + budget_seconds
        requested_timeout_ms = self._clamp_timeout_ms(
            init_timeout_ms if init_timeout_ms is not None else self.init_timeout_ms_default,
            default=self.init_timeout_ms_default,
            min_value=1000,
            max_value=self.connect_init_timeout_cap_ms,
        )
        last_attempt = 0
        for attempt in range(1, attempts + 1):
            last_attempt = attempt
            remaining_before_attempt = deadline - time.monotonic()
            if remaining_before_attempt <= 0:
                LOGGER.warning(
                    "MT5 connect budget exhausted before attempt (attempts=%s budget=%.1fs)",
                    attempts,
                    budget_seconds,
                )
                self._watchdog_debug(
                    "connect_budget_exhausted_pre_attempt",
                    attempts=attempts,
                    budget_seconds=budget_seconds,
                )
                break
            if remaining_before_attempt < 1.0:
                LOGGER.warning(
                    "MT5 connect budget too low for another initialize cycle (remaining=%.3fs)",
                    remaining_before_attempt,
                )
                self._watchdog_debug(
                    "connect_budget_too_low_pre_attempt",
                    remaining_seconds=remaining_before_attempt,
                )
                break
            effective_init_timeout_ms = self._clamp_timeout_ms(
                min(requested_timeout_ms, int(remaining_before_attempt * 1000)),
                default=requested_timeout_ms,
                min_value=1000,
                max_value=self.connect_init_timeout_cap_ms,
            )

            if self.force_shutdown_on_reconnect and not self.attach_only:
                try:
                    self._trace_api("shutdown", "start", reason="reconnect_attempt")
                    mt5.shutdown()
                    self._trace_api("shutdown", "ok", reason="reconnect_attempt")
                except Exception as exc:
                    self._trace_api("shutdown", "error", reason="reconnect_attempt")
                    self._watchdog_debug("shutdown_exception", attempt=attempt, error=repr(exc))

            try:
                init_ok = self._initialize_terminal_once(timeout_ms=effective_init_timeout_ms)
            except Exception as exc:
                self._watchdog_debug(
                    "connect_initialize_exception",
                    attempt=attempt,
                    timeout_ms=effective_init_timeout_ms,
                    error=repr(exc),
                )
                init_ok = False

            if not init_ok:
                code, message = self._last_error()
                LOGGER.warning("MT5 initialize failed (attempt=%s/%s code=%s msg=%s)", attempt, attempts, code, message)
                self.connected = False
                remaining_for_sleep = deadline - time.monotonic()
                self._sleep_backoff(attempt, max_sleep_seconds=remaining_for_sleep)
                continue

            try:
                login_ok = self._login_once()
            except Exception as exc:
                self._watchdog_debug("connect_login_exception", attempt=attempt, error=repr(exc))
                login_ok = False

            if login_ok:
                try:
                    terminal = mt5.terminal_info()
                    account = mt5.account_info()
                except Exception as exc:
                    self._watchdog_debug("connect_session_probe_exception", attempt=attempt, error=repr(exc))
                    terminal = None
                    account = None
                if terminal is None or account is None:
                    code, message = self._last_error()
                    LOGGER.warning(
                        "MT5 session probe failed after login (attempt=%s/%s code=%s msg=%s)",
                        attempt,
                        attempts,
                        code,
                        message,
                    )
                    self.connected = False
                    remaining_for_sleep = deadline - time.monotonic()
                    self._sleep_backoff(attempt, max_sleep_seconds=remaining_for_sleep)
                    continue

                if not self._validate_account_session(terminal=terminal, account=account, context="connect"):
                    return False

                self.connected = True
                self._mark_session_healthy()
                self._last_heartbeat_ts = time.time()
                self._next_reconnect_ts = 0.0
                LOGGER.info(
                    "MT5 live gateway connected. login=%s server=%s",
                    getattr(account, "login", None),
                    getattr(account, "server", None),
                )
                return True

            code, message = self._last_error()
            LOGGER.warning(
                "MT5 login failed (attempt=%s/%s code=%s msg=%s)",
                attempt,
                attempts,
                code,
                message,
            )
            remaining_for_sleep = deadline - time.monotonic()
            self._sleep_backoff(attempt, max_sleep_seconds=remaining_for_sleep)

        self.connected = False
        if time.monotonic() >= deadline:
            self._watchdog_debug(
                "connect_budget_exhausted_post_loop",
                attempts=attempts,
                budget_seconds=budget_seconds,
            )
        self._schedule_next_reconnect(attempt=max(1, last_attempt))
        return False

    def disconnect(self) -> None:
        self.connected = False
        if mt5 is not None:
            try:
                mt5.shutdown()
            except Exception:
                pass

    def _is_ipc_error(self, code: int, message: str) -> bool:
        normalized_code = self._safe_int(code, -1)
        if normalized_code in self.ipc_timeout_codes:
            return True
        msg = str(message or "").lower()
        return "ipc" in msg or "timeout" in msg or "send failed" in msg or "recv failed" in msg

    def _reconnect(self, reason: str) -> bool:
        if self._fatal_reason:
            return False
        if time.time() < self._next_reconnect_ts:
            return False

        LOGGER.warning("Attempting MT5 reconnect. reason=%s", reason)
        self._notify_system(f"MT5 reconnect triggered: {reason}")
        self.connected = False
        ok = self._connect_with_attempt_limit(
            self.reconnect_attempts_per_cycle,
            init_timeout_ms=self.reconnect_init_timeout_ms,
        )
        if ok:
            self._notify_system("MT5 reconnect successful.")
        else:
            if not self._fatal_reason:
                self._notify_error("MT5 reconnect failed after retries.")
        return ok

    def _ensure_connected(self) -> bool:
        if self._fatal_reason:
            return False
        if not self.connected:
            if time.time() < self._next_reconnect_ts:
                return False
            return self._reconnect("not_connected")
        return True

    def heartbeat(self) -> bool:
        if not self._ensure_connected():
            return False
        if mt5 is None:
            return False

        try:
            self._trace_api("terminal_info", "start", context="heartbeat")
            terminal = mt5.terminal_info()
            self._trace_api("account_info", "start", context="heartbeat")
            account = mt5.account_info()
        except Exception as exc:
            self._watchdog_debug("heartbeat_exception", error=repr(exc))
            terminal = None
            account = None
        if terminal is not None and account is not None:
            if not self._validate_account_session(terminal=terminal, account=account, context="heartbeat"):
                return False
            self.connected = True
            self._mark_session_healthy()
            self._last_heartbeat_ts = time.time()
            return True

        code, message = self._last_error()
        LOGGER.warning("MT5 heartbeat failed. code=%s message=%s", code, message)
        if self._is_ipc_error(code, message):
            self._handle_ipc_failure(context="heartbeat", code=code, message=message)
            return False
        self.connected = False
        self._schedule_next_reconnect(attempt=1)
        return False

    def _heartbeat_if_due(self) -> bool:
        now = time.time()
        interval = self.heartbeat_seconds
        if now - self._last_heartbeat_ts >= interval:
            return self.heartbeat()
        return True

    def account_info(self) -> Dict[str, Any]:
        if not self._ensure_connected() or not self._heartbeat_if_due() or mt5 is None:
            return {}
        try:
            self._trace_api("terminal_info", "start", context="account_info")
            terminal = mt5.terminal_info()
            self._trace_api("account_info", "start", context="account_info")
            info = mt5.account_info()
        except Exception as exc:
            self._watchdog_debug("account_info_probe_exception", error=repr(exc))
            terminal = None
            info = None
        if info is None or terminal is None:
            code, message = self._last_error()
            if self._is_ipc_error(code, message):
                self._handle_ipc_failure(context="account_info", code=code, message=message)
            else:
                self.connected = False
                self._schedule_next_reconnect(attempt=1)
            return {}
        if not self._validate_account_session(terminal=terminal, account=info, context="account_info"):
            return {}
        self._mark_session_healthy()
        self._last_heartbeat_ts = time.time()
        try:
            raw_positions = mt5.positions_get()
        except Exception as exc:
            self._watchdog_debug("account_info_positions_exception", error=repr(exc))
            raw_positions = None
            code, message = self._last_error()
            if self._is_ipc_error(code, message):
                self._handle_ipc_failure(context="account_info:positions_get", code=code, message=message)
            else:
                self.connected = False
                self._schedule_next_reconnect(attempt=1)
            return {}
        open_positions = 0 if raw_positions is None else len(list(raw_positions))
        balance = float(getattr(info, "balance", 0.0) or 0.0)
        equity = float(getattr(info, "equity", 0.0) or 0.0)
        return {
            "balance": balance,
            "equity": equity,
            "margin": float(getattr(info, "margin", 0.0) or 0.0),
            "free_margin": float(getattr(info, "margin_free", 0.0) or 0.0),
            "floating_pnl": 0.0 if open_positions == 0 else equity - balance,
            "open_positions": open_positions,
            "currency": str(getattr(info, "currency", "") or ""),
            "login": self._int_or_none(getattr(info, "login", None)),
            "server": str(getattr(info, "server", "") or getattr(terminal, "server", "") or ""),
            "trade_allowed": bool(getattr(terminal, "trade_allowed", True)),
        }

    def get_latest_price(self, symbol: str) -> Optional[float]:
        if not self._ensure_connected() or mt5 is None:
            return None
        if not self._ensure_symbol_ready(symbol):
            return None
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return None
        # Prefer last traded price if available, otherwise midpoint.
        for candidate in (getattr(tick, "last", None),):
            try:
                value = float(candidate)
            except (TypeError, ValueError):
                value = 0.0
            if value > 0 and math.isfinite(value):
                return value
        try:
            bid = float(getattr(tick, "bid", 0.0) or 0.0)
            ask = float(getattr(tick, "ask", 0.0) or 0.0)
        except (TypeError, ValueError):
            return None
        if bid > 0 and ask > 0 and math.isfinite(bid) and math.isfinite(ask):
            return (bid + ask) / 2.0
        value = bid if bid > 0 else ask
        if value > 0 and math.isfinite(value):
            return value
        return None

    def get_live_spread(self, symbol: str) -> Optional[float]:
        if not self._ensure_connected() or mt5 is None:
            return None
        if not self._ensure_symbol_ready(symbol):
            return None
        info = mt5.symbol_info(symbol)
        if info is None:
            return None
        return float(getattr(info, "spread", 0.0))

    def _ensure_symbol_ready(self, symbol: str) -> bool:
        if mt5 is None:
            return False
        try:
            info = mt5.symbol_info(symbol)
            if info is None:
                self._trace_api("symbol_info", "missing", symbol=symbol)
                code, message = self._last_error()
                if self._is_ipc_error(code, message):
                    self._handle_ipc_failure(context=f"ensure_symbol_ready:{symbol}", code=code, message=message)
                return False
            if not bool(getattr(info, "visible", True)):
                self._trace_api("symbol_select", "start", symbol=symbol, select=True)
                selected = mt5.symbol_select(symbol, True)
                self._trace_api("symbol_select", "result", symbol=symbol, selected=bool(selected))
                if not selected:
                    code, message = self._last_error()
                    if self._is_ipc_error(code, message):
                        self._handle_ipc_failure(context=f"ensure_symbol_ready:{symbol}", code=code, message=message)
                    return False
                time.sleep(0.25)
                info = mt5.symbol_info(symbol)
                if info is None or not bool(getattr(info, "visible", True)):
                    self._trace_api("symbol_info", "still_invisible", symbol=symbol)
                    code, message = self._last_error()
                    if self._is_ipc_error(code, message):
                        self._handle_ipc_failure(context=f"ensure_symbol_ready:{symbol}", code=code, message=message)
                    return False
            return True
        except Exception as exc:
            self._watchdog_debug("ensure_symbol_ready_exception", symbol=symbol, error=repr(exc))
            code, message = self._last_error()
            if self._is_ipc_error(code, message):
                self._handle_ipc_failure(context=f"ensure_symbol_ready:{symbol}", code=code, message=message)
            else:
                self.connected = False
                self._schedule_next_reconnect(attempt=1)
            return False

    def fetch_bars(self, symbol: str, timeframe: str, bars: int) -> Optional[pd.DataFrame]:
        if not self._ensure_connected() or not self._heartbeat_if_due() or mt5 is None:
            return None
        tf_attr = getattr(mt5, timeframe, None)
        if tf_attr is None:
            LOGGER.error("Invalid timeframe: %s", timeframe)
            return None
        if not self._ensure_symbol_ready(symbol):
            # Silence warning for nasdaq stocks outside US hours
            nasdaq_symbols = [s.upper() for s in (self.config.get("nasdaq_universe", []) or [])]
            if symbol.upper() in nasdaq_symbols:
                now_utc = datetime.now(timezone.utc)
                if now_utc.hour < 13 or now_utc.hour >= 22:
                    return None
            LOGGER.warning("Symbol not ready for market data. symbol=%s timeframe=%s", symbol, timeframe)
            return None
        try:
            self._trace_api("copy_rates_from_pos", "start", symbol=symbol, timeframe=timeframe, bars=bars)
            rates = mt5.copy_rates_from_pos(symbol, tf_attr, 0, bars)
            if rates is None or len(rates) == 0:
                code, message = self._last_error()
                self._trace_api("copy_rates_from_pos", "empty", symbol=symbol, timeframe=timeframe, code=code, message=message)
                if self._is_ipc_error(code, message):
                    self._handle_ipc_failure(context=f"fetch_bars:{symbol}", code=code, message=message)
                return None
            df = pd.DataFrame(rates)
            df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
            self._mark_session_healthy()
            self._trace_api("copy_rates_from_pos", "ok", symbol=symbol, timeframe=timeframe, rows=len(df))
            return df
        except Exception as exc:
            self._watchdog_debug("fetch_bars_exception", symbol=symbol, timeframe=timeframe, error=repr(exc))
            code, message = self._last_error()
            if self._is_ipc_error(code, message):
                self._handle_ipc_failure(context=f"fetch_bars:{symbol}", code=code, message=message)
            return None

    def get_positions(self, symbol: Optional[str] = None) -> List[Position]:
        if not self._ensure_connected() or not self._heartbeat_if_due() or mt5 is None:
            return []
        try:
            kwargs = {}
            if symbol:
                kwargs["symbol"] = symbol
            self._trace_api("positions_get", "start", symbol=symbol or "*")
            raw = mt5.positions_get(**kwargs)
            if raw is None:
                code, message = self._last_error()
                self._trace_api("positions_get", "none", symbol=symbol or "*", code=code, message=message)
                if self._is_ipc_error(code, message):
                    self._handle_ipc_failure(context=f"get_positions:{symbol or '*'}", code=code, message=message)
                return []
            out = []
            for p in raw:
                side = Side.BUY if p.type == mt5.POSITION_TYPE_BUY else Side.SELL
                out.append(
                    Position(
                        ticket=int(p.ticket),
                        symbol=str(p.symbol),
                        side=side,
                        volume=float(p.volume),
                        price_open=float(p.price_open),
                        sl=float(p.sl) if p.sl else None,
                        tp=float(p.tp) if p.tp else None,
                        comment=str(p.comment),
                        magic=int(p.magic),
                        time_open_utc=datetime.fromtimestamp(int(getattr(p, "time", 0)) - self.server_offset_seconds, tz=timezone.utc)
                        if int(getattr(p, "time", 0) or 0) > 0
                        else None,
                        metadata={
                            "floating_pnl": float(getattr(p, "profit", 0.0) or 0.0),
                            "price_current": float(getattr(p, "price_current", 0.0) or 0.0),
                            "swap": float(getattr(p, "swap", 0.0) or 0.0),
                            "commission": float(getattr(p, "commission", 0.0) or 0.0),
                        },
                    )
                )
            self._mark_session_healthy()
            self._trace_api("positions_get", "ok", symbol=symbol or "*", count=len(out))
            return out
        except Exception as exc:
            self._watchdog_debug("get_positions_exception", symbol=symbol or "*", error=repr(exc))
            code, message = self._last_error()
            if self._is_ipc_error(code, message):
                self._handle_ipc_failure(context=f"get_positions:{symbol or '*'}", code=code, message=message)
            return []

    def submit_order(self, intent: OrderIntent) -> OrderResult:
        blocked = self._blocked_order_result("submit_order", symbol=intent.symbol)
        if blocked is not None:
            return blocked
        if not self._ensure_connected() or mt5 is None:
            return OrderResult(ok=False, status="ERROR", message="Not connected")

        # Mapping sides
        order_type = mt5.ORDER_TYPE_BUY if intent.side == Side.BUY else mt5.ORDER_TYPE_SELL
        tick = mt5.symbol_info_tick(intent.symbol)
        if tick is None:
            code, message = self._last_error()
            if self._is_ipc_error(code, message):
                self._handle_ipc_failure(context=f"submit_order:{intent.symbol}", code=code, message=message)
            return OrderResult(ok=False, status="ERROR", message="symbol_info_tick returned None")
        price = tick.ask if intent.side == Side.BUY else tick.bid

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": intent.symbol,
            "volume": float(intent.volume),
            "type": order_type,
            "price": float(price),
            "magic": int(intent.magic or 0),
            "comment": str(intent.comment or ""),
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        if intent.sl:
            request["sl"] = float(intent.sl)
        if intent.tp:
            request["tp"] = float(intent.tp)

        try:
            self._trace_api("order_send", "start", symbol=intent.symbol, volume=float(intent.volume), side=intent.side.value)
            res, send_meta, err_code, err_message = self._order_send_with_comment_retry(
                request=request,
                context=f"submit_order:{intent.symbol}",
            )
            if res is None:
                code = int(err_code or -1)
                message = str(err_message or "")
                if self._is_ipc_error(code, message):
                    self._handle_ipc_failure(context=f"submit_order:{intent.symbol}", code=code, message=message)
                return OrderResult(
                    ok=False,
                    status="ERROR",
                    message=f"order_send returned None (code={code} msg={message})",
                    retcode=code,
                    raw={"_meta": send_meta},
                )

            self._trace_api(
                "order_send",
                "result",
                symbol=intent.symbol,
                retcode=int(getattr(res, "retcode", -1)),
                comment=str(getattr(res, "comment", "")),
            )
            ok = res.retcode == mt5.TRADE_RETCODE_DONE
            status = "FILLED" if ok else "REJECTED"
            return OrderResult(
                ok=ok,
                status=status,
                message=res.comment,
                ticket=getattr(res, "order", None),
                retcode=int(res.retcode),
                filled_price=float(res.price) if res.price else None,
                raw={
                    **(res._asdict() if hasattr(res, "_asdict") else {"raw": str(res)}),
                    "_meta": send_meta,
                },
            )
        except Exception as e:
            self._watchdog_debug("submit_order_exception", symbol=intent.symbol, error=repr(e))
            code, message = self._last_error()
            if self._is_ipc_error(code, message):
                self._handle_ipc_failure(context=f"submit_order:{intent.symbol}", code=code, message=message)
            return OrderResult(ok=False, status="ERROR", message=str(e))

    def close_position(self, position: Position, reason: str = "") -> OrderResult:
        blocked = self._blocked_order_result(
            "close_position",
            symbol=position.symbol,
            ticket=int(position.ticket),
            reason=str(reason or ""),
        )
        if blocked is not None:
            return blocked
        if not self._ensure_connected() or mt5 is None:
            return OrderResult(ok=False, status="ERROR", message="Not connected")

        order_type = mt5.ORDER_TYPE_SELL if position.side == Side.BUY else mt5.ORDER_TYPE_BUY
        tick = mt5.symbol_info_tick(position.symbol)
        if tick is None:
            code, message = self._last_error()
            if self._is_ipc_error(code, message):
                self._handle_ipc_failure(context=f"close_position:{position.symbol}", code=code, message=message)
            return OrderResult(ok=False, status="ERROR", message="symbol_info_tick returned None")
        price = tick.bid if position.side == Side.BUY else tick.ask

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": position.symbol,
            "volume": float(position.volume),
            "type": order_type,
            "position": int(position.ticket),
            "price": float(price),
            "magic": int(position.magic),
            # Keep close request comment empty for maximum broker compatibility.
            "comment": "",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        try:
            self._trace_api("order_send", "start_close", symbol=position.symbol, ticket=int(position.ticket), volume=float(position.volume))
            res, send_meta, err_code, err_message = self._order_send_with_comment_retry(
                request=request,
                context=f"close_position:{position.symbol}",
            )
            if res is None:
                code = int(err_code or -1)
                message = str(err_message or "")
                if self._is_ipc_error(code, message):
                    self._handle_ipc_failure(context=f"close_position:{position.symbol}", code=code, message=message)
                return OrderResult(
                    ok=False,
                    status="ERROR",
                    message=f"order_send returned None (code={code} msg={message})",
                    retcode=code,
                    raw={"_meta": send_meta},
                )
            self._trace_api(
                "order_send",
                "result_close",
                symbol=position.symbol,
                retcode=int(getattr(res, "retcode", -1)),
                comment=str(getattr(res, "comment", "")),
            )
            ok = res.retcode == mt5.TRADE_RETCODE_DONE
            pnl_value: Optional[float] = None
            if ok:
                try:
                    deals = mt5.history_deals_get(position=int(position.ticket))
                except Exception:
                    deals = None
                if deals:
                    total = 0.0
                    found = False
                    for deal in deals:
                        profit = float(getattr(deal, "profit", 0.0) or 0.0)
                        swap = float(getattr(deal, "swap", 0.0) or 0.0)
                        commission = float(getattr(deal, "commission", 0.0) or 0.0)
                        fee = float(getattr(deal, "fee", 0.0) or 0.0)
                        total += profit + swap + commission + fee
                        found = True
                    if found and math.isfinite(total):
                        pnl_value = total
            return OrderResult(
                ok=ok,
                status="CLOSED" if ok else "REJECTED",
                message=res.comment,
                ticket=getattr(res, "order", None),
                retcode=int(res.retcode),
                filled_price=float(res.price) if res.price else None,
                pnl=pnl_value,
                raw={
                    **(res._asdict() if hasattr(res, "_asdict") else {"raw": str(res)}),
                    "_meta": send_meta,
                },
            )
        except Exception as e:
            self._watchdog_debug("close_position_exception", symbol=position.symbol, ticket=position.ticket, error=repr(e))
            code, message = self._last_error()
            if self._is_ipc_error(code, message):
                self._handle_ipc_failure(context=f"close_position:{position.symbol}", code=code, message=message)
            return OrderResult(ok=False, status="ERROR", message=str(e))

    def get_position_close_info(self, ticket: int) -> Optional[Dict[str, Any]]:
        """
        Best-effort close summary for a position ticket using MT5 deal history.
        This is required to reconcile broker-side SL/TP closes that bypass bot close_position().
        """
        if not self._ensure_connected() or mt5 is None:
            return None
        try:
            deals = mt5.history_deals_get(position=int(ticket))
        except Exception:
            deals = None
        if not deals:
            return None

        total = 0.0
        last_price: Optional[float] = None
        last_time_utc: Optional[str] = None
        close_reason: Optional[str] = None
        # Prefer OUT deals for exit price/time, but still sum all costs.
        for deal in deals:
            profit = float(getattr(deal, "profit", 0.0) or 0.0)
            swap = float(getattr(deal, "swap", 0.0) or 0.0)
            commission = float(getattr(deal, "commission", 0.0) or 0.0)
            fee = float(getattr(deal, "fee", 0.0) or 0.0)
            total += profit + swap + commission + fee

            entry = getattr(deal, "entry", None)  # DEAL_ENTRY_IN/OUT/INOUT
            price = getattr(deal, "price", None)
            if price is not None:
                try:
                    price_f = float(price)
                except Exception:
                    price_f = None
            else:
                price_f = None

            # MT5 times can be int seconds; keep string ISO for runtime logs.
            t = getattr(deal, "time", None)
            if isinstance(t, (int, float)) and t > 0:
                try:
                    dt = datetime.fromtimestamp(float(t), tz=timezone.utc)
                    last_time_utc = dt.isoformat()
                except Exception:
                    pass

            if entry == getattr(mt5, "DEAL_ENTRY_OUT", None):
                if price_f is not None:
                    last_price = price_f
                r = getattr(deal, "reason", None)
                if r is not None:
                    close_reason = str(r)

        if not math.isfinite(total):
            return None
        return {
            "ticket": int(ticket),
            "pnl": float(total),
            "exit_price": last_price,
            "close_time_utc": last_time_utc,
            "close_reason": close_reason,
        }

    def fetch_ticks(
        self,
        symbol: str,
        from_dt: datetime,
        to_dt: datetime,
        max_ticks: int = 2000,
    ) -> List[MarketTick]:
        if not self._ensure_connected() or mt5 is None:
            return []
        symbol_text = str(symbol or "").strip()
        if not symbol_text:
            return []
        if not self._ensure_symbol_ready(symbol_text):
            return []

        def _field(rec: Any, name: str) -> Any:
            try:
                return rec[name]
            except Exception:
                return getattr(rec, name, None)

        ticks: List[MarketTick] = []
        raw = None
        try:
            flags = getattr(mt5, "COPY_TICKS_ALL", 0)
            raw = mt5.copy_ticks_range(symbol_text, from_dt, to_dt, flags)
        except Exception:
            raw = None

        if raw is not None:
            try:
                seq = list(raw)
            except Exception:
                seq = []
            try:
                limit = max(1, int(max_ticks))
            except Exception:
                limit = 2000
            if limit > 0 and len(seq) > limit:
                seq = seq[-limit:]
            for rec in seq:
                time_msc = _field(rec, "time_msc")
                time_sec = _field(rec, "time")
                dt = None
                dt_msc: Optional[int] = None
                if isinstance(time_msc, (int, float)) and float(time_msc) > 0:
                    try:
                        dt_msc = int(time_msc)
                        dt = datetime.fromtimestamp(float(time_msc) / 1000.0, tz=timezone.utc)
                    except Exception:
                        dt = None
                        dt_msc = None
                elif isinstance(time_sec, (int, float)) and float(time_sec) > 0:
                    try:
                        dt = datetime.fromtimestamp(float(time_sec), tz=timezone.utc)
                        dt_msc = int(float(time_sec) * 1000.0)
                    except Exception:
                        dt = None
                        dt_msc = None
                if dt is None:
                    continue

                bid_raw = _field(rec, "bid")
                ask_raw = _field(rec, "ask")
                last_raw = _field(rec, "last")
                vol_raw = _field(rec, "volume_real")
                if vol_raw is None:
                    vol_raw = _field(rec, "volume")
                flags_raw = _field(rec, "flags")

                def _to_float(value: Any) -> Optional[float]:
                    try:
                        out = float(value)
                    except (TypeError, ValueError):
                        return None
                    return out if math.isfinite(out) else None

                bid = _to_float(bid_raw)
                ask = _to_float(ask_raw)
                last = _to_float(last_raw)
                vol = _to_float(vol_raw)
                try:
                    flags_val = int(flags_raw) if flags_raw is not None else None
                except Exception:
                    flags_val = None

                ticks.append(
                    MarketTick(
                        time_utc=dt,
                        bid=bid,
                        ask=ask,
                        last=last,
                        volume=vol,
                        time_msc=dt_msc,
                        flags=flags_val,
                    )
                )
            return ticks

        # Fallback: single tick snapshot.
        try:
            snap = mt5.symbol_info_tick(symbol_text)
        except Exception:
            snap = None
        if snap is None:
            return []

        try:
            snap_time_msc = getattr(snap, "time_msc", None)
        except Exception:
            snap_time_msc = None
        try:
            snap_time = getattr(snap, "time", None)
        except Exception:
            snap_time = None

        dt = None
        dt_msc = None
        if isinstance(snap_time_msc, (int, float)) and float(snap_time_msc) > 0:
            try:
                dt_msc = int(snap_time_msc)
                dt = datetime.fromtimestamp(float(snap_time_msc) / 1000.0, tz=timezone.utc)
            except Exception:
                dt = None
                dt_msc = None
        elif isinstance(snap_time, (int, float)) and float(snap_time) > 0:
            try:
                dt = datetime.fromtimestamp(float(snap_time), tz=timezone.utc)
                dt_msc = int(float(snap_time) * 1000.0)
            except Exception:
                dt = None
                dt_msc = None
        if dt is None:
            dt = datetime.now(timezone.utc)
        bid = getattr(snap, "bid", None)
        ask = getattr(snap, "ask", None)
        last = getattr(snap, "last", None)
        vol = getattr(snap, "volume", None)
        try:
            bid_f = float(bid) if bid is not None else None
        except Exception:
            bid_f = None
        try:
            ask_f = float(ask) if ask is not None else None
        except Exception:
            ask_f = None
        try:
            last_f = float(last) if last is not None else None
        except Exception:
            last_f = None
        try:
            vol_f = float(vol) if vol is not None else None
        except Exception:
            vol_f = None
        return [
            MarketTick(
                time_utc=dt,
                bid=bid_f,
                ask=ask_f,
                last=last_f,
                volume=vol_f,
                time_msc=dt_msc,
            )
        ]

    def get_symbol_constraints(self, symbol: str) -> Optional[SymbolConstraints]:
        if not self._ensure_connected() or mt5 is None:
            return None
        if not self._ensure_symbol_ready(symbol):
            return None
        info = mt5.symbol_info(symbol)
        if info is None:
            return None
        # Map MT5 symbol fields onto our SymbolConstraints schema.
        point = getattr(info, "point", None)
        if point is None:
            point = getattr(info, "trade_tick_size", 0.0)

        return SymbolConstraints(
            min_volume=float(getattr(info, "volume_min", 0.01) or 0.01),
            max_volume=float(getattr(info, "volume_max", 100.0) or 100.0),
            volume_step=float(getattr(info, "volume_step", 0.01) or 0.01),
            point=float(point or 0.0),
            digits=int(getattr(info, "digits", 0) or 0),
            contract_size=float(getattr(info, "trade_contract_size", 1.0) or 1.0),
            base_currency=str(getattr(info, "currency_base", "") or ""),
            quote_currency=str(getattr(info, "currency_profit", "") or ""),
            profit_currency=str(getattr(info, "currency_profit", "") or ""),
            trade_stops_level=self._safe_float(getattr(info, 'trade_stops_level', None), 0.0),
            trade_freeze_level=self._safe_float(getattr(info, 'trade_freeze_level', None), 0.0),
        )

    def precheck_order(self, intent: OrderIntent) -> OrderResult:
        blocked = self._blocked_order_result("precheck_order", symbol=intent.symbol)
        if blocked is not None:
            return blocked
        if not self._ensure_connected() or mt5 is None:
            return OrderResult(ok=False, status="ERROR", message="Not connected")
        try:
            order_type = mt5.ORDER_TYPE_BUY if intent.side == Side.BUY else mt5.ORDER_TYPE_SELL
            tick = mt5.symbol_info_tick(intent.symbol)
            if tick is None:
                code, message = self._last_error()
                if self._is_ipc_error(code, message):
                    self._handle_ipc_failure(context=f"precheck_order:{intent.symbol}", code=code, message=message)
                return OrderResult(ok=False, status="ERROR", message="symbol_info_tick returned None")

            # Spread Guard (Monday/News Protection)
            # Default 500 points (e.g. 50 pips for 5-digit forex, or 50 cents for gold)
            spread = getattr(tick, "spread", 0)
            max_spread = self._safe_int(self.mt5_cfg.get("max_spread_points", 500), 500)
            if max_spread > 0 and spread > max_spread:
                return OrderResult(
                    ok=False,
                    status="REJECTED_SPREAD",
                    message=f"Current spread {spread} exceeds limit {max_spread}"
                )

            price = tick.ask if intent.side == Side.BUY else tick.bid
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": intent.symbol,
                "volume": float(intent.volume),
                "type": order_type,
                "price": float(price),
                "magic": int(intent.magic or 0),
                "comment": str(intent.comment or ""),
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            sanitized_comment, comment_changed = self._sanitize_comment(request.get("comment", ""))
            request["comment"] = sanitized_comment
            if intent.sl:
                request["sl"] = float(intent.sl)
            if intent.tp:
                request["tp"] = float(intent.tp)
            self._trace_api("order_check", "start", symbol=intent.symbol, volume=float(intent.volume), side=intent.side.value)
            check = mt5.order_check(request)
            if check is None:
                code, message = self._last_error()
                if self._is_ipc_error(code, message):
                    self._handle_ipc_failure(context=f"precheck_order:{intent.symbol}", code=code, message=message)
                return OrderResult(
                    ok=False,
                    status="ERROR",
                    message=f"order_check returned None (code={code} msg={message})",
                    retcode=int(code),
                    raw={
                        "_meta": {
                            "comment_original": str(intent.comment or ""),
                            "comment_sanitized": str(sanitized_comment or ""),
                            "comment_sanitized_changed": bool(comment_changed),
                            "retried_without_comment": False,
                            "retry_success": False,
                        }
                    },
                )
            self._trace_api(
                "order_check",
                "result",
                symbol=intent.symbol,
                retcode=int(getattr(check, "retcode", -1)),
                comment=str(getattr(check, "comment", "")),
            )
            ok = int(getattr(check, "retcode", -1)) in (0, mt5.TRADE_RETCODE_DONE, mt5.TRADE_RETCODE_PLACED)
            return OrderResult(
                ok=ok,
                status="OK" if ok else "REJECTED",
                message=str(getattr(check, "comment", "")),
                retcode=int(getattr(check, "retcode", -1)),
                raw={
                    **(check._asdict() if hasattr(check, "_asdict") else {"raw": str(check)}),
                    "_meta": {
                        "comment_original": str(intent.comment or ""),
                        "comment_sanitized": str(sanitized_comment or ""),
                        "comment_sanitized_changed": bool(comment_changed),
                        "retried_without_comment": False,
                        "retry_success": False,
                    },
                },
            )
        except Exception as e:
            self._watchdog_debug("precheck_order_exception", symbol=intent.symbol, error=repr(e))
            code, message = self._last_error()
            if self._is_ipc_error(code, message):
                self._handle_ipc_failure(context=f"precheck_order:{intent.symbol}", code=code, message=message)
            return OrderResult(ok=False, status="ERROR", message=str(e))

    def send_order(self, intent: OrderIntent) -> OrderResult:
        return self.submit_order(intent)

    def modify_position_sl_tp(
        self,
        position: Position,
        sl: Optional[float],
        tp: Optional[float],
        reason: str,
    ) -> OrderResult:
        blocked = self._blocked_order_result(
            "modify_position_sl_tp",
            symbol=position.symbol,
            ticket=int(position.ticket),
            reason=str(reason or ""),
        )
        if blocked is not None:
            return blocked
        if not self._ensure_connected() or mt5 is None:
            return OrderResult(ok=False, status="ERROR", message="Not connected")
        try:
            request = {
                "action": mt5.TRADE_ACTION_SLTP,
                "symbol": position.symbol,
                "position": int(position.ticket),
                "sl": float(sl) if sl is not None else 0.0,
                "tp": float(tp) if tp is not None else 0.0,
                # Keep modify request comment empty for maximum broker compatibility.
                "comment": "",
            }
            self._trace_api("order_send", "start_sltp", symbol=position.symbol, ticket=int(position.ticket))
            res, send_meta, err_code, err_message = self._order_send_with_comment_retry(
                request=request,
                context=f"modify_position_sl_tp:{position.symbol}",
            )
            if res is None:
                code = int(err_code or -1)
                message = str(err_message or "")
                if self._is_ipc_error(code, message):
                    self._handle_ipc_failure(context=f"modify_position_sl_tp:{position.symbol}", code=code, message=message)
                return OrderResult(
                    ok=False,
                    status="ERROR",
                    message=f"order_send returned None (code={code} msg={message})",
                    retcode=code,
                    raw={"_meta": send_meta},
                )
            self._trace_api(
                "order_send",
                "result_sltp",
                symbol=position.symbol,
                retcode=int(getattr(res, "retcode", -1)),
                comment=str(getattr(res, "comment", "")),
            )
            ok = res.retcode == mt5.TRADE_RETCODE_DONE
            return OrderResult(
                ok=ok,
                status="MODIFIED" if ok else "REJECTED",
                message=res.comment,
                ticket=getattr(res, "order", None),
                retcode=int(res.retcode),
                filled_price=float(res.price) if getattr(res, "price", None) else None,
                raw={
                    **(res._asdict() if hasattr(res, "_asdict") else {"raw": str(res)}),
                    "_meta": send_meta,
                },
            )
        except Exception as e:
            self._watchdog_debug(
                "modify_position_sl_tp_exception",
                symbol=position.symbol,
                ticket=position.ticket,
                error=repr(e),
            )
            code, message = self._last_error()
            if self._is_ipc_error(code, message):
                self._handle_ipc_failure(context=f"modify_position_sl_tp:{position.symbol}", code=code, message=message)
            return OrderResult(ok=False, status="ERROR", message=str(e))

    def close_all_positions(self, reason: str) -> List[OrderResult]:
        blocked = self._blocked_order_result("close_all_positions", reason=str(reason or ""))
        if blocked is not None:
            return [blocked]
        results: List[OrderResult] = []
        positions = self.get_positions(symbol=None)
        for p in positions:
            results.append(self.close_position(p, reason=reason))
        return results
