from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo


class ExecutionChurnGuard:
    def __init__(self, config: Dict[str, Any], snapshot: Optional[Dict[str, Any]] = None) -> None:
        self.enabled = True
        self.reentry_cooldown_seconds = 180.0
        self.flip_reentry_cooldown_seconds = 30.0
        self.max_entries_per_symbol_per_hour = 4
        self.max_entries_per_symbol_per_day = 12
        self.max_entries_global_per_day = 10
        self.daily_reset_timezone = "UTC"
        self.min_hold_bars_floor = 2
        self._min_hold_bars_floor_by_symbol: Dict[str, int] = {}
        self.tiny_pnl_threshold_usd = 2.0
        self.quick_exit_window_seconds = 300.0
        self.tiny_pnl_max_count_per_hour = 2
        self.tiny_pnl_cooldown_seconds = 3600.0
        self.protection_failure_lock_seconds = 180.0
        self.loss_reentry_lock_seconds = 180.0
        self.protection_retry_interval_seconds = 2.0
        self.protection_retry_max_attempts = 8
        self._entry_timestamps_by_symbol: Dict[str, List[float]] = {}
        self._daily_entries_by_symbol: Dict[str, Dict[str, int]] = {}
        self._global_daily_counts: Dict[str, int] = {}
        self._per_symbol_daily_limits: Dict[str, int] = {}
        self._tiny_pnl_exit_timestamps_by_symbol: Dict[str, List[float]] = {}
        self._cooldown_until_by_symbol: Dict[str, float] = {}
        self._cooldown_reason_by_symbol: Dict[str, str] = {}

        self.update_config(config)
        self._restore_snapshot(snapshot or {})

    def update_config(self, config: Dict[str, Any]) -> None:
        cfg = dict(config or {})
        self.enabled = bool(cfg.get("enabled", True))
        try:
            self.reentry_cooldown_seconds = max(0.0, float(cfg.get("reentry_cooldown_seconds", 180.0)))
        except (TypeError, ValueError):
            self.reentry_cooldown_seconds = 180.0
        try:
            self.flip_reentry_cooldown_seconds = max(0.0, float(cfg.get("flip_reentry_cooldown_seconds", 30.0)))
        except (TypeError, ValueError):
            self.flip_reentry_cooldown_seconds = 30.0
        try:
            self.max_entries_per_symbol_per_hour = max(1, int(cfg.get("max_entries_per_symbol_per_hour", 4)))
        except (TypeError, ValueError):
            self.max_entries_per_symbol_per_hour = 4
        try:
            self.max_entries_per_symbol_per_day = max(1, int(cfg.get("max_entries_per_symbol_per_day", 12)))
        except (TypeError, ValueError):
            self.max_entries_per_symbol_per_day = 12
        try:
            self.max_entries_global_per_day = max(1, int(cfg.get("max_entries_global_per_day", 10)))
        except (TypeError, ValueError):
            self.max_entries_global_per_day = 10
        tz = str(cfg.get("daily_reset_timezone", "UTC") or "UTC").strip()
        try:
            ZoneInfo(tz)
            self.daily_reset_timezone = tz
        except Exception:
            self.daily_reset_timezone = "UTC"
        try:
            self.min_hold_bars_floor = max(1, int(cfg.get("min_hold_bars_floor", 2)))
        except (TypeError, ValueError):
            self.min_hold_bars_floor = 2
        raw_min_hold_by_symbol = cfg.get("min_hold_bars_floor_by_symbol", {})
        parsed_min_hold_by_symbol: Dict[str, int] = {}
        if isinstance(raw_min_hold_by_symbol, dict):
            for symbol, value in raw_min_hold_by_symbol.items():
                key = str(symbol or "").strip().upper()
                if not key:
                    continue
                try:
                    parsed_min_hold_by_symbol[key] = max(1, int(value))
                except (TypeError, ValueError):
                    continue
        self._min_hold_bars_floor_by_symbol = parsed_min_hold_by_symbol
        try:
            self.tiny_pnl_threshold_usd = max(0.0, float(cfg.get("tiny_pnl_threshold_usd", 2.0)))
        except (TypeError, ValueError):
            self.tiny_pnl_threshold_usd = 2.0
        try:
            self.quick_exit_window_seconds = max(1.0, float(cfg.get("quick_exit_window_seconds", 300.0)))
        except (TypeError, ValueError):
            self.quick_exit_window_seconds = 300.0
        try:
            self.tiny_pnl_max_count_per_hour = max(1, int(cfg.get("tiny_pnl_max_count_per_hour", 2)))
        except (TypeError, ValueError):
            self.tiny_pnl_max_count_per_hour = 2
        try:
            self.tiny_pnl_cooldown_seconds = max(1.0, float(cfg.get("tiny_pnl_cooldown_seconds", 3600.0)))
        except (TypeError, ValueError):
            self.tiny_pnl_cooldown_seconds = 3600.0
        try:
            self.protection_failure_lock_seconds = max(
                1.0, float(cfg.get("protection_failure_lock_seconds", 180.0))
            )
        except (TypeError, ValueError):
            self.protection_failure_lock_seconds = 180.0
        try:
            self.loss_reentry_lock_seconds = max(1.0, float(cfg.get("loss_reentry_lock_seconds", 180.0)))
        except (TypeError, ValueError):
            self.loss_reentry_lock_seconds = 180.0
        try:
            self.protection_retry_interval_seconds = max(
                0.1, float(cfg.get("protection_retry_interval_seconds", 2.0))
            )
        except (TypeError, ValueError):
            self.protection_retry_interval_seconds = 2.0
        try:
            self.protection_retry_max_attempts = max(1, int(cfg.get("protection_retry_max_attempts", 8)))
        except (TypeError, ValueError):
            self.protection_retry_max_attempts = 8
        per_symbol = cfg.get("per_symbol_daily_limits", {})
        parsed: Dict[str, int] = {}
        if isinstance(per_symbol, dict):
            for symbol, value in per_symbol.items():
                key = str(symbol or "").strip().upper()
                if not key:
                    continue
                try:
                    parsed[key] = max(0, int(value))
                except (TypeError, ValueError):
                    continue
        # Compatibility key requested by user.
        try:
            eth_limit = int(cfg.get("max_entries_per_symbol_per_day_eth", 0) or 0)
            if eth_limit > 0:
                parsed["ETHUSD"] = eth_limit
        except (TypeError, ValueError):
            pass
        self._per_symbol_daily_limits = parsed

    def _restore_snapshot(self, snapshot: Dict[str, Any]) -> None:
        raw = snapshot.get("entry_timestamps_by_symbol")
        if not isinstance(raw, dict):
            raw = {}
        restored_ts: Dict[str, List[float]] = {}
        for symbol, values in raw.items():
            if not isinstance(values, list):
                continue
            parsed = []
            for value in values:
                try:
                    parsed.append(float(value))
                except (TypeError, ValueError):
                    continue
            restored_ts[str(symbol).upper()] = parsed
        self._entry_timestamps_by_symbol = restored_ts

        daily = snapshot.get("daily_entries_by_symbol")
        if not isinstance(daily, dict):
            daily = {}
        restored_daily: Dict[str, Dict[str, int]] = {}
        for symbol, day_map in daily.items():
            if not isinstance(day_map, dict):
                continue
            clean_map: Dict[str, int] = {}
            for day_key, count in day_map.items():
                try:
                    clean_map[str(day_key)] = max(0, int(count))
                except (TypeError, ValueError):
                    continue
            restored_daily[str(symbol).upper()] = clean_map
        self._daily_entries_by_symbol = restored_daily

        global_daily = snapshot.get("global_daily_counts")
        if not isinstance(global_daily, dict):
            global_daily = {}
        restored_global_daily: Dict[str, int] = {}
        for day_key, count in global_daily.items():
            try:
                restored_global_daily[str(day_key)] = max(0, int(count))
            except (TypeError, ValueError):
                continue
        self._global_daily_counts = restored_global_daily

        tiny = snapshot.get("tiny_pnl_exit_timestamps_by_symbol")
        if not isinstance(tiny, dict):
            tiny = {}
        restored_tiny: Dict[str, List[float]] = {}
        for symbol, values in tiny.items():
            if not isinstance(values, list):
                continue
            parsed = []
            for value in values:
                try:
                    parsed.append(float(value))
                except (TypeError, ValueError):
                    continue
            restored_tiny[str(symbol).upper()] = parsed
        self._tiny_pnl_exit_timestamps_by_symbol = restored_tiny

        cooldowns = snapshot.get("cooldown_until_by_symbol")
        if not isinstance(cooldowns, dict):
            cooldowns = {}
        restored_cooldowns: Dict[str, float] = {}
        for symbol, value in cooldowns.items():
            try:
                restored_cooldowns[str(symbol).upper()] = float(value)
            except (TypeError, ValueError):
                continue
        self._cooldown_until_by_symbol = restored_cooldowns
        raw_reasons = snapshot.get("cooldown_reason_by_symbol")
        restored_reasons: Dict[str, str] = {}
        if isinstance(raw_reasons, dict):
            for symbol, value in raw_reasons.items():
                key = str(symbol or "").upper()
                text = str(value or "").strip().upper()
                if key and text:
                    restored_reasons[key] = text
        self._cooldown_reason_by_symbol = restored_reasons

    def snapshot(self) -> Dict[str, Any]:
        return {
            "enabled": bool(self.enabled),
            "reentry_cooldown_seconds": float(self.reentry_cooldown_seconds),
            "flip_reentry_cooldown_seconds": float(self.flip_reentry_cooldown_seconds),
            "max_entries_per_symbol_per_hour": int(self.max_entries_per_symbol_per_hour),
            "max_entries_per_symbol_per_day": int(self.max_entries_per_symbol_per_day),
            "max_entries_global_per_day": int(self.max_entries_global_per_day),
            "daily_reset_timezone": str(self.daily_reset_timezone),
            "per_symbol_daily_limits": dict(self._per_symbol_daily_limits),
            "min_hold_bars_floor": int(self.min_hold_bars_floor),
            "min_hold_bars_floor_by_symbol": dict(self._min_hold_bars_floor_by_symbol),
            "tiny_pnl_threshold_usd": float(self.tiny_pnl_threshold_usd),
            "quick_exit_window_seconds": float(self.quick_exit_window_seconds),
            "tiny_pnl_max_count_per_hour": int(self.tiny_pnl_max_count_per_hour),
            "tiny_pnl_cooldown_seconds": float(self.tiny_pnl_cooldown_seconds),
            "protection_failure_lock_seconds": float(self.protection_failure_lock_seconds),
            "loss_reentry_lock_seconds": float(self.loss_reentry_lock_seconds),
            "protection_retry_interval_seconds": float(self.protection_retry_interval_seconds),
            "protection_retry_max_attempts": int(self.protection_retry_max_attempts),
            "global_daily_count": int(self._global_daily_counts.get(self._day_key(timezone_now_ts()), 0)),
            "global_daily_counts": dict(self._global_daily_counts),
            "entry_timestamps_by_symbol": {
                symbol: list(values) for symbol, values in self._entry_timestamps_by_symbol.items()
            },
            "daily_entries_by_symbol": {
                symbol: dict(values) for symbol, values in self._daily_entries_by_symbol.items()
            },
            "tiny_pnl_exit_timestamps_by_symbol": {
                symbol: list(values) for symbol, values in self._tiny_pnl_exit_timestamps_by_symbol.items()
            },
            "cooldown_until_by_symbol": dict(self._cooldown_until_by_symbol),
            "cooldown_reason_by_symbol": dict(self._cooldown_reason_by_symbol),
        }

    def _day_key(self, now_ts: float) -> str:
        tz = ZoneInfo(self.daily_reset_timezone)
        current = datetime.fromtimestamp(float(now_ts), tz=timezone.utc).astimezone(tz)
        return current.date().isoformat()

    def _daily_limit_for_symbol(self, symbol: str) -> int:
        key = str(symbol or "").upper()
        return int(self._per_symbol_daily_limits.get(key, self.max_entries_per_symbol_per_day))

    def _prune(self, symbol: str, now_ts: float) -> List[float]:
        key = str(symbol or "").upper()
        values = self._entry_timestamps_by_symbol.get(key, [])
        horizon = now_ts - 3600.0
        kept = [ts for ts in values if ts >= horizon]
        self._entry_timestamps_by_symbol[key] = kept

        day_key = self._day_key(now_ts)
        daily_map = self._daily_entries_by_symbol.get(key, {})
        if isinstance(daily_map, dict):
            self._daily_entries_by_symbol[key] = {k: int(v) for k, v in daily_map.items() if str(k) == day_key}
        else:
            self._daily_entries_by_symbol[key] = {}
        self._global_daily_counts = {k: int(v) for k, v in self._global_daily_counts.items() if str(k) == day_key}
        tiny_values = self._tiny_pnl_exit_timestamps_by_symbol.get(key, [])
        tiny_horizon = now_ts - 3600.0
        self._tiny_pnl_exit_timestamps_by_symbol[key] = [ts for ts in tiny_values if ts >= tiny_horizon]
        cooldown_until = float(self._cooldown_until_by_symbol.get(key, 0.0) or 0.0)
        if cooldown_until <= now_ts:
            self._cooldown_until_by_symbol.pop(key, None)
            self._cooldown_reason_by_symbol.pop(key, None)
        return kept

    def should_block_entry(self, symbol: str, now_ts: float, is_flip: bool = False) -> Optional[str]:
        if not self.enabled:
            return None
        key = str(symbol or "").upper()
        values = self._prune(symbol=key, now_ts=now_ts)
        cooldown_until = float(self._cooldown_until_by_symbol.get(key, 0.0) or 0.0)
        if now_ts < cooldown_until:
            reason = str(self._cooldown_reason_by_symbol.get(key, "CHURN_TINY_PNL_COOLDOWN") or "CHURN_TINY_PNL_COOLDOWN")
            return reason
        if values:
            since_last = now_ts - max(values)
            reentry_cooldown = self.flip_reentry_cooldown_seconds if bool(is_flip) else self.reentry_cooldown_seconds
            if since_last < reentry_cooldown:
                return "CHURN_COOLDOWN"
        if len(values) >= self.max_entries_per_symbol_per_hour:
            return "CHURN_HOURLY_LIMIT"
        day_key = self._day_key(now_ts)
        day_count = int(self._daily_entries_by_symbol.get(key, {}).get(day_key, 0))
        if day_count >= self._daily_limit_for_symbol(key):
            return "CHURN_DAILY_LIMIT"
        global_count = int(self._global_daily_counts.get(day_key, 0))
        if global_count >= self.max_entries_global_per_day:
            return "CHURN_DAILY_LIMIT"
        return None

    def record_entry(self, symbol: str, now_ts: float) -> None:
        key = str(symbol or "").upper()
        values = self._prune(symbol=key, now_ts=now_ts)
        values.append(float(now_ts))
        self._entry_timestamps_by_symbol[key] = values
        day_key = self._day_key(now_ts)
        daily_map = self._daily_entries_by_symbol.get(key, {})
        if not isinstance(daily_map, dict):
            daily_map = {}
        daily_map[day_key] = int(daily_map.get(day_key, 0)) + 1
        self._daily_entries_by_symbol[key] = daily_map
        self._global_daily_counts[day_key] = int(self._global_daily_counts.get(day_key, 0)) + 1

    def record_close(
        self,
        symbol: str,
        now_ts: float,
        realized_pnl: Optional[float],
        hold_seconds: Optional[float],
    ) -> Optional[Dict[str, Any]]:
        if not self.enabled:
            return None
        key = str(symbol or "").upper()
        self._prune(symbol=key, now_ts=now_ts)
        try:
            pnl_abs = abs(float(realized_pnl))
        except (TypeError, ValueError):
            return None
        try:
            held = float(hold_seconds) if hold_seconds is not None else float("inf")
        except (TypeError, ValueError):
            held = float("inf")
        if pnl_abs > self.tiny_pnl_threshold_usd:
            return None
        if held > self.quick_exit_window_seconds:
            return None

        tiny_list = self._tiny_pnl_exit_timestamps_by_symbol.get(key, [])
        tiny_list.append(float(now_ts))
        tiny_horizon = now_ts - 3600.0
        tiny_list = [ts for ts in tiny_list if ts >= tiny_horizon]
        self._tiny_pnl_exit_timestamps_by_symbol[key] = tiny_list

        if len(tiny_list) >= self.tiny_pnl_max_count_per_hour:
            cooldown_until = now_ts + self.tiny_pnl_cooldown_seconds
            self._cooldown_until_by_symbol[key] = float(cooldown_until)
            self._cooldown_reason_by_symbol[key] = "CHURN_TINY_PNL_COOLDOWN"
            return {
                "symbol": key,
                "reason": "CHURN_TINY_PNL_COOLDOWN",
                "tiny_count_last_hour": len(tiny_list),
                "cooldown_until_ts": float(cooldown_until),
                "realized_pnl": float(realized_pnl),
                "hold_seconds": float(held),
            }
        if realized_pnl is not None:
            try:
                pnl_v = float(realized_pnl)
            except (TypeError, ValueError):
                pnl_v = 0.0
            if pnl_v < 0.0:
                cooldown_until = now_ts + self.loss_reentry_lock_seconds
                self._cooldown_until_by_symbol[key] = max(
                    float(self._cooldown_until_by_symbol.get(key, 0.0) or 0.0),
                    float(cooldown_until),
                )
                self._cooldown_reason_by_symbol[key] = "CHURN_LOSS_REENTRY_LOCK"
                return {
                    "symbol": key,
                    "reason": "CHURN_LOSS_REENTRY_LOCK",
                    "cooldown_until_ts": float(self._cooldown_until_by_symbol[key]),
                    "realized_pnl": float(pnl_v),
                    "hold_seconds": float(held),
                }
        return None

    def record_protection_failure(self, symbol: str, now_ts: float) -> Dict[str, Any]:
        key = str(symbol or "").upper()
        cooldown_until = float(now_ts) + self.protection_failure_lock_seconds
        self._cooldown_until_by_symbol[key] = max(
            float(self._cooldown_until_by_symbol.get(key, 0.0) or 0.0),
            cooldown_until,
        )
        self._cooldown_reason_by_symbol[key] = "CHURN_PROTECTION_LOCK"
        return {
            "symbol": key,
            "reason": "CHURN_PROTECTION_LOCK",
            "cooldown_until_ts": float(self._cooldown_until_by_symbol[key]),
        }

    def is_protection_locked(self, symbol: str, now_ts: float) -> bool:
        key = str(symbol or "").upper()
        self._prune(symbol=key, now_ts=float(now_ts))
        cooldown_until = float(self._cooldown_until_by_symbol.get(key, 0.0) or 0.0)
        return float(now_ts) < cooldown_until

    def enforce_min_hold(self, configured_min_hold: Optional[int], symbol: Optional[str] = None) -> int:
        current = int(configured_min_hold) if configured_min_hold is not None else self.min_hold_bars_floor
        symbol_floor = self.min_hold_bars_floor
        symbol_key = str(symbol or "").strip().upper()
        if symbol_key:
            symbol_floor = int(self._min_hold_bars_floor_by_symbol.get(symbol_key, symbol_floor))
        return max(self.min_hold_bars_floor, symbol_floor, current)


def timezone_now_ts() -> float:
    return datetime.now(timezone.utc).timestamp()
