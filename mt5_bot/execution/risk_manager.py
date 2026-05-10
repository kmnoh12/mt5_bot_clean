from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
try:
    from zoneinfo import ZoneInfo
except ImportError:
    from datetime import timezone as ZoneInfo  # Fallback (won't work for strings but prevents crash on import)

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from core.models import OrderResult, SymbolConstraints
try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None


@dataclass
class RiskStatus:
    halted: bool
    reason: str
    session_start_equity: Optional[float]
    daily_start_equity: Optional[float]
    daily_date_utc: str
    consecutive_losses: int
    equity_peak: Optional[float]


class RiskEngine:
    INVALID_CONSTRAINTS_OR_SCALE = "INVALID_CONSTRAINTS_OR_SCALE"

    def __init__(self, config: Dict[str, Any], snapshot: Optional[Dict[str, Any]] = None) -> None:
        cfg = self._as_mapping(config)
        snap = self._as_mapping(snapshot)

        self.risk_per_trade_pct = 0.05
        self.max_risk_per_trade_pct = 0.15
        self.daily_loss_limit_pct = 0.12
        self.session_loss_limit_pct = 0.25
        self.max_consecutive_losses = 5
        self.per_symbol_risk_per_trade_pct: Dict[str, float] = {}
        self.dynamic_risk_cap_enabled = False
        self.dynamic_risk_min_pct = 0.003
        self.dynamic_risk_max_pct = 0.012
        self.dynamic_risk_lookback_hours = 24
        self.dynamic_risk_min_samples = 8
        self.dynamic_risk_winrate_floor = 0.35
        self.dynamic_risk_winrate_ceiling = 0.65
        self._recent_closed_trades: list[Dict[str, Any]] = []

        self.kelly_enabled = True
        self.kelly_fraction = 0.5
        self.kelly_win_probability: Optional[float] = None
        self.kelly_payoff_ratio: Optional[float] = None
        self.dd_hard_stop_pct = 0.30
        self.daily_reset_timezone = "Asia/Seoul"
        self.auto_resume_daily_drawdown_on_new_day = True
        self.auto_resume_after_halt_minutes = 0
        self.manual_resume_resets_daily_baseline = False
        self.dynamic_lot_enabled = True
        self.dynamic_lot_fail_mode = "fallback"
        self.dynamic_lot_default_lot = 0.01
        self.dynamic_lot_min_volume_policy = "block"
        self._last_dynamic_lot_meta: Dict[str, Any] = {}

        self.update_config(cfg)

        self._halted = bool(snap.get("halted", False))
        self._halt_reason = str(snap.get("halt_reason", "") or "")
        self._halted_since_utc = str(snap.get("halted_since_utc", "") or "")
        self._session_start_equity = self._safe_float(snap.get("session_start_equity"))
        self._daily_start_equity = self._safe_float(snap.get("daily_start_equity"))
        self._daily_date_utc = str(snap.get("daily_date_utc", "") or "")
        self._consecutive_losses = self._safe_int(snap.get("consecutive_losses", 0), 0, min_value=0)
        self._equity_peak = self._safe_float(snap.get("equity_peak"))
        self._recent_closed_trades = self._normalize_recent_closed_trades(snap.get("recent_closed_trades"))

    def update_config(self, config: Dict[str, Any]) -> None:
        cfg = self._as_mapping(config)
        self.risk_per_trade_pct = self._safe_float(
            cfg.get("risk_per_trade_pct", self.risk_per_trade_pct),
            self.risk_per_trade_pct,
            min_value=0.001,
        ) or 0.001
        self.max_risk_per_trade_pct = self._safe_float(
            cfg.get("max_risk_per_trade_pct", self.max_risk_per_trade_pct),
            self.max_risk_per_trade_pct,
            min_value=0.001,
        ) or 0.001
        self.daily_loss_limit_pct = self._safe_float(
            cfg.get("daily_loss_limit_pct", self.daily_loss_limit_pct),
            self.daily_loss_limit_pct,
            min_value=0.001,
        ) or 0.001
        self.session_loss_limit_pct = self._safe_float(
            cfg.get("session_loss_limit_pct", self.session_loss_limit_pct),
            self.session_loss_limit_pct,
            min_value=0.001,
        ) or 0.001
        self.max_consecutive_losses = self._safe_int(
            cfg.get("max_consecutive_losses", self.max_consecutive_losses),
            self.max_consecutive_losses,
            min_value=1,
        )
        self.per_symbol_risk_per_trade_pct = self._normalize_per_symbol_risk_map(
            cfg.get("per_symbol_risk_per_trade_pct", self.per_symbol_risk_per_trade_pct)
        )
        self.dynamic_risk_cap_enabled = bool(cfg.get("dynamic_risk_cap_enabled", self.dynamic_risk_cap_enabled))
        dynamic_risk_min_pct = self._safe_float(
            cfg.get("dynamic_risk_min_pct", self.dynamic_risk_min_pct),
            self.dynamic_risk_min_pct,
            min_value=0.001,
            max_value=0.1,
        )
        if dynamic_risk_min_pct is not None:
            self.dynamic_risk_min_pct = float(dynamic_risk_min_pct)
        self.dynamic_risk_max_pct = self._safe_float(
            cfg.get("dynamic_risk_max_pct", self.dynamic_risk_max_pct),
            self.dynamic_risk_max_pct,
            min_value=0.001,
            max_value=0.2,
        ) or 0.012
        if self.dynamic_risk_max_pct < self.dynamic_risk_min_pct:
            self.dynamic_risk_max_pct = self.dynamic_risk_min_pct
        self.dynamic_risk_lookback_hours = self._safe_int(
            cfg.get("dynamic_risk_lookback_hours", self.dynamic_risk_lookback_hours),
            self.dynamic_risk_lookback_hours,
            min_value=1,
            max_value=24 * 30,
        )
        self.dynamic_risk_min_samples = self._safe_int(
            cfg.get("dynamic_risk_min_samples", self.dynamic_risk_min_samples),
            self.dynamic_risk_min_samples,
            min_value=1,
            max_value=10000,
        )
        dynamic_risk_winrate_floor = self._safe_float(
            cfg.get("dynamic_risk_winrate_floor", self.dynamic_risk_winrate_floor),
            self.dynamic_risk_winrate_floor,
            min_value=0.0,
            max_value=1.0,
        )
        if dynamic_risk_winrate_floor is not None:
            self.dynamic_risk_winrate_floor = float(dynamic_risk_winrate_floor)
        dynamic_risk_winrate_ceiling = self._safe_float(
            cfg.get("dynamic_risk_winrate_ceiling", self.dynamic_risk_winrate_ceiling),
            self.dynamic_risk_winrate_ceiling,
            min_value=0.0,
            max_value=1.0,
        )
        if dynamic_risk_winrate_ceiling is not None:
            self.dynamic_risk_winrate_ceiling = float(dynamic_risk_winrate_ceiling)
        if self.dynamic_risk_winrate_ceiling < self.dynamic_risk_winrate_floor:
            self.dynamic_risk_winrate_ceiling = self.dynamic_risk_winrate_floor

        self.kelly_enabled = bool(cfg.get("kelly_enabled", self.kelly_enabled))
        self.kelly_fraction = self._safe_float(
            cfg.get("kelly_fraction", self.kelly_fraction),
            self.kelly_fraction,
            min_value=0.0,
            max_value=1.0,
        ) or 0.0
        self.kelly_win_probability = self._clamp_probability(
            cfg.get("kelly_win_probability", self.kelly_win_probability)
        )
        self.kelly_payoff_ratio = self._positive_optional_float(
            cfg.get("kelly_payoff_ratio", self.kelly_payoff_ratio)
        )
        self.dd_hard_stop_pct = self._safe_float(
            cfg.get("dd_hard_stop_pct", self.dd_hard_stop_pct),
            self.dd_hard_stop_pct,
            min_value=0.01,
            max_value=0.95,
        ) or 0.30
        self.daily_reset_timezone = str(cfg.get("daily_reset_timezone", self.daily_reset_timezone) or "Asia/Seoul")
        self.auto_resume_daily_drawdown_on_new_day = bool(
            cfg.get("auto_resume_daily_drawdown_on_new_day", self.auto_resume_daily_drawdown_on_new_day)
        )
        auto_resume_after_halt_minutes = self._safe_int(
            cfg.get("auto_resume_after_halt_minutes", self.auto_resume_after_halt_minutes),
            self.auto_resume_after_halt_minutes,
            min_value=0,
            max_value=60 * 24 * 14,
        )
        self.auto_resume_after_halt_minutes = int(auto_resume_after_halt_minutes or 0)
        self.manual_resume_resets_daily_baseline = bool(
            cfg.get("manual_resume_resets_daily_baseline", self.manual_resume_resets_daily_baseline)
        )
        self.dynamic_lot_enabled = bool(cfg.get("dynamic_lot_enabled", self.dynamic_lot_enabled))
        fail_mode = str(cfg.get("dynamic_lot_fail_mode", self.dynamic_lot_fail_mode) or "fallback").strip().lower()
        if fail_mode not in {"fallback", "block"}:
            fail_mode = "fallback"
        self.dynamic_lot_fail_mode = fail_mode
        self.dynamic_lot_default_lot = self._safe_float(
            cfg.get("dynamic_lot_default_lot", self.dynamic_lot_default_lot),
            self.dynamic_lot_default_lot,
            min_value=0.0000001,
        ) or 0.01
        min_volume_policy = str(
            cfg.get("dynamic_lot_min_volume_policy", self.dynamic_lot_min_volume_policy) or "block"
        ).strip().lower()
        if min_volume_policy not in {"block", "allow"}:
            min_volume_policy = "block"
        self.dynamic_lot_min_volume_policy = min_volume_policy

    @staticmethod
    def _as_mapping(value: Any) -> Dict[str, Any]:
        if isinstance(value, dict):
            return dict(value)
        if value is None:
            return {}
        if isinstance(value, (str, bytes, bytearray)):
            try:
                text = value.decode("utf-8", errors="replace") if isinstance(value, (bytes, bytearray)) else str(value)
            except Exception as exc:
                RiskEngine._watchdog_debug_log(
                    "risk_engine.as_mapping.decode_exception",
                    error=exc,
                    extra={"value_type": type(value).__name__},
                )
                return {}
            if not text.strip():
                RiskEngine._watchdog_debug_log("risk_engine.as_mapping.empty_json_input")
                return {}
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError as exc:
                RiskEngine._watchdog_debug_log(
                    "risk_engine.as_mapping.json_decode_error",
                    error=exc,
                    extra={"input_length": len(text)},
                )
                return {}
            except Exception as exc:
                RiskEngine._watchdog_debug_log(
                    "risk_engine.as_mapping.json_parse_exception",
                    error=exc,
                    extra={"input_length": len(text)},
                )
                return {}
            if isinstance(parsed, dict):
                return dict(parsed)
            RiskEngine._watchdog_debug_log(
                "risk_engine.as_mapping.non_mapping_json",
                extra={"parsed_type": type(parsed).__name__},
            )
            return {}
        try:
            return dict(value)
        except Exception as exc:
            RiskEngine._watchdog_debug_log(
                "risk_engine.as_mapping.dict_cast_exception",
                error=exc,
                extra={"value_type": type(value).__name__},
            )
            return {}

    @staticmethod
    def _as_float(value: Any) -> Optional[float]:
        try:
            out = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(out):
            return None
        return out

    @classmethod
    def _safe_float(
        cls,
        value: Any,
        default: Optional[float] = None,
        *,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
    ) -> Optional[float]:
        out = cls._as_float(value)
        if out is None:
            out = cls._as_float(default)
        if out is None:
            return None
        if min_value is not None:
            out = max(min_value, out)
        if max_value is not None:
            out = min(max_value, out)
        return out

    @classmethod
    def _safe_int(
        cls,
        value: Any,
        default: int = 0,
        *,
        min_value: Optional[int] = None,
        max_value: Optional[int] = None,
    ) -> int:
        out_float = cls._as_float(value)
        if out_float is None:
            out_float = cls._as_float(default)
        out = int(out_float) if out_float is not None else int(default)
        if min_value is not None:
            out = max(min_value, out)
        if max_value is not None:
            out = min(max_value, out)
        return out

    @classmethod
    def _safe_multiply(cls, left: Any, right: Any) -> Optional[float]:
        left_num = cls._as_float(left)
        right_num = cls._as_float(right)
        if left_num is None or right_num is None:
            return None
        out = left_num * right_num
        if not math.isfinite(out):
            return None
        return out

    @classmethod
    def _safe_divide(cls, numerator: Any, denominator: Any) -> Optional[float]:
        numer = cls._as_float(numerator)
        denom = cls._as_float(denominator)
        if numer is None or denom is None or denom == 0:
            return None
        out = numer / denom
        if not math.isfinite(out):
            return None
        return out

    @staticmethod
    def _safe_getattr(source: Any, name: str) -> Any:
        if isinstance(source, dict):
            try:
                return source.get(name)
            except Exception:
                return None
        try:
            return getattr(source, name)
        except Exception:
            return None

    @staticmethod
    def _watchdog_debug_log(event: str, error: Optional[BaseException] = None, extra: Optional[Dict[str, Any]] = None) -> None:
        try:
            log_path = Path(__file__).resolve().parents[1] / "watchdog_debug.log"
            stamp = datetime.now(timezone.utc).isoformat()
            parts = [f"[{stamp}] {event}"]
            if error is not None:
                parts.append(f"{type(error).__name__}: {error}")
            if extra:
                fields = []
                for key, value in extra.items():
                    try:
                        fields.append(f"{key}={value!r}")
                    except Exception:
                        fields.append(f"{key}=<unrepr>")
                if fields:
                    parts.append(" ".join(fields))
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(" | ".join(parts) + "\n")
        except Exception:
            return

    @classmethod
    def _normalize_constraints(cls, constraints: Any) -> Tuple[Dict[str, float], bool]:
        normalized = {
            "min_volume": 0.01,
            "max_volume": 100.0,
            "volume_step": 0.01,
            "contract_size": 1.0,
        }
        try:
            raw_min = cls._safe_float(cls._safe_getattr(constraints, "min_volume"))
            raw_max = cls._safe_float(cls._safe_getattr(constraints, "max_volume"))
            raw_step = cls._safe_float(cls._safe_getattr(constraints, "volume_step"))
            raw_contract = cls._safe_float(cls._safe_getattr(constraints, "contract_size"))

            valid = True
            if raw_min is None or raw_min < 0:
                valid = False
            if raw_max is None or raw_max <= 0:
                valid = False
            if raw_step is None or raw_step <= 0:
                valid = False
            if raw_contract is None or raw_contract <= 0:
                valid = False

            min_v = max(0.0, raw_min if raw_min is not None else normalized["min_volume"])
            max_v = raw_max if raw_max is not None else max(min_v, normalized["max_volume"])
            if max_v < min_v:
                valid = False
                max_v = min_v
            step = raw_step if raw_step is not None and raw_step > 0 else normalized["volume_step"]
            contract_size = raw_contract if raw_contract is not None and raw_contract > 0 else normalized["contract_size"]

            normalized["min_volume"] = float(min_v)
            normalized["max_volume"] = float(max_v)
            normalized["volume_step"] = float(step)
            normalized["contract_size"] = max(1e-9, float(contract_size))
            if not valid:
                cls._watchdog_debug_log(
                    "risk_engine.normalize_constraints.invalid_values",
                    extra={
                        "raw_min": raw_min,
                        "raw_max": raw_max,
                        "raw_step": raw_step,
                        "raw_contract": raw_contract,
                        "resolved_min": normalized["min_volume"],
                        "resolved_max": normalized["max_volume"],
                        "resolved_step": normalized["volume_step"],
                        "resolved_contract": normalized["contract_size"],
                    },
                )
            return normalized, valid
        except Exception as exc:
            cls._watchdog_debug_log("risk_engine.normalize_constraints.exception", error=exc)
            return normalized, False

    @classmethod
    def _normalize_volume_scale(cls, volume_scale: Any) -> Tuple[float, bool]:
        if volume_scale is None:
            return 1.0, True
        scale = cls._safe_float(volume_scale)
        if scale is None:
            return 1.0, False
        # Preserve prior behavior: 0 -> default 1.0, negative -> clamped up to 0.1.
        if scale == 0:
            scale = 1.0
        return max(0.1, scale), True

    @classmethod
    def _positive_optional_float(cls, value: Any) -> Optional[float]:
        out = cls._as_float(value)
        if out is None or out <= 0:
            return None
        return out

    @classmethod
    def _clamp_probability(cls, value: Any) -> Optional[float]:
        out = cls._as_float(value)
        if out is None:
            return None
        return min(1.0, max(0.0, out))

    @staticmethod
    def _normalize_symbol_key(symbol: Any) -> str:
        return str(symbol or "").strip().upper()

    @classmethod
    def _normalize_per_symbol_risk_map(cls, raw_map: Any) -> Dict[str, float]:
        if not isinstance(raw_map, dict):
            return {}
        out: Dict[str, float] = {}
        for raw_symbol, raw_value in raw_map.items():
            symbol = cls._normalize_symbol_key(raw_symbol)
            if not symbol:
                continue
            value = cls._safe_float(raw_value, None, min_value=0.001)
            if value is None:
                continue
            out[symbol] = float(value)
        return out

    def _resolve_risk_per_trade_pct(self, symbol: Optional[str]) -> float:
        key = self._normalize_symbol_key(symbol)
        static_risk = float(self.risk_per_trade_pct)
        if key:
            per_symbol = self.per_symbol_risk_per_trade_pct.get(key)
            if per_symbol is not None:
                static_risk = float(per_symbol)
        dynamic_risk = self._resolve_dynamic_risk_pct(symbol=key)
        if dynamic_risk is not None:
            return float(dynamic_risk)
        return static_risk

    @classmethod
    def _parse_iso_utc(cls, value: Any) -> Optional[datetime]:
        text = str(value or "").strip()
        if not text:
            return None
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except Exception:
            return None
        return parsed.astimezone(timezone.utc) if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)

    def _normalize_recent_closed_trades(self, raw: Any) -> list[Dict[str, Any]]:
        if not isinstance(raw, list):
            return []
        out: list[Dict[str, Any]] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            pnl = self._as_float(item.get("pnl"))
            ts = self._parse_iso_utc(item.get("ts_utc"))
            if pnl is None or ts is None:
                continue
            out.append(
                {
                    "symbol": self._normalize_symbol_key(item.get("symbol")),
                    "pnl": float(pnl),
                    "ts_utc": ts.isoformat(),
                }
            )
        return out[-5000:]

    def _prune_recent_closed_trades(self, now_utc: datetime) -> None:
        horizon = now_utc - timedelta(hours=max(1, int(self.dynamic_risk_lookback_hours)))
        pruned: list[Dict[str, Any]] = []
        for item in self._recent_closed_trades[-5000:]:
            ts = self._parse_iso_utc(item.get("ts_utc"))
            if ts is None:
                continue
            if ts >= horizon:
                pruned.append(item)
        self._recent_closed_trades = pruned[-5000:]

    def _record_closed_trade(self, *, pnl: float, symbol: Optional[str], now_utc: datetime) -> None:
        self._recent_closed_trades.append(
            {
                "symbol": self._normalize_symbol_key(symbol),
                "pnl": float(pnl),
                "ts_utc": now_utc.isoformat(),
            }
        )
        self._prune_recent_closed_trades(now_utc)

    def _recent_winrate(self, *, symbol: Optional[str], now_utc: datetime) -> Tuple[Optional[float], int]:
        self._prune_recent_closed_trades(now_utc)
        key = self._normalize_symbol_key(symbol)
        trades: list[float] = []
        if key:
            for item in self._recent_closed_trades:
                if self._normalize_symbol_key(item.get("symbol")) != key:
                    continue
                pnl = self._as_float(item.get("pnl"))
                if pnl is not None:
                    trades.append(float(pnl))
        if len(trades) < self.dynamic_risk_min_samples:
            trades = []
            for item in self._recent_closed_trades:
                pnl = self._as_float(item.get("pnl"))
                if pnl is not None:
                    trades.append(float(pnl))
        total = len(trades)
        if total < self.dynamic_risk_min_samples:
            return None, total
        wins = sum(1 for pnl in trades if pnl > 0.0)
        return (float(wins) / float(total)) if total > 0 else None, total

    def _resolve_dynamic_risk_pct(self, symbol: Optional[str]) -> Optional[float]:
        if not self.dynamic_risk_cap_enabled:
            return None
        now_utc = datetime.now(timezone.utc)
        winrate, sample_size = self._recent_winrate(symbol=symbol, now_utc=now_utc)
        if winrate is None or sample_size < self.dynamic_risk_min_samples:
            return None
        floor = float(self.dynamic_risk_winrate_floor)
        ceiling = float(self.dynamic_risk_winrate_ceiling)
        min_pct = float(self.dynamic_risk_min_pct)
        max_pct = float(self.dynamic_risk_max_pct)
        if ceiling <= floor:
            normalized = 1.0 if winrate >= ceiling else 0.0
        else:
            normalized = (float(winrate) - floor) / (ceiling - floor)
        normalized = min(1.0, max(0.0, normalized))
        return min_pct + ((max_pct - min_pct) * normalized)

    @staticmethod
    def _drawdown_pct(current: float, baseline: float) -> float:
        if baseline <= 0:
            return 0.0
        return (current - baseline) / baseline

    @staticmethod
    def _peak_drawdown_pct(current: float, peak: Optional[float]) -> float:
        if peak is None or peak <= 0:
            return 0.0
        return max(0.0, (peak - current) / peak)

    @staticmethod
    def _dd_multiplier(drawdown_pct: float) -> float:
        if drawdown_pct < 0.10:
            return 1.5
        if drawdown_pct < 0.15:
            return 1.0
        if drawdown_pct < 0.25:
            return 0.5
        return 0.0

    @staticmethod
    def _streak_multiplier(loss_streak: int) -> float:
        streak = max(0, int(loss_streak))
        if streak <= 1:
            return 1.0
        return 0.85 ** float(streak - 1)

    def _kelly_risk_pct(self, win_probability: Optional[float], payoff_ratio: Optional[float]) -> Optional[float]:
        if not self.kelly_enabled:
            return None

        p = self._clamp_probability(win_probability)
        b = self._positive_optional_float(payoff_ratio)
        if p is None:
            p = self.kelly_win_probability
        if b is None:
            b = self.kelly_payoff_ratio
        if p is None or b is None or b <= 0:
            return None

        f_star = ((b * p) - (1.0 - p)) / b
        kelly_fractional = self.kelly_fraction * max(0.0, f_star)
        return min(self.max_risk_per_trade_pct, max(0.0, kelly_fractional))

    def _resolved_peak(self, equity: Optional[float]) -> Optional[float]:
        current = self._as_float(equity)
        peak = self._equity_peak
        if current is None or current <= 0:
            return peak
        if peak is None or peak <= 0:
            return current
        return max(peak, current)

    def _effective_risk_components(
        self,
        *,
        equity: float,
        symbol: Optional[str],
        win_probability: Optional[float],
        payoff_ratio: Optional[float],
    ) -> Dict[str, float]:
        resolved_risk_pct = self._resolve_risk_per_trade_pct(symbol)
        base_risk_pct = min(resolved_risk_pct, self.max_risk_per_trade_pct)
        kelly_risk_pct = self._kelly_risk_pct(win_probability=win_probability, payoff_ratio=payoff_ratio)
        pre_multiplier_risk_pct = base_risk_pct if kelly_risk_pct is None else min(self.max_risk_per_trade_pct, kelly_risk_pct)

        effective_peak = self._resolved_peak(equity)
        drawdown_pct = self._peak_drawdown_pct(equity, effective_peak)
        dd_multiplier = self._dd_multiplier(drawdown_pct)
        streak_multiplier = self._streak_multiplier(self._consecutive_losses)
        # Hard-cap final risk to max_risk_per_trade_pct even after multipliers.
        effective_risk_pct = min(
            self.max_risk_per_trade_pct,
            pre_multiplier_risk_pct * dd_multiplier * streak_multiplier,
        )

        kelly_multiplier = 1.0
        if base_risk_pct > 0:
            kelly_multiplier = pre_multiplier_risk_pct / base_risk_pct

        return {
            "base_risk_pct": float(base_risk_pct),
            "resolved_risk_pct": float(resolved_risk_pct),
            "kelly_risk_pct": float(pre_multiplier_risk_pct),
            "kelly_multiplier": float(kelly_multiplier),
            "drawdown_pct": float(drawdown_pct),
            "dd_multiplier": float(dd_multiplier),
            "streak_multiplier": float(streak_multiplier),
            "effective_risk_pct": float(effective_risk_pct),
            "equity_peak": float(effective_peak) if effective_peak is not None else 0.0,
            "loss_streak": float(max(0, int(self._consecutive_losses))),
        }

    def snapshot(self) -> Dict[str, Any]:
        return {
            "halted": self._halted,
            "halt_reason": self._halt_reason,
            "halted_since_utc": self._halted_since_utc,
            "session_start_equity": self._session_start_equity,
            "daily_start_equity": self._daily_start_equity,
            "daily_date_utc": self._daily_date_utc,
            "consecutive_losses": self._consecutive_losses,
            "equity_peak": self._equity_peak,
            "recent_closed_trades": list(self._recent_closed_trades[-5000:]),
        }

    def status(self) -> RiskStatus:
        return RiskStatus(
            halted=self._halted,
            reason=self._halt_reason,
            session_start_equity=self._session_start_equity,
            daily_start_equity=self._daily_start_equity,
            daily_date_utc=self._daily_date_utc,
            consecutive_losses=self._consecutive_losses,
            equity_peak=self._equity_peak,
        )

    def resume(self) -> None:
        self._halted = False
        self._halt_reason = ""
        self._halted_since_utc = ""
        if self.manual_resume_resets_daily_baseline:
            self._daily_start_equity = None
            self._daily_date_utc = ""

    def _parse_halted_since(self) -> Optional[datetime]:
        raw = str(self._halted_since_utc or "").strip()
        if not raw:
            return None
        try:
            parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                return parsed.replace(tzinfo=timezone.utc)
            return parsed.astimezone(timezone.utc)
        except Exception:
            return None

    def _maybe_auto_resume_halt(self, now_day: str) -> None:
        if not self._halted:
            return
        reason = str(self._halt_reason or "")
        if "DAILY_DRAWDOWN_LIMIT_BREACH" not in reason:
            return

        if self.auto_resume_daily_drawdown_on_new_day and self._daily_date_utc and self._daily_date_utc != now_day:
            self.resume()
            return

        if self.auto_resume_after_halt_minutes > 0:
            halted_since = self._parse_halted_since()
            if halted_since is None:
                return
            elapsed = datetime.now(timezone.utc) - halted_since
            if elapsed >= timedelta(minutes=self.auto_resume_after_halt_minutes):
                self.resume()

    def halt(self, reason: str) -> None:
        self._halted = True
        self._halt_reason = str(reason or "RISK_GUARD_HALT")
        self._halted_since_utc = datetime.now(timezone.utc).isoformat()

    def on_order_result(self, result: Optional[OrderResult]) -> None:
        if result is None:
            return
        if int(result.retcode or 0) == 10027:
            self.halt("AUTOTRADING_DISABLED_CLIENT_10027")

    def on_trade_close(self, pnl: Optional[float], symbol: Optional[str] = None) -> None:
        value = self._as_float(pnl)
        if value is None:
            return
        self._record_closed_trade(pnl=float(value), symbol=symbol, now_utc=datetime.now(timezone.utc))
        if value < 0:
            self._consecutive_losses += 1
            if self._consecutive_losses >= self.max_consecutive_losses:
                self.halt(f"MAX_CONSECUTIVE_LOSSES_{self._consecutive_losses}")
        else:
            self._consecutive_losses = 0

    def sync_account(self, account_info: Dict[str, Any]) -> None:
        try:
            tz = ZoneInfo(self.daily_reset_timezone)
        except Exception:
            tz = timezone.utc
        now_day = datetime.now(tz).date().isoformat()

        equity = self._as_float((account_info or {}).get("equity"))
        if equity is None:
            equity = self._as_float((account_info or {}).get("balance"))
        if equity is None or equity <= 0:
            return

        if self._session_start_equity is None or self._session_start_equity <= 0:
            self._session_start_equity = equity

        if self._daily_date_utc != now_day or self._daily_start_equity is None or self._daily_start_equity <= 0:
            self._daily_date_utc = now_day
            self._daily_start_equity = equity

        if self._equity_peak is None or self._equity_peak <= 0:
            self._equity_peak = equity
        else:
            self._equity_peak = max(self._equity_peak, equity)

    def evaluate_limits(self, account_info: Dict[str, Any]) -> Optional[str]:
        try:
            tz = ZoneInfo(self.daily_reset_timezone)
        except Exception:
            tz = timezone.utc
        now_day = datetime.now(tz).date().isoformat()
        self._maybe_auto_resume_halt(now_day)
        self.sync_account(account_info)
        if self._halted:
            return self._halt_reason or "RISK_HALTED"

        equity = self._as_float((account_info or {}).get("equity"))
        if equity is None:
            equity = self._as_float((account_info or {}).get("balance"))
        if equity is None or equity <= 0:
            return None

        peak_drawdown = self._peak_drawdown_pct(equity, self._equity_peak)
        if peak_drawdown >= self.dd_hard_stop_pct:
            self.halt(f"EQUITY_DRAWDOWN_HARD_STOP_{peak_drawdown:.4f}")
            return self._halt_reason

        if self._session_start_equity and self._session_start_equity > 0:
            session_dd = self._drawdown_pct(equity, self._session_start_equity)
            if session_dd <= -self.session_loss_limit_pct:
                self.halt(f"SESSION_DRAWDOWN_LIMIT_BREACH_{session_dd:.4f}")
                return self._halt_reason

        if self._daily_start_equity and self._daily_start_equity > 0:
            daily_dd = self._drawdown_pct(equity, self._daily_start_equity)
            if daily_dd <= -self.daily_loss_limit_pct:
                self.halt(f"DAILY_DRAWDOWN_LIMIT_BREACH_{daily_dd:.4f}")
                return self._halt_reason

        if self._consecutive_losses >= self.max_consecutive_losses:
            self.halt(f"MAX_CONSECUTIVE_LOSSES_{self._consecutive_losses}")
            return self._halt_reason

        return None

    def can_trade(self, account_info: Dict[str, Any]) -> Tuple[bool, str]:
        reason = self.evaluate_limits(account_info)
        if reason:
            return False, reason
        return True, "OK"

    @classmethod
    def _quantize_volume(cls, raw_volume: float, constraints: SymbolConstraints) -> float:
        try:
            normalized, _ = cls._normalize_constraints(constraints)
            min_v = normalized["min_volume"]
            max_v = normalized["max_volume"]
            step = normalized["volume_step"]

            volume_in = cls._safe_float(raw_volume, min_v)
            if volume_in is None:
                volume_in = min_v
            volume = min(max(volume_in, min_v), max_v)
            units = round((volume - min_v) / step)
            quantized = min_v + (units * step)
            quantized = min(max(quantized, min_v), max_v)

            text = f"{step:.12f}".rstrip("0")
            precision = len(text.split(".")[1]) if "." in text else 0
            rounded = round(quantized, precision)
            out = cls._safe_float(rounded, min_v)
            if out is None:
                return min_v
            return min(max(out, min_v), max_v)
        except Exception as exc:
            cls._watchdog_debug_log("risk_engine.quantize_volume.exception", error=exc)
            fallback, _ = cls._normalize_constraints(constraints)
            return max(0.0, float(fallback["min_volume"]))

    def repair_volume_for_10014(self, constraints: SymbolConstraints) -> float:
        # Safe fallback: minimum valid tradable quantity.
        normalized, _ = self._normalize_constraints(constraints)
        return self._quantize_volume(normalized["min_volume"], constraints)

    @staticmethod
    def _step_precision(step: float) -> int:
        try:
            text = f"{float(step):.12f}".rstrip("0").rstrip(".")
        except Exception:
            return 2
        if "." not in text:
            return 0
        return max(0, len(text.split(".", 1)[1]))

    @staticmethod
    def _safe_mt5_last_error() -> Any:
        if mt5 is None:
            return ("MT5_IMPORT_ERROR", "MetaTrader5 package not installed")
        try:
            return mt5.last_error()
        except Exception as exc:
            return ("MT5_LAST_ERROR_EXCEPTION", repr(exc))

    def _set_dynamic_lot_meta(self, **kwargs: Any) -> None:
        try:
            self._last_dynamic_lot_meta = dict(kwargs)
        except Exception:
            self._last_dynamic_lot_meta = {}

    def _normalize_lot_with_symbol_info(self, raw_lot: Any, symbol_info: Any) -> Optional[float]:
        lot = self._as_float(raw_lot)
        if lot is None or lot <= 0:
            return None
        min_volume = self._safe_float(self._safe_getattr(symbol_info, "volume_min"), 0.01, min_value=0.0) or 0.01
        max_volume = self._safe_float(self._safe_getattr(symbol_info, "volume_max"), min_volume, min_value=min_volume) or min_volume
        step = self._safe_float(self._safe_getattr(symbol_info, "volume_step"), 0.01, min_value=0.0000001) or 0.01
        units = math.floor(max(0.0, lot) / step)
        stepped = units * step
        precision = self._step_precision(step)
        stepped = round(stepped, precision)
        clamped = min(max(stepped, min_volume), max_volume)
        clamped = round(clamped, precision)
        out = self._as_float(clamped)
        if out is None or out <= 0:
            return None
        return out

    def calculate_dynamic_lot(
        self,
        symbol: str,
        side: str,
        sl_price: float,
        entry_price: Optional[float] = None,
        risk_pct: Optional[float] = None,
        default_lot: float = 0.01,
        fail_mode: str = "fallback",
    ) -> Optional[float]:
        symbol_text = str(symbol or "").strip()
        side_text = str(side or "").strip().lower()
        resolved_fail_mode = str(fail_mode or "fallback").strip().lower()
        if resolved_fail_mode not in {"fallback", "block"}:
            resolved_fail_mode = "fallback"

        if mt5 is None:
            err = self._safe_mt5_last_error()
            self._set_dynamic_lot_meta(
                reason="MT5_UNAVAILABLE",
                mt5_last_error=err,
                fail_mode=resolved_fail_mode,
            )
            self._watchdog_debug_log(
                "risk_engine.dynamic_lot.mt5_unavailable",
                extra={"symbol": symbol_text, "side": side_text, "mt5_last_error": err},
            )
            return None if resolved_fail_mode == "block" else max(0.0, float(default_lot or 0.01))

        info = mt5.symbol_info(symbol_text)
        if info is None:
            try:
                mt5.symbol_select(symbol_text, True)
            except Exception:
                pass
            info = mt5.symbol_info(symbol_text)
        if info is None:
            err = self._safe_mt5_last_error()
            self._set_dynamic_lot_meta(
                reason="SYMBOL_INFO_UNAVAILABLE",
                mt5_last_error=err,
                fail_mode=resolved_fail_mode,
            )
            self._watchdog_debug_log(
                "risk_engine.dynamic_lot.symbol_info_unavailable",
                extra={"symbol": symbol_text, "side": side_text, "mt5_last_error": err},
            )
            return None if resolved_fail_mode == "block" else max(0.0, float(default_lot or 0.01))

        def _fallback_lot() -> Optional[float]:
            normalized = self._normalize_lot_with_symbol_info(default_lot, info)
            if normalized is not None:
                return normalized
            return self._normalize_lot_with_symbol_info(self.dynamic_lot_default_lot, info)

        sl = self._as_float(sl_price)
        if sl is None:
            self._set_dynamic_lot_meta(reason="INVALID_SL_PRICE", fail_mode=resolved_fail_mode)
            return None if resolved_fail_mode == "block" else _fallback_lot()

        tick = mt5.symbol_info_tick(symbol_text)
        if tick is None:
            err = self._safe_mt5_last_error()
            self._set_dynamic_lot_meta(
                reason="TICK_UNAVAILABLE",
                mt5_last_error=err,
                fail_mode=resolved_fail_mode,
            )
            self._watchdog_debug_log(
                "risk_engine.dynamic_lot.tick_unavailable",
                extra={"symbol": symbol_text, "side": side_text, "mt5_last_error": err},
            )
            return None if resolved_fail_mode == "block" else _fallback_lot()

        account = mt5.account_info()
        if account is None:
            err = self._safe_mt5_last_error()
            self._set_dynamic_lot_meta(
                reason="ACCOUNT_UNAVAILABLE",
                mt5_last_error=err,
                fail_mode=resolved_fail_mode,
            )
            self._watchdog_debug_log(
                "risk_engine.dynamic_lot.account_unavailable",
                extra={"symbol": symbol_text, "side": side_text, "mt5_last_error": err},
            )
            return None if resolved_fail_mode == "block" else _fallback_lot()

        action: Optional[int] = None
        if side_text == "buy":
            action = int(mt5.ORDER_TYPE_BUY)
        elif side_text == "sell":
            action = int(mt5.ORDER_TYPE_SELL)
        if action is None:
            self._set_dynamic_lot_meta(reason="INVALID_SIDE", side=side_text, fail_mode=resolved_fail_mode)
            return None if resolved_fail_mode == "block" else _fallback_lot()

        if entry_price is None:
            if side_text == "buy":
                price_open = self._as_float(self._safe_getattr(tick, "ask"))
            else:
                price_open = self._as_float(self._safe_getattr(tick, "bid"))
        else:
            price_open = self._as_float(entry_price)
        if price_open is None or price_open <= 0:
            self._set_dynamic_lot_meta(reason="INVALID_ENTRY_PRICE", fail_mode=resolved_fail_mode)
            return None if resolved_fail_mode == "block" else _fallback_lot()

        if side_text == "buy" and sl >= price_open:
            self._set_dynamic_lot_meta(
                reason="INVALID_SL_DIRECTION",
                side=side_text,
                price_open=price_open,
                sl_price=sl,
                fail_mode=resolved_fail_mode,
            )
            return None if resolved_fail_mode == "block" else _fallback_lot()
        if side_text == "sell" and sl <= price_open:
            self._set_dynamic_lot_meta(
                reason="INVALID_SL_DIRECTION",
                side=side_text,
                price_open=price_open,
                sl_price=sl,
                fail_mode=resolved_fail_mode,
            )
            return None if resolved_fail_mode == "block" else _fallback_lot()

        point = self._safe_float(self._safe_getattr(info, "point"), 0.0, min_value=0.0) or 0.0
        points = 0.0
        if point > 0:
            points = abs(price_open - sl) / point

        try:
            pnl_1lot = mt5.order_calc_profit(action, symbol_text, 1.0, float(price_open), float(sl))
        except Exception as exc:
            pnl_1lot = None
            self._watchdog_debug_log(
                "risk_engine.dynamic_lot.order_calc_profit_exception",
                error=exc,
                extra={"symbol": symbol_text, "side": side_text},
            )
        if pnl_1lot is None:
            err = self._safe_mt5_last_error()
            self._set_dynamic_lot_meta(
                reason="ORDER_CALC_PROFIT_NONE",
                mt5_last_error=err,
                points=points,
                price_open=price_open,
                sl_price=sl,
                fail_mode=resolved_fail_mode,
            )
            return None if resolved_fail_mode == "block" else _fallback_lot()

        expected_loss_1lot = abs(self._as_float(pnl_1lot) or 0.0)
        if expected_loss_1lot <= 0:
            self._set_dynamic_lot_meta(
                reason="NON_POSITIVE_EXPECTED_LOSS_1LOT",
                pnl_1lot=pnl_1lot,
                points=points,
                fail_mode=resolved_fail_mode,
            )
            return None if resolved_fail_mode == "block" else _fallback_lot()

        account_balance = self._as_float(self._safe_getattr(account, "balance")) or 0.0
        account_currency = str(self._safe_getattr(account, "currency") or "")

        if risk_pct is None:
            resolved_risk_pct = float(self._resolve_risk_per_trade_pct(symbol_text)) * 100.0
        else:
            resolved_risk_pct = self._as_float(risk_pct) or 0.0
        if resolved_risk_pct <= 0:
            self._set_dynamic_lot_meta(reason="INVALID_RISK_PCT", risk_pct=resolved_risk_pct, fail_mode=resolved_fail_mode)
            return None if resolved_fail_mode == "block" else _fallback_lot()

        risk_amount = account_balance * (resolved_risk_pct / 100.0)
        raw_lot = risk_amount / expected_loss_1lot if expected_loss_1lot > 0 else 0.0

        volume_min = self._safe_float(self._safe_getattr(info, "volume_min"), 0.01, min_value=0.0) or 0.01
        volume_max = self._safe_float(self._safe_getattr(info, "volume_max"), volume_min, min_value=volume_min) or volume_min
        volume_step = self._safe_float(self._safe_getattr(info, "volume_step"), 0.01, min_value=0.0000001) or 0.01
        precision = self._step_precision(volume_step)
        stepped = math.floor(max(0.0, raw_lot) / volume_step) * volume_step
        stepped = round(stepped, precision)
        final_lot = min(max(stepped, volume_min), volume_max)
        final_lot = round(final_lot, precision)
        over_risk = False

        if raw_lot + 1e-12 < volume_min:
            pnl_min = None
            try:
                pnl_min = mt5.order_calc_profit(action, symbol_text, float(volume_min), float(price_open), float(sl))
            except Exception:
                pnl_min = None
            expected_loss_min = abs(self._as_float(pnl_min) or 0.0)
            if self.dynamic_lot_min_volume_policy == "block" and expected_loss_min > (risk_amount + 1e-12):
                err = self._safe_mt5_last_error() if pnl_min is None else None
                self._set_dynamic_lot_meta(
                    reason="MIN_VOLUME_EXCEEDS_RISK_LIMIT",
                    symbol=symbol_text,
                    side=side_text,
                    price_open=price_open,
                    sl_price=sl,
                    points=points,
                    account_balance=account_balance,
                    risk_pct=resolved_risk_pct,
                    risk_amount=risk_amount,
                    pnl_1lot=self._as_float(pnl_1lot),
                    expected_loss_1lot=expected_loss_1lot,
                    raw_lot=raw_lot,
                    final_lot=None,
                    expected_pnl_usd=None,
                    account_currency=account_currency,
                    mt5_last_error=err,
                    over_risk=True,
                    blocked=True,
                )
                self._watchdog_debug_log(
                    "risk_engine.dynamic_lot.blocked_min_volume_over_risk",
                    extra=dict(self._last_dynamic_lot_meta),
                )
                return None
            over_risk = expected_loss_min > (risk_amount + 1e-12)
            final_lot = round(volume_min, precision)

        try:
            expected_pnl_final = mt5.order_calc_profit(action, symbol_text, float(final_lot), float(price_open), float(sl))
        except Exception as exc:
            expected_pnl_final = None
            self._watchdog_debug_log(
                "risk_engine.dynamic_lot.order_calc_profit_final_exception",
                error=exc,
                extra={"symbol": symbol_text, "side": side_text, "final_lot": final_lot},
            )
        if expected_pnl_final is None:
            err = self._safe_mt5_last_error()
            self._set_dynamic_lot_meta(
                reason="ORDER_CALC_PROFIT_FINAL_NONE",
                symbol=symbol_text,
                side=side_text,
                price_open=price_open,
                sl_price=sl,
                points=points,
                account_balance=account_balance,
                risk_pct=resolved_risk_pct,
                risk_amount=risk_amount,
                pnl_1lot=self._as_float(pnl_1lot),
                expected_loss_1lot=expected_loss_1lot,
                raw_lot=raw_lot,
                final_lot=final_lot,
                expected_pnl_usd=None,
                account_currency=account_currency,
                mt5_last_error=err,
                over_risk=over_risk,
                fail_mode=resolved_fail_mode,
            )
            return None if resolved_fail_mode == "block" else _fallback_lot()

        self._set_dynamic_lot_meta(
            reason="OK",
            symbol=symbol_text,
            side=side_text,
            price_open=price_open,
            sl_price=sl,
            points=points,
            account_balance=account_balance,
            risk_pct=resolved_risk_pct,
            risk_amount=risk_amount,
            pnl_1lot=self._as_float(pnl_1lot),
            expected_loss_1lot=expected_loss_1lot,
            raw_lot=raw_lot,
            final_lot=final_lot,
            expected_pnl_usd=self._as_float(expected_pnl_final),
            account_currency=account_currency,
            mt5_last_error=None,
            over_risk=over_risk,
            fail_mode=resolved_fail_mode,
        )
        self._watchdog_debug_log(
            "risk_engine.dynamic_lot.success",
            extra=dict(self._last_dynamic_lot_meta),
        )
        return final_lot

    def plan_entry_volume(
        self,
        constraints: SymbolConstraints,
        equity: Optional[float],
        entry_price: Optional[float],
        sl_price: Optional[float],
        requested_volume: Optional[float],
        side: Optional[str] = None,
        volume_scale: float = 1.0,
        quote_to_account_rate: Optional[float] = None,
        require_fx_rate: bool = False,
        win_probability: Optional[float] = None,
        payoff_ratio: Optional[float] = None,
        symbol: Optional[str] = None,
    ) -> Tuple[Optional[float], Optional[str], Dict[str, float]]:
        try:
            eq = self._as_float(equity)
            entry = self._as_float(entry_price)
            sl = self._as_float(sl_price)
            req = self._as_float(requested_volume)
            if equity is not None and eq is None:
                self._watchdog_debug_log(
                    "risk_engine.plan_entry_volume.invalid_equity",
                    extra={"equity": equity},
                )
            if entry_price is not None and entry is None:
                self._watchdog_debug_log(
                    "risk_engine.plan_entry_volume.invalid_entry_price",
                    extra={"entry_price": entry_price},
                )
            if sl_price is not None and sl is None:
                self._watchdog_debug_log(
                    "risk_engine.plan_entry_volume.invalid_sl_price",
                    extra={"sl_price": sl_price},
                )
            if requested_volume is not None and req is None:
                self._watchdog_debug_log(
                    "risk_engine.plan_entry_volume.invalid_requested_volume",
                    extra={"requested_volume": requested_volume},
                )

            side_text = str(side or "").strip().lower()
            if self.dynamic_lot_enabled and sl is not None and side_text in {"buy", "sell"} and str(symbol or "").strip():
                dynamic_default_lot = req if req is not None and req > 0 else self.dynamic_lot_default_lot
                dynamic_lot = self.calculate_dynamic_lot(
                    symbol=str(symbol or ""),
                    side=side_text,
                    sl_price=float(sl),
                    entry_price=entry,
                    risk_pct=None,
                    default_lot=float(dynamic_default_lot),
                    fail_mode=self.dynamic_lot_fail_mode,
                )
                dynamic_meta = dict(self._last_dynamic_lot_meta or {})
                if dynamic_lot is not None:
                    meta: Dict[str, Any] = {
                        "volume_source": "dynamic_order_calc_profit",
                    }
                    for key in (
                        "points",
                        "account_balance",
                        "risk_pct",
                        "risk_amount",
                        "pnl_1lot",
                        "expected_loss_1lot",
                        "raw_lot",
                        "final_lot",
                        "expected_pnl_usd",
                    ):
                        value = self._as_float(dynamic_meta.get(key))
                        if value is not None:
                            meta[key] = float(value)
                    return float(dynamic_lot), None, meta
                if self.dynamic_lot_fail_mode == "block" or bool(dynamic_meta.get("blocked")):
                    reason = str(dynamic_meta.get("reason") or "DYNAMIC_LOT_BLOCKED")
                    out_meta: Dict[str, Any] = {}
                    for key in (
                        "points",
                        "account_balance",
                        "risk_pct",
                        "risk_amount",
                        "pnl_1lot",
                        "expected_loss_1lot",
                        "raw_lot",
                        "expected_pnl_usd",
                    ):
                        value = self._as_float(dynamic_meta.get(key))
                        if value is not None:
                            out_meta[key] = float(value)
                    return None, reason, out_meta

            if entry is None or sl is None:
                return None, "MISSING_ENTRY_OR_SL", {}

            stop_distance = abs(entry - sl)
            if stop_distance <= 0:
                self._watchdog_debug_log(
                    "risk_engine.plan_entry_volume.invalid_stop_distance",
                    extra={"entry": entry, "sl": sl, "stop_distance": stop_distance},
                )
                return None, "INVALID_STOP_DISTANCE", {}

            normalized, constraints_valid = self._normalize_constraints(constraints)
            scale, scale_valid = self._normalize_volume_scale(volume_scale)
            min_volume = normalized["min_volume"]
            contract_size = normalized["contract_size"]
            if not constraints_valid or not scale_valid:
                self._watchdog_debug_log(
                    "risk_engine.plan_entry_volume.invalid_constraints_or_scale",
                    extra={
                        "constraints_valid": constraints_valid,
                        "scale_valid": scale_valid,
                        "volume_scale": volume_scale,
                        "resolved_scale": scale,
                    },
                )
                return None, self.INVALID_CONSTRAINTS_OR_SCALE, {
                    "constraints_valid": 1.0 if constraints_valid else 0.0,
                    "scale_valid": 1.0 if scale_valid else 0.0,
                    "min_volume": normalized["min_volume"],
                    "max_volume": normalized["max_volume"],
                    "volume_step": normalized["volume_step"],
                    "contract_size": normalized["contract_size"],
                    "volume_scale": scale,
                }

            if eq is None or eq <= 0:
                # Fallback for unknown equity in dry environments.
                raw_volume = req if req is not None and req > 0 else min_volume
                scaled_raw_volume = self._safe_multiply(raw_volume, scale)
                if scaled_raw_volume is None:
                    self._watchdog_debug_log(
                        "risk_engine.plan_entry_volume.invalid_fallback_scaled_volume",
                        extra={"raw_volume": raw_volume, "scale": scale},
                    )
                    return None, self.INVALID_CONSTRAINTS_OR_SCALE, {}
                planned = self._quantize_volume(scaled_raw_volume, constraints)
                return planned, None, {
                    "stop_distance": stop_distance,
                    "risk_amount": 0.0,
                    "volume_source": "fallback_no_equity",
                    "loss_streak": float(max(0, int(self._consecutive_losses))),
                }

            self.sync_account({"equity": eq})
            components = self._effective_risk_components(
                equity=eq,
                symbol=symbol,
                win_probability=win_probability,
                payoff_ratio=payoff_ratio,
            )

            if components["drawdown_pct"] >= self.dd_hard_stop_pct or components["dd_multiplier"] <= 0.0:
                return None, "DRAWDOWN_HARD_STOP", {
                    "drawdown_pct": components["drawdown_pct"],
                    "equity_peak": components["equity_peak"],
                    "loss_streak": components["loss_streak"],
                }

            effective_risk_pct = components["effective_risk_pct"]
            if effective_risk_pct <= 0:
                return None, "RISK_MULTIPLIER_ZERO", {
                    "drawdown_pct": components["drawdown_pct"],
                    "dd_multiplier": components["dd_multiplier"],
                    "streak_multiplier": components["streak_multiplier"],
                    "kelly_multiplier": components["kelly_multiplier"],
                    "loss_streak": components["loss_streak"],
                }

            risk_amount = self._safe_multiply(eq, effective_risk_pct)
            if risk_amount is None:
                self._watchdog_debug_log(
                    "risk_engine.plan_entry_volume.invalid_risk_amount",
                    extra={"equity": eq, "effective_risk_pct": effective_risk_pct},
                )
                return None, self.INVALID_CONSTRAINTS_OR_SCALE, {}
            fx_rate = self._as_float(quote_to_account_rate)
            if quote_to_account_rate is not None and fx_rate is None:
                self._watchdog_debug_log(
                    "risk_engine.plan_entry_volume.invalid_quote_to_account_rate",
                    extra={"quote_to_account_rate": quote_to_account_rate},
                )
            if bool(require_fx_rate) and (fx_rate is None or fx_rate <= 0):
                return None, "MISSING_FX_RATE", {}
            if fx_rate is None or fx_rate <= 0:
                fx_rate = 1.0

            denom = self._safe_multiply(stop_distance, contract_size)
            denom = self._safe_multiply(denom, fx_rate) if denom is not None else None
            if denom is None or denom <= 0:
                self._watchdog_debug_log(
                    "risk_engine.plan_entry_volume.invalid_risk_denom",
                    extra={
                        "stop_distance": stop_distance,
                        "contract_size": contract_size,
                        "fx_rate": fx_rate,
                        "denom": denom,
                    },
                )
                return None, "INVALID_RISK_DENOM", {}

            risk_based_volume = self._safe_divide(risk_amount, denom)
            if risk_based_volume is None:
                self._watchdog_debug_log(
                    "risk_engine.plan_entry_volume.invalid_risk_based_volume",
                    extra={"risk_amount": risk_amount, "denom": denom},
                )
                return None, self.INVALID_CONSTRAINTS_OR_SCALE, {}

            raw_volume = risk_based_volume
            source = "risk_based"
            if req is not None and req > 0:
                raw_volume = min(raw_volume, req)
                source = "risk_capped_by_requested"

            scaled_raw_volume = self._safe_multiply(raw_volume, scale)
            if scaled_raw_volume is None:
                self._watchdog_debug_log(
                    "risk_engine.plan_entry_volume.invalid_scaled_raw_volume",
                    extra={"raw_volume": raw_volume, "scale": scale},
                )
                return None, self.INVALID_CONSTRAINTS_OR_SCALE, {}

            if scaled_raw_volume + 1e-12 < min_volume:
                min_required_risk_amount = self._safe_multiply(min_volume, stop_distance)
                min_required_risk_amount = (
                    self._safe_multiply(min_required_risk_amount, contract_size)
                    if min_required_risk_amount is not None
                    else None
                )
                min_required_risk_amount = (
                    self._safe_multiply(min_required_risk_amount, fx_rate)
                    if min_required_risk_amount is not None
                    else None
                )
                if min_required_risk_amount is None:
                    min_required_risk_amount = 0.0

                # If account risk budget is enough for the broker minimum lot,
                # prefer taking the minimum valid lot instead of hard-stopping.
                # This preserves edge exposure while avoiding silent rejection when
                # sizing is clipped below minimum by requested/scale constraints.
                
                if risk_amount >= (min_required_risk_amount or 0.0):
                    floor_volume = self._quantize_volume(min_volume, constraints)
                    if floor_volume >= min_volume:
                        return floor_volume, None, {
                            "stop_distance": stop_distance,
                            "risk_amount": risk_amount,
                            "raw_volume": max(0.0, scaled_raw_volume),
                            "min_volume": min_volume,
                            "min_required_risk_amount": min_required_risk_amount,
                            "volume_source": "min_volume_risk_floor",
                            "quote_to_account_rate": fx_rate,
                            "risk_pct_effective": effective_risk_pct,
                            "risk_pct_base": components["base_risk_pct"],
                            "risk_pct_pre_multipliers": components["kelly_risk_pct"],
                            "kelly_multiplier": components["kelly_multiplier"],
                            "dd_multiplier": components["dd_multiplier"],
                            "streak_multiplier": components["streak_multiplier"],
                            "drawdown_pct": components["drawdown_pct"],
                            "equity_peak": components["equity_peak"],
                            "loss_streak": components["loss_streak"],
                        }

                return None, "MIN_VOLUME_EXCEEDS_RISK_LIMIT", {
                    "stop_distance": stop_distance,
                    "risk_amount": risk_amount,
                    "raw_volume": max(0.0, scaled_raw_volume),
                    "min_volume": min_volume,
                    "min_required_risk_amount": min_required_risk_amount,
                    "quote_to_account_rate": fx_rate,
                    "risk_pct_effective": effective_risk_pct,
                    "risk_pct_base": components["base_risk_pct"],
                    "risk_pct_pre_multipliers": components["kelly_risk_pct"],
                    "kelly_multiplier": components["kelly_multiplier"],
                    "dd_multiplier": components["dd_multiplier"],
                    "streak_multiplier": components["streak_multiplier"],
                    "drawdown_pct": components["drawdown_pct"],
                    "equity_peak": components["equity_peak"],
                    "loss_streak": components["loss_streak"],
                }

            planned = self._quantize_volume(scaled_raw_volume, constraints)
            if planned <= 0:
                return None, "NON_POSITIVE_VOLUME", {}

            risk_amount_quote = self._safe_divide(risk_amount, fx_rate) if fx_rate > 0 else risk_amount
            if risk_amount_quote is None:
                risk_amount_quote = risk_amount
            return planned, None, {
                "stop_distance": stop_distance,
                "risk_amount": risk_amount,
                "risk_amount_quote": risk_amount_quote,
                "quote_to_account_rate": fx_rate,
                "volume_source": source,
                "risk_pct_base": components["base_risk_pct"],
                "risk_pct_pre_multipliers": components["kelly_risk_pct"],
                "risk_pct_effective": effective_risk_pct,
                "kelly_multiplier": components["kelly_multiplier"],
                "dd_multiplier": components["dd_multiplier"],
                "streak_multiplier": components["streak_multiplier"],
                "drawdown_pct": components["drawdown_pct"],
                "equity_peak": components["equity_peak"],
                "loss_streak": components["loss_streak"],
            }
        except Exception as exc:
            self._watchdog_debug_log(
                "risk_engine.plan_entry_volume.exception",
                error=exc,
                extra={
                    "entry_price": entry_price,
                    "sl_price": sl_price,
                    "requested_volume": requested_volume,
                    "volume_scale": volume_scale,
                },
            )
            return None, self.INVALID_CONSTRAINTS_OR_SCALE, {}
