from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.models import DecisionAction, OrderResult, Position, Side, StrategyDecision
from strategies.liquidity_sweep_reversal import LiquiditySweepReversalStrategy


TARGET_SYMBOLS = ("GOLD", "BTCUSD", "GBPJPY")
SYMBOL_ALIASES: Dict[str, tuple[str, ...]] = {
    "GOLD": ("GOLD", "XAUUSD", "XAUUSDM"),
    "BTCUSD": ("BTCUSD", "BTCUSDM"),
    "GBPJPY": ("GBPJPY", "GBPJPYM"),
}

BASE_LSR_PARAMS: Dict[str, Any] = {
    "enabled": True,
    "atr_period": 14,
    "pivot_lookback_sec": 1800,
    "swing_window": 60,
    "sweep_buffer_atr": 0.25,
    "reclaim_buffer_atr": 0.05,
    "reclaim_window_sec": 15,
    "displacement_mult": 1.6,
    "displacement_lookback": 20,
    "sl_atr_mult": 0.8,
    "stop_buffer_atr": 0.05,
    "tp_R1": 1.2,
    "tp_R2": 2.5,
    "be_at_R": 1.0,
    "max_hold_bars": 120,
    "min_hold_bars": 1,
    "min_cooldown_bars": 5,
}

FALLBACK_PARAMS: Dict[str, Dict[str, Any]] = {
    "GOLD": {
        "atr_period": 14,
        "pivot_lookback_sec": 1800,
        "swing_window": 60,
        "sweep_buffer_atr": 0.20,
        "reclaim_buffer_atr": 0.05,
        "reclaim_window_sec": 60,
        "displacement_mult": 1.35,
        "displacement_lookback": 20,
        "sl_atr_mult": 0.75,
        "stop_buffer_atr": 0.05,
        "tp_R1": 1.20,
        "tp_R2": 2.50,
        "be_at_R": 1.00,
        "max_hold_bars": 120,
        "min_hold_bars": 1,
        "min_cooldown_bars": 5,
    },
    "BTCUSD": {
        "atr_period": 14,
        "pivot_lookback_sec": 1800,
        "swing_window": 60,
        "sweep_buffer_atr": 0.25,
        "reclaim_buffer_atr": 0.06,
        "reclaim_window_sec": 120,
        "displacement_mult": 1.55,
        "displacement_lookback": 20,
        "sl_atr_mult": 0.95,
        "stop_buffer_atr": 0.08,
        "tp_R1": 1.30,
        "tp_R2": 2.80,
        "be_at_R": 1.10,
        "max_hold_bars": 180,
        "min_hold_bars": 1,
        "min_cooldown_bars": 5,
    },
    "GBPJPY": {
        "atr_period": 14,
        "pivot_lookback_sec": 1800,
        "swing_window": 60,
        "sweep_buffer_atr": 0.22,
        "reclaim_buffer_atr": 0.05,
        "reclaim_window_sec": 90,
        "displacement_mult": 1.45,
        "displacement_lookback": 20,
        "sl_atr_mult": 0.85,
        "stop_buffer_atr": 0.06,
        "tp_R1": 1.20,
        "tp_R2": 2.60,
        "be_at_R": 1.00,
        "max_hold_bars": 150,
        "min_hold_bars": 1,
        "min_cooldown_bars": 5,
    },
}

SEARCH_SPACE: Dict[str, tuple[Any, ...]] = {
    "sweep_buffer_atr": (0.14, 0.20, 0.28),
    "reclaim_window_sec": (30, 60, 120),
    "displacement_mult": (1.25, 1.40, 1.60),
    "sl_atr_mult": (0.60, 0.80, 1.00),
    "tp_R1": (1.00, 1.20, 1.40),
    "tp_R2": (2.20, 2.60, 3.00),
    "be_at_R": (0.80, 1.00, 1.20),
    "max_hold_bars": (80, 120, 180),
}

SEARCH_SPACE_QUICK: Dict[str, tuple[Any, ...]] = {
    "sweep_buffer_atr": (0.14, 0.22),
    "reclaim_window_sec": (30, 90),
    "displacement_mult": (1.25, 1.50),
    "sl_atr_mult": (0.60, 0.90),
    "tp_R2": (2.20, 2.80),
}

@dataclass
class CoverageInfo:
    file_path: str
    rows: int
    used_rows: int
    start_utc: str
    end_utc: str
    timeframe_seconds: int
    mode: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "file_path": self.file_path,
            "rows": int(self.rows),
            "used_rows": int(self.used_rows),
            "start_utc": self.start_utc,
            "end_utc": self.end_utc,
            "timeframe_seconds": int(self.timeframe_seconds),
            "mode": self.mode,
        }


def _normalize_symbol(text: Any) -> str:
    raw = str(text or "").strip().upper()
    if not raw:
        return ""
    for canonical, aliases in SYMBOL_ALIASES.items():
        if raw in aliases:
            return canonical
    return raw


def _parse_symbol_filter(raw: str) -> list[str]:
    values = []
    for token in str(raw or "").split(","):
        symbol = _normalize_symbol(token)
        if symbol and symbol not in values:
            values.append(symbol)
    return values if values else list(TARGET_SYMBOLS)


def _discover_csv_files(data_path: Path) -> list[Path]:
    if data_path.is_file():
        return [data_path] if data_path.suffix.lower() == ".csv" else []
    if not data_path.exists():
        return []
    return sorted(path for path in data_path.rglob("*.csv") if path.is_file())


def _infer_symbol_from_filename(path: Path) -> str:
    stem = path.stem.upper()
    match = re.match(r"^([A-Z0-9]+)[_-]TIMEFRAME", stem)
    if match:
        return _normalize_symbol(match.group(1))
    token = re.split(r"[_-]", stem)[0]
    return _normalize_symbol(token)


def _infer_timeframe_hint(path: Path) -> str:
    stem = path.stem.upper()
    match = re.search(r"TIMEFRAME[_-]([A-Z0-9]+)", stem)
    if match:
        return match.group(1)
    return ""


def _coerce_ohlc_frame(raw: pd.DataFrame) -> Optional[pd.DataFrame]:
    if raw is None or raw.empty:
        return None
    lowered = {str(col).lower(): col for col in raw.columns}
    required = ("open", "high", "low", "close")
    if any(name not in lowered for name in required):
        return None

    out = pd.DataFrame()
    for name in required:
        out[name] = pd.to_numeric(raw[lowered[name]], errors="coerce")

    time_col = ""
    for candidate in ("time", "timestamp", "date", "datetime"):
        if candidate in lowered:
            time_col = lowered[candidate]
            break
    if time_col:
        parsed = pd.to_datetime(raw[time_col], errors="coerce", utc=True)
        if parsed.notna().sum() <= 0:
            as_num = pd.to_numeric(raw[time_col], errors="coerce")
            parsed = pd.to_datetime(as_num, unit="s", errors="coerce", utc=True)
        out["time"] = parsed
    else:
        out["time"] = pd.date_range(
            start=datetime.now(timezone.utc),
            periods=len(out),
            freq="min",
            tz="UTC",
        )

    out.dropna(subset=["time", "open", "high", "low", "close"], inplace=True)
    out.sort_values(by="time", inplace=True, kind="stable")
    out.reset_index(drop=True, inplace=True)
    if out.empty:
        return None
    return out


def _median_timeframe_seconds(frame: pd.DataFrame) -> int:
    if frame is None or len(frame) < 2:
        return 0
    raw = pd.to_datetime(frame["time"], utc=True, errors="coerce").dropna()
    if len(raw) < 2:
        return 0
    diff = raw.diff().dropna().dt.total_seconds()
    if diff.empty:
        return 0
    try:
        value = float(diff.median())
    except (TypeError, ValueError):
        return 0
    if not math.isfinite(value) or value <= 0:
        return 0
    return int(round(value))


def _best_symbol_data(
    files: Iterable[Path],
    symbol: str,
    mode: str,
) -> tuple[Optional[pd.DataFrame], Optional[CoverageInfo], Optional[str]]:
    symbol_upper = _normalize_symbol(symbol)
    best: Optional[pd.DataFrame] = None
    best_coverage: Optional[CoverageInfo] = None
    rejection_reason = "missing_data_file"

    for path in files:
        inferred_symbol = _infer_symbol_from_filename(path)
        if inferred_symbol != symbol_upper:
            continue

        timeframe_hint = _infer_timeframe_hint(path)
        if mode == "m1" and timeframe_hint and timeframe_hint != "M1":
            rejection_reason = "no_m1_data"
            continue

        try:
            raw = pd.read_csv(path)
        except Exception:
            rejection_reason = "csv_read_failed"
            continue
        frame = _coerce_ohlc_frame(raw)
        if frame is None or frame.empty:
            rejection_reason = "ohlc_invalid"
            continue

        tf_seconds = _median_timeframe_seconds(frame)
        if mode == "m1" and tf_seconds and tf_seconds > 90:
            rejection_reason = "timeframe_not_m1"
            continue

        mode_frame = frame
        mode_name = mode
        if mode == "tick":
            mode_frame = _expand_tick_like_bars(frame, tf_seconds if tf_seconds > 0 else 60)
            mode_name = "tick"
            tf_seconds = max(1, int(round((tf_seconds if tf_seconds > 0 else 60) / 4.0)))

        if best is None or len(mode_frame) > len(best):
            start = pd.to_datetime(mode_frame["time"].iloc[0], utc=True, errors="coerce")
            end = pd.to_datetime(mode_frame["time"].iloc[-1], utc=True, errors="coerce")
            best = mode_frame
            best_coverage = CoverageInfo(
                file_path=str(path),
                rows=int(len(frame)),
                used_rows=int(len(mode_frame)),
                start_utc=start.isoformat() if pd.notna(start) else "",
                end_utc=end.isoformat() if pd.notna(end) else "",
                timeframe_seconds=max(1, tf_seconds) if tf_seconds else 60,
                mode=mode_name,
            )

    return best, best_coverage, rejection_reason


def _expand_tick_like_bars(frame: pd.DataFrame, bar_seconds: int) -> pd.DataFrame:
    interval = max(1, int(bar_seconds))
    micro_step_ms = max(1, int((interval * 1000) // 4))
    rows: list[dict[str, Any]] = []

    for row in frame.itertuples(index=False):
        time_value = pd.to_datetime(getattr(row, "time"), utc=True, errors="coerce")
        if pd.isna(time_value):
            continue
        open_price = float(getattr(row, "open"))
        high_price = float(getattr(row, "high"))
        low_price = float(getattr(row, "low"))
        close_price = float(getattr(row, "close"))
        if close_price >= open_price:
            points = (open_price, low_price, high_price, close_price)
        else:
            points = (open_price, high_price, low_price, close_price)
        for idx in range(3):
            p_open = float(points[idx])
            p_close = float(points[idx + 1])
            rows.append(
                {
                    "time": time_value + timedelta(milliseconds=micro_step_ms * idx),
                    "open": p_open,
                    "high": max(p_open, p_close),
                    "low": min(p_open, p_close),
                    "close": p_close,
                }
            )

    out = pd.DataFrame(rows)
    out.sort_values(by="time", inplace=True, kind="stable")
    out.reset_index(drop=True, inplace=True)
    return out


def _apply_trade_close(
    *,
    side: Side,
    entry_price: float,
    exit_price: float,
    close_volume: float,
) -> float:
    direction = 1.0 if side == Side.BUY else -1.0
    return (float(exit_price) - float(entry_price)) * direction * float(close_volume)


def _trade_score(metrics: Dict[str, Any]) -> float:
    trades = int(metrics.get("trades", 0))
    if trades <= 0:
        return -9999.0

    total_r = float(metrics.get("total_r", 0.0))
    expectancy = float(metrics.get("expectancy_r", 0.0))
    win_rate = float(metrics.get("win_rate", 0.0))
    max_drawdown = float(metrics.get("max_drawdown_r", 0.0))
    profit_factor = float(metrics.get("profit_factor", 0.0))
    profit_factor = min(3.0, max(0.0, profit_factor))

    score = (expectancy * 1.8) + (total_r * 0.35) + (win_rate * 0.5) + (profit_factor * 0.2) - (max_drawdown * 0.8)
    if trades < 3:
        score -= (3 - trades) * 0.8
    return float(score)


def _simulate_symbol(
    symbol: str,
    frame: pd.DataFrame,
    params: Dict[str, Any],
) -> Dict[str, Any]:
    sim = frame.copy().reset_index(drop=True)
    if len(sim) > 6000:
        sim = sim.tail(6000).reset_index(drop=True)

    strategy = LiquiditySweepReversalStrategy(config=dict(params))
    ticket = 1
    position: Optional[Position] = None
    trade_state: Optional[Dict[str, Any]] = None
    trades_r: list[float] = []
    entries = 0
    partial_closes = 0

    window = 450
    for closed_idx in range(120, len(sim) - 1):
        current_bar = sim.iloc[closed_idx]
        current_close = float(current_bar["close"])
        current_high = float(current_bar["high"])
        current_low = float(current_bar["low"])

        if position is not None and trade_state is not None:
            stop_hit = False
            target_hit = False
            stop_price = float(position.sl) if position.sl is not None else None
            target_price = float(position.tp) if position.tp is not None else None

            if position.side == Side.BUY:
                stop_hit = stop_price is not None and current_low <= stop_price
                target_hit = target_price is not None and current_high >= target_price
            else:
                stop_hit = stop_price is not None and current_high >= stop_price
                target_hit = target_price is not None and current_low <= target_price

            auto_exit_price = None
            auto_exit_reason = ""
            if stop_hit and target_hit:
                auto_exit_price = stop_price
                auto_exit_reason = "AUTO_STOP_HIT_SAME_BAR"
            elif stop_hit:
                auto_exit_price = stop_price
                auto_exit_reason = "AUTO_STOP_HIT"
            elif target_hit:
                auto_exit_price = target_price
                auto_exit_reason = "AUTO_TARGET_HIT"

            if auto_exit_price is not None:
                close_volume = float(position.volume)
                pnl_value = _apply_trade_close(
                    side=position.side,
                    entry_price=float(trade_state["entry_price"]),
                    exit_price=float(auto_exit_price),
                    close_volume=close_volume,
                )
                trade_state["pnl_value"] = float(trade_state.get("pnl_value", 0.0)) + pnl_value
                result = OrderResult(
                    ok=True,
                    status="CLOSED_AUTO",
                    ticket=position.ticket,
                    filled_price=float(auto_exit_price),
                    pnl=float(pnl_value),
                    message=auto_exit_reason,
                )
                decision = StrategyDecision(
                    action=DecisionAction.EXIT,
                    reason=auto_exit_reason,
                    strategy=strategy.name,
                    metadata={"is_partial": False},
                )
                strategy.apply_order_result(symbol, decision, result)
                risk_value = max(1e-9, float(trade_state["risk_value"]))
                trades_r.append(float(trade_state["pnl_value"]) / risk_value)
                position = None
                trade_state = None

        start = max(0, closed_idx + 2 - window)
        bars_window = sim.iloc[start : closed_idx + 2].reset_index(drop=True)
        decision = strategy.evaluate(symbol=symbol, bars=bars_window, position=position)

        if position is not None and decision.action == DecisionAction.HOLD:
            if decision.sl is not None:
                position.sl = float(decision.sl)
            if decision.tp is not None:
                position.tp = float(decision.tp)

        if decision.action in {DecisionAction.BUY, DecisionAction.SELL} and position is None:
            if decision.sl is None:
                continue
            entry_price = float(decision.metadata.get("signal_close", current_close))
            side = Side.BUY if decision.action == DecisionAction.BUY else Side.SELL
            risk_per_unit = abs(entry_price - float(decision.sl))
            if risk_per_unit <= 0:
                continue
            entry_volume = 1.0
            position = Position(
                ticket=ticket,
                symbol=symbol,
                side=side,
                volume=entry_volume,
                price_open=entry_price,
                sl=float(decision.sl),
                tp=float(decision.tp) if decision.tp is not None else None,
                metadata={"min_volume": 0.01, "volume_step": 0.01},
            )
            trade_state = {
                "entry_price": entry_price,
                "risk_value": risk_per_unit * entry_volume,
                "pnl_value": 0.0,
            }
            fill = OrderResult(ok=True, status="FILLED_SIM", ticket=ticket, filled_price=entry_price)
            strategy.apply_order_result(symbol, decision, fill)
            ticket += 1
            entries += 1
            continue

        if decision.action == DecisionAction.EXIT and position is not None and trade_state is not None:
            requested = float(decision.volume) if decision.volume is not None else float(position.volume)
            close_volume = min(float(position.volume), max(0.0, requested))
            if close_volume <= 0.0:
                continue

            pnl_value = _apply_trade_close(
                side=position.side,
                entry_price=float(trade_state["entry_price"]),
                exit_price=current_close,
                close_volume=close_volume,
            )
            trade_state["pnl_value"] = float(trade_state.get("pnl_value", 0.0)) + pnl_value

            is_partial = close_volume < float(position.volume) - 1e-9
            result = OrderResult(
                ok=True,
                status="CLOSED_PARTIAL_SIM" if is_partial else "CLOSED_SIM",
                ticket=position.ticket,
                filled_price=current_close,
                pnl=float(pnl_value),
                message=decision.reason,
            )
            strategy.apply_order_result(symbol, decision, result)

            if is_partial:
                position.volume = max(0.0, float(position.volume) - close_volume)
                partial_closes += 1
            else:
                risk_value = max(1e-9, float(trade_state["risk_value"]))
                trades_r.append(float(trade_state["pnl_value"]) / risk_value)
                position = None
                trade_state = None

    if position is not None and trade_state is not None:
        final_close = float(sim.iloc[-1]["close"])
        close_volume = float(position.volume)
        pnl_value = _apply_trade_close(
            side=position.side,
            entry_price=float(trade_state["entry_price"]),
            exit_price=final_close,
            close_volume=close_volume,
        )
        trade_state["pnl_value"] = float(trade_state.get("pnl_value", 0.0)) + pnl_value
        risk_value = max(1e-9, float(trade_state["risk_value"]))
        trades_r.append(float(trade_state["pnl_value"]) / risk_value)

    wins = [value for value in trades_r if value > 0]
    losses = [value for value in trades_r if value < 0]
    total_r = float(sum(trades_r))
    expectancy = float(total_r / len(trades_r)) if trades_r else 0.0
    win_rate = float(len(wins) / len(trades_r)) if trades_r else 0.0
    gross_win = float(sum(wins))
    gross_loss = float(abs(sum(losses)))
    profit_factor = float(gross_win / gross_loss) if gross_loss > 1e-12 else (99.0 if gross_win > 0 else 0.0)

    equity = 0.0
    peak = 0.0
    max_dd = 0.0
    for trade_r in trades_r:
        equity += float(trade_r)
        peak = max(peak, equity)
        max_dd = max(max_dd, peak - equity)

    metrics = {
        "trades": int(len(trades_r)),
        "entries": int(entries),
        "partial_closes": int(partial_closes),
        "total_r": float(total_r),
        "expectancy_r": float(expectancy),
        "win_rate": float(win_rate),
        "profit_factor": float(profit_factor),
        "max_drawdown_r": float(max_dd),
    }
    metrics["score"] = _trade_score(metrics)
    return metrics


def _coerce_candidate_params(candidate: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(BASE_LSR_PARAMS)
    out.update(candidate or {})
    out["enabled"] = True
    out["tp_R1"] = max(0.1, float(out.get("tp_R1", BASE_LSR_PARAMS["tp_R1"])))
    out["tp_R2"] = max(float(out["tp_R1"]) + 0.1, float(out.get("tp_R2", BASE_LSR_PARAMS["tp_R2"])))
    out["min_hold_bars"] = max(1, int(out.get("min_hold_bars", BASE_LSR_PARAMS["min_hold_bars"])))
    out["max_hold_bars"] = max(2, int(out.get("max_hold_bars", BASE_LSR_PARAMS["max_hold_bars"])))
    out["min_cooldown_bars"] = max(1, int(out.get("min_cooldown_bars", BASE_LSR_PARAMS["min_cooldown_bars"])))
    out["atr_period"] = max(3, int(out.get("atr_period", BASE_LSR_PARAMS["atr_period"])))
    out["swing_window"] = max(5, int(out.get("swing_window", BASE_LSR_PARAMS["swing_window"])))
    out["displacement_lookback"] = max(5, int(out.get("displacement_lookback", BASE_LSR_PARAMS["displacement_lookback"])))
    out["pivot_lookback_sec"] = max(120, int(out.get("pivot_lookback_sec", BASE_LSR_PARAMS["pivot_lookback_sec"])))
    out["reclaim_window_sec"] = max(1, int(out.get("reclaim_window_sec", BASE_LSR_PARAMS["reclaim_window_sec"])))
    out["displacement_mult"] = max(1.0, float(out.get("displacement_mult", BASE_LSR_PARAMS["displacement_mult"])))
    out["sl_atr_mult"] = max(0.1, float(out.get("sl_atr_mult", BASE_LSR_PARAMS["sl_atr_mult"])))
    out["be_at_R"] = max(0.1, float(out.get("be_at_R", BASE_LSR_PARAMS["be_at_R"])))
    out["sweep_buffer_atr"] = max(0.0, float(out.get("sweep_buffer_atr", BASE_LSR_PARAMS["sweep_buffer_atr"])))
    out["reclaim_buffer_atr"] = max(0.0, float(out.get("reclaim_buffer_atr", BASE_LSR_PARAMS["reclaim_buffer_atr"])))
    out["stop_buffer_atr"] = max(0.0, float(out.get("stop_buffer_atr", BASE_LSR_PARAMS["stop_buffer_atr"])))
    return out


def _tune_symbol(
    symbol: str,
    frame: pd.DataFrame,
    search_space: Dict[str, tuple[Any, ...]],
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    params = _coerce_candidate_params(dict(FALLBACK_PARAMS.get(symbol, BASE_LSR_PARAMS)))
    best_metrics = _simulate_symbol(symbol=symbol, frame=frame, params=params)

    for key, candidates in search_space.items():
        local_best_params = dict(params)
        local_best_metrics = dict(best_metrics)
        for candidate in candidates:
            proposal = dict(params)
            proposal[key] = candidate
            proposal = _coerce_candidate_params(proposal)
            metrics = _simulate_symbol(symbol=symbol, frame=frame, params=proposal)

            current_key = (
                float(metrics.get("score", -9999.0)),
                int(metrics.get("trades", 0)),
                float(metrics.get("total_r", -9999.0)),
            )
            best_key = (
                float(local_best_metrics.get("score", -9999.0)),
                int(local_best_metrics.get("trades", 0)),
                float(local_best_metrics.get("total_r", -9999.0)),
            )
            if current_key > best_key:
                local_best_params = proposal
                local_best_metrics = metrics

        params = local_best_params
        best_metrics = local_best_metrics

    return params, best_metrics


def _fallback_payload(symbol: str, reason: str) -> Dict[str, Any]:
    params = _coerce_candidate_params(dict(FALLBACK_PARAMS.get(symbol, BASE_LSR_PARAMS)))
    return {
        "symbol": symbol,
        "source": "fallback",
        "fallback": True,
        "fallback_reason": reason,
        "params": params,
        "metrics": {},
        "coverage": None,
    }


def run_simulation(
    data_path: Path,
    symbols: list[str],
    mode: str,
    *,
    max_rows_per_symbol: int = 0,
    quick: bool = False,
) -> Dict[str, Any]:
    files = _discover_csv_files(data_path)
    results: Dict[str, Any] = {}

    for symbol in symbols:
        frame, coverage, data_reason = _best_symbol_data(files=files, symbol=symbol, mode=mode)
        if frame is None or coverage is None:
            results[symbol] = _fallback_payload(symbol, data_reason or "missing_data")
            continue

        if max_rows_per_symbol > 0 and len(frame) > max_rows_per_symbol:
            frame = frame.tail(max_rows_per_symbol).reset_index(drop=True)
        min_rows = 900 if mode == "m1" else 700
        if len(frame) < min_rows:
            results[symbol] = _fallback_payload(symbol, f"insufficient_rows:{len(frame)}<{min_rows}")
            results[symbol]["coverage"] = coverage.to_dict()
            continue

        tuned_params, metrics = _tune_symbol(
            symbol=symbol,
            frame=frame,
            search_space=SEARCH_SPACE_QUICK if quick else SEARCH_SPACE,
        )
        if int(metrics.get("trades", 0)) < 3:
            results[symbol] = _fallback_payload(symbol, f"insufficient_trades:{int(metrics.get('trades', 0))}")
            results[symbol]["coverage"] = coverage.to_dict()
            results[symbol]["metrics"] = metrics
            continue

        results[symbol] = {
            "symbol": symbol,
            "source": "tuned",
            "fallback": False,
            "fallback_reason": "",
            "params": tuned_params,
            "metrics": metrics,
            "coverage": coverage.to_dict(),
        }

    tuned_symbols = [symbol for symbol, payload in results.items() if not bool(payload.get("fallback", True))]
    fallback_symbols = [symbol for symbol, payload in results.items() if bool(payload.get("fallback", False))]

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_path": str(data_path.resolve()),
        "mode": mode,
        "quick": bool(quick),
        "max_rows_per_symbol": int(max_rows_per_symbol),
        "symbols": symbols,
        "tuned_symbols": tuned_symbols,
        "fallback_symbols": fallback_symbols,
        "results": results,
    }
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Mini LSR backtest simulator and parameter tuner.")
    parser.add_argument(
        "--data-path",
        default=str(PROJECT_ROOT / "data"),
        help="CSV file or directory that contains historical bars/tick-like data.",
    )
    parser.add_argument(
        "--symbol-filter",
        default="GOLD,BTCUSD,GBPJPY",
        help="Comma-separated symbol filter. Default: GOLD,BTCUSD,GBPJPY",
    )
    parser.add_argument(
        "--mode",
        choices=("m1", "tick"),
        default="m1",
        help="Replay mode: m1 for 1-minute bars, tick for synthetic tick-like replay.",
    )
    parser.add_argument(
        "--output-json",
        default="",
        help="Optional output JSON file path. Prints JSON to stdout regardless.",
    )
    parser.add_argument(
        "--max-rows-per-symbol",
        type=int,
        default=0,
        help="Limit replay rows per symbol (tail rows). 0 means full data.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use a reduced search space for fast first-pass tuning.",
    )
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    symbols = _parse_symbol_filter(args.symbol_filter)
    data_path = Path(str(args.data_path)).expanduser()
    summary = run_simulation(
        data_path=data_path,
        symbols=symbols,
        mode=str(args.mode),
        max_rows_per_symbol=max(0, int(args.max_rows_per_symbol or 0)),
        quick=bool(args.quick),
    )
    encoded = json.dumps(summary, indent=2, sort_keys=True)
    print(encoded)

    output_path = str(args.output_json or "").strip()
    if output_path:
        out = Path(output_path).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(encoded + "\n", encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
