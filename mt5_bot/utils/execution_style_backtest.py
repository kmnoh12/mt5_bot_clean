from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any, Dict, Iterable, List, Optional, Sequence

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None  # type: ignore[assignment]

from core.no_trade_guard import NoTradeBiasGuard
from core.risk_model import FeeAwareRiskInput, FeeAwareRiskModel
from execution.daily_bleed_guard import DailyBleedGuard
from execution.position_sizer import NetRiskPositionSizeInput, NetRiskPositionSizer, SymbolVolumeSpec
from strategies.entry_filter import FeeAwareEntryFilter
from strategies.opportunity_scanner import TradeOpportunityScanner


@dataclass(frozen=True)
class ExecutionStyleBacktestConfig:
    symbol: str = "TESTUSD"
    timeframe: str = "M1"
    target_net_loss_usd: float = 1.0
    hard_max_net_loss_usd: float = 1.25
    max_daily_net_loss_usd: float = 3.0
    min_reward_to_net_risk_ratio: float = 3.0
    min_signal_score: float = 70.0
    tick_size: float = 0.01
    tick_value: float = 0.01
    contract_size: float = 1.0
    volume_min: float = 0.01
    volume_step: float = 0.01
    volume_max: float = 100.0
    spread_points: float = 0.0
    commission_per_lot: float = 0.0
    expected_slippage_points: float = 0.0
    max_open_positions_total: int = 1
    max_bars_in_trade: int = 60
    scanner_lookback_bars: int = 20
    scanner_atr_period: int = 14
    sweep_buffer_atr: float = 0.05
    stop_buffer_atr: float = 0.05
    initial_balance: float = 10000.0
    account_equity: float = 10000.0
    account_leverage: float = 500.0
    max_effective_leverage: float = 0.0
    max_margin_used_pct: float = 0.0


@dataclass
class SimulatedTrade:
    symbol: str
    direction: str
    entry_index: int
    exit_index: int
    entry_time: str
    exit_time: str
    entry_price: float
    exit_price: float
    sl_price: float
    tp_price: Optional[float]
    lot: float
    net_pnl: float
    gross_pnl: float
    cost_usd: float
    spread_cost_usd: float
    slippage_cost_usd: float
    commission_cost_usd: float
    implementation_shortfall_usd: float
    exit_reason: str
    max_favorable_excursion: float = 0.0
    max_adverse_excursion: float = 0.0
    profit_lock_sl_moves: int = 0
    profit_lock_saved_pnl: float = 0.0
    block_reasons_before_entry: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class _OpenPosition:
    opportunity: Any
    direction: str
    entry_index: int
    entry_time: str
    entry_price: float
    sl_price: float
    tp_price: Optional[float]
    original_sl_price: float
    lot: float
    cost_usd: float
    spread_cost_usd: float
    slippage_cost_usd: float
    commission_cost_usd: float
    implementation_shortfall_usd: float
    units: float
    max_favorable: float = 0.0
    max_adverse: float = 0.0
    sl_moves: int = 0
    profit_lock_saved_pnl: float = 0.0
    highest_locked_pnl: float = -1.0
    block_reasons_before_entry: List[str] = field(default_factory=list)


PROFIT_LOCK_STAGES: Sequence[tuple[float, float, Optional[float]]] = (
    (30.0, 15.0, 45.0),
    (20.0, 10.0, 30.0),
    (10.0, 5.0, 20.0),
    (5.0, 2.0, None),
    (3.0, 1.0, None),
    (2.0, 0.0, None),
)


def run_execution_style_backtest(
    bars: Any,
    config: Optional[ExecutionStyleBacktestConfig | Dict[str, Any]] = None,
) -> Dict[str, Any]:
    cfg = _coerce_config(config)
    frame = _coerce_frame(bars)
    records = frame.to_dict("records")
    scanner = TradeOpportunityScanner(
        lookback_bars=cfg.scanner_lookback_bars,
        atr_period=cfg.scanner_atr_period,
        min_signal_score=cfg.min_signal_score,
        sweep_buffer_atr=cfg.sweep_buffer_atr,
        stop_buffer_atr=cfg.stop_buffer_atr,
    )
    risk_model = FeeAwareRiskModel()
    sizer = NetRiskPositionSizer(risk_model)
    entry_filter = FeeAwareEntryFilter(
        {
            "min_signal_score": cfg.min_signal_score,
            "min_reward_to_net_risk_ratio": cfg.min_reward_to_net_risk_ratio,
            "hard_max_net_loss_usd": cfg.hard_max_net_loss_usd,
            "max_open_positions": cfg.max_open_positions_total,
        }
    )
    bleed_guard = DailyBleedGuard(
        {
            "enabled": True,
            "max_daily_net_loss_usd": cfg.max_daily_net_loss_usd,
            "stop_after_consecutive_losses": 3,
            "cooldown_after_loss_minutes": 30,
            "cooldown_after_same_setup_loss_minutes": 60,
            "same_direction_loss_limit_per_day": 2,
            "same_symbol_loss_limit_per_day": 3,
        }
    )
    no_trade_guard = NoTradeBiasGuard({"warning_no_trade_hours": 24.0, "failure_no_trade_hours": 48.0})
    symbol_spec = SymbolVolumeSpec(
        volume_min=cfg.volume_min,
        volume_step=cfg.volume_step,
        volume_max=cfg.volume_max,
        tick_size=cfg.tick_size,
        tick_value=cfg.tick_value,
        contract_size=cfg.contract_size,
    )

    trades: List[SimulatedTrade] = []
    block_counts: Dict[str, int] = {}
    daily_pnl: Dict[str, float] = {}
    raw_signal_count = scored_signal_count = eligible_signal_count = 0
    open_position: Optional[_OpenPosition] = None
    spread_price = cfg.spread_points * cfg.tick_size

    scan_start = max(cfg.scanner_lookback_bars + 1, cfg.scanner_atr_period + 1)
    scan_window = max(cfg.scanner_lookback_bars + 1, cfg.scanner_atr_period + 1)
    for idx in range(scan_start, len(records)):
        current = records[idx]
        now_ts = _row_ts(current)
        if open_position is not None:
            closed = _advance_position(open_position, current, idx, cfg)
            if closed is not None:
                trades.append(closed)
                daily_pnl[_date_key(closed.exit_time)] = daily_pnl.get(_date_key(closed.exit_time), 0.0) + closed.net_pnl
                bleed_guard.record_trade_close(
                    symbol=cfg.symbol,
                    realized_pnl=closed.net_pnl,
                    now_ts=now_ts,
                    direction=open_position.direction,
                    setup_key="v4_execution_style_backtest",
                )
                open_position = None
            else:
                continue

        # Scanner only needs the latest short context. Passing frame.iloc[:idx]
        # made full CSV stress runs O(n^2) because every tick reconverted the
        # entire prefix to dict records.
        window = records[max(0, idx - scan_window + 1) : idx + 1]
        opportunities = scanner.scan(
            symbol=cfg.symbol,
            timeframe=cfg.timeframe,
            bars=window,
            spread=spread_price,
            min_signal_score=cfg.min_signal_score,
            detected_at_utc=_row_datetime(current),
        )
        for opportunity in opportunities:
            raw_signal_count += 1
            scored_signal_count += 1
            no_trade_guard.record_raw_signal(opportunity, now_ts=now_ts)
            no_trade_guard.record_scored_signal(opportunity, now_ts=now_ts)
            entry = float(opportunity.entry_price)
            stop = float(opportunity.invalidation_price)
            direction = "long" if opportunity.direction == "long" else "short"
            size = sizer.size(
                NetRiskPositionSizeInput(
                    symbol=cfg.symbol,
                    target_net_loss_usd=cfg.target_net_loss_usd,
                    hard_max_net_loss_usd=cfg.hard_max_net_loss_usd,
                    entry_price=entry,
                    stop_price=stop,
                    direction=direction,
                    symbol_spec=symbol_spec,
                    spread=spread_price,
                    commission_per_lot=cfg.commission_per_lot,
                    expected_slippage_points=cfg.expected_slippage_points,
                )
            )
            candidate = _candidate_from_opportunity(opportunity)
            if not size.passed or size.recommended_lot is None:
                candidate.update(
                    {
                        "estimated_sl_net_loss": size.estimated_net_loss,
                        "min_lot_estimated_sl_net_loss": size.estimated_net_loss,
                        "position_size_feasible": False,
                    }
                )
            else:
                lot = float(size.recommended_lot)
                tp_price = float(opportunity.target_reference_price)
                risk = risk_model.estimate(
                    FeeAwareRiskInput(
                        symbol=cfg.symbol,
                        entry_price=entry,
                        stop_price=stop,
                        direction=direction,
                        lot=lot,
                        spread=spread_price,
                        commission_per_lot=cfg.commission_per_lot,
                        expected_slippage_points=cfg.expected_slippage_points,
                        tick_size=cfg.tick_size,
                        tick_value=cfg.tick_value,
                        contract_size=cfg.contract_size,
                        take_profit_price=tp_price,
                        hard_max_net_loss_usd=cfg.hard_max_net_loss_usd,
                    )
                )
                candidate.update(
                    {
                        "recommended_lot": lot,
                        "estimated_sl_net_loss": risk.estimated_net_loss_usd,
                        "fee_adjusted_rr": risk.fee_adjusted_rr or 0.0,
                        "estimated_net_profit_at_tp_usd": risk.estimated_net_profit_at_tp_usd,
                        "position_size_feasible": True,
                    }
                )
            context = {
                "open_positions_count": 1 if open_position is not None else 0,
                "max_open_positions": cfg.max_open_positions_total,
                "daily_bleed_guard": bleed_guard,
                "now_ts": now_ts,
                "direction": direction,
                "setup_key": "v4_execution_style_backtest",
                "live_gate_open": True,
                "paper_only_mode": False,
            }
            decision = entry_filter.evaluate(candidate, context)
            no_trade_guard.record_filter_decision(candidate, decision, now_ts=now_ts)
            if not decision.allow:
                _count_reasons(block_counts, decision.reasons)
                continue
            eligible_signal_count += 1
            if size.passed and size.recommended_lot is not None:
                cap_reason = _leverage_margin_block_reason(
                    entry_price=entry,
                    lot=float(size.recommended_lot),
                    contract_size=cfg.contract_size,
                    account_equity=cfg.account_equity,
                    account_leverage=cfg.account_leverage,
                    max_effective_leverage=cfg.max_effective_leverage,
                    max_margin_used_pct=cfg.max_margin_used_pct,
                )
                if cap_reason:
                    _count_reasons(block_counts, [cap_reason])
                    no_trade_guard.record_rejection(candidate, cap_reason, now_ts=now_ts)
                    continue
                cost_components = _round_trip_cost_components(float(size.recommended_lot), cfg, spread_price)
                open_position = _OpenPosition(
                    opportunity=opportunity,
                    direction=direction,
                    entry_index=idx,
                    entry_time=_row_time_iso(current),
                    entry_price=entry,
                    sl_price=stop,
                    tp_price=float(opportunity.target_reference_price),
                    original_sl_price=stop,
                    lot=float(size.recommended_lot),
                    cost_usd=cost_components["total_cost_usd"],
                    spread_cost_usd=cost_components["spread_cost_usd"],
                    slippage_cost_usd=cost_components["slippage_cost_usd"],
                    commission_cost_usd=cost_components["commission_cost_usd"],
                    implementation_shortfall_usd=cost_components["implementation_shortfall_usd"],
                    units=float(size.recommended_lot) * cfg.contract_size,
                    block_reasons_before_entry=list(decision.reasons) if hasattr(decision, "reasons") else [],
                )
                no_trade_guard.record_executed_trade(candidate, now_ts=now_ts)
                break

    if open_position is not None and len(frame) > 0:
        last_idx = len(frame) - 1
        closed = _close_position(open_position, records[last_idx], last_idx, cfg, "end_of_data")
        trades.append(closed)
        daily_pnl[_date_key(closed.exit_time)] = daily_pnl.get(_date_key(closed.exit_time), 0.0) + closed.net_pnl

    metrics = _compute_metrics(trades, frame, raw_signal_count, scored_signal_count, eligible_signal_count, block_counts, daily_pnl, cfg)
    no_trade_snapshot = no_trade_guard.snapshot()
    metrics.update(
        {
            "raw_signal_count": raw_signal_count,
            "scored_signal_count": scored_signal_count,
            "eligible_signal_count": eligible_signal_count,
            "executed_trade_count": len(trades),
            "block_reasons": dict(sorted(block_counts.items())),
            "no_trade_guard": no_trade_snapshot,
        }
    )
    return {
        "config": cfg.__dict__,
        "metrics": metrics,
        "trades": [trade.to_dict() for trade in trades],
    }


def run_cost_stress_scenarios(bars: Any, base_config: Optional[ExecutionStyleBacktestConfig | Dict[str, Any]] = None) -> Dict[str, Any]:
    cfg = _coerce_config(base_config)
    spread_values = [cfg.spread_points, cfg.spread_points * 1.5, cfg.spread_points * 2.0, cfg.spread_points * 4.0]
    slippage_values = [0.0, cfg.spread_points * 0.5, cfg.spread_points, cfg.spread_points * 2.0]
    commission_values = [cfg.commission_per_lot, cfg.commission_per_lot * 1.5]
    scenarios = []
    metrics_cache: Dict[tuple[float, float, float], Dict[str, Any]] = {}
    for spread in spread_values:
        for slippage in slippage_values:
            for commission in commission_values:
                key = (float(spread), float(slippage), float(commission))
                metrics = metrics_cache.get(key)
                if metrics is None:
                    scenario_cfg = ExecutionStyleBacktestConfig(**{**cfg.__dict__, "spread_points": spread, "expected_slippage_points": slippage, "commission_per_lot": commission})
                    result = run_execution_style_backtest(bars, scenario_cfg)
                    metrics = result["metrics"]
                    metrics_cache[key] = metrics
                scenarios.append(
                    {
                        "spread_points": float(spread),
                        "expected_slippage_points": float(slippage),
                        "commission_per_lot": float(commission),
                        "net_profit_factor": metrics.get("profit_factor"),
                        "trade_count": metrics.get("executed_trade_count"),
                        "no_trade_days": metrics.get("no_trade_days_count"),
                        "max_single_trade_loss": metrics.get("max_single_trade_loss"),
                        "cost_zero_trade": int(metrics.get("executed_trade_count", 0)) == 0 and bool(metrics.get("block_reasons")),
                    }
                )
    return {"scenario_count": len(scenarios), "unique_scenario_count": len(metrics_cache), "scenarios": scenarios}


def build_tradability_verdict(metrics: Dict[str, Any], criteria: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    c = {
        "min_total_oos_trades": 200,
        "min_median_trades_per_week": 5,
        "max_no_trade_days_pct": 50.0,
        "hard_max_net_loss_usd": 1.25,
        "max_daily_net_loss_usd": 3.0,
    }
    c.update(criteria or {})
    failures: List[str] = []
    warnings: List[str] = []
    trade_count = int(metrics.get("executed_trade_count", metrics.get("total_trades", 0)) or 0)
    if trade_count < int(c["min_total_oos_trades"]):
        failures.append("insufficient_oos_trades")
    if float(metrics.get("median_trades_per_week", 0.0) or 0.0) < float(c["min_median_trades_per_week"]):
        failures.append("median_trades_per_week_too_low")
    if float(metrics.get("no_trade_days_pct", 0.0) or 0.0) > float(c["max_no_trade_days_pct"]):
        failures.append("no_trade_days_pct_too_high")
    if abs(float(metrics.get("max_single_trade_loss", 0.0) or 0.0)) > float(c["hard_max_net_loss_usd"]) + 1e-9:
        failures.append("hard_max_single_trade_loss_exceeded")
    if abs(float(metrics.get("daily_max_loss", 0.0) or 0.0)) > float(c["max_daily_net_loss_usd"]) + 1e-9:
        failures.append("daily_max_loss_exceeded")
    if not metrics.get("block_reasons") and trade_count == 0:
        failures.append("unexplained_zero_trade_period")
    pf = float(metrics.get("profit_factor", 0.0) or 0.0)
    if pf < 1.15 and trade_count > 0:
        warnings.append("profit_factor_below_reference")
    verdict = "GO" if not failures and pf >= 1.15 else "REDESIGN" if not failures else "KILL" if trade_count == 0 else "REDESIGN"
    return {"verdict": verdict, "failures": failures, "warnings": warnings, "criteria": c, "metrics": metrics}


def write_execution_style_reports(result: Dict[str, Any], output_dir: str | Path) -> Dict[str, Path]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / "execution_style_backtest.json"
    md_path = out / "execution_style_backtest.md"
    verdict_path = out / "tradability_verdict.md"
    json_path.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    md_path.write_text(_backtest_markdown(result), encoding="utf-8")
    verdict = build_tradability_verdict(result.get("metrics", {}))
    verdict_path.write_text(_verdict_markdown(verdict), encoding="utf-8")
    return {"json": json_path, "markdown": md_path, "verdict": verdict_path}


def _coerce_config(config: Optional[ExecutionStyleBacktestConfig | Dict[str, Any]]) -> ExecutionStyleBacktestConfig:
    if isinstance(config, ExecutionStyleBacktestConfig):
        return config
    return ExecutionStyleBacktestConfig(**dict(config or {}))


def _coerce_frame(bars: Any):
    if pd is None:
        raise RuntimeError("pandas is required for execution-style backtest")
    frame = bars.copy() if hasattr(bars, "copy") else pd.DataFrame(list(bars or []))
    if frame is None or frame.empty:
        raise ValueError("bars must not be empty")
    for col in ("open", "high", "low", "close"):
        if col not in frame.columns:
            raise ValueError(f"missing OHLC column: {col}")
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    if "time" not in frame.columns:
        frame["time"] = pd.date_range(start="2026-01-01T00:00:00Z", periods=len(frame), freq="min")
    frame["time"] = pd.to_datetime(frame["time"], utc=True, errors="coerce")
    frame.dropna(subset=["time", "open", "high", "low", "close"], inplace=True)
    frame.sort_values("time", inplace=True, kind="stable")
    frame.reset_index(drop=True, inplace=True)
    return frame


def _row_datetime(row: Any) -> datetime:
    value = row.get("time") if hasattr(row, "get") else None
    if hasattr(value, "to_pydatetime"):
        return value.to_pydatetime().astimezone(timezone.utc)
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc) if value.tzinfo else value.replace(tzinfo=timezone.utc)
    return datetime.now(timezone.utc)


def _row_time_iso(row: Any) -> str:
    return _row_datetime(row).isoformat()


def _row_ts(row: Any) -> float:
    return _row_datetime(row).timestamp()


def _date_key(value: str) -> str:
    return str(value or "")[:10]


def _candidate_from_opportunity(opportunity: Any) -> Dict[str, Any]:
    components = dict(getattr(opportunity, "components", {}) or {})
    return {
        "symbol": getattr(opportunity, "symbol", ""),
        "direction": getattr(opportunity, "direction", ""),
        "signal_score": getattr(opportunity, "signal_score", 0.0),
        "fee_adjusted_rr": components.get("fee_adjusted_rr_value", 0.0),
        "spread": components.get("spread", 0.0),
        "late_entry": getattr(opportunity, "late_entry", False),
        "entry_price": getattr(opportunity, "entry_price", None),
        "invalidation_price": getattr(opportunity, "invalidation_price", None),
        "target_reference_price": getattr(opportunity, "target_reference_price", None),
        "reason": getattr(opportunity, "reason", ""),
    }


def _round_trip_cost_components(lot: float, cfg: ExecutionStyleBacktestConfig, spread_price: float) -> Dict[str, float]:
    spread_cost = (spread_price / cfg.tick_size) * cfg.tick_value * lot if cfg.tick_size > 0 else 0.0
    slippage_cost = cfg.expected_slippage_points * cfg.tick_value * lot
    commission_cost = cfg.commission_per_lot * lot * 2.0
    return {
        "spread_cost_usd": float(spread_cost),
        "slippage_cost_usd": float(slippage_cost),
        "commission_cost_usd": float(commission_cost),
        "implementation_shortfall_usd": float(slippage_cost),
        "total_cost_usd": float(spread_cost + slippage_cost + commission_cost),
    }


def _leverage_margin_block_reason(
    *,
    entry_price: float,
    lot: float,
    contract_size: float,
    account_equity: float,
    account_leverage: float,
    max_effective_leverage: float,
    max_margin_used_pct: float,
) -> Optional[str]:
    if account_equity <= 0.0:
        return None
    notional = abs(float(entry_price) * float(lot) * float(contract_size))
    effective = notional / float(account_equity)
    if max_effective_leverage > 0.0 and effective > float(max_effective_leverage) + 1e-12:
        return "effective_leverage_cap_exceeded"
    margin = notional / max(float(account_leverage), 1e-12)
    margin_pct = margin / float(account_equity) * 100.0
    if max_margin_used_pct > 0.0 and margin_pct > float(max_margin_used_pct) + 1e-12:
        return "margin_cap_exceeded"
    return None


def _advance_position(pos: _OpenPosition, row: Any, idx: int, cfg: ExecutionStyleBacktestConfig) -> Optional[SimulatedTrade]:
    high = float(row["high"])
    low = float(row["low"])
    close = float(row["close"])
    if pos.direction == "long":
        favorable_price = high
        adverse_price = low
        current_net = (close - pos.entry_price) * pos.units - pos.cost_usd
    else:
        favorable_price = low
        adverse_price = high
        current_net = (pos.entry_price - close) * pos.units - pos.cost_usd
    pos.max_favorable = max(pos.max_favorable, _pnl_at_price(pos, favorable_price, cfg))
    pos.max_adverse = min(pos.max_adverse, _pnl_at_price(pos, adverse_price, cfg))
    _apply_profit_lock(pos, current_net)
    hit_sl = low <= pos.sl_price if pos.direction == "long" else high >= pos.sl_price
    hit_tp = False
    if pos.tp_price is not None:
        hit_tp = high >= pos.tp_price if pos.direction == "long" else low <= pos.tp_price
    if hit_sl:
        return _close_position(pos, row, idx, cfg, "stop_loss_or_profit_lock", price=pos.sl_price)
    if hit_tp:
        return _close_position(pos, row, idx, cfg, "take_profit", price=pos.tp_price)
    if idx - pos.entry_index >= cfg.max_bars_in_trade:
        return _close_position(pos, row, idx, cfg, "max_bars_in_trade")
    return None


def _apply_profit_lock(pos: _OpenPosition, current_net: float) -> None:
    for trigger, lock, target in PROFIT_LOCK_STAGES:
        if current_net + 1e-12 >= trigger and lock > pos.highest_locked_pnl:
            if pos.units <= 0:
                return
            if pos.direction == "long":
                new_sl = pos.entry_price + (lock + pos.cost_usd) / pos.units
                pos.sl_price = max(pos.sl_price, new_sl)
                if target is not None:
                    pos.tp_price = max(pos.tp_price or new_sl, pos.entry_price + (target + pos.cost_usd) / pos.units)
            else:
                new_sl = pos.entry_price - (lock + pos.cost_usd) / pos.units
                pos.sl_price = min(pos.sl_price, new_sl)
                if target is not None:
                    pos.tp_price = min(pos.tp_price or new_sl, pos.entry_price - (target + pos.cost_usd) / pos.units)
            original_lock = _pnl_at_price(pos, pos.original_sl_price, None)
            pos.profit_lock_saved_pnl = max(pos.profit_lock_saved_pnl, lock - original_lock)
            pos.highest_locked_pnl = lock
            pos.sl_moves += 1
            return


def _pnl_at_price(pos: _OpenPosition, price: float, cfg: Optional[ExecutionStyleBacktestConfig]) -> float:
    if pos.direction == "long":
        gross = (float(price) - pos.entry_price) * pos.units
    else:
        gross = (pos.entry_price - float(price)) * pos.units
    return gross - pos.cost_usd


def _close_position(pos: _OpenPosition, row: Any, idx: int, cfg: ExecutionStyleBacktestConfig, reason: str, price: Optional[float] = None) -> SimulatedTrade:
    exit_price = float(price if price is not None else row["close"])
    gross = (exit_price - pos.entry_price) * pos.units if pos.direction == "long" else (pos.entry_price - exit_price) * pos.units
    net = gross - pos.cost_usd
    return SimulatedTrade(
        symbol=cfg.symbol,
        direction=pos.direction,
        entry_index=pos.entry_index,
        exit_index=idx,
        entry_time=pos.entry_time,
        exit_time=_row_time_iso(row),
        entry_price=pos.entry_price,
        exit_price=exit_price,
        sl_price=pos.sl_price,
        tp_price=pos.tp_price,
        lot=pos.lot,
        net_pnl=float(net),
        gross_pnl=float(gross),
        cost_usd=float(pos.cost_usd),
        spread_cost_usd=float(pos.spread_cost_usd),
        slippage_cost_usd=float(pos.slippage_cost_usd),
        commission_cost_usd=float(pos.commission_cost_usd),
        implementation_shortfall_usd=float(pos.implementation_shortfall_usd),
        exit_reason=reason,
        max_favorable_excursion=float(pos.max_favorable),
        max_adverse_excursion=float(pos.max_adverse),
        profit_lock_sl_moves=int(pos.sl_moves),
        profit_lock_saved_pnl=float(pos.profit_lock_saved_pnl),
    )


def _count_reasons(counts: Dict[str, int], reasons: Iterable[str]) -> None:
    for reason in reasons:
        text = str(reason)
        counts[text] = counts.get(text, 0) + 1


def _compute_metrics(
    trades: Sequence[SimulatedTrade],
    frame: Any,
    raw_signal_count: int,
    scored_signal_count: int,
    eligible_signal_count: int,
    block_counts: Dict[str, int],
    daily_pnl: Dict[str, float],
    cfg: ExecutionStyleBacktestConfig,
) -> Dict[str, Any]:
    pnls = [float(t.net_pnl) for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    dates = sorted({_row_time_iso(row)[:10] for _, row in frame.iterrows()})
    trade_dates = {_date_key(t.exit_time) for t in trades}
    no_trade_days = len([d for d in dates if d[:10] not in trade_dates])
    equity = cfg.initial_balance
    peak = equity
    max_dd = 0.0
    consecutive = 0
    max_consecutive = 0
    for pnl in pnls:
        equity += pnl
        peak = max(peak, equity)
        max_dd = max(max_dd, peak - equity)
        if pnl < 0:
            consecutive += 1
            max_consecutive = max(max_consecutive, consecutive)
        else:
            consecutive = 0
    per_week = _median_trades_per_week(trades)
    spread_cost = sum(float(t.spread_cost_usd) for t in trades)
    slippage_cost = sum(float(t.slippage_cost_usd) for t in trades)
    commission_cost = sum(float(t.commission_cost_usd) for t in trades)
    implementation_shortfall = sum(float(t.implementation_shortfall_usd) for t in trades)
    return {
        "total_net_pnl": sum(pnls),
        "profit_factor": gross_profit / gross_loss if gross_loss > 0 else float("inf") if gross_profit > 0 else 0.0,
        "max_drawdown": max_dd,
        "max_single_trade_loss": min(pnls) if pnls else 0.0,
        "avg_loss": sum(losses) / len(losses) if losses else 0.0,
        "avg_win": sum(wins) / len(wins) if wins else 0.0,
        "payoff_ratio": (sum(wins) / len(wins)) / abs(sum(losses) / len(losses)) if wins and losses else 0.0,
        "win_rate": len(wins) / len(pnls) if pnls else 0.0,
        "median_trades_per_day": _median_trades_per_day(trades),
        "median_trades_per_week": per_week,
        "no_trade_days_count": no_trade_days,
        "no_trade_days_pct": (no_trade_days / len(dates) * 100.0) if dates else 0.0,
        "consecutive_losses_max": max_consecutive,
        "daily_max_loss": min(daily_pnl.values()) if daily_pnl else 0.0,
        "raw_signal_count": raw_signal_count,
        "scored_signal_count": scored_signal_count,
        "eligible_signal_count": eligible_signal_count,
        "executed_trade_count": len(trades),
        "block_reasons": dict(block_counts),
        "mfe_max": max((t.max_favorable_excursion for t in trades), default=0.0),
        "mae_min": min((t.max_adverse_excursion for t in trades), default=0.0),
        "winner_capture_ratio": _winner_capture_ratio(trades),
        "profit_lock_sl_moves": sum(t.profit_lock_sl_moves for t in trades),
        "profit_lock_saved_pnl": sum(t.profit_lock_saved_pnl for t in trades),
        "spread_cost": spread_cost,
        "slippage_cost": slippage_cost,
        "commission_cost": commission_cost,
        "total_cost": spread_cost + slippage_cost + commission_cost,
        "implementation_shortfall_cost": implementation_shortfall,
        "implementation_shortfall_per_trade": implementation_shortfall / len(trades) if trades else 0.0,
        "hard_max_respected": all(abs(t.net_pnl) <= cfg.hard_max_net_loss_usd + 1e-9 for t in trades if t.net_pnl < 0),
        "daily_bleed_respected": all(abs(v) <= cfg.max_daily_net_loss_usd + cfg.hard_max_net_loss_usd + 1e-9 for v in daily_pnl.values() if v < 0),
    }


def _median_trades_per_day(trades: Sequence[SimulatedTrade]) -> float:
    counts: Dict[str, int] = {}
    for trade in trades:
        counts[_date_key(trade.exit_time)] = counts.get(_date_key(trade.exit_time), 0) + 1
    return float(median(counts.values())) if counts else 0.0


def _median_trades_per_week(trades: Sequence[SimulatedTrade]) -> float:
    counts: Dict[str, int] = {}
    for trade in trades:
        dt = datetime.fromisoformat(trade.exit_time.replace("Z", "+00:00"))
        year, week, _ = dt.isocalendar()
        key = f"{year}-W{week:02d}"
        counts[key] = counts.get(key, 0) + 1
    return float(median(counts.values())) if counts else 0.0


def _winner_capture_ratio(trades: Sequence[SimulatedTrade]) -> float:
    wins = [t for t in trades if t.net_pnl > 0 and t.max_favorable_excursion > 0]
    if not wins:
        return 0.0
    return sum(t.net_pnl / t.max_favorable_excursion for t in wins) / len(wins)


def _backtest_markdown(result: Dict[str, Any]) -> str:
    metrics = result.get("metrics", {})
    lines = ["# Execution Style Backtest", ""]
    for key in (
        "executed_trade_count",
        "total_net_pnl",
        "profit_factor",
        "max_single_trade_loss",
        "daily_max_loss",
        "no_trade_days_count",
        "profit_lock_sl_moves",
        "profit_lock_saved_pnl",
        "spread_cost",
        "slippage_cost",
        "commission_cost",
        "implementation_shortfall_cost",
    ):
        lines.append(f"- {key}: {metrics.get(key)}")
    lines.extend(["", "## Block Reasons"])
    for reason, count in sorted((metrics.get("block_reasons") or {}).items()):
        lines.append(f"- {reason}: {count}")
    return "\n".join(lines) + "\n"


def _verdict_markdown(verdict: Dict[str, Any]) -> str:
    lines = ["# Tradability Verdict", "", f"- verdict: {verdict.get('verdict')}", "", "## Failures"]
    lines.extend([f"- {item}" for item in verdict.get("failures", [])] or ["- none"])
    lines.extend(["", "## Warnings"])
    lines.extend([f"- {item}" for item in verdict.get("warnings", [])] or ["- none"])
    return "\n".join(lines) + "\n"
