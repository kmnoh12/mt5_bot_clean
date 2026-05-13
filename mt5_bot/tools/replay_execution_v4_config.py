from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.optimization.execution_metrics import aggregate_metrics, metrics_from_backtest
from core.optimization.leverage_margin import analyze_trades
from core.optimization.oos_validator import attach_oos_metrics, train_oos_split
from tools.optimize_execution_v4 import _backtest_config, _load_frames
from utils.execution_style_backtest import run_execution_style_backtest


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Replay a fixed v4 optimization config with optional cost stress.")
    parser.add_argument("--config-json", required=True)
    parser.add_argument("--config-key", default="balanced")
    parser.add_argument("--symbols", default="BTCUSD")
    parser.add_argument("--data-dir", default="data/binance")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-rows", type=int, default=20000)
    parser.add_argument("--account-equity", type=float, default=140.0)
    parser.add_argument("--account-leverage", type=float, default=500.0)
    parser.add_argument("--spread-points", type=float, default=0.0)
    parser.add_argument("--slippage-points", type=float, default=0.0)
    parser.add_argument("--commission-per-lot", type=float, default=0.0)
    args = parser.parse_args(list(argv) if argv is not None else None)

    config_payload = json.loads(Path(args.config_json).read_text(encoding="utf-8"))
    selected = config_payload.get(args.config_key)
    if not isinstance(selected, dict) or not isinstance(selected.get("config"), dict):
        raise ValueError(f"config key not found or invalid: {args.config_key}")
    config = dict(selected["config"])
    frames = _load_frames(Path(args.data_dir), _parse_symbols(args.symbols), args.max_rows)
    result = replay_config(config, frames, args)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "fixed_replay_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, default=str, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (out_dir / "fixed_replay_summary.md").write_text(_markdown(result, selected), encoding="utf-8")
    print(json.dumps({"ok": True, "output_dir": str(out_dir), "symbols": sorted(frames)}, ensure_ascii=False))
    return 0


def replay_config(config: Mapping[str, Any], frames: Mapping[str, pd.DataFrame], args: argparse.Namespace) -> Dict[str, Any]:
    train_items = []
    oos_items = []
    symbols: Dict[str, Any] = {}
    for symbol, frame in frames.items():
        train_frame, oos_frame = train_oos_split(frame, 0.7)
        cfg = _backtest_config(symbol, config, args)
        train_result = run_execution_style_backtest(train_frame, cfg)
        oos_result = run_execution_style_backtest(oos_frame, cfg)
        max_daily_loss_usd = float(config.get("risk", {}).get("target_net_loss_usd", 1.0)) * float(config.get("daily_bleed", {}).get("max_daily_loss_R", 3))
        train_metrics = metrics_from_backtest(
            train_result,
            hard_max_net_loss_usd=float(config.get("risk", {}).get("hard_max_net_loss_usd", 1.25)),
            max_daily_loss_usd=max_daily_loss_usd,
            max_effective_leverage=float(config.get("risk", {}).get("max_effective_leverage", 1.0)),
            max_margin_used_pct=float(config.get("risk", {}).get("max_margin_used_pct", 5.0)),
            leverage_metrics=analyze_trades(
                train_result.get("trades", []),
                account_equity=args.account_equity,
                account_leverage=args.account_leverage,
                contract_size=cfg.contract_size,
            ),
        )
        oos_metrics = metrics_from_backtest(
            oos_result,
            hard_max_net_loss_usd=float(config.get("risk", {}).get("hard_max_net_loss_usd", 1.25)),
            max_daily_loss_usd=max_daily_loss_usd,
            max_effective_leverage=float(config.get("risk", {}).get("max_effective_leverage", 1.0)),
            max_margin_used_pct=float(config.get("risk", {}).get("max_margin_used_pct", 5.0)),
            leverage_metrics=analyze_trades(
                oos_result.get("trades", []),
                account_equity=args.account_equity,
                account_leverage=args.account_leverage,
                contract_size=cfg.contract_size,
            ),
        )
        train_items.append(train_metrics)
        oos_items.append(oos_metrics)
        symbols[symbol] = {
            "rows": int(len(frame)),
            "train_rows": int(len(train_frame)),
            "oos_rows": int(len(oos_frame)),
            "train_metrics": train_metrics,
            "oos_metrics": oos_metrics,
            "train_trades": train_result.get("trades", []),
            "oos_trades": oos_result.get("trades", []),
            "daily_bleed_analysis": _daily_bleed_analysis(train_result.get("trades", []) + oos_result.get("trades", []), config),
        }
    train_agg = aggregate_metrics(train_items)
    oos_agg = aggregate_metrics(oos_items)
    combined = attach_oos_metrics(train_agg, oos_agg)
    return {
        "config": dict(config),
        "costs": {
            "spread_points": float(args.spread_points),
            "slippage_points": float(args.slippage_points),
            "commission_per_lot": float(args.commission_per_lot),
        },
        "train_metrics": train_agg,
        "oos_metrics": oos_agg,
        "combined_metrics": combined,
        "symbols": symbols,
    }


def _daily_bleed_analysis(trades: Iterable[Mapping[str, Any]], config: Mapping[str, Any]) -> Dict[str, Any]:
    stop_after = int(config.get("daily_bleed", {}).get("stop_after_consecutive_losses", 3) or 3)
    max_daily_loss_r = float(config.get("daily_bleed", {}).get("max_daily_loss_R", 3) or 3)
    target_loss = float(config.get("risk", {}).get("target_net_loss_usd", 1.0) or 1.0)
    daily_limit = max_daily_loss_r * target_loss
    sorted_trades = sorted(trades, key=lambda item: str(item.get("exit_time", "")))
    max_consecutive = 0
    current = 0
    daily: Dict[str, float] = {}
    theoretical_halts = 0
    for trade in sorted_trades:
        pnl = float(trade.get("net_pnl", 0.0) or 0.0)
        day = str(trade.get("exit_time", ""))[:10]
        daily[day] = daily.get(day, 0.0) + pnl
        if pnl < 0.0:
            current += 1
            max_consecutive = max(max_consecutive, current)
            if current >= stop_after:
                theoretical_halts += 1
        elif pnl > 0.0:
            current = 0
    return {
        "configured_stop_after_consecutive_losses": stop_after,
        "observed_max_consecutive_losses": max_consecutive,
        "theoretical_consecutive_loss_halt_events": theoretical_halts,
        "configured_daily_loss_limit_usd": daily_limit,
        "worst_daily_pnl": min(daily.values()) if daily else 0.0,
        "daily_loss_limit_breached_days": sum(1 for value in daily.values() if value <= -abs(daily_limit)),
        "why_halt_count_can_be_zero": (
            "The backtest currently records DailyBleedGuard blocks as entry-filter block reasons; "
            "daily_bleed_halt_count is not incremented as a separate counter in the simulation metrics."
        ),
    }


def _markdown(result: Mapping[str, Any], selected: Mapping[str, Any]) -> str:
    combined = result.get("combined_metrics", {})
    train = result.get("train_metrics", {})
    oos = result.get("oos_metrics", {})
    lines = [
        "# BTCUSD Fixed Replay v4",
        "",
        "- live 적용 금지: paper-forward 검증 전용입니다.",
        f"- source_trial_id: {selected.get('trial_id')}",
        f"- costs: {result.get('costs')}",
        "",
        "## Train",
        f"- total_net_pnl: {train.get('total_net_pnl')}",
        f"- net_profit_factor: {train.get('net_profit_factor')}",
        f"- gross_profit: {train.get('gross_profit')}",
        f"- gross_loss: {train.get('gross_loss')}",
        f"- win_count: {train.get('win_count')}",
        f"- loss_count: {train.get('loss_count')}",
        f"- max_single_trade_net_loss: {train.get('max_single_trade_net_loss')}",
        f"- no_trade_days_pct: {train.get('no_trade_days_pct')}",
        f"- profit_lock_saved_pnl: {train.get('profit_lock_saved_pnl')}",
        "",
        "## OOS",
        f"- total_trades: {oos.get('total_trades')}",
        f"- expectancy_net: {oos.get('expectancy_net')}",
        f"- net_profit_factor: {oos.get('net_profit_factor')}",
        f"- gross_profit: {oos.get('gross_profit')}",
        f"- gross_loss: {oos.get('gross_loss')}",
        f"- win_count: {oos.get('win_count')}",
        f"- loss_count: {oos.get('loss_count')}",
        f"- max_single_trade_net_loss: {oos.get('max_single_trade_net_loss')}",
        f"- no_trade_days_pct: {oos.get('no_trade_days_pct')}",
        f"- profit_lock_saved_pnl: {oos.get('profit_lock_saved_pnl')}",
        "",
        "## Combined",
        f"- oos_decay_pct: {combined.get('oos_decay_pct')}",
        f"- effective_leverage_max: {combined.get('effective_leverage_max')}",
        f"- margin_used_pct_max: {combined.get('margin_used_pct_max')}",
        "",
        "## Daily Bleed Analysis",
    ]
    for symbol, payload in (result.get("symbols") or {}).items():
        lines.append(f"### {symbol}")
        for key, value in (payload.get("daily_bleed_analysis") or {}).items():
            lines.append(f"- {key}: {value}")
    return "\n".join(lines) + "\n"


def _parse_symbols(raw: str) -> list[str]:
    return ["".join(ch for ch in item.upper() if ch.isalnum()) for item in str(raw).split(",") if item.strip()]


if __name__ == "__main__":
    raise SystemExit(main())

