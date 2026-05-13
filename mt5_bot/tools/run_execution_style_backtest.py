from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.execution_style_backtest import (  # noqa: E402
    ExecutionStyleBacktestConfig,
    build_tradability_verdict,
    run_cost_stress_scenarios,
    run_execution_style_backtest,
    write_execution_style_reports,
)


DEFAULT_SYMBOL_CONFIG: Dict[str, Dict[str, Any]] = {
    "BTCUSD": {"tick_size": 0.01, "tick_value": 0.01, "contract_size": 1.0, "volume_min": 0.01, "volume_step": 0.01},
    "ETHUSD": {"tick_size": 0.01, "tick_value": 0.01, "contract_size": 1.0, "volume_min": 0.01, "volume_step": 0.01},
    "SOLUSD": {"tick_size": 0.01, "tick_value": 0.01, "contract_size": 1.0, "volume_min": 0.01, "volume_step": 0.01},
}


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run v4 execution-style MT5-free backtest reports from CSV data.")
    parser.add_argument("--data-dir", default="mt5_bot/data/binance", help="Directory containing SYMBOL_TIMEFRAME_M1.csv files")
    parser.add_argument("--symbols", default="BTCUSD,ETHUSD,SOLUSD", help="Comma-separated symbols to run")
    parser.add_argument("--output-dir", default="reports/execution_style", help="Report output directory")
    parser.add_argument("--timeframe", default="TIMEFRAME_M1")
    parser.add_argument("--max-rows", type=int, default=20000, help="Use tail N rows per symbol; 0 means all rows")
    parser.add_argument("--min-rr", type=float, default=1.5, help="Minimum fee-adjusted RR for the proxy simulation")
    parser.add_argument("--min-score", type=float, default=70.0)
    parser.add_argument("--target-loss", type=float, default=1.0)
    parser.add_argument("--hard-max-loss", type=float, default=1.25)
    parser.add_argument("--daily-max-loss", type=float, default=3.0)
    parser.add_argument("--spread-points", type=float, default=0.0)
    parser.add_argument("--commission-per-lot", type=float, default=0.0)
    parser.add_argument("--slippage-points", type=float, default=0.0)
    parser.add_argument("--stress", action="store_true", help="Also run 32 scenario cost stress per symbol")
    parser.add_argument("--stress-max-rows", type=int, default=None, help="Use a separate tail row cap for stress only; defaults to --max-rows/full frame")
    args = parser.parse_args(list(argv) if argv is not None else None)

    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    symbols = _parse_symbols(args.symbols)
    combined: Dict[str, Any] = {"symbols": {}, "summary": {}, "data_dir": str(data_dir)}

    for symbol in symbols:
        csv_path = _find_csv(data_dir, symbol)
        if csv_path is None:
            combined["symbols"][symbol] = {"error": "csv_not_found"}
            continue
        frame = pd.read_csv(csv_path)
        frame = _normalize_frame(frame)
        if args.max_rows and args.max_rows > 0 and len(frame) > args.max_rows:
            frame = frame.tail(args.max_rows).reset_index(drop=True)
        cfg = _config_for_symbol(symbol, args)
        result = run_execution_style_backtest(frame, cfg)
        symbol_dir = out_dir / symbol
        paths = write_execution_style_reports(result, symbol_dir)
        verdict = build_tradability_verdict(result.get("metrics", {}), {"min_total_oos_trades": 20})
        (symbol_dir / "tradability_verdict.json").write_text(json.dumps(verdict, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
        payload = {
            "csv_path": str(csv_path),
            "rows_used": int(len(frame)),
            "metrics": result.get("metrics", {}),
            "verdict": verdict,
            "reports": {key: str(path) for key, path in paths.items()},
        }
        if args.stress:
            stress_frame = frame
            if args.stress_max_rows is not None and args.stress_max_rows > 0 and len(frame) > args.stress_max_rows:
                stress_frame = frame.tail(args.stress_max_rows).reset_index(drop=True)
            stress = run_cost_stress_scenarios(stress_frame, cfg)
            (symbol_dir / "cost_stress_execution_style.json").write_text(
                json.dumps(stress, indent=2, sort_keys=True, default=str) + "\n",
                encoding="utf-8",
            )
            payload["stress"] = {
                "scenario_count": stress.get("scenario_count"),
                "unique_scenario_count": stress.get("unique_scenario_count"),
                "rows_used": int(len(stress_frame)),
                "json": str(symbol_dir / "cost_stress_execution_style.json"),
            }
        combined["symbols"][symbol] = payload

    combined["summary"] = _combined_summary(combined["symbols"])
    combined_path = out_dir / "execution_style_summary.json"
    combined_path.write_text(json.dumps(combined, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    (out_dir / "execution_style_summary.md").write_text(_summary_markdown(combined), encoding="utf-8")
    print(json.dumps({"ok": True, "summary_json": str(combined_path), "symbols": symbols}, ensure_ascii=False))
    return 0


def _parse_symbols(raw: str) -> List[str]:
    out: List[str] = []
    for token in str(raw or "").split(","):
        symbol = re.sub(r"[^A-Za-z0-9]", "", token).upper()
        if symbol and symbol not in out:
            out.append(symbol)
    return out


def _find_csv(data_dir: Path, symbol: str) -> Optional[Path]:
    candidates = [data_dir / f"{symbol}_TIMEFRAME_M1.csv", data_dir / f"{symbol}.csv"]
    for path in candidates:
        if path.exists():
            return path
    matches = sorted(data_dir.glob(f"{symbol}*.csv")) if data_dir.exists() else []
    return matches[0] if matches else None


def _normalize_frame(frame: pd.DataFrame) -> pd.DataFrame:
    cols = {str(col).strip(): str(col).strip().lower() for col in frame.columns}
    frame = frame.rename(columns=cols)
    for col in ("open", "high", "low", "close"):
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame["time"] = pd.to_datetime(frame["time"], utc=True, errors="coerce") if "time" in frame.columns else pd.date_range("2026-01-01", periods=len(frame), freq="min", tz="UTC")
    frame = frame.dropna(subset=["time", "open", "high", "low", "close"]).sort_values("time", kind="stable").reset_index(drop=True)
    return frame


def _config_for_symbol(symbol: str, args: argparse.Namespace) -> ExecutionStyleBacktestConfig:
    values = dict(DEFAULT_SYMBOL_CONFIG.get(symbol.upper(), DEFAULT_SYMBOL_CONFIG["BTCUSD"]))
    values.update(
        {
            "symbol": symbol.upper(),
            "timeframe": args.timeframe,
            "target_net_loss_usd": args.target_loss,
            "hard_max_net_loss_usd": args.hard_max_loss,
            "max_daily_net_loss_usd": args.daily_max_loss,
            "min_reward_to_net_risk_ratio": args.min_rr,
            "min_signal_score": args.min_score,
            "spread_points": args.spread_points,
            "commission_per_lot": args.commission_per_lot,
            "expected_slippage_points": args.slippage_points,
            "scanner_atr_period": 14,
            "scanner_lookback_bars": 20,
            "max_bars_in_trade": 120,
        }
    )
    return ExecutionStyleBacktestConfig(**values)


def _combined_summary(symbols: Dict[str, Any]) -> Dict[str, Any]:
    ok_items = {s: p for s, p in symbols.items() if isinstance(p, dict) and "metrics" in p}
    return {
        "symbol_count": len(symbols),
        "completed_symbol_count": len(ok_items),
        "total_trades": sum(int(p.get("metrics", {}).get("executed_trade_count", 0) or 0) for p in ok_items.values()),
        "verdict_by_symbol": {s: p.get("verdict", {}).get("verdict") for s, p in ok_items.items()},
        "max_single_trade_loss_by_symbol": {s: p.get("metrics", {}).get("max_single_trade_loss") for s, p in ok_items.items()},
        "no_trade_days_pct_by_symbol": {s: p.get("metrics", {}).get("no_trade_days_pct") for s, p in ok_items.items()},
    }


def _summary_markdown(combined: Dict[str, Any]) -> str:
    lines = ["# Execution Style Summary", "", f"- data_dir: {combined.get('data_dir')}", ""]
    summary = combined.get("summary", {})
    lines.append(f"- completed_symbol_count: {summary.get('completed_symbol_count')}")
    lines.append(f"- total_trades: {summary.get('total_trades')}")
    lines.extend(["", "## Symbols"])
    for symbol, payload in sorted((combined.get("symbols") or {}).items()):
        if "error" in payload:
            lines.append(f"- {symbol}: ERROR {payload.get('error')}")
            continue
        metrics = payload.get("metrics", {})
        verdict = payload.get("verdict", {}).get("verdict")
        lines.append(
            f"- {symbol}: verdict={verdict}, trades={metrics.get('executed_trade_count')}, "
            f"pf={metrics.get('profit_factor')}, max_loss={metrics.get('max_single_trade_loss')}, "
            f"no_trade_days_pct={metrics.get('no_trade_days_pct')}"
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
