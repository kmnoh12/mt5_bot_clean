from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.optimization.execution_metrics import aggregate_metrics, metrics_from_backtest
from core.optimization.leverage_margin import analyze_trades, config_margin_verdict
from core.optimization.objective_v4 import hard_reject_reasons, score_trials
from core.optimization.oos_validator import attach_oos_metrics, train_oos_split
from core.optimization.pareto_ranker import pareto_frontier, select_recommendations
from core.optimization.search_space_v4 import flatten_config, grid_trial_configs, random_trial_configs
from core.optimization.trial_result import TrialResult, flatten_trial_for_csv
from utils.execution_style_backtest import ExecutionStyleBacktestConfig, run_execution_style_backtest


DEFAULT_SYMBOL_CONFIG: Dict[str, Dict[str, Any]] = {
    "BTCUSD": {"tick_size": 0.01, "tick_value": 0.01, "contract_size": 1.0, "volume_min": 0.01, "volume_step": 0.01},
    "ETHUSD": {"tick_size": 0.01, "tick_value": 0.01, "contract_size": 1.0, "volume_min": 0.01, "volume_step": 0.01},
    "SOLUSD": {"tick_size": 0.01, "tick_value": 0.01, "contract_size": 1.0, "volume_min": 0.01, "volume_step": 0.01},
}


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run offline v4 execution optimization without MT5 order APIs.")
    parser.add_argument("--trials", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20260513)
    parser.add_argument("--mode", choices=["random", "grid", "optuna"], default="random")
    parser.add_argument("--symbols", default="BTCUSD,ETHUSD")
    parser.add_argument("--timeframes", default="M1")
    parser.add_argument("--train-start", default=None)
    parser.add_argument("--train-end", default=None)
    parser.add_argument("--oos-start", default=None)
    parser.add_argument("--oos-end", default=None)
    parser.add_argument("--output-dir", default="reports/optimization_v4")
    parser.add_argument("--data-dir", default="data/binance")
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-rows", type=int, default=6000)
    parser.add_argument("--account-equity", type=float, default=140.0)
    parser.add_argument("--account-leverage", type=float, default=500.0)
    parser.add_argument("--spread-points", type=float, default=0.0)
    parser.add_argument("--commission-per-lot", type=float, default=0.0)
    parser.add_argument("--slippage-points", type=float, default=0.0)
    parser.add_argument("--min-total-trades", type=int, default=100)
    parser.add_argument("--min-oos-trades", type=int, default=100)
    parser.add_argument("--min-oos-profit-factor", type=float, default=1.05)
    parser.add_argument("--min-oos-expectancy-net", type=float, default=0.0)
    parser.add_argument("--max-no-trade-days-pct", type=float, default=60.0)
    parser.add_argument("--min-train-profit-factor", type=float, default=0.0)
    parser.add_argument("--require-train-net-pnl-positive", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    symbols = _parse_symbols(args.symbols)
    frames = _load_frames(Path(args.data_dir), symbols, args.max_rows)
    configs = _trial_configs(args.mode, args.trials, args.seed)
    if args.dry_run:
        _write_preflight_reports(out_dir, symbols, frames, [])
        print(json.dumps({"ok": True, "dry_run": True, "trials": len(configs)}, ensure_ascii=False))
        return 0

    trials: List[TrialResult] = []
    for idx, config in enumerate(configs, start=1):
        trial = _run_trial(idx, args.seed, config, frames, args)
        trials.append(trial)

    scored = score_trials(trials)
    frontier = pareto_frontier(scored)
    recommendations = select_recommendations(scored)
    _write_outputs(out_dir, scored, frontier, recommendations, symbols, frames, args)
    print(
        json.dumps(
            {
                "ok": True,
                "trials": len(scored),
                "accepted": sum(1 for t in scored if not t.rejected),
                "output_dir": str(out_dir),
            },
            ensure_ascii=False,
        )
    )
    return 0


def _run_trial(
    trial_id: int,
    seed: int,
    config: Mapping[str, Any],
    frames: Mapping[str, pd.DataFrame],
    args: argparse.Namespace,
) -> TrialResult:
    train_items: List[Dict[str, Any]] = []
    oos_items: List[Dict[str, Any]] = []
    symbol_metrics: Dict[str, Dict[str, Any]] = {}
    for symbol, frame in frames.items():
        train_frame, oos_frame = _date_or_ratio_split(frame, args)
        cfg = _backtest_config(symbol, config, args)
        train_result = run_execution_style_backtest(train_frame, cfg)
        oos_result = run_execution_style_backtest(oos_frame, cfg)
        leverage_train = analyze_trades(
            train_result.get("trades", []),
            account_equity=args.account_equity,
            account_leverage=args.account_leverage,
            contract_size=cfg.contract_size,
        )
        leverage_oos = analyze_trades(
            oos_result.get("trades", []),
            account_equity=args.account_equity,
            account_leverage=args.account_leverage,
            contract_size=cfg.contract_size,
        )
        max_daily_loss_usd = float(config.get("risk", {}).get("target_net_loss_usd", 1.0)) * float(config.get("daily_bleed", {}).get("max_daily_loss_R", 3))
        train_metrics = metrics_from_backtest(
            train_result,
            hard_max_net_loss_usd=float(config.get("risk", {}).get("hard_max_net_loss_usd", 1.25)),
            max_daily_loss_usd=max_daily_loss_usd,
            max_effective_leverage=float(config.get("risk", {}).get("max_effective_leverage", 1.0)),
            max_margin_used_pct=float(config.get("risk", {}).get("max_margin_used_pct", 5.0)),
            leverage_metrics=leverage_train,
        )
        oos_metrics = metrics_from_backtest(
            oos_result,
            hard_max_net_loss_usd=float(config.get("risk", {}).get("hard_max_net_loss_usd", 1.25)),
            max_daily_loss_usd=max_daily_loss_usd,
            max_effective_leverage=float(config.get("risk", {}).get("max_effective_leverage", 1.0)),
            max_margin_used_pct=float(config.get("risk", {}).get("max_margin_used_pct", 5.0)),
            leverage_metrics=leverage_oos,
        )
        train_items.append(train_metrics)
        oos_items.append(oos_metrics)
        symbol_metrics[symbol] = {"train": train_metrics, "oos": oos_metrics}
    train_agg = aggregate_metrics(train_items)
    oos_agg = aggregate_metrics(oos_items)
    metrics = attach_oos_metrics(train_agg, oos_agg)
    config_with_validation = dict(config)
    config_with_validation["validation"] = {
        "min_total_trades": int(args.min_total_trades),
        "min_oos_trades": int(args.min_oos_trades),
        "min_oos_profit_factor": float(args.min_oos_profit_factor),
        "min_oos_expectancy_net": float(args.min_oos_expectancy_net),
        "max_no_trade_days_pct": float(args.max_no_trade_days_pct),
        "min_train_profit_factor": float(args.min_train_profit_factor),
        "require_train_net_pnl_positive": bool(args.require_train_net_pnl_positive),
    }
    reasons = hard_reject_reasons(metrics, config_with_validation)
    return TrialResult(
        trial_id=trial_id,
        seed=seed,
        config=config_with_validation,
        metrics=metrics,
        train_metrics=train_agg,
        oos_metrics=oos_agg,
        symbol_metrics=symbol_metrics,
        rejected=bool(reasons),
        reject_reasons=reasons,
    )


def _backtest_config(symbol: str, config: Mapping[str, Any], args: argparse.Namespace) -> ExecutionStyleBacktestConfig:
    spec = dict(DEFAULT_SYMBOL_CONFIG.get(symbol, DEFAULT_SYMBOL_CONFIG["BTCUSD"]))
    risk = dict(config.get("risk", {}) or {})
    entry = dict(config.get("entry", {}) or {})
    initial_exit = dict(config.get("initial_exit", {}) or {})
    lsr = dict(config.get("lsr", {}) or {})
    daily = dict(config.get("daily_bleed", {}) or {})
    max_daily_loss = float(risk.get("target_net_loss_usd", 1.0)) * float(daily.get("max_daily_loss_R", 3.0))
    values = {
        **spec,
        "symbol": symbol,
        "timeframe": "TIMEFRAME_M1",
        "target_net_loss_usd": float(risk.get("target_net_loss_usd", 1.0)),
        "hard_max_net_loss_usd": float(risk.get("hard_max_net_loss_usd", 1.25)),
        "max_daily_net_loss_usd": max_daily_loss,
        "min_reward_to_net_risk_ratio": float(initial_exit.get("min_reward_to_net_risk_ratio", entry.get("min_fee_adjusted_rr", 2.0))),
        "min_signal_score": float(entry.get("min_signal_score", 70.0)),
        "spread_points": float(args.spread_points),
        "commission_per_lot": float(args.commission_per_lot),
        "expected_slippage_points": float(args.slippage_points),
        "scanner_lookback_bars": int(lsr.get("swing_window", 20)),
        "scanner_atr_period": 14,
        "sweep_buffer_atr": float(lsr.get("sweep_buffer_atr", 0.05)),
        "stop_buffer_atr": float(lsr.get("stop_buffer_atr", 0.05)),
        "max_bars_in_trade": int(lsr.get("max_hold_bars", 80)),
        "initial_balance": 10000.0,
        "account_equity": float(args.account_equity),
        "account_leverage": float(args.account_leverage),
        "max_effective_leverage": float(risk.get("max_effective_leverage", 0.0) or 0.0),
        "max_margin_used_pct": float(risk.get("max_margin_used_pct", 0.0) or 0.0),
    }
    return ExecutionStyleBacktestConfig(**values)


def _write_outputs(
    out_dir: Path,
    trials: List[TrialResult],
    frontier: List[TrialResult],
    recommendations: Mapping[str, Optional[TrialResult]],
    symbols: List[str],
    frames: Mapping[str, pd.DataFrame],
    args: argparse.Namespace,
) -> None:
    _write_preflight_reports(out_dir, symbols, frames, trials)
    _write_json(out_dir / "optimization_v4_summary.json", {"trials": [t.to_dict() for t in trials], "summary": _summary(trials)})
    _write_csv(out_dir / "optimization_v4_trials.csv", [flatten_trial_for_csv(t) for t in trials])
    _write_csv(out_dir / "optimization_v4_pareto_frontier.csv", [flatten_trial_for_csv(t) for t in frontier])
    _write_csv(out_dir / "optimization_v4_rejected_trials.csv", [flatten_trial_for_csv(t) for t in trials if t.rejected])
    _write_markdown(out_dir / "optimization_v4_summary.md", _summary_md(trials, symbols, args))
    _write_markdown(out_dir / "optimization_v4_top20.md", _top20_md(trials, frontier))
    _write_markdown(out_dir / "optimization_v4_oos_verdict.md", _oos_verdict_md(trials, frontier, recommendations, args))
    rec_payload = {name: trial.to_dict() if trial else None for name, trial in recommendations.items()}
    _write_json(out_dir / "recommended_configs_v4.json", rec_payload)
    _write_markdown(out_dir / "recommended_configs_v4.md", _recommendations_md(recommendations))


def _write_preflight_reports(out_dir: Path, symbols: List[str], frames: Mapping[str, pd.DataFrame], trials: List[TrialResult]) -> None:
    lines = [
        "# Repo Audit for v4 Optimization",
        "",
        "- 상태: 현재 작업 폴더는 git repo가 아닌 MT5 bot runtime 폴더입니다.",
        "- 라이브 주문 API 경로: brokers/mt5_live.py, execution/order_manager.py, execution/manual_position_guard.py.",
        "- 이번 최적화 CLI는 CSV와 pure-Python backtest만 사용하며 order_send/order_check/modify/close_position을 호출하지 않습니다.",
        "- 기존 fee-aware risk: core/risk_model.py, execution/position_sizer.py.",
        "- 기존 v4-style simulation: utils/execution_style_backtest.py.",
        "- 새 optimization modules: core/optimization/*, tools/optimize_execution_v4.py.",
        "",
        "## 데이터",
    ]
    for symbol, frame in frames.items():
        lines.append(f"- {symbol}: rows={len(frame)}")
    lines.extend(["", "## 최적화 후보 파라미터", "- entry/risk/initial_exit/profit_lock/trailing/daily_bleed/lsr search space 사용"])
    _write_markdown(out_dir / "repo_audit_for_v4_optimization.md", "\n".join(lines) + "\n")

    accepted = [t for t in trials if not t.rejected]
    source = accepted[0] if accepted else (trials[0] if trials else None)
    margin_lines = [
        "# Leverage / Margin Analysis",
        "",
        "- account_info().leverage는 브로커 계좌 속성이며 최적화 대상이 아닙니다.",
        "- 최적화/검증 대상은 max_effective_leverage, max_margin_used_pct, lot, SL distance, risk cap입니다.",
        "- effective_leverage = position_notional / account_equity",
        "- margin_used_pct = required_margin / account_equity * 100",
        "",
    ]
    if source is not None:
        margin_lines.append("## 관측치")
        margin_lines.append(f"- effective_leverage_max: {source.metrics.get('effective_leverage_max')}")
        margin_lines.append(f"- margin_used_pct_max: {source.metrics.get('margin_used_pct_max')}")
        margin_lines.append(f"- verdict: {config_margin_verdict(source.metrics, source.config)}")
    _write_markdown(out_dir / "leverage_margin_analysis.md", "\n".join(margin_lines) + "\n")


def _summary(trials: List[TrialResult]) -> Dict[str, Any]:
    accepted = [t for t in trials if not t.rejected]
    return {
        "trial_count": len(trials),
        "accepted_count": len(accepted),
        "rejected_count": len(trials) - len(accepted),
        "best_robust_score": max((t.robust_score for t in trials), default=0.0),
        "best_total_net_pnl": max((float(t.metrics.get("total_net_pnl", 0.0) or 0.0) for t in trials), default=0.0),
    }


def _summary_md(trials: List[TrialResult], symbols: List[str], args: argparse.Namespace) -> str:
    s = _summary(trials)
    return "\n".join(
        [
            "# Optimization v4 Summary",
            "",
            f"- symbols: {', '.join(symbols)}",
            f"- trials: {s['trial_count']}",
            f"- accepted: {s['accepted_count']}",
            f"- rejected: {s['rejected_count']}",
            f"- seed: {args.seed}",
            f"- mode: {args.mode}",
            "- live trading/order APIs: not used",
        ]
    ) + "\n"


def _top20_md(trials: List[TrialResult], frontier: List[TrialResult]) -> str:
    lines = ["# Optimization v4 Top 20", "", "## Top by robust_score"]
    for trial in sorted(trials, key=lambda t: t.robust_score, reverse=True)[:20]:
        lines.append(_trial_line(trial))
    lines.extend(["", "## Top by total_net_pnl"])
    for trial in sorted(trials, key=lambda t: float(t.metrics.get("total_net_pnl", 0.0) or 0.0), reverse=True)[:20]:
        lines.append(_trial_line(trial))
    lines.extend(["", "## Top by net_profit_factor"])
    for trial in sorted(trials, key=lambda t: float(t.metrics.get("net_profit_factor", 0.0) or 0.0), reverse=True)[:20]:
        lines.append(_trial_line(trial))
    lines.extend(["", "## Top by low_drawdown"])
    for trial in sorted(trials, key=lambda t: float(t.metrics.get("max_drawdown_pct", 0.0) or 0.0))[:20]:
        lines.append(_trial_line(trial))
    lines.extend(["", "## Top by tradability"])
    for trial in sorted(trials, key=lambda t: (float(t.metrics.get("median_trades_per_day", 0.0) or 0.0), -float(t.metrics.get("no_trade_days_pct", 0.0) or 0.0)), reverse=True)[:20]:
        lines.append(_trial_line(trial))
    lines.extend(["", "## Pareto frontier"])
    for trial in frontier[:50]:
        lines.append(_trial_line(trial))
    return "\n".join(lines) + "\n"


def _oos_verdict_md(
    trials: List[TrialResult],
    frontier: List[TrialResult],
    recommendations: Mapping[str, Optional[TrialResult]],
    args: argparse.Namespace,
) -> str:
    lines = [
        "# Optimization v4 OOS Verdict",
        "",
        "## 1. 요약 결론",
        "- 이 결과는 live 적용 config가 아니라 paper forward 후보입니다.",
        "- 모든 trial은 MT5 live order API 없이 CSV 기반 offline simulation으로 실행했습니다.",
        "",
        "## 2. 현재 config 리스크/레버리지/로트 상태",
        f"- account_equity 가정: {args.account_equity}",
        f"- account_leverage 가정: {args.account_leverage}",
        "- account leverage 자체는 최적화 대상으로 취급하지 않았습니다.",
        "",
        "## 3. 구현/변경 파일 목록",
        "- core/optimization/*.py",
        "- tools/optimize_execution_v4.py",
        "- utils/execution_style_backtest.py",
        "",
        "## 4. search space 요약",
        "- entry, risk, initial_exit, profit_lock, trailing, daily_bleed, lsr 파라미터 random/grid search",
        "",
        "## 5. trial 실행 조건",
        f"- trials={len(trials)}, seed={args.seed}, mode={args.mode}",
        "",
        "## 6. hard reject 조건",
        "- hard max loss breach, daily loss breach, no-trade, OOS PF/expectancy/trade count, leverage/margin cap",
        "",
        "## 7. 전체 trial 결과 요약",
        f"- accepted={sum(1 for t in trials if not t.rejected)}",
        f"- rejected={sum(1 for t in trials if t.rejected)}",
        "",
        "## 8. top by robust score",
    ]
    for trial in sorted(trials, key=lambda t: t.robust_score, reverse=True)[:10]:
        lines.append(_trial_line(trial))
    lines.extend(["", "## 9. top by net profit"])
    for trial in sorted(trials, key=lambda t: float(t.metrics.get("total_net_pnl", 0.0) or 0.0), reverse=True)[:10]:
        lines.append(_trial_line(trial))
    lines.extend(["", "## 10. Pareto frontier"])
    for trial in frontier[:20]:
        lines.append(_trial_line(trial))
    for section, name in (("11", "aggressive"), ("12", "balanced"), ("13", "conservative")):
        lines.extend(["", f"## {section}. {name} 추천 config"])
        lines.append(_recommendation_block(name, recommendations.get(name)))
    lines.extend(
        [
            "",
            "## 14. 각 후보의 장단점",
            "- aggressive: 수익/빈도 우선, DD와 daily bleed 리스크를 paper forward에서 집중 확인.",
            "- balanced: robust score 우선, 1순위 paper forward 후보.",
            "- conservative: DD/단일손실/마진 압력 우선 축소.",
            "",
            "## 15. OOS 성능 및 decay",
            "- oos_decay_pct, oos_total_trades, oos_net_profit_factor, oos_expectancy_net를 trial별 저장.",
            "",
            "## 16. no-trade / tradability 분석",
            "- 거래 0건과 no_trade_days_pct가 높은 trial은 성공으로 보지 않고 hard reject 또는 penalty.",
            "",
            "## 17. DailyBleedGuard 분석",
            "- max_daily_loss_R 기반 daily loss breach를 reject 조건에 반영.",
            "",
            "## 18. profit-lock trailing 분석",
            "- profit_lock_move_count, profit_lock_saved_pnl, winner_capture_ratio 저장.",
            "",
            "## 19. effective leverage / margin 분석",
            "- effective_leverage_max와 margin_used_pct_max가 각 trial cap을 넘으면 reject.",
            "",
            "## 20. 왜 단순 최고수익 후보를 선택하지 않았는가",
            "- hard max loss, no-trade, OOS decay, margin pressure를 동시에 통과해야 paper forward 후보가 됩니다.",
            "",
            "## 21. live 적용 금지 및 paper forward 조건",
            "- 추천 config는 live 적용 금지. 최소 1주 paper forward에서 OOS trade count, max loss, DD, margin pressure 재검증 필요.",
            "",
            "## 22. 다음 작업 제안",
            "- paper forward 로그 수집 후 동일 CLI로 재검증하고, 통과 후보만 별도 승인 절차로 live config patch 검토.",
        ]
    )
    return "\n".join(lines) + "\n"


def _recommendations_md(recommendations: Mapping[str, Optional[TrialResult]]) -> str:
    lines = ["# Recommended Configs v4", "", "- live 적용 금지: paper forward 검증 후보입니다.", ""]
    for name in ("aggressive", "balanced", "conservative"):
        lines.append(f"## {name}")
        lines.append(_recommendation_block(name, recommendations.get(name)))
        lines.append("")
    return "\n".join(lines)


def _recommendation_block(name: str, trial: Optional[TrialResult]) -> str:
    if trial is None:
        return "- 후보 없음: hard reject를 통과한 trial이 없습니다."
    cfg = json.dumps(trial.config, indent=2, sort_keys=True, ensure_ascii=False)
    m = trial.metrics
    return "\n".join(
        [
            f"- trial_id: {trial.trial_id}",
            f"- robust_score: {trial.robust_score:.6f}",
            f"- total_net_pnl: {m.get('total_net_pnl')}",
            f"- total_trades: {m.get('total_trades')}",
            f"- max_single_trade_net_loss: {m.get('max_single_trade_net_loss')}",
            f"- max_drawdown_pct: {m.get('max_drawdown_pct')}",
            f"- no_trade_days_pct: {m.get('no_trade_days_pct')}",
            f"- oos_total_trades: {m.get('oos_total_trades')}",
            f"- oos_net_profit_factor: {m.get('oos_net_profit_factor')}",
            f"- oos_expectancy_net: {m.get('oos_expectancy_net')}",
            f"- effective_leverage_max: {m.get('effective_leverage_max')}",
            f"- margin_used_pct_max: {m.get('margin_used_pct_max')}",
            "",
            "```json",
            cfg,
            "```",
        ]
    )


def _trial_line(trial: TrialResult) -> str:
    m = trial.metrics
    return (
        f"- trial={trial.trial_id} rejected={trial.rejected} score={trial.robust_score:.4f} "
        f"pnl={float(m.get('total_net_pnl', 0.0) or 0.0):.4f} "
        f"pf={float(m.get('net_profit_factor', 0.0) or 0.0):.4f} "
        f"trades={int(m.get('total_trades', 0) or 0)} "
        f"oos_trades={int(m.get('oos_total_trades', 0) or 0)} "
        f"no_trade={float(m.get('no_trade_days_pct', 0.0) or 0.0):.2f}% "
        f"eff_lev={float(m.get('effective_leverage_max', 0.0) or 0.0):.4f} "
        f"margin={float(m.get('margin_used_pct_max', 0.0) or 0.0):.4f}% "
        f"reasons={','.join(trial.reject_reasons[:3])}"
    )


def _trial_configs(mode: str, trials: int, seed: int) -> List[Dict[str, Any]]:
    if mode == "grid":
        return grid_trial_configs(trials)
    # Optuna is deliberately treated as deterministic random unless the optional
    # dependency is introduced explicitly later.
    return random_trial_configs(trials, seed)


def _load_frames(data_dir: Path, symbols: List[str], max_rows: int) -> Dict[str, pd.DataFrame]:
    frames: Dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        path = _find_csv(data_dir, symbol)
        if path is None:
            continue
        frame = pd.read_csv(path)
        frame = _normalize_frame(frame)
        if max_rows and max_rows > 0 and len(frame) > max_rows:
            frame = frame.tail(max_rows).reset_index(drop=True)
        if len(frame) >= 50:
            frames[symbol] = frame
    if not frames:
        raise FileNotFoundError(f"no usable CSV data found under {data_dir}")
    return frames


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
    if "time" in frame.columns:
        frame["time"] = pd.to_datetime(frame["time"], utc=True, errors="coerce")
    else:
        frame["time"] = pd.date_range("2026-01-01", periods=len(frame), freq="min", tz="UTC")
    return frame.dropna(subset=["time", "open", "high", "low", "close"]).sort_values("time", kind="stable").reset_index(drop=True)


def _date_or_ratio_split(frame: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    if args.train_start or args.train_end or args.oos_start or args.oos_end:
        train = _date_filter(frame, args.train_start, args.train_end)
        oos = _date_filter(frame, args.oos_start, args.oos_end)
        if len(train) >= 30 and len(oos) >= 30:
            return train, oos
    return train_oos_split(frame, 0.7)


def _date_filter(frame: pd.DataFrame, start: Optional[str], end: Optional[str]) -> pd.DataFrame:
    out = frame
    if start:
        out = out[out["time"] >= pd.Timestamp(start, tz="UTC")]
    if end:
        out = out[out["time"] <= pd.Timestamp(end, tz="UTC")]
    return out.reset_index(drop=True)


def _parse_symbols(raw: str) -> List[str]:
    out: List[str] = []
    for item in str(raw or "").split(","):
        symbol = "".join(ch for ch in item.upper() if ch.isalnum())
        if symbol and symbol not in out:
            out.append(symbol)
    return out


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_markdown(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    keys = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


if __name__ == "__main__":
    raise SystemExit(main())
