from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover
    yaml = None


DEFAULT_CONFIG: Dict[str, Any] = {
    "general": {
        "mode": "live",
        "dry_run": True,
        "log_level": "INFO",
        "poll_seconds": 1,
        "bars_per_request": 300,
        "heartbeat_seconds": 10,
        "reconnect": {
            "max_attempts": 6,
            "attempts_per_cycle": 1,
            "base_delay_seconds": 1.0,
            "max_delay_seconds": 30.0,
            "cooldown_seconds": 2.0,
            "jitter_seconds": 0.5,
            "max_ipc_failures_before_halt": 8,
            "force_shutdown_on_reconnect": False,
        },
    },
    "mt5": {
        "login": 0,
        "password": "",
        "server": "",
        "path": "",
        "init_timeout_ms": 120000,
        "reconnect_init_timeout_ms": 15000,
        "attach_only": True,
        "allow_programmatic_login": False,
        "allow_terminal_launch": False,
        "account_guard_enabled": True,
        "require_trade_enabled": True,
        "ipc_timeout_codes": [-10005],
    },
    "backtest": {
        "data_dir": "./data",
        "initial_balance": 10000.0,
        "spread_points": 8.0,
        "commission_per_lot": 0.0,
        "contract_size": 1.0,
        "fast_mode": True,
    },
    "execution": {
        "dry_run": False,
        "live_trading_enabled": False,
        "deviation": 20,
        "magic": 20260206,
        "comment_prefix": "quant_bot",
        "order_retry_attempts": 3,
        "default_volume": 0.01,
        "allow_opposite_position": False,
        "max_positions_per_symbol": 1,
        "close_positions_on_exit": True,
        "bar_close_only": True,
    },
    "execution_style": {
        "name": "fee_aware_fixed_risk_profit_lock",
        "long_enabled": True,
        "short_enabled": True,
        "objective": "Find high-quality long/short entries, cap net loss per trade, lock profit aggressively.",
    },
    "risk_per_trade": {
        "target_net_loss_usd": 1.0,
        "hard_max_net_loss_usd": 1.25,
        "spread_points": 0.0,
        "commission_per_lot": 0.0,
        "expected_slippage_points": 0.0,
    },
    "entry_quality": {
        "min_reward_to_net_risk_ratio": 3.0,
        "min_signal_score": 70,
        "forbid_late_entry": True,
        "require_deterministic_signal": True,
        "allow_llm_discretionary_entry": False,
        "allow_llm_discretionary_veto": False,
    },
    "initial_exit": {
        "take_profit": {
            "min_profit_usd": 3.0,
            "preferred_profit_usd": 5.0,
            "allow_tp_extension": True,
        },
    },
    "profit_lock": {
        "enabled": True,
        "evaluate_on_tick": True,
        "use_net_unrealized_pnl_after_estimated_exit_costs": True,
        "never_move_sl_backward": True,
        "min_seconds_between_sltp_updates": 10,
    },
    "daily_bleed_guard": {
        "enabled": True,
        "max_daily_net_loss_usd": 3.0,
        "stop_after_consecutive_losses": 3,
        "cooldown_after_loss_minutes": 30,
        "cooldown_after_same_setup_loss_minutes": 60,
        "same_direction_loss_limit_per_day": 2,
        "same_symbol_loss_limit_per_day": 3,
    },
    "opportunity_scanner": {
        "enabled": True,
        "drive_entries": True,
        "lookback_bars": 20,
        "atr_period": 14,
        "sweep_buffer_atr": 0.05,
        "stop_buffer_atr": 0.05,
        "late_entry_atr_mult": 0.75,
        "late_entry_min_rr": 1.0,
    },
    "fee_aware_entry_filter": {
        "enabled": True,
        "max_signal_age_seconds": 300,
    },
    "no_trade_bias_guard": {
        "enabled": True,
        "warning_no_trade_hours": 24.0,
        "failure_no_trade_hours": 48.0,
        "top_rejected_limit": 5,
    },
    "reports": {
        "enabled": True,
        "output_dir": "reports",
    },
    "risk_guard": {
        "risk_per_trade_pct": 0.003,
        "max_risk_per_trade_pct": 0.005,
        "daily_loss_limit_pct": 0.02,
        "session_loss_limit_pct": 0.03,
        "max_consecutive_losses": 4,
        "dynamic_risk_cap_enabled": True,
        "dynamic_risk_min_pct": 0.001,
        "dynamic_risk_max_pct": 0.005,
        "dynamic_risk_lookback_hours": 24,
        "dynamic_risk_min_samples": 8,
        "dynamic_risk_winrate_floor": 0.35,
        "dynamic_risk_winrate_ceiling": 0.65,
    },
    "trailing_profit_guard": {
        "enabled": True,
        "retain_ratio": 0.5,
        "min_activation_profit_usd": 3.0,
        "stage_a_activation_usd": 3.0,
        "stage_b_activation_usd": 15.0,
        "stage_b_retain_ratio": 0.10,
        "stage_c_activation_usd": 35.0,
        "stage_c_retain_ratio": 0.35,
        "break_even_enabled": True,
        "break_even_activation_profit_usd": 3.0,
        "break_even_lock_profit_usd": 0.2,
        "break_even_sync_sl": True,
        "min_hold_seconds_for_exit": 300.0,
        "min_drawdown_usd_for_exit": 2.5,
        "min_breach_count_for_exit": 2,
    },
    "manual_position_guard": {
        "enabled": False,
        "symbols": [],
        "manual_magic_values": [0],
        "retain_ratio": 0.8,
        "min_activation_profit_usd": 5.0,
        "breach_close": True,
        "sync_sl_to_lock_line": True,
        "close_retry_cooldown_seconds": 30,
        "block_strategy_for_protected_symbols": False,
    },
    "execution_churn_guard": {
        "enabled": True,
        "reentry_cooldown_seconds": 120,
        "flip_reentry_cooldown_seconds": 30,
        "max_entries_per_symbol_per_hour": 2,
        "max_entries_per_symbol_per_day": 3,
        "max_entries_per_symbol_per_day_eth": 0,
        "max_entries_global_per_day": 1,
        "daily_reset_timezone": "Asia/Seoul",
        "per_symbol_daily_limits": {"BTCUSD": 1, "ETHUSD": 0, "GOLD": 0},
        "min_hold_bars_floor": 2,
        "min_hold_bars_floor_by_symbol": {"BTCUSD": 3, "ETHUSD": 3, "GOLD": 2},
        "tiny_pnl_threshold_usd": 2.0,
        "quick_exit_window_seconds": 300,
        "tiny_pnl_max_count_per_hour": 2,
        "tiny_pnl_cooldown_seconds": 3600,
        "protection_failure_lock_seconds": 180,
        "loss_reentry_lock_seconds": 180,
    },
    "entry_quality_guard": {
        "enabled": True,
        "trend_only_symbols": ["BTCUSD"],
        "min_score": 0.62,
        "min_score_risk_off": 0.68,
        "min_score_risk_on": 0.58,
        "lookback_closed_trades": 200,
        "min_winner_pnl_usd": 5.0,
        "max_churn_abs_pnl_usd": 2.0,
        "max_churn_hold_seconds": 300,
    },
    "cost_edge_guard": {
        "enabled": True,
        "min_edge_to_cost_ratio_default": 3.0,
        "min_edge_to_cost_ratio_by_symbol": {"GOLD": 2.5},
        "spread_sample_bars": 120,
        "use_recent_deal_cost_stats": True,
    },
    "exit_quality_guard": {
        "enabled": True,
        "tiny_profit_block_usd": 2.0,
        "min_hold_seconds_for_soft_exit": 300,
        "m5_reverse_confirm_bars": 2,
    },
    "broker_request_guard": {
        "enabled": True,
        "comment_ascii_only": True,
        "comment_max_len": 24,
        "retry_without_comment_on_invalid_comment": True,
    },
    "mtf_confirm": {
        "enabled": True,
        "symbols": ["BTCUSD"],
        "confirm_timeframe": "TIMEFRAME_M5",
        "fast_ema": 20,
        "slow_ema": 50,
    },
    "trade_journal": {
        "enabled": True,
        "output_dir": "./reports/trade_journal",
        "tiny_pnl_threshold_usd": 2.0,
        "quick_exit_window_seconds": 300,
        "big_loss_threshold_usd": -10.0,
    },
    "telegram": {
        "enabled": False,
        "bot_token": "",
        "chat_id": "",
        "notify_trade": True,
        "notify_error": True,
        "notify_system": True,
    },
    "external_signals": {
        "enabled": False,
        "source": "none",
        "allowed_symbols": [],
        "json_file": {
            "path": "./signals/inbox.json",
            "consume_mode": "mark_used",
        },
        "socket": {
            "host": "127.0.0.1",
            "port": 8765,
            "max_queue_size": 500,
        },
    },
    "llm_assist": {
        "enabled": False,
        "provider": "gemini",
        "model": "gemini-3-flash",
        "api_key": "",
        "api_key_env": "GEMINI_API_KEY",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "timeout_ms": 800,
        "approve_confidence": 0.55,
        "veto_confidence": 0.65,
        "scale_on_ambiguous": 0.5,
        "max_bars_for_prompt": 60,
        "settings_path": "./dashboard_settings.json",
    },
    "dashboard": {
        "enabled": True,
        "host": "127.0.0.1",
        "port": 8010,
        "control_path": "./runtime_control.json",
        "settings_path": "./dashboard_settings.json",
    },
    "auto_tuning": {
        "enabled": False,
        "target_symbols": ["BTCUSD"],
        "tune_interval_seconds": 300,
        "lookback_bars": 120,
        "min_bars": 80,
        "smoothing_alpha": 0.25,
        "parameter_bounds": {
            "trend_score_threshold": {"min": 0.08, "max": 0.85},
            "trend_strength_threshold": {"min": 0.12, "max": 0.90},
            "meanrev_max_strength": {"min": 0.12, "max": 0.80},
            "breakout_lookback": {"min": 3, "max": 30},
            "trend_sl_atr_mult": {"min": 0.6, "max": 3.2},
            "trend_tp_r_multiple": {"min": 0.8, "max": 5.0},
            "trailing_atr_mult": {"min": 0.4, "max": 3.0},
            "trailing_start_rr": {"min": 0.1, "max": 3.0},
            "regime_flip_exit_threshold": {"min": 0.05, "max": 0.55},
        },
    },
    "validation": {
        "require_oos_pass": False,
        "report_path": "./validation/oos_report.json",
    },
    "storage": {
        "state_path": "./state.json",
        "events_path": "./events.jsonl",
    },
    "required_active_symbols": ["BTCUSD"],
    "nasdaq_universe": ["BTCUSD"],
    "symbol_profiles": {},
    "universe": [
        {
            "symbol": "BTCUSD",
            "strategy": "trend_regime_sm",
            "timeframe": "TIMEFRAME_M1",
            "volume": 0.01,
        },

    ],
    "strategies": {
        "mean_reversion_sm": {
            "enabled": False,
            "rsi_period": 14,
            "rsi_oversold": 30,
            "rsi_overbought": 70,
            "bollinger_period": 20,
            "bollinger_stddev": 2.0,
            "sl_atr_mult": 1.0,
            "tp_atr_mult": 1.5,
            "min_cooldown_bars": 3,
            "min_hold_bars": 1,
        },
        "vol_breakout_sm": {
            "enabled": False,
            "lookback": 20,
            "atr_period": 14,
            "breakout_atr_multiple": 1.0,
            "sl_atr_mult": 1.0,
            "tp_atr_mult": 2.0,
            "trailing_exit_atr_multiple": 1.2,
            "min_cooldown_bars": 2,
            "min_hold_bars": 1,
        },
        "trend_regime_sm": {
            "enabled": True,
            "fast_ema_period": 20,
            "slow_ema_period": 80,
            "atr_period": 14,
            "adx_period": 14,
            "rsi_period": 14,
            "slope_lookback": 8,
            "breakout_lookback": 5,
            "meanrev_lookback": 10,
            "adx_floor": 14.0,
            "adx_ceiling": 35.0,
            "atr_pct_floor": 0.0006,
            "atr_pct_ceiling": 0.02,
            "ema_gap_norm_divisor": 3.0,
            "slope_norm_divisor": 0.5,
            "trend_score_threshold": 0.32,
            "trend_strength_threshold": 0.42,
            "meanrev_max_strength": 0.40,
            "meanrev_entry_distance_atr": 0.85,
            "meanrev_rsi_oversold": 35.0,
            "meanrev_rsi_overbought": 65.0,
            "trend_sl_atr_mult": 1.2,
            "trend_tp_r_multiple": 2.1,
            "meanrev_sl_atr_mult": 1.0,
            "meanrev_tp_r_multiple": 1.4,
            "sl_atr_mult": 1.2,
            "tp_r_multiple": 2.0,
            "break_even_rr": 1.0,
            "break_even_offset_atr": 0.05,
            "trailing_atr_mult": 1.0,
            "trailing_start_rr": 0.8,
            "time_stop_bars": 60,
            "regime_flip_exit_threshold": 0.18,
            "min_hold_bars": 1,
            "min_cooldown_bars": 2,
        },
    },
}


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _migrate_legacy(raw: Dict[str, Any]) -> Dict[str, Any]:
    migrated = dict(raw or {})

    account_cfg = migrated.get("account")
    if isinstance(account_cfg, dict):
        mt5_cfg = dict(migrated.get("mt5", {}))
        if not mt5_cfg.get("login"):
            mt5_cfg["login"] = account_cfg.get("login", 0)
        if not mt5_cfg.get("password"):
            mt5_cfg["password"] = account_cfg.get("password", "")
        if not mt5_cfg.get("server"):
            mt5_cfg["server"] = account_cfg.get("server", "")
        migrated["mt5"] = mt5_cfg

    return migrated


def _normalize_symbol(symbol: Any) -> str:
    text = str(symbol or "").strip()
    return text.upper()


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _apply_symbol_profiles(config: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(config or {})
    profiles = out.get("symbol_profiles")
    if not isinstance(profiles, dict):
        return out

    universe = list(out.get("universe", [])) if isinstance(out.get("universe"), list) else []
    risk_guard = dict(out.get("risk_guard", {})) if isinstance(out.get("risk_guard"), dict) else {}
    churn_guard = (
        dict(out.get("execution_churn_guard", {}))
        if isinstance(out.get("execution_churn_guard"), dict)
        else {}
    )
    cost_guard = dict(out.get("cost_edge_guard", {})) if isinstance(out.get("cost_edge_guard"), dict) else {}
    strategies = dict(out.get("strategies", {})) if isinstance(out.get("strategies"), dict) else {}

    per_symbol_risk = (
        dict(risk_guard.get("per_symbol_risk_per_trade_pct", {}))
        if isinstance(risk_guard.get("per_symbol_risk_per_trade_pct"), dict)
        else {}
    )
    per_symbol_daily_limits = (
        dict(churn_guard.get("per_symbol_daily_limits", {}))
        if isinstance(churn_guard.get("per_symbol_daily_limits"), dict)
        else {}
    )
    min_hold_floor_by_symbol = (
        dict(churn_guard.get("min_hold_bars_floor_by_symbol", {}))
        if isinstance(churn_guard.get("min_hold_bars_floor_by_symbol"), dict)
        else {}
    )
    min_edge_ratio_by_symbol = (
        dict(cost_guard.get("min_edge_to_cost_ratio_by_symbol", {}))
        if isinstance(cost_guard.get("min_edge_to_cost_ratio_by_symbol"), dict)
        else {}
    )

    for raw_symbol, raw_profile in profiles.items():
        symbol = _normalize_symbol(raw_symbol)
        if not symbol or not isinstance(raw_profile, dict):
            continue

        profile_universe = raw_profile.get("universe")
        if isinstance(profile_universe, dict):
            found_index = None
            current_entry: Dict[str, Any] = {}
            for idx, item in enumerate(universe):
                if isinstance(item, dict) and _normalize_symbol(item.get("symbol")) == symbol:
                    found_index = idx
                    current_entry = dict(item)
                    break

            merged_entry = deep_merge(current_entry, profile_universe)
            merged_entry["symbol"] = symbol
            if found_index is None:
                universe.append(merged_entry)
            else:
                universe[found_index] = merged_entry

        profile_risk = raw_profile.get("risk_guard")
        if isinstance(profile_risk, dict):
            risk_per_trade_pct = profile_risk.get("risk_per_trade_pct")
            if _is_number(risk_per_trade_pct):
                per_symbol_risk[symbol] = float(risk_per_trade_pct)

        profile_churn = raw_profile.get("execution_churn_guard")
        if isinstance(profile_churn, dict):
            per_symbol_daily_limit = profile_churn.get("per_symbol_daily_limit")
            if _is_number(per_symbol_daily_limit):
                per_symbol_daily_limits[symbol] = int(per_symbol_daily_limit)
            min_hold_bars_floor = profile_churn.get("min_hold_bars_floor")
            if _is_number(min_hold_bars_floor):
                min_hold_floor_by_symbol[symbol] = int(min_hold_bars_floor)

        profile_cost = raw_profile.get("cost_edge_guard")
        if isinstance(profile_cost, dict):
            min_edge_ratio = profile_cost.get("min_edge_to_cost_ratio")
            if _is_number(min_edge_ratio):
                min_edge_ratio_by_symbol[symbol] = float(min_edge_ratio)

        profile_strategies = raw_profile.get("strategies")
        if isinstance(profile_strategies, dict):
            for raw_strategy_name, strategy_override in profile_strategies.items():
                strategy_name = str(raw_strategy_name or "").strip()
                if not strategy_name or not isinstance(strategy_override, dict):
                    continue
                strategy_cfg = (
                    dict(strategies.get(strategy_name, {}))
                    if isinstance(strategies.get(strategy_name), dict)
                    else {}
                )
                symbol_params = (
                    dict(strategy_cfg.get("symbol_params", {}))
                    if isinstance(strategy_cfg.get("symbol_params"), dict)
                    else {}
                )
                current_symbol_params = (
                    dict(symbol_params.get(symbol, {}))
                    if isinstance(symbol_params.get(symbol), dict)
                    else {}
                )
                symbol_params[symbol] = deep_merge(current_symbol_params, strategy_override)
                strategy_cfg["symbol_params"] = symbol_params
                strategies[strategy_name] = strategy_cfg

    risk_guard["per_symbol_risk_per_trade_pct"] = per_symbol_risk
    churn_guard["per_symbol_daily_limits"] = per_symbol_daily_limits
    churn_guard["min_hold_bars_floor_by_symbol"] = min_hold_floor_by_symbol
    cost_guard["min_edge_to_cost_ratio_by_symbol"] = min_edge_ratio_by_symbol

    out["universe"] = universe
    out["risk_guard"] = risk_guard
    out["execution_churn_guard"] = churn_guard
    out["cost_edge_guard"] = cost_guard
    out["strategies"] = strategies
    return out


def load_config(config_path: Path) -> Dict[str, Any]:
    if yaml is None:
        raise RuntimeError("PyYAML is not installed. Install with `pip install PyYAML`.")

    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError("Configuration root must be a dictionary.")

    raw = _migrate_legacy(raw)
    config = deep_merge(DEFAULT_CONFIG, raw)
    config = _apply_symbol_profiles(config)
    return normalize_paths(config, config_path.parent)


def normalize_paths(config: Dict[str, Any], base_dir: Path) -> Dict[str, Any]:
    normalized = dict(config)

    def resolve(raw: Any) -> str:
        candidate = Path(str(raw or "")).expanduser()
        if candidate.is_absolute():
            return str(candidate)
        return str((base_dir / candidate).resolve())

    storage_cfg = dict(normalized.get("storage", {}))
    storage_cfg["state_path"] = resolve(storage_cfg.get("state_path", "./state.json"))
    storage_cfg["events_path"] = resolve(storage_cfg.get("events_path", "./events.jsonl"))
    normalized["storage"] = storage_cfg

    signals_cfg = dict(normalized.get("external_signals", {}))
    json_cfg = dict(signals_cfg.get("json_file", {}))
    json_cfg["path"] = resolve(json_cfg.get("path", "./signals/inbox.json"))
    signals_cfg["json_file"] = json_cfg
    normalized["external_signals"] = signals_cfg

    backtest_cfg = dict(normalized.get("backtest", {}))
    backtest_cfg["data_dir"] = resolve(backtest_cfg.get("data_dir", "./data"))
    normalized["backtest"] = backtest_cfg

    dashboard_cfg = dict(normalized.get("dashboard", {}))
    dashboard_cfg["control_path"] = resolve(dashboard_cfg.get("control_path", "./runtime_control.json"))
    dashboard_cfg["settings_path"] = resolve(dashboard_cfg.get("settings_path", "./dashboard_settings.json"))
    normalized["dashboard"] = dashboard_cfg

    llm_cfg = dict(normalized.get("llm_assist", {}))
    if not llm_cfg.get("settings_path"):
        llm_cfg["settings_path"] = dashboard_cfg["settings_path"]
    llm_cfg["settings_path"] = resolve(llm_cfg.get("settings_path", "./dashboard_settings.json"))
    normalized["llm_assist"] = llm_cfg

    validation_cfg = dict(normalized.get("validation", {}))
    validation_cfg["report_path"] = resolve(validation_cfg.get("report_path", "./validation/oos_report.json"))
    normalized["validation"] = validation_cfg

    journal_cfg = dict(normalized.get("trade_journal", {}))
    journal_cfg["output_dir"] = resolve(journal_cfg.get("output_dir", "./reports/trade_journal"))
    normalized["trade_journal"] = journal_cfg

    return normalized
