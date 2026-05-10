from __future__ import annotations

from typing import Any, Dict

from strategies.deep_wave_sm import DeepWaveStrategy
from strategies.liquidity_sweep_reversal import LiquiditySweepReversalStrategy
from strategies.liquidity_sweep_reversal_tick import LiquiditySweepReversalTickStrategy
from strategies.mean_reversion_sm import MeanReversionStateMachine
from strategies.trend_regime_sm import TrendRegimeStateMachine
from strategies.vol_breakout_sm import VolBreakoutStateMachine


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _normalize_symbol(symbol: Any) -> str:
    return str(symbol or "").strip().upper()


def _apply_symbol_profile_strategy_overrides(config: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    strategy_cfg = dict(config.get("strategies", {})) if isinstance(config.get("strategies"), dict) else {}
    symbol_profiles = (
        dict(config.get("symbol_profiles", {}))
        if isinstance(config.get("symbol_profiles"), dict)
        else {}
    )

    for raw_symbol, raw_profile in symbol_profiles.items():
        symbol = _normalize_symbol(raw_symbol)
        if not symbol or not isinstance(raw_profile, dict):
            continue
        profile_strategies = raw_profile.get("strategies")
        if not isinstance(profile_strategies, dict):
            continue

        for raw_strategy_name, strategy_override in profile_strategies.items():
            strategy_name = str(raw_strategy_name or "").strip()
            if not strategy_name or not isinstance(strategy_override, dict):
                continue

            current_strategy_cfg = (
                dict(strategy_cfg.get(strategy_name, {}))
                if isinstance(strategy_cfg.get(strategy_name), dict)
                else {}
            )
            symbol_params = (
                dict(current_strategy_cfg.get("symbol_params", {}))
                if isinstance(current_strategy_cfg.get("symbol_params"), dict)
                else {}
            )
            current_symbol_params = (
                dict(symbol_params.get(symbol, {}))
                if isinstance(symbol_params.get(symbol), dict)
                else {}
            )
            symbol_params[symbol] = _deep_merge(current_symbol_params, strategy_override)
            current_strategy_cfg["symbol_params"] = symbol_params
            strategy_cfg[strategy_name] = current_strategy_cfg

    return strategy_cfg


def build_strategies(config: Dict[str, Any], state_snapshot: Dict[str, Any]) -> Dict[str, Any]:
    strategy_cfg = _apply_symbol_profile_strategy_overrides(config) if isinstance(config, dict) else {}
    snapshot = state_snapshot if isinstance(state_snapshot, dict) else {}
    return {
        "mean_reversion_sm": MeanReversionStateMachine(
            config=dict(strategy_cfg.get("mean_reversion_sm", {})),
            snapshot=dict(snapshot.get("mean_reversion_sm", {})),
        ),
        "vol_breakout_sm": VolBreakoutStateMachine(
            config=dict(strategy_cfg.get("vol_breakout_sm", {})),
            snapshot=dict(snapshot.get("vol_breakout_sm", {})),
        ),
        "trend_regime_sm": TrendRegimeStateMachine(
            config=dict(strategy_cfg.get("trend_regime_sm", {})),
            snapshot=dict(snapshot.get("trend_regime_sm", {})),
        ),
        "deep_wave_sm": DeepWaveStrategy(
            config=dict(strategy_cfg.get("deep_wave_sm", {})),
            snapshot=dict(snapshot.get("deep_wave_sm", {})),
        ),
        "liquidity_sweep_reversal": LiquiditySweepReversalStrategy(
            config=dict(strategy_cfg.get("liquidity_sweep_reversal", {})),
            snapshot=dict(snapshot.get("liquidity_sweep_reversal", {})),
        ),
        "liquidity_sweep_reversal_tick": LiquiditySweepReversalTickStrategy(
            config=dict(strategy_cfg.get("liquidity_sweep_reversal_tick", {})),
            snapshot=dict(snapshot.get("liquidity_sweep_reversal_tick", {})),
        ),
    }
