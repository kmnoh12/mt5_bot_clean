from __future__ import annotations

import itertools
import random
from typing import Any, Dict, Iterable, List, Mapping


SEARCH_SPACE_V4: Dict[str, Dict[str, List[Any]]] = {
    "entry": {
        "min_signal_score": [55, 60, 65, 70, 75, 80],
        "min_fee_adjusted_rr": [1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
    },
    "risk": {
        "target_net_loss_usd": [0.75, 1.00, 1.25],
        "hard_max_net_loss_usd": [1.00, 1.25, 1.50],
        "max_effective_leverage": [3.0, 5.0, 8.0, 10.0, 15.0, 20.0, 30.0],
        "max_margin_used_pct": [5, 10, 15, 20, 30],
    },
    "initial_exit": {
        "initial_tp_R": [3, 4, 5, 6, 8, 10],
        "min_reward_to_net_risk_ratio": [2.0, 2.5, 3.0, 3.5],
    },
    "profit_lock": {
        "breakeven_trigger_R": [1.2, 1.5, 2.0, 2.5],
        "breakeven_lock_R": [0.0, 0.2, 0.5],
        "lock1_trigger_R": [2.5, 3.0, 3.5, 4.0],
        "lock1_sl_R": [0.5, 1.0, 1.5],
        "lock2_trigger_R": [5.0, 6.0, 8.0],
        "lock2_sl_R": [2.0, 3.0, 4.0],
        "runner_trigger_R": [5.0, 8.0, 10.0],
        "runner_sl_R": [3.0, 5.0, 6.0],
        "runner_tp_R": [15, 20, 25, 30],
    },
    "trailing": {
        "method": ["fixed_R_ladder", "atr_trailing", "structure_trailing", "hybrid"],
        "atr_trail_mult": [0.5, 0.8, 1.0, 1.5, 2.0],
        "trail_start_R": [2.0, 3.0, 5.0],
    },
    "daily_bleed": {
        "max_daily_loss_R": [2, 3, 4, 5],
        "stop_after_consecutive_losses": [2, 3, 4],
        "cooldown_after_loss_minutes": [15, 30, 60, 90],
    },
    "lsr": {
        "swing_window": [20, 30, 45, 60],
        "sweep_buffer_atr": [0.02, 0.05, 0.08, 0.1, 0.16],
        "reclaim_buffer_atr": [0.0, 0.05, 0.1, 0.16],
        "reclaim_window_sec": [60, 120, 300, 600, 900],
        "sl_atr_mult": [0.25, 0.35, 0.45, 0.6, 0.9],
        "stop_buffer_atr": [0.02, 0.05, 0.08, 0.12, 0.16],
        "max_hold_bars": [30, 80, 160, 240, 720],
        "min_cooldown_bars": [3, 8, 15, 30],
    },
}


def random_trial_configs(trials: int, seed: int, space: Mapping[str, Mapping[str, List[Any]]] = SEARCH_SPACE_V4) -> List[Dict[str, Any]]:
    rng = random.Random(int(seed))
    return [_sample_one(rng, space) for _ in range(max(0, int(trials)))]


def grid_trial_configs(limit: int, space: Mapping[str, Mapping[str, List[Any]]] = SEARCH_SPACE_V4) -> List[Dict[str, Any]]:
    keys: List[tuple[str, str, List[Any]]] = []
    for section, values in space.items():
        for key, choices in values.items():
            keys.append((section, key, list(choices)))
    out: List[Dict[str, Any]] = []
    for combo in itertools.product(*(item[2] for item in keys)):
        cfg: Dict[str, Any] = {}
        for (section, key, _), value in zip(keys, combo):
            cfg.setdefault(section, {})[key] = value
        out.append(cfg)
        if len(out) >= int(limit):
            break
    return out


def flatten_config(config: Mapping[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for section, values in config.items():
        if isinstance(values, Mapping):
            for key, value in values.items():
                out[f"{section}.{key}"] = value
        else:
            out[str(section)] = values
    return out


def _sample_one(rng: random.Random, space: Mapping[str, Mapping[str, List[Any]]]) -> Dict[str, Any]:
    cfg: Dict[str, Any] = {}
    for section, values in space.items():
        cfg[section] = {key: rng.choice(list(choices)) for key, choices in values.items()}
    return cfg
