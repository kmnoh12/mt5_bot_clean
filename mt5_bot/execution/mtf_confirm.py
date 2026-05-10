from __future__ import annotations

from typing import Any, Dict, Iterable, Set

import pandas as pd

from core.models import DecisionAction
from utils.indicators import compute_ema, sanitize_ohlc


class MtfDirectionConfirm:
    def __init__(self, config: Dict[str, Any]) -> None:
        self.enabled = True
        self.symbols: Set[str] = {"BTCUSD", "ETHUSD"}
        self.confirm_timeframe = "TIMEFRAME_M5"
        self.fast_ema = 20
        self.slow_ema = 50
        self.update_config(config)

    def update_config(self, config: Dict[str, Any]) -> None:
        cfg = dict(config or {})
        self.enabled = bool(cfg.get("enabled", True))
        self.confirm_timeframe = str(cfg.get("confirm_timeframe", "TIMEFRAME_M5") or "TIMEFRAME_M5").strip()
        try:
            self.fast_ema = max(2, int(cfg.get("fast_ema", 20)))
        except (TypeError, ValueError):
            self.fast_ema = 20
        try:
            self.slow_ema = max(self.fast_ema + 1, int(cfg.get("slow_ema", 50)))
        except (TypeError, ValueError):
            self.slow_ema = 50

        raw_symbols = cfg.get("symbols", ["BTCUSD", "ETHUSD"])
        parsed = []
        if isinstance(raw_symbols, Iterable) and not isinstance(raw_symbols, (str, bytes)):
            for item in raw_symbols:
                text = str(item or "").strip().upper()
                if text and text not in parsed:
                    parsed.append(text)
        self.symbols = set(parsed) if parsed else {"BTCUSD", "ETHUSD"}

    def is_symbol_enabled(self, symbol: str) -> bool:
        return self.enabled and str(symbol or "").strip().upper() in self.symbols

    def allow_entry(self, symbol: str, action: DecisionAction, bars: pd.DataFrame) -> bool:
        if action not in {DecisionAction.BUY, DecisionAction.SELL}:
            return True
        if not self.is_symbol_enabled(symbol):
            return True
        clean = sanitize_ohlc(bars)
        if clean is None or len(clean) < (self.slow_ema + 2):
            return False
        closed = clean.iloc[:-1].copy()
        if len(closed) < (self.slow_ema + 1):
            return False

        ema_fast = compute_ema(closed, period=self.fast_ema)
        ema_slow = compute_ema(closed, period=self.slow_ema)
        if ema_fast is None or ema_slow is None:
            return False

        if action == DecisionAction.BUY:
            return bool(float(ema_fast) > float(ema_slow))
        return bool(float(ema_fast) < float(ema_slow))
