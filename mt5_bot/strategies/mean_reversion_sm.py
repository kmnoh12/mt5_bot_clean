from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd

from core.models import Position, Side, StrategyDecision, StrategyState, StrategySymbolState
from strategies.base import BaseStateMachineStrategy
from utils.indicators import compute_atr, compute_bollinger_bands, compute_rsi, last_close, parse_bar_time, sanitize_ohlc


class MeanReversionStateMachine(BaseStateMachineStrategy):
    def __init__(self, config: Dict[str, Any], snapshot: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(name="mean_reversion_sm", config=config, snapshot=snapshot)
        cfg = config or {}
        self.rsi_period = max(2, int(cfg.get("rsi_period", 14)))
        self.rsi_oversold = float(cfg.get("rsi_oversold", 30))
        self.rsi_overbought = float(cfg.get("rsi_overbought", 70))
        self.bollinger_period = max(5, int(cfg.get("bollinger_period", 20)))
        self.bollinger_stddev = max(0.1, float(cfg.get("bollinger_stddev", 2.0)))
        self.sl_atr_mult = max(0.2, float(cfg.get("sl_atr_mult", 1.0)))
        self.tp_atr_mult = max(0.2, float(cfg.get("tp_atr_mult", 1.5)))
        self.min_hold_bars = max(1, int(cfg.get("min_hold_bars", 1)))

    def _evaluate_impl(
        self,
        symbol: str,
        bars: pd.DataFrame,
        position: Optional[Position],
        st: StrategySymbolState,
    ) -> StrategyDecision:
        clean = sanitize_ohlc(bars)
        if clean is None or len(clean) < self.bollinger_period + 2:
            return self._hold("INSUFFICIENT_BARS")

        closed = clean.iloc[:-1].copy() if len(clean) > 1 else clean.copy()
        if closed.empty:
            return self._hold("NO_CLOSED_BAR")

        close_price = last_close(closed)
        rsi = compute_rsi(closed, period=self.rsi_period)
        middle, upper, lower = compute_bollinger_bands(
            closed,
            period=self.bollinger_period,
            stddev=self.bollinger_stddev,
        )
        atr = compute_atr(closed, period=14)
        if None in (close_price, rsi, middle, upper, lower, atr):
            return self._hold("INDICATORS_UNAVAILABLE")

        signal_bar_time = parse_bar_time(closed.iloc[-1].get("time")) if "time" in closed.columns else None

        if st.state == StrategyState.HALTED:
            return self._hold("HALTED_MANUAL_RECOVERY_REQUIRED")

        if st.state == StrategyState.COOLDOWN:
            if st.cooldown_bars_remaining > 0:
                st.cooldown_bars_remaining -= 1
                return self._hold("COOLDOWN_ACTIVE", {"remaining": st.cooldown_bars_remaining})
            self._transition(st, StrategyState.IDLE, "COOLDOWN_COMPLETE")

        if st.state == StrategyState.IDLE:
            if close_price <= lower and rsi <= self.rsi_oversold:
                st.bias = Side.BUY
                self._transition(st, StrategyState.SETUP, "OVERSOLD_SETUP")
                return self._hold("BUY_SETUP_CREATED", {"rsi": rsi, "close": close_price, "atr": atr})
            if close_price >= upper and rsi >= self.rsi_overbought:
                st.bias = Side.SELL
                self._transition(st, StrategyState.SETUP, "OVERBOUGHT_SETUP")
                return self._hold("SELL_SETUP_CREATED", {"rsi": rsi, "close": close_price, "atr": atr})
            return self._hold("NO_SETUP")

        if st.state == StrategyState.SETUP:
            if st.bias == Side.BUY:
                if close_price > lower and rsi >= (self.rsi_oversold + 2):
                    self._transition(st, StrategyState.ENTRY_READY, "BUY_CONFIRM")
                elif close_price > middle:
                    self._transition(st, StrategyState.IDLE, "BUY_SETUP_INVALIDATED")
                return self._hold("BUY_SETUP_MONITOR", {"rsi": rsi, "close": close_price, "atr": atr})
            if st.bias == Side.SELL:
                if close_price < upper and rsi <= (self.rsi_overbought - 2):
                    self._transition(st, StrategyState.ENTRY_READY, "SELL_CONFIRM")
                elif close_price < middle:
                    self._transition(st, StrategyState.IDLE, "SELL_SETUP_INVALIDATED")
                return self._hold("SELL_SETUP_MONITOR", {"rsi": rsi, "close": close_price, "atr": atr})
            self._transition(st, StrategyState.IDLE, "SETUP_WITHOUT_BIAS")
            return self._hold("SETUP_RESET")

        if st.state == StrategyState.ENTRY_READY:
            if st.bias is None:
                self._transition(st, StrategyState.IDLE, "ENTRY_READY_NO_BIAS")
                return self._hold("ENTRY_READY_RESET")

            st.pending_order = True
            st.entry_price = close_price
            st.entry_bar_time = signal_bar_time
            st.metadata["bars_in_trade"] = 0
            self._transition(st, StrategyState.ENTRY_PENDING, "ENTRY_EMITTED")

            if st.bias == Side.BUY:
                sl = close_price - (atr * self.sl_atr_mult)
                tp = close_price + (atr * self.tp_atr_mult)
            else:
                sl = close_price + (atr * self.sl_atr_mult)
                tp = close_price - (atr * self.tp_atr_mult)

            return self._emit_entry(
                side=st.bias,
                reason=f"MEAN_REVERSION_ENTRY_{st.bias.value}",
                confidence=0.65,
                sl=sl,
                tp=tp,
                signal_bar_time=signal_bar_time,
                min_hold_bars=self.min_hold_bars,
                metadata={"signal_close": close_price, "atr": atr, "risk_per_unit": abs(close_price - sl)},
            )

        if st.state == StrategyState.ENTRY_PENDING:
            if position is None:
                return self._hold("ENTRY_PENDING_WAIT_FILL")
            self._transition(st, StrategyState.IN_POSITION, "POSITION_DETECTED_WHILE_PENDING")

        if st.state == StrategyState.IN_POSITION:
            if position is None:
                self._transition(st, StrategyState.COOLDOWN, "POSITION_NOT_FOUND")
                st.cooldown_bars_remaining = self.min_cooldown_bars
                return self._hold("WAITING_RECONCILIATION")

            bars_in_trade = int(st.metadata.get("bars_in_trade", 0)) + 1
            st.metadata["bars_in_trade"] = bars_in_trade

            if position.side == Side.BUY:
                if bars_in_trade >= self.min_hold_bars and (close_price >= middle or rsi >= 55):
                    self._transition(st, StrategyState.EXIT_READY, "BUY_MEAN_REVERSION_TARGET")
                else:
                    return self._hold("BUY_POSITION_MONITOR", {"rsi": rsi, "close": close_price, "atr": atr, "bars_in_trade": bars_in_trade})
            else:
                if bars_in_trade >= self.min_hold_bars and (close_price <= middle or rsi <= 45):
                    self._transition(st, StrategyState.EXIT_READY, "SELL_MEAN_REVERSION_TARGET")
                else:
                    return self._hold("SELL_POSITION_MONITOR", {"rsi": rsi, "close": close_price, "atr": atr, "bars_in_trade": bars_in_trade})

        if st.state == StrategyState.EXIT_READY:
            st.pending_order = False
            return self._emit_exit(reason="MEAN_REVERSION_EXIT", confidence=0.7)

        return self._hold("NO_ACTION")
