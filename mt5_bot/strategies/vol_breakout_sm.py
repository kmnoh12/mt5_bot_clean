from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd

from core.models import Position, Side, StrategyDecision, StrategyState, StrategySymbolState
from strategies.base import BaseStateMachineStrategy
from utils.indicators import compute_atr, last_close, parse_bar_time, rolling_high_low, sanitize_ohlc


class VolBreakoutStateMachine(BaseStateMachineStrategy):
    def __init__(self, config: Dict[str, Any], snapshot: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(name="vol_breakout_sm", config=config, snapshot=snapshot)
        cfg = config or {}
        self.lookback = max(10, int(cfg.get("lookback", 20)))
        self.atr_period = max(5, int(cfg.get("atr_period", 14)))
        self.breakout_atr_multiple = max(0.1, float(cfg.get("breakout_atr_multiple", 1.0)))
        self.trailing_exit_atr_multiple = max(0.2, float(cfg.get("trailing_exit_atr_multiple", 1.2)))
        self.sl_atr_mult = max(0.2, float(cfg.get("sl_atr_mult", 1.0)))
        self.tp_atr_mult = max(0.2, float(cfg.get("tp_atr_mult", 2.0)))
        self.min_hold_bars = max(1, int(cfg.get("min_hold_bars", 1)))

    def _evaluate_impl(
        self,
        symbol: str,
        bars: pd.DataFrame,
        position: Optional[Position],
        st: StrategySymbolState,
    ) -> StrategyDecision:
        clean = sanitize_ohlc(bars)
        if clean is None or len(clean) < self.lookback + 5:
            return self._hold("INSUFFICIENT_BARS")

        closed = clean.iloc[:-1].copy() if len(clean) > 1 else clean.copy()
        if len(closed) < self.lookback + 2:
            return self._hold("INSUFFICIENT_CLOSED_BARS")

        signal_close = last_close(closed)
        atr = compute_atr(closed, period=self.atr_period)
        reference = closed.iloc[-(self.lookback + 1) : -1]
        ref_high, ref_low = rolling_high_low(reference, lookback=self.lookback)
        if None in (signal_close, atr, ref_high, ref_low):
            return self._hold("INDICATORS_UNAVAILABLE")

        upper_trigger = ref_high + (atr * self.breakout_atr_multiple)
        lower_trigger = ref_low - (atr * self.breakout_atr_multiple)
        signal_bar_time = parse_bar_time(closed.iloc[-1].get("time")) if "time" in closed.columns else None

        if st.state == StrategyState.HALTED:
            return self._hold("HALTED_MANUAL_RECOVERY_REQUIRED")

        if st.state == StrategyState.COOLDOWN:
            if st.cooldown_bars_remaining > 0:
                st.cooldown_bars_remaining -= 1
                return self._hold("COOLDOWN_ACTIVE", {"remaining": st.cooldown_bars_remaining})
            self._transition(st, StrategyState.IDLE, "COOLDOWN_COMPLETE")

        if st.state == StrategyState.IDLE:
            if signal_close > upper_trigger:
                st.bias = Side.BUY
                self._transition(st, StrategyState.SETUP, "UP_BREAKOUT_SETUP")
                return self._hold("BUY_BREAKOUT_SETUP")
            if signal_close < lower_trigger:
                st.bias = Side.SELL
                self._transition(st, StrategyState.SETUP, "DOWN_BREAKOUT_SETUP")
                return self._hold("SELL_BREAKOUT_SETUP")
            return self._hold("NO_BREAKOUT_SETUP")

        if st.state == StrategyState.SETUP:
            if st.bias == Side.BUY:
                if signal_close > upper_trigger:
                    self._transition(st, StrategyState.ENTRY_READY, "BUY_BREAKOUT_CONFIRM")
                elif signal_close < ref_high:
                    self._transition(st, StrategyState.IDLE, "BUY_SETUP_INVALIDATED")
                return self._hold("BUY_SETUP_MONITOR")
            if st.bias == Side.SELL:
                if signal_close < lower_trigger:
                    self._transition(st, StrategyState.ENTRY_READY, "SELL_BREAKOUT_CONFIRM")
                elif signal_close > ref_low:
                    self._transition(st, StrategyState.IDLE, "SELL_SETUP_INVALIDATED")
                return self._hold("SELL_SETUP_MONITOR")
            self._transition(st, StrategyState.IDLE, "SETUP_WITHOUT_BIAS")
            return self._hold("SETUP_RESET")

        if st.state == StrategyState.ENTRY_READY:
            if st.bias is None:
                self._transition(st, StrategyState.IDLE, "ENTRY_READY_NO_BIAS")
                return self._hold("ENTRY_READY_RESET")
            st.pending_order = True
            st.entry_price = signal_close
            st.entry_bar_time = signal_bar_time
            st.peak_price = signal_close
            st.trough_price = signal_close
            st.metadata["bars_in_trade"] = 0
            self._transition(st, StrategyState.ENTRY_PENDING, "ENTRY_EMITTED")

            if st.bias == Side.BUY:
                sl = signal_close - (atr * self.sl_atr_mult)
                tp = signal_close + (atr * self.tp_atr_mult)
            else:
                sl = signal_close + (atr * self.sl_atr_mult)
                tp = signal_close - (atr * self.tp_atr_mult)

            return self._emit_entry(
                side=st.bias,
                reason=f"VOL_BREAKOUT_ENTRY_{st.bias.value}",
                confidence=0.7,
                sl=sl,
                tp=tp,
                signal_bar_time=signal_bar_time,
                min_hold_bars=self.min_hold_bars,
                metadata={
                    "signal_close": signal_close,
                    "atr": atr,
                    "risk_per_unit": abs(signal_close - sl),
                },
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
                st.peak_price = max(st.peak_price or signal_close, signal_close)
                trailing_stop = (st.peak_price or signal_close) - (atr * self.trailing_exit_atr_multiple)
                if bars_in_trade >= self.min_hold_bars and signal_close <= trailing_stop:
                    self._transition(st, StrategyState.EXIT_READY, "BUY_TRAILING_EXIT")
                else:
                    return self._hold(
                        "BUY_POSITION_MONITOR",
                        {"peak": st.peak_price, "trailing_stop": trailing_stop, "bars_in_trade": bars_in_trade},
                        sl=max(float(position.sl) if position.sl is not None else trailing_stop, trailing_stop),
                        tp=position.tp,
                    )
            else:
                st.trough_price = min(st.trough_price or signal_close, signal_close)
                trailing_stop = (st.trough_price or signal_close) + (atr * self.trailing_exit_atr_multiple)
                if bars_in_trade >= self.min_hold_bars and signal_close >= trailing_stop:
                    self._transition(st, StrategyState.EXIT_READY, "SELL_TRAILING_EXIT")
                else:
                    return self._hold(
                        "SELL_POSITION_MONITOR",
                        {"trough": st.trough_price, "trailing_stop": trailing_stop, "bars_in_trade": bars_in_trade},
                        sl=min(float(position.sl) if position.sl is not None else trailing_stop, trailing_stop),
                        tp=position.tp,
                    )

        if st.state == StrategyState.EXIT_READY:
            st.pending_order = False
            return self._emit_exit(reason="VOL_BREAKOUT_EXIT", confidence=0.75)

        return self._hold("NO_ACTION")
