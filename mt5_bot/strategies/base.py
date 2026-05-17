from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import pandas as pd

from core.models import (
    DecisionAction,
    ExternalSignal,
    OrderResult,
    Position,
    Side,
    StrategyEvaluationContext,
    StrategyDecision,
    StrategyState,
    StrategySymbolState,
)
from utils.indicators import parse_bar_time


LOGGER = logging.getLogger(__name__)


class BaseStateMachineStrategy(ABC):
    def __init__(self, name: str, config: Dict[str, Any], snapshot: Optional[Dict[str, Any]] = None) -> None:
        self.name = name
        self.config = config or {}
        self.min_cooldown_bars = max(1, int(self.config.get("min_cooldown_bars", 2)))
        self.default_min_hold_bars = max(1, int(self.config.get("min_hold_bars", 1)))
        self._states: Dict[str, StrategySymbolState] = {}
        self._restore_snapshot(snapshot or {})

    def _restore_snapshot(self, snapshot: Dict[str, Any]) -> None:
        for symbol, payload in (snapshot or {}).items():
            if not isinstance(payload, dict):
                continue
            state_text = str(payload.get("state", StrategyState.IDLE.value)).upper()
            state = StrategyState.__members__.get(state_text, StrategyState.IDLE)
            bias = None
            bias_text = str(payload.get("bias", "")).upper()
            if bias_text in Side.__members__:
                bias = Side[bias_text]

            updated_at = self._parse_dt(payload.get("updated_at_utc")) or datetime.now(timezone.utc)

            self._states[symbol] = StrategySymbolState(
                state=state,
                bias=bias,
                cooldown_bars_remaining=max(0, int(payload.get("cooldown_bars_remaining", 0))),
                last_reason=str(payload.get("last_reason", "") or ""),
                entry_price=float(payload["entry_price"]) if payload.get("entry_price") is not None else None,
                peak_price=float(payload["peak_price"]) if payload.get("peak_price") is not None else None,
                trough_price=float(payload["trough_price"]) if payload.get("trough_price") is not None else None,
                last_closed_bar_time=self._parse_dt(payload.get("last_closed_bar_time")),
                entry_bar_time=self._parse_dt(payload.get("entry_bar_time")),
                pending_order=bool(payload.get("pending_order", False)),
                updated_at_utc=updated_at,
                metadata=dict(payload.get("metadata", {})) if isinstance(payload.get("metadata"), dict) else {},
            )

    @staticmethod
    def _parse_dt(raw: Any) -> Optional[datetime]:
        if raw is None:
            return None
        if isinstance(raw, datetime):
            return raw.astimezone(timezone.utc) if raw.tzinfo else raw.replace(tzinfo=timezone.utc)
        text = str(raw)
        if not text:
            return None
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(text)
        except Exception:
            return None
        return parsed.astimezone(timezone.utc) if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)

    @staticmethod
    def _finite_float(value: Any) -> Optional[float]:
        try:
            out = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(out):
            return None
        return out

    @classmethod
    def _is_partial_exit_decision(cls, decision: StrategyDecision) -> bool:
        metadata = decision.metadata if isinstance(decision.metadata, dict) else {}
        if bool(metadata.get("is_partial", False)):
            return True

        requested_volume = cls._finite_float(decision.volume)
        if requested_volume is None or requested_volume <= 0:
            return False

        total_volume = cls._finite_float(metadata.get("position_volume_before"))
        if total_volume is None:
            total_volume = cls._finite_float(metadata.get("position_volume"))
        if total_volume is None:
            total_volume = cls._finite_float(metadata.get("current_position_volume"))
        if total_volume is None or total_volume <= 0:
            return False

        tolerance = max(1e-9, abs(total_volume) * 1e-9)
        return requested_volume < (total_volume - tolerance)

    def snapshot(self) -> Dict[str, Any]:
        return {symbol: state.to_dict() for symbol, state in self._states.items()}

    def get_symbol_state(self, symbol: str) -> StrategySymbolState:
        return self._symbol_state(symbol)

    def get_all_symbol_states(self) -> Dict[str, Any]:
        return self.snapshot()

    def _symbol_state(self, symbol: str) -> StrategySymbolState:
        if symbol not in self._states:
            self._states[symbol] = StrategySymbolState()
        return self._states[symbol]

    def mark_closed_bar(self, symbol: str, closed_time: Optional[datetime]) -> None:
        if closed_time is None:
            return
        st = self._symbol_state(symbol)
        st.last_closed_bar_time = closed_time.astimezone(timezone.utc)

    def _transition(self, st: StrategySymbolState, to_state: StrategyState, reason: str) -> None:
        if st.state != to_state:
            LOGGER.info("%s state transition %s -> %s (%s)", self.name, st.state.value, to_state.value, reason)
        st.state = to_state
        st.last_reason = reason
        st.updated_at_utc = datetime.now(timezone.utc)

    def _hold(
        self,
        reason: str,
        metadata: Optional[Dict[str, Any]] = None,
        sl: Optional[float] = None,
        tp: Optional[float] = None,
    ) -> StrategyDecision:
        return StrategyDecision(
            action=DecisionAction.HOLD,
            reason=reason,
            strategy=self.name,
            confidence=0.0,
            sl=sl,
            tp=tp,
            metadata=metadata or {},
        )

    def _emit_entry(
        self,
        side: Side,
        reason: str,
        confidence: float = 1.0,
        volume: Optional[float] = None,
        sl: Optional[float] = None,
        tp: Optional[float] = None,
        signal_bar_time: Optional[datetime] = None,
        min_hold_bars: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> StrategyDecision:
        action = DecisionAction.BUY if side == Side.BUY else DecisionAction.SELL
        return StrategyDecision(
            action=action,
            reason=reason,
            strategy=self.name,
            confidence=confidence,
            volume=volume,
            sl=sl,
            tp=tp,
            signal_bar_time=signal_bar_time,
            min_hold_bars=min_hold_bars,
            metadata=metadata or {},
        )

    def _emit_exit(self, reason: str, confidence: float = 1.0, metadata: Optional[Dict[str, Any]] = None) -> StrategyDecision:
        return StrategyDecision(
            action=DecisionAction.EXIT,
            reason=reason,
            strategy=self.name,
            confidence=confidence,
            metadata=metadata or {},
        )

    def _apply_external_signal(
        self,
        st: StrategySymbolState,
        signal: ExternalSignal,
        has_position: bool,
    ) -> Optional[StrategyDecision]:
        if signal.action == DecisionAction.HOLD:
            return None
        if signal.action == DecisionAction.EXIT:
            if has_position:
                self._transition(st, StrategyState.COOLDOWN, f"EXTERNAL_EXIT:{signal.signal_id}")
                st.cooldown_bars_remaining = self.min_cooldown_bars
                return StrategyDecision(
                    action=DecisionAction.EXIT,
                    reason=f"EXTERNAL_SIGNAL_EXIT:{signal.reason or signal.signal_id}",
                    strategy=self.name,
                    confidence=signal.confidence,
                    metadata={"external_signal_id": signal.signal_id},
                )
            return self._hold("EXTERNAL_EXIT_IGNORED_NO_POSITION")

        if signal.action in {DecisionAction.BUY, DecisionAction.SELL}:
            side = Side.BUY if signal.action == DecisionAction.BUY else Side.SELL
            st.bias = side
            st.pending_order = True
            self._transition(st, StrategyState.ENTRY_PENDING, f"EXTERNAL_ENTRY:{signal.signal_id}")
            return StrategyDecision(
                action=signal.action,
                reason=f"EXTERNAL_SIGNAL_ENTRY:{signal.reason or signal.signal_id}",
                strategy=self.name,
                confidence=signal.confidence,
                volume=signal.volume,
                signal_bar_time=st.last_closed_bar_time,
                min_hold_bars=self.default_min_hold_bars,
                metadata={"external_signal_id": signal.signal_id},
            )
        return None

    def apply_order_result(
        self,
        symbol: str,
        decision: StrategyDecision,
        result: Optional[OrderResult],
    ) -> None:
        st = self._symbol_state(symbol)

        if decision.action in {DecisionAction.BUY, DecisionAction.SELL}:
            st.pending_order = False
            if result is not None and result.ok:
                st.bias = Side.BUY if decision.action == DecisionAction.BUY else Side.SELL
                if result.filled_price is not None and math.isfinite(float(result.filled_price)):
                    st.entry_price = float(result.filled_price)
                if decision.signal_bar_time is not None:
                    st.entry_bar_time = decision.signal_bar_time.astimezone(timezone.utc)
                elif st.last_closed_bar_time is not None:
                    st.entry_bar_time = st.last_closed_bar_time
                if decision.min_hold_bars is not None:
                    st.metadata["min_hold_bars"] = max(1, int(decision.min_hold_bars))
                st.metadata["bars_in_trade"] = 0
                st.metadata["last_manage_bar_time"] = ""
                self._transition(st, StrategyState.IN_POSITION, "ENTRY_FILLED")
                return

            code = int((result.retcode or 0) if result is not None else 0)
            if code == 10027:
                self._transition(st, StrategyState.HALTED, "ENTRY_REJECTED_10027")
                return

            status = str(getattr(result, "status", "") or "").strip().upper() if result is not None else ""
            no_trade_statuses = {
                "ENTRY_QUALITY_BLOCK",
                "EDGE_TOO_LOW",
                "EXPECTED_LOSS_CAP",
                "EXPECTED_LOSS_CAP_AFTER_STOP_ADJUSTMENT",
                "RISK_PLAN_FAILED",
                "CHECK_REJECTED",
            }
            if status in no_trade_statuses:
                self.mark_entry_rejected_no_trade(symbol=symbol, reason=status or "ENTRY_REJECTED_NO_TRADE")
                return

            st.cooldown_bars_remaining = self.min_cooldown_bars
            self._transition(st, StrategyState.COOLDOWN, "ENTRY_FAILED")
            return

        if decision.action == DecisionAction.EXIT:
            if result is not None and result.ok:
                st.pending_order = False
                if self._is_partial_exit_decision(decision):
                    self._transition(st, StrategyState.IN_POSITION, "PARTIAL_EXIT_FILLED")
                    return
                st.peak_price = None
                st.trough_price = None
                st.entry_price = None
                st.entry_bar_time = None
                st.cooldown_bars_remaining = self.min_cooldown_bars
                st.metadata["bars_in_trade"] = 0
                st.metadata["last_manage_bar_time"] = ""
                self._transition(st, StrategyState.COOLDOWN, "EXIT_FILLED")
            return

    def mark_entry_rejected_no_trade(self, symbol: str, reason: str) -> None:
        st = self._symbol_state(symbol)
        st.pending_order = False
        st.cooldown_bars_remaining = max(0, int(st.cooldown_bars_remaining or 0))
        if st.state == StrategyState.ENTRY_PENDING:
            self._transition(st, StrategyState.IDLE, str(reason or "ENTRY_REJECTED_NO_TRADE"))
            return
        st.last_reason = str(reason or "ENTRY_REJECTED_NO_TRADE")
        st.updated_at_utc = datetime.now(timezone.utc)

    def evaluate(
        self,
        symbol: str,
        bars: pd.DataFrame,
        position: Optional[Position],
        external_signal: Optional[ExternalSignal] = None,
        context: Optional[StrategyEvaluationContext] = None,
    ) -> StrategyDecision:
        context_payload: Dict[str, Any] = {}
        if context is not None:
            if isinstance(context, StrategyEvaluationContext):
                context_payload = context.to_dict()
            elif isinstance(context, dict):
                context_payload = dict(context)

        risk_context: Dict[str, Any] = {}
        for key in ("equity", "equity_peak", "daily_pnl"):
            value = context_payload.get(key)
            if value is not None:
                risk_context[key] = value
        if "loss_streak" in context_payload:
            risk_context["loss_streak"] = max(0, int(context_payload.get("loss_streak", 0)))

        if context_payload and bars is not None:
            try:
                bars.attrs["context"] = dict(context_payload)
                mtf_info = context_payload.get("mtf_info")
                if isinstance(mtf_info, dict):
                    bars.attrs["mtf_info"] = dict(mtf_info)
                if risk_context:
                    bars.attrs["risk_context"] = dict(risk_context)
            except Exception:
                pass

        st = self._symbol_state(symbol)
        has_position = position is not None

        bar_time = None
        if bars is not None and not bars.empty and "time" in bars.columns and len(bars) >= 2:
            bar_time = parse_bar_time(bars.iloc[-2].get("time"))
        if bar_time is not None:
            st.last_closed_bar_time = bar_time

        if st.state == StrategyState.IN_POSITION and not has_position:
            self._transition(st, StrategyState.COOLDOWN, "POSITION_MISSING")
            st.cooldown_bars_remaining = self.min_cooldown_bars
        elif st.state in {StrategyState.IDLE, StrategyState.SETUP, StrategyState.ENTRY_READY, StrategyState.ENTRY_PENDING} and has_position:
            st.bias = position.side
            st.pending_order = False
            if st.entry_bar_time is None:
                st.entry_bar_time = st.last_closed_bar_time
            self._transition(st, StrategyState.IN_POSITION, "POSITION_DETECTED")

        if external_signal is not None:
            decision = self._apply_external_signal(st=st, signal=external_signal, has_position=has_position)
            if decision is not None:
                if not isinstance(decision.metadata, dict):
                    decision.metadata = {}
                mtf_info = context_payload.get("mtf_info")
                if isinstance(mtf_info, dict) and mtf_info:
                    decision.metadata.setdefault("mtf_info", dict(mtf_info))
                if risk_context:
                    decision.metadata.setdefault("risk_context", dict(risk_context))
                decision.state = st.state
                return decision

        try:
            decision = self._evaluate_impl(symbol=symbol, bars=bars, position=position, st=st)
        except Exception as exc:
            LOGGER.exception("Strategy %s failed for %s", self.name, symbol)
            self._transition(st, StrategyState.HALTED, f"EXCEPTION:{exc}")
            decision = self._hold("STRATEGY_EXCEPTION", {"error": str(exc)})

        if not isinstance(decision.metadata, dict):
            decision.metadata = {}
        mtf_info = context_payload.get("mtf_info")
        if isinstance(mtf_info, dict) and mtf_info:
            decision.metadata.setdefault("mtf_info", dict(mtf_info))
        if risk_context:
            decision.metadata.setdefault("risk_context", dict(risk_context))
        decision.state = st.state
        return decision

    @abstractmethod
    def _evaluate_impl(
        self,
        symbol: str,
        bars: pd.DataFrame,
        position: Optional[Position],
        st: StrategySymbolState,
    ) -> StrategyDecision:
        raise NotImplementedError
