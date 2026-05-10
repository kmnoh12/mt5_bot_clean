from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from core.models import DecisionAction, Position, StrategyDecision
from execution.trailing_guard import TrailingGuardSignal


@dataclass(frozen=True)
class ExitEngineDecision:
    decision: StrategyDecision
    source: str
    priority: int


class ExitEngine:
    """Single place that chooses one exit decision per symbol/position."""

    def __init__(self) -> None:
        self.trailing_priority = 100
        self.strategy_priority = 50

    def _from_trailing_signal(self, signal: TrailingGuardSignal) -> ExitEngineDecision:
        decision = StrategyDecision(
            action=DecisionAction.EXIT,
            reason=signal.reason,
            strategy="profit_lock_guard",
            metadata={
                "peak_pnl_usd": float(signal.peak_pnl_usd),
                "current_pnl_usd": float(signal.current_pnl_usd),
                "trigger_pnl_usd": float(signal.trigger_pnl_usd),
                "drawdown_usd": float(signal.drawdown_usd),
                "threshold_usd": float(signal.threshold_usd),
                "exit_source": "profit_lock_guard",
            },
        )
        return ExitEngineDecision(decision=decision, source="profit_lock_guard", priority=self.trailing_priority)

    def choose(
        self,
        *,
        position: Optional[Position],
        strategy_decision: StrategyDecision,
        trailing_signal: Optional[TrailingGuardSignal],
    ) -> StrategyDecision:
        if position is None:
            return strategy_decision

        candidates: list[ExitEngineDecision] = []
        if trailing_signal is not None:
            candidates.append(self._from_trailing_signal(trailing_signal))
        if strategy_decision.action == DecisionAction.EXIT:
            candidates.append(
                ExitEngineDecision(
                    decision=strategy_decision,
                    source="strategy",
                    priority=self.strategy_priority,
                )
            )

        if not candidates:
            return strategy_decision

        chosen = max(candidates, key=lambda item: int(item.priority))
        if chosen.source == "strategy":
            return strategy_decision
        return chosen.decision
