from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from core.models import Position


class ExitQualityGuard:
    def __init__(self, config: Dict[str, Any]) -> None:
        self.enabled = True
        self.tiny_profit_block_usd = 2.0
        self.min_hold_seconds_for_soft_exit = 300.0
        self.m5_reverse_confirm_bars = 2
        self.update_config(config)

    @staticmethod
    def _to_float(value: Any, default: float = 0.0) -> float:
        try:
            out = float(value)
        except (TypeError, ValueError):
            return float(default)
        if not math.isfinite(out):
            return float(default)
        return float(out)

    def update_config(self, config: Dict[str, Any]) -> None:
        cfg = dict(config or {})
        self.enabled = bool(cfg.get("enabled", True))
        self.tiny_profit_block_usd = max(0.0, self._to_float(cfg.get("tiny_profit_block_usd", 2.0), 2.0))
        self.min_hold_seconds_for_soft_exit = max(
            1.0, self._to_float(cfg.get("min_hold_seconds_for_soft_exit", 300.0), 300.0)
        )
        self.m5_reverse_confirm_bars = max(1, int(self._to_float(cfg.get("m5_reverse_confirm_bars", 2), 2)))

    @staticmethod
    def _is_soft_exit(reason: str) -> bool:
        text = str(reason or "").upper()
        return ("REGIME_FLIP" in text) or ("TRAIL_BREACH" in text) or ("PROFIT_LOCK" in text)

    def should_block_exit(
        self,
        *,
        position: Position,
        reason: str,
        m5_reverse_confirmed: bool = False,
    ) -> Dict[str, Any]:
        if not self.enabled:
            return {"allow": True, "reason": "EXIT_QUALITY_DISABLED"}
        if not self._is_soft_exit(reason):
            return {"allow": True, "reason": "NOT_SOFT_EXIT"}

        metadata = position.metadata if isinstance(position.metadata, dict) else {}
        floating = self._to_float(metadata.get("floating_pnl", metadata.get("profit", 0.0)), 0.0)
        swap = self._to_float(metadata.get("swap"), 0.0)
        commission = self._to_float(metadata.get("commission"), 0.0)
        net_pnl = floating + swap + commission

        hold_seconds = float("inf")
        if position.time_open_utc is not None:
            opened = position.time_open_utc.astimezone(timezone.utc)
            hold_seconds = (datetime.now(timezone.utc) - opened).total_seconds()

        if m5_reverse_confirmed:
            return {
                "allow": True,
                "reason": "M5_REVERSE_CONFIRMED",
                "net_pnl_usd": float(net_pnl),
                "hold_seconds": float(hold_seconds),
            }

        if net_pnl <= self.tiny_profit_block_usd and hold_seconds < self.min_hold_seconds_for_soft_exit:
            return {
                "allow": False,
                "reason": "SOFT_EXIT_BLOCKED",
                "net_pnl_usd": float(net_pnl),
                "hold_seconds": float(hold_seconds),
                "tiny_profit_block_usd": float(self.tiny_profit_block_usd),
                "min_hold_seconds_for_soft_exit": float(self.min_hold_seconds_for_soft_exit),
            }
        return {
            "allow": True,
            "reason": "SOFT_EXIT_ALLOWED",
            "net_pnl_usd": float(net_pnl),
            "hold_seconds": float(hold_seconds),
        }
