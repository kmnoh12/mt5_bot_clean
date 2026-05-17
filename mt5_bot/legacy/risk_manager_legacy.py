from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, Optional


LOGGER = logging.getLogger(__name__)


@dataclass
class BucketRiskConfig:
    name: str
    risk_per_trade_pct: float
    min_lot: float
    max_lot: float


class RiskManager:
    def __init__(self, execution_config: Dict[str, Any]) -> None:
        cfg = execution_config or {}
        self.stop_loss_atr_multiple = max(0.1, float(cfg.get("stop_loss_atr_multiple", 1.5)))
        self.take_profit_atr_multiple = max(0.1, float(cfg.get("take_profit_atr_multiple", 2.0)))
        self.fallback_stop_loss_pct = max(0.05, float(cfg.get("fallback_stop_loss_pct", 1.0)))

    @staticmethod
    def _safe_float(value: Any, default: float) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return default
        if not math.isfinite(number):
            return default
        return number

    @staticmethod
    def _quantize_volume(raw_volume: float, min_volume: float, max_volume: float, step: float) -> float:
        min_volume = max(0.0, min_volume)
        max_volume = max(min_volume, max_volume)
        step = step if step > 0 else 0.01
        volume = min(max(raw_volume, min_volume), max_volume)

        units = round((volume - min_volume) / step)
        quantized = min_volume + (units * step)
        quantized = min(max(quantized, min_volume), max_volume)

        step_text = f"{step:.10f}".rstrip("0")
        precision = len(step_text.split(".")[1]) if "." in step_text else 0
        return round(quantized, precision)

    @staticmethod
    def _round_price(price: float, digits: int) -> float:
        digits = max(0, int(digits))
        return round(price, digits)

    def build_order_plan(
        self,
        symbol_info: Any,
        side: str,
        price: float,
        atr: Optional[float],
        equity: float,
        bucket_risk: BucketRiskConfig,
    ) -> Optional[Dict[str, float]]:
        side = str(side).upper()
        if side not in {"BUY", "SELL"}:
            LOGGER.warning("Invalid order side for risk plan: %s", side)
            return None

        price = self._safe_float(price, 0.0)
        equity = self._safe_float(equity, 0.0)
        if price <= 0 or equity <= 0:
            LOGGER.warning("Risk plan skipped due to non-positive price/equity. price=%s equity=%s", price, equity)
            return None

        digits = int(getattr(symbol_info, "digits", 5))
        point = self._safe_float(getattr(symbol_info, "point", 10 ** (-digits)), 10 ** (-digits))
        contract_size = self._safe_float(getattr(symbol_info, "trade_contract_size", 1.0), 1.0)
        symbol_min = self._safe_float(getattr(symbol_info, "volume_min", 0.01), 0.01)
        symbol_max = self._safe_float(getattr(symbol_info, "volume_max", bucket_risk.max_lot), bucket_risk.max_lot)
        volume_step = self._safe_float(getattr(symbol_info, "volume_step", 0.01), 0.01)

        min_volume = max(symbol_min, float(bucket_risk.min_lot))
        max_volume = min(symbol_max, float(bucket_risk.max_lot))
        if max_volume < min_volume:
            max_volume = min_volume

        risk_pct = max(0.01, float(bucket_risk.risk_per_trade_pct))
        risk_amount = equity * (risk_pct / 100.0)
        if risk_amount <= 0:
            LOGGER.warning("Risk amount is non-positive for bucket '%s'.", bucket_risk.name)
            return None

        atr_value = self._safe_float(atr, 0.0)
        if atr_value > 0:
            stop_distance = atr_value * self.stop_loss_atr_multiple
        else:
            stop_distance = price * (self.fallback_stop_loss_pct / 100.0)
        stop_distance = max(stop_distance, point * 10)
        if stop_distance <= 0:
            LOGGER.warning("Stop distance is non-positive for '%s'.", bucket_risk.name)
            return None

        denom = stop_distance * max(contract_size, 1e-9)
        raw_volume = risk_amount / denom if denom > 0 else min_volume
        if raw_volume <= 0 or not math.isfinite(raw_volume):
            raw_volume = min_volume

        volume = self._quantize_volume(raw_volume, min_volume=min_volume, max_volume=max_volume, step=volume_step)

        tp_ratio = self.take_profit_atr_multiple / max(self.stop_loss_atr_multiple, 1e-9)
        tp_distance = max(stop_distance * tp_ratio, point * 10)

        if side == "BUY":
            sl = self._round_price(price - stop_distance, digits)
            tp = self._round_price(price + tp_distance, digits)
        else:
            sl = self._round_price(price + stop_distance, digits)
            tp = self._round_price(price - tp_distance, digits)

        return {
            "volume": float(volume),
            "sl": float(sl),
            "tp": float(tp),
            "risk_amount": float(risk_amount),
            "stop_distance": float(stop_distance),
        }
