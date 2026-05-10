from __future__ import annotations

import json
import logging
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd

from core.models import DecisionAction, Position, StrategyDecision, parse_action
from llm.client import build_chat_client


LOGGER = logging.getLogger(__name__)
SUPPORTED_PROVIDERS = {"openai", "gemini"}


class LlmAssistService:
    def __init__(self, config: Dict[str, Any]) -> None:
        self.base_config = dict(config or {})
        self.enabled = bool(self.base_config.get("enabled", False))
        self.provider = str(self.base_config.get("provider", "openai") or "openai").strip().lower()
        self.approve_confidence = float(self.base_config.get("approve_confidence", 0.55))
        self.veto_confidence = float(self.base_config.get("veto_confidence", 0.65))
        self.scale_on_ambiguous = float(self.base_config.get("scale_on_ambiguous", 0.5))
        self.max_bars_for_prompt = max(10, int(self.base_config.get("max_bars_for_prompt", 60)))
        self.settings_path = Path(str(self.base_config.get("settings_path", "") or "")).resolve() if self.base_config.get("settings_path") else None

    def _load_runtime_overrides(self) -> Dict[str, Any]:
        if self.settings_path is None or not self.settings_path.exists():
            return {}
        try:
            with self.settings_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except Exception:
            return {}
        if not isinstance(payload, dict):
            return {}
        llm_cfg = payload.get("llm_assist")
        return dict(llm_cfg) if isinstance(llm_cfg, dict) else {}

    def _effective_config(self) -> Dict[str, Any]:
        merged = dict(self.base_config)
        merged.update(self._load_runtime_overrides())
        return merged

    def _build_prompt(self, symbol: str, decision: StrategyDecision, bars: pd.DataFrame, position: Optional[Position]) -> Dict[str, Any]:
        frame = bars.copy() if bars is not None else pd.DataFrame()
        if not frame.empty and len(frame) > self.max_bars_for_prompt:
            frame = frame.tail(self.max_bars_for_prompt).copy()

        cols = [col for col in ["time", "open", "high", "low", "close", "tick_volume"] if col in frame.columns]
        serial_rows = []
        if cols:
            for _, row in frame[cols].iterrows():
                serial_rows.append({col: (row[col].isoformat() if hasattr(row[col], "isoformat") else row[col]) for col in cols})

        return {
            "symbol": symbol,
            "rule_action": decision.action.value,
            "rule_reason": decision.reason,
            "rule_confidence": decision.confidence,
            "rule_sl": decision.sl,
            "rule_tp": decision.tp,
            "position": {
                "side": position.side.value if position else None,
                "volume": position.volume if position else None,
                "entry": position.price_open if position else None,
                "sl": position.sl if position else None,
                "tp": position.tp if position else None,
            },
            "metadata": decision.metadata,
            "bars": serial_rows,
        }

    def _clone(self, decision: StrategyDecision, **kwargs: Any) -> StrategyDecision:
        return replace(decision, **kwargs)

    def apply(
        self,
        symbol: str,
        decision: StrategyDecision,
        bars: pd.DataFrame,
        position: Optional[Position],
    ) -> Tuple[StrategyDecision, Optional[Dict[str, Any]]]:
        if decision.action not in {DecisionAction.BUY, DecisionAction.SELL}:
            return decision, None

        cfg = self._effective_config()
        enabled = bool(cfg.get("enabled", self.enabled))
        if not enabled:
            return decision, None

        provider = str(cfg.get("provider", self.provider)).strip().lower()
        if provider not in SUPPORTED_PROVIDERS:
            return decision, {"event": "llm_assist", "status": "provider_unsupported"}

        client = build_chat_client(cfg)
        if client is None:
            return decision, {"event": "llm_assist", "status": "api_key_missing", "provider": provider}

        started = time.perf_counter()
        raw = client.infer(self._build_prompt(symbol=symbol, decision=decision, bars=bars, position=position))
        latency_ms = int((time.perf_counter() - started) * 1000)
        if raw is None:
            return decision, {"event": "llm_assist", "status": "no_response", "provider": provider, "latency_ms": latency_ms}

        llm_action = parse_action(raw.action) or DecisionAction.HOLD
        llm_conf = max(0.0, min(1.0, float(raw.confidence)))
        llm_reason = str(raw.reason or "")

        outcome = {
            "event": "llm_assist",
            "status": "evaluated",
            "symbol": symbol,
            "rule_action": decision.action.value,
            "llm_action": llm_action.value,
            "llm_confidence": llm_conf,
            "llm_reason": llm_reason,
            "provider": provider,
            "latency_ms": latency_ms,
        }

        if llm_action == decision.action and llm_conf >= float(cfg.get("approve_confidence", self.approve_confidence)):
            merged_meta = dict(decision.metadata)
            merged_meta.update({"llm_approved": True, "llm_confidence": llm_conf, "llm_reason": llm_reason})
            return self._clone(decision, metadata=merged_meta), outcome

        opposite = (
            (decision.action == DecisionAction.BUY and llm_action == DecisionAction.SELL)
            or (decision.action == DecisionAction.SELL and llm_action == DecisionAction.BUY)
            or (llm_action == DecisionAction.HOLD)
        )
        if opposite and llm_conf >= float(cfg.get("veto_confidence", self.veto_confidence)):
            veto = StrategyDecision(
                action=DecisionAction.HOLD,
                reason=f"LLM_VETO:{llm_reason or llm_action.value}",
                strategy=decision.strategy,
                confidence=decision.confidence,
                signal_bar_time=decision.signal_bar_time,
                min_hold_bars=decision.min_hold_bars,
                metadata={
                    **decision.metadata,
                    "llm_veto": True,
                    "llm_confidence": llm_conf,
                    "llm_reason": llm_reason,
                },
            )
            outcome["status"] = "veto"
            return veto, outcome

        scale = max(0.1, min(1.0, float(cfg.get("scale_on_ambiguous", self.scale_on_ambiguous))))
        new_volume = decision.volume * scale if decision.volume is not None else None
        merged_meta = dict(decision.metadata)
        merged_meta["volume_scale"] = float(merged_meta.get("volume_scale", 1.0)) * scale
        merged_meta["llm_ambiguous"] = True
        merged_meta["llm_confidence"] = llm_conf
        merged_meta["llm_reason"] = llm_reason
        outcome["status"] = "scaled"
        outcome["scale"] = scale
        return self._clone(decision, volume=new_volume, metadata=merged_meta), outcome
