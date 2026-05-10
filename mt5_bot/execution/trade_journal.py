from __future__ import annotations

from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, Optional


class TradeJournal:
    def __init__(self, config: Dict[str, Any]) -> None:
        cfg = dict(config or {})
        self.enabled = bool(cfg.get("enabled", True))
        self.output_dir = Path(str(cfg.get("output_dir", "./reports/trade_journal")))
        self.tiny_pnl_threshold_usd = max(0.0, float(cfg.get("tiny_pnl_threshold_usd", 2.0) or 2.0))
        self.quick_exit_window_seconds = max(1.0, float(cfg.get("quick_exit_window_seconds", 300.0) or 300.0))
        self.big_loss_threshold_usd = float(cfg.get("big_loss_threshold_usd", -10.0) or -10.0)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _fmt_ts(now_utc: datetime) -> str:
        utc_text = now_utc.astimezone(timezone.utc).strftime("%H:%M:%S UTC")
        kst_text = now_utc.astimezone(timezone(timedelta(hours=9))).strftime("%H:%M:%S KST")
        return f"{utc_text} / {kst_text}"

    @staticmethod
    def _safe_float(value: Any) -> Optional[float]:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _classify(self, pnl: Optional[float], hold_seconds: Optional[float]) -> Dict[str, Any]:
        pnl_v = self._safe_float(pnl)
        hold_v = self._safe_float(hold_seconds)
        tiny = pnl_v is not None and abs(pnl_v) <= self.tiny_pnl_threshold_usd
        quick = hold_v is not None and hold_v <= self.quick_exit_window_seconds
        big_loss = pnl_v is not None and pnl_v <= self.big_loss_threshold_usd

        if tiny and quick:
            return {
                "label": "CHURN",
                "improvement": "다음 진입은 추세 정렬(M5)과 신호 강도 조건이 충족될 때만 허용",
                "is_anomaly": True,
            }
        if big_loss:
            return {
                "label": "BIG_LOSS",
                "improvement": "손절 거리와 포지션 사이즈를 축소하고 동일 구간 재진입을 금지",
                "is_anomaly": True,
            }
        if pnl_v is not None and pnl_v < 0:
            return {
                "label": "NORMAL_LOSS",
                "improvement": "진입 근거와 청산 근거의 일관성을 점검",
                "is_anomaly": True,
            }
        if pnl_v is not None and pnl_v > 0:
            return {
                "label": "WIN",
                "improvement": "수익 구간 홀딩 규칙을 유지하고 성급한 조기청산을 피함",
                "is_anomaly": False,
            }
        return {
            "label": "UNKNOWN",
            "improvement": "데이터 누락을 점검하고 기록 완결성을 보강",
            "is_anomaly": False,
        }

    def record_trade(
        self,
        *,
        symbol: str,
        reason: str,
        pnl: Optional[float],
        hold_seconds: Optional[float],
        entry_price: Optional[float],
        exit_price: Optional[float],
        now_utc: Optional[datetime] = None,
    ) -> Optional[Dict[str, Any]]:
        if not self.enabled:
            return None
        now = now_utc or datetime.now(timezone.utc)
        day_key = now.astimezone(timezone.utc).date().isoformat()
        target = self.output_dir / f"{day_key}.md"
        if not target.exists():
            target.write_text(f"# Trade Reflection {day_key}\n\n", encoding="utf-8")

        classification = self._classify(pnl=pnl, hold_seconds=hold_seconds)
        pnl_text = "N/A" if pnl is None else f"{float(pnl):.2f}"
        hold_text = "N/A" if hold_seconds is None else f"{float(hold_seconds):.0f}s"
        entry_text = "N/A" if entry_price is None else f"{float(entry_price):.5f}"
        exit_text = "N/A" if exit_price is None else f"{float(exit_price):.5f}"

        line = (
            f"- [{self._fmt_ts(now)}] {symbol} | pnl={pnl_text} | hold={hold_text} | "
            f"entry={entry_text} -> exit={exit_text} | reason={reason} | "
            f"class={classification['label']} | next_action={classification['improvement']}\n"
        )
        with target.open("a", encoding="utf-8") as handle:
            handle.write(line)

        return {
            "path": str(target),
            "classification": classification["label"],
            "is_anomaly": bool(classification["is_anomaly"]),
        }
