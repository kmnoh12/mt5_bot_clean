from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

import pandas as pd


class ClosedBarGate:
    """Allows strategy evaluation only when a new closed bar appears per symbol."""

    def __init__(self, snapshot: Optional[Dict[str, str]] = None) -> None:
        self._last_closed_key: Dict[str, str] = dict(snapshot or {})

    @staticmethod
    def _normalize_time(raw: Any) -> Optional[datetime]:
        if raw is None:
            return None
        if isinstance(raw, datetime):
            return raw.astimezone(timezone.utc) if raw.tzinfo else raw.replace(tzinfo=timezone.utc)
        try:
            parsed = pd.to_datetime(raw, utc=True, errors="coerce")
        except Exception:
            return None
        if pd.isna(parsed):
            return None
        if isinstance(parsed, pd.Timestamp):
            return parsed.to_pydatetime().astimezone(timezone.utc)
        return None

    def _closed_bar_time(self, bars: pd.DataFrame) -> Optional[datetime]:
        if bars is None or bars.empty or len(bars) < 2:
            return None
        if "time" in bars.columns:
            return self._normalize_time(bars.iloc[-2].get("time"))
        # Fallback key when no time column exists.
        return datetime.fromtimestamp(float(len(bars) - 2), tz=timezone.utc)

    def should_evaluate(self, symbol: str, bars: pd.DataFrame) -> Tuple[bool, Optional[datetime]]:
        closed_time = self._closed_bar_time(bars)
        if closed_time is None:
            # Fail-open to avoid fully blocking a symbol under malformed data.
            return True, None

        key = f"{closed_time.isoformat()}"
        previous = self._last_closed_key.get(symbol)
        if previous == key:
            return False, closed_time

        self._last_closed_key[symbol] = key
        return True, closed_time

    def snapshot(self) -> Dict[str, str]:
        return dict(self._last_closed_key)
