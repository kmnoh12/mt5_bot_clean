from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

try:
    import MetaTrader5 as mt5
except ImportError:  # pragma: no cover
    mt5 = None


LOGGER = logging.getLogger(__name__)


class MT5Gateway:
    def __init__(
        self,
        config: Dict[str, Any],
        max_retries: int = 3,
        retry_delay_seconds: float = 1.5,
    ) -> None:
        self.config = config or {}
        self.max_retries = max(1, int(max_retries))
        self.retry_delay_seconds = max(0.0, float(retry_delay_seconds))
        self.connected = False

    def last_error(self) -> Any:
        if mt5 is None:
            return ("MT5_IMPORT_ERROR", "MetaTrader5 package not installed")
        try:
            return mt5.last_error()
        except Exception as exc:  # pragma: no cover
            return ("MT5_LAST_ERROR_FAILED", str(exc))

    def _sleep(self, attempt: int) -> None:
        if attempt >= self.max_retries:
            return
        delay = self.retry_delay_seconds * attempt
        time.sleep(max(self.retry_delay_seconds, delay))

    def _initialize_terminal_once(self, terminal_path: str) -> bool:
        timeout_ms = int(self.config.get("init_timeout_ms", 120000))
        try:
            mt5.shutdown()
        except Exception:
            pass

        kwargs: Dict[str, Any] = {"timeout": timeout_ms}
        if terminal_path:
            kwargs["path"] = terminal_path

        try:
            return bool(mt5.initialize(**kwargs))
        except TypeError:
            # Some MT5 package builds reject `timeout`.
            kwargs.pop("timeout", None)
            return bool(mt5.initialize(**kwargs))

    def _initialize_terminal(self) -> bool:
        configured_path = str(self.config.get("path", "")).strip()
        candidate_paths: List[str] = []

        if configured_path:
            candidate = Path(configured_path)
            if candidate.exists():
                candidate_paths.append(str(candidate))
            else:
                LOGGER.warning("Configured MT5 path does not exist: %s", configured_path)
        candidate_paths.append("")

        for candidate_path in candidate_paths:
            for attempt in range(1, self.max_retries + 1):
                try:
                    ok = self._initialize_terminal_once(candidate_path)
                except Exception:
                    LOGGER.exception(
                        "MT5 initialize exception (path=%s, attempt=%s/%s)",
                        candidate_path or "auto",
                        attempt,
                        self.max_retries,
                    )
                    ok = False

                if ok:
                    LOGGER.info(
                        "MT5 initialize success (path=%s, attempt=%s/%s)",
                        candidate_path or "auto",
                        attempt,
                        self.max_retries,
                    )
                    return True

                LOGGER.warning(
                    "MT5 initialize failed (path=%s, attempt=%s/%s): %s",
                    candidate_path or "auto",
                    attempt,
                    self.max_retries,
                    self.last_error(),
                )
                self._sleep(attempt)
        return False

    def _login(self) -> bool:
        login_raw = self.config.get("login", 0)
        password = str(self.config.get("password", ""))
        server = str(self.config.get("server", "")).strip()

        try:
            login = int(login_raw)
        except (TypeError, ValueError):
            login = 0

        if login <= 0 or not server:
            LOGGER.info("No explicit MT5 login/server configured. Using current terminal session.")
            return True

        for attempt in range(1, self.max_retries + 1):
            try:
                ok = bool(mt5.login(login, password=password, server=server))
            except Exception:
                LOGGER.exception("MT5 login exception (attempt=%s/%s)", attempt, self.max_retries)
                ok = False

            if ok:
                LOGGER.info("MT5 login success (%s@%s)", login, server)
                return True

            LOGGER.warning(
                "MT5 login failed (attempt=%s/%s): %s",
                attempt,
                self.max_retries,
                self.last_error(),
            )
            self._sleep(attempt)
        return False

    def connect(self) -> bool:
        if mt5 is None:
            LOGGER.error("MetaTrader5 package is not installed.")
            return False

        if not self._initialize_terminal():
            LOGGER.error("MT5 terminal initialization failed after retries.")
            return False

        if not self._login():
            LOGGER.error("MT5 login failed after retries.")
            try:
                mt5.shutdown()
            except Exception:
                pass
            return False

        self.connected = True
        return True

    def shutdown(self) -> None:
        if mt5 is None:
            self.connected = False
            return
        try:
            mt5.shutdown()
        except Exception:
            LOGGER.exception("MT5 shutdown failed.")
        finally:
            self.connected = False

    def resolve_timeframe(self, timeframe_name: str) -> Optional[int]:
        if mt5 is None:
            return None
        value = getattr(mt5, str(timeframe_name), None)
        if value is None:
            LOGGER.error("Unknown timeframe: %s", timeframe_name)
            return None
        return int(value)

    def ensure_symbol(self, symbol: str) -> bool:
        if mt5 is None:
            return False
        symbol = str(symbol).strip()
        if not symbol:
            return False

        for attempt in range(1, self.max_retries + 1):
            info = mt5.symbol_info(symbol)
            if info is None:
                LOGGER.warning(
                    "%s: symbol_info unavailable (attempt=%s/%s).",
                    symbol,
                    attempt,
                    self.max_retries,
                )
                self._sleep(attempt)
                continue
            if getattr(info, "visible", False):
                return True
            try:
                if mt5.symbol_select(symbol, True):
                    LOGGER.info("%s: symbol selected.", symbol)
                    return True
            except Exception:
                LOGGER.exception("%s: symbol_select raised exception.", symbol)
            self._sleep(attempt)
        return False

    def copy_rates(self, symbol: str, timeframe: int, bars: int) -> Optional[pd.DataFrame]:
        if mt5 is None:
            return None
        bars = max(10, int(bars))

        for attempt in range(1, self.max_retries + 1):
            try:
                rates = mt5.copy_rates_from_pos(symbol, int(timeframe), 0, bars)
            except Exception:
                LOGGER.exception(
                    "%s: copy_rates_from_pos exception (attempt=%s/%s).",
                    symbol,
                    attempt,
                    self.max_retries,
                )
                rates = None

            if rates is None or len(rates) == 0:
                LOGGER.warning(
                    "%s: empty OHLC data (attempt=%s/%s): %s",
                    symbol,
                    attempt,
                    self.max_retries,
                    self.last_error(),
                )
                self._sleep(attempt)
                continue

            frame = pd.DataFrame(rates)
            required_cols = {"open", "high", "low", "close"}
            missing = required_cols.difference(frame.columns)
            if missing:
                LOGGER.warning("%s: OHLC columns missing: %s", symbol, sorted(missing))
                return None

            for col in required_cols:
                frame[col] = pd.to_numeric(frame[col], errors="coerce")
            frame.dropna(subset=list(required_cols), inplace=True)
            if frame.empty:
                LOGGER.warning("%s: OHLC frame empty after numeric cleanup.", symbol)
                return None

            if "time" in frame.columns:
                frame["time"] = pd.to_datetime(frame["time"], unit="s", utc=True, errors="coerce")
            return frame.reset_index(drop=True)
        return None

    def symbol_info(self, symbol: str) -> Any:
        if mt5 is None:
            return None
        try:
            return mt5.symbol_info(symbol)
        except Exception:
            LOGGER.exception("%s: symbol_info exception.", symbol)
            return None

    def symbol_tick(self, symbol: str) -> Any:
        if mt5 is None:
            return None
        try:
            return mt5.symbol_info_tick(symbol)
        except Exception:
            LOGGER.exception("%s: symbol_info_tick exception.", symbol)
            return None

    def positions_get(self, symbol: Optional[str] = None) -> List[Any]:
        if mt5 is None:
            return []
        try:
            result = mt5.positions_get(symbol=symbol) if symbol else mt5.positions_get()
        except Exception:
            LOGGER.exception("positions_get exception (symbol=%s).", symbol)
            return []
        if result is None:
            return []
        return list(result)

    def account_equity(self) -> Optional[float]:
        if mt5 is None:
            return None
        for attempt in range(1, self.max_retries + 1):
            try:
                info = mt5.account_info()
            except Exception:
                LOGGER.exception("account_info exception (attempt=%s/%s).", attempt, self.max_retries)
                info = None

            if info is None:
                self._sleep(attempt)
                continue

            equity = getattr(info, "equity", None)
            balance = getattr(info, "balance", None)
            chosen = equity if equity not in (None, 0) else balance
            try:
                return float(chosen)
            except (TypeError, ValueError):
                LOGGER.warning("account_info returned non-numeric equity/balance: %r", chosen)
                self._sleep(attempt)
        return None

    def send_order(self, request: Dict[str, Any]) -> Dict[str, Any]:
        if mt5 is None:
            return {"ok": False, "status": "MT5_NOT_INSTALLED"}
        if not self.connected:
            return {"ok": False, "status": "MT5_NOT_CONNECTED"}

        success_codes = {
            int(getattr(mt5, "TRADE_RETCODE_DONE", 10009)),
            int(getattr(mt5, "TRADE_RETCODE_DONE_PARTIAL", 10010)),
        }
        retry_codes = {
            int(getattr(mt5, "TRADE_RETCODE_REQUOTE", 10004)),
            int(getattr(mt5, "TRADE_RETCODE_PRICE_CHANGED", 10020)),
            int(getattr(mt5, "TRADE_RETCODE_REJECT", 10006)),
        }

        last_payload: Dict[str, Any] = {}
        for attempt in range(1, self.max_retries + 1):
            try:
                result = mt5.order_send(request)
            except Exception:
                LOGGER.exception("order_send exception (attempt=%s/%s).", attempt, self.max_retries)
                result = None

            if result is None:
                LOGGER.warning(
                    "order_send returned None (attempt=%s/%s): %s",
                    attempt,
                    self.max_retries,
                    self.last_error(),
                )
                self._sleep(attempt)
                continue

            payload = result._asdict() if hasattr(result, "_asdict") else {"repr": repr(result)}
            retcode = int(getattr(result, "retcode", -1))
            last_payload = payload

            if retcode in success_codes:
                return {"ok": True, "status": "FILLED", "retcode": retcode, "result": payload}

            LOGGER.warning(
                "order rejected (retcode=%s, attempt=%s/%s).",
                retcode,
                attempt,
                self.max_retries,
            )
            if retcode not in retry_codes:
                return {
                    "ok": False,
                    "status": "REJECTED",
                    "retcode": retcode,
                    "result": payload,
                }
            self._sleep(attempt)

        return {
            "ok": False,
            "status": "FAILED_AFTER_RETRY",
            "error": self.last_error(),
            "result": last_payload,
        }
