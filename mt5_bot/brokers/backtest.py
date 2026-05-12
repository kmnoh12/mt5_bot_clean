from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from brokers.base import BrokerGateway
from core.models import OrderIntent, OrderResult, Position, Side, SymbolConstraints


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class InstrumentKey:
    symbol: str
    timeframe: str


class HistoricalDataLoader:
    REQUIRED_COLUMNS = {"open", "high", "low", "close"}

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = Path(data_dir)

    def _candidate_paths(self, symbol: str, timeframe: str) -> List[Path]:
        filenames = [
            f"{symbol}_{timeframe}.csv",
            f"{str(symbol).upper()}_{str(timeframe).upper()}.csv",
            f"{symbol}.csv",
            f"{str(symbol).upper()}.csv",
        ]
        candidates = [self.data_dir / name for name in filenames]
        if self.data_dir.exists():
            wanted = {name.lower() for name in filenames}
            for path in self.data_dir.glob("*.csv"):
                if path.name.lower() in wanted and path not in candidates:
                    candidates.append(path)
        return candidates

    def load(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        for path in self._candidate_paths(symbol=symbol, timeframe=timeframe):
            if not path.exists() or not path.is_file():
                continue
            try:
                frame = pd.read_csv(path)
            except Exception as exc:
                LOGGER.warning("Failed to read backtest CSV %s: %s", path, exc)
                continue
            if frame is None or frame.empty:
                continue

            normalized_columns = {str(col).strip(): str(col).strip().lower() for col in frame.columns}
            frame = frame.rename(columns=normalized_columns)
            missing = self.REQUIRED_COLUMNS - set(frame.columns)
            if missing:
                LOGGER.warning("Backtest CSV %s missing columns: %s", path, sorted(missing))
                continue

            for column in ["open", "high", "low", "close", "tick_volume", "volume"]:
                if column in frame.columns:
                    frame[column] = pd.to_numeric(frame[column], errors="coerce")
            frame = frame.dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)
            if frame.empty:
                continue

            time_column = "time" if "time" in frame.columns else "timestamp" if "timestamp" in frame.columns else ""
            if time_column:
                numeric_time = pd.to_numeric(frame[time_column], errors="coerce")
                if numeric_time.notna().all():
                    frame[time_column] = pd.to_datetime(numeric_time, unit="s", utc=True, errors="coerce")
                else:
                    frame[time_column] = pd.to_datetime(frame[time_column], utc=True, errors="coerce")
                frame = frame.sort_values(time_column).reset_index(drop=True)
            return frame
        return None


class BacktestGateway(BrokerGateway):
    mode = "backtest"

    def __init__(
        self,
        universe: List[Dict[str, Any]],
        general_cfg: Dict[str, Any],
        backtest_cfg: Dict[str, Any],
        execution_cfg: Dict[str, Any],
    ) -> None:
        self.universe = list(universe or [])
        self.general_cfg = general_cfg or {}
        self.backtest_cfg = backtest_cfg or {}
        self.execution_cfg = execution_cfg or {}

        self.loader = HistoricalDataLoader(Path(str(self.backtest_cfg.get("data_dir", "./data"))))
        self.warmup_bars = max(50, int(self.general_cfg.get("bars_per_request", 300)))
        self.spread_points = max(0.0, float(self.backtest_cfg.get("spread_points", 8.0)))
        self.commission_per_lot = max(0.0, float(self.backtest_cfg.get("commission_per_lot", 0.0)))
        self.contract_size = max(1e-9, float(self.backtest_cfg.get("contract_size", 1.0)))

        self.balance = float(self.backtest_cfg.get("initial_balance", 10000.0))
        self.account_currency = str(self.backtest_cfg.get("account_currency", "USD") or "USD").upper()
        self.connected = False
        self.current_index = 0
        self.max_index = 0
        self._ticket_counter = 1
        self._data: Dict[InstrumentKey, pd.DataFrame] = {}
        self._symbol_default_key: Dict[str, InstrumentKey] = {}
        self._positions: Dict[int, Position] = {}

        self._load_data()

    def _load_data(self) -> None:
        lengths: List[int] = []
        for item in self.universe:
            symbol = str(item.get("symbol", "")).strip()
            timeframe = str(item.get("timeframe", "TIMEFRAME_M5")).strip()
            if not symbol:
                continue
            key = InstrumentKey(symbol=symbol, timeframe=timeframe)
            frame = self.loader.load(symbol=symbol, timeframe=timeframe)
            if frame is None or frame.empty:
                continue
            self._data[key] = frame
            lengths.append(len(frame))
            if symbol not in self._symbol_default_key:
                self._symbol_default_key[symbol] = key

        if not self._data:
            raise RuntimeError("Backtest mode requires historical CSV data files.")

        min_length = min(lengths)
        if min_length <= self.warmup_bars:
            raise RuntimeError(
                f"Backtest data too short. Need > {self.warmup_bars} bars, got min={min_length}."
            )

        self.current_index = self.warmup_bars
        self.max_index = min_length - 1
        LOGGER.info(
            "Backtest data loaded. instruments=%s warmup=%s max_index=%s",
            len(self._data),
            self.warmup_bars,
            self.max_index,
        )

    def connect(self) -> bool:
        self.connected = True
        return True

    def disconnect(self) -> None:
        self.connected = False

    def heartbeat(self) -> bool:
        return self.connected

    def _point_for_symbol(self, symbol: str) -> float:
        text = symbol.upper()
        if "XAU" in text or "GOLD" in text:
            return 0.01
        if "BTC" in text or "ETH" in text:
            return 0.01
        if "JPY" in text:
            return 0.001
        return 0.0001

    def _digits_for_symbol(self, symbol: str) -> int:
        point = self._point_for_symbol(symbol)
        text = f"{point:.10f}".rstrip("0")
        if "." not in text:
            return 0
        return len(text.split(".")[1])

    @staticmethod
    def _guess_quote_currency(symbol: str) -> str:
        text = "".join(ch for ch in str(symbol or "").upper() if ch.isalpha())
        if len(text) >= 6:
            tail = text[-3:]
            if tail in {"USD", "EUR", "JPY", "GBP", "KRW", "CHF", "CAD", "AUD", "NZD"}:
                return tail
        if text.endswith(("GOLD", "SILVER", "XAU", "XAG", "BTC", "ETH")):
            return "USD"
        return ""

    def _resolve_key(self, symbol: str, timeframe: str) -> Optional[InstrumentKey]:
        target = InstrumentKey(symbol=symbol, timeframe=timeframe)
        if target in self._data:
            return target
        return self._symbol_default_key.get(symbol)

    def fetch_bars(self, symbol: str, timeframe: str, bars: int) -> Optional[pd.DataFrame]:
        key = self._resolve_key(symbol=symbol, timeframe=timeframe)
        if key is None:
            return None
        frame = self._data[key]
        bars = max(10, int(bars))
        start = max(0, self.current_index - bars + 1)
        end = self.current_index + 1
        if end > len(frame):
            return None
        return frame.iloc[start:end].copy().reset_index(drop=True)

    def _current_close(self, symbol: str) -> Optional[float]:
        key = self._symbol_default_key.get(symbol)
        if key is None:
            return None
        frame = self._data[key]
        if self.current_index >= len(frame):
            return None
        value = frame.iloc[self.current_index]["close"]
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _quote(self, symbol: str) -> Optional[Tuple[float, float]]:
        mid = self._current_close(symbol)
        if mid is None:
            return None
        point = self._point_for_symbol(symbol)
        spread = self.spread_points * point
        bid = mid - (spread / 2.0)
        ask = mid + (spread / 2.0)
        return bid, ask

    def _position_floating_pnl(self, position: Position) -> Optional[float]:
        quote = self._quote(position.symbol)
        if quote is None:
            return None
        bid, ask = quote
        mark = bid if position.side == Side.BUY else ask
        direction = 1.0 if position.side == Side.BUY else -1.0
        return (mark - position.price_open) * direction * float(position.volume) * self.contract_size

    def get_positions(self, symbol: Optional[str] = None) -> List[Position]:
        positions = list(self._positions.values())
        for pos in positions:
            floating = self._position_floating_pnl(pos)
            if not isinstance(pos.metadata, dict):
                pos.metadata = {}
            if floating is not None:
                pos.metadata["floating_pnl"] = float(floating)
        if symbol is None:
            return positions
        return [pos for pos in positions if pos.symbol == symbol]

    def get_symbol_constraints(self, symbol: str) -> Optional[SymbolConstraints]:
        quote = self._guess_quote_currency(symbol)
        return SymbolConstraints(
            min_volume=0.01,
            max_volume=100.0,
            volume_step=0.01,
            point=self._point_for_symbol(symbol),
            digits=self._digits_for_symbol(symbol),
            contract_size=self.contract_size,
            quote_currency=quote,
            profit_currency=quote,
        )

    def get_latest_price(self, symbol: str) -> Optional[float]:
        return self._current_close(symbol)

    def _is_valid_step(self, volume: float, constraints: SymbolConstraints) -> bool:
        min_v = float(constraints.min_volume)
        step = float(constraints.volume_step)
        if step <= 0:
            return True
        units = round((volume - min_v) / step)
        snapped = min_v + (units * step)
        return abs(snapped - volume) <= max(1e-9, step * 1e-6)

    def precheck_order(self, intent: OrderIntent) -> OrderResult:
        quote = self._quote(intent.symbol)
        if quote is None:
            return OrderResult(ok=False, status="CHECK_NO_QUOTE", message="Missing quote in backtest precheck.")

        constraints = self.get_symbol_constraints(intent.symbol)
        if constraints is None:
            return OrderResult(ok=False, status="CHECK_NO_CONSTRAINTS", message="Missing symbol constraints.")

        vol = float(intent.volume)
        if vol < constraints.min_volume or vol > constraints.max_volume or not self._is_valid_step(vol, constraints):
            return OrderResult(
                ok=False,
                status="CHECK_REJECTED",
                message="invalid_volume",
                retcode=10014,
            )

        bid, ask = quote
        price = ask if intent.side == Side.BUY else bid

        if intent.side == Side.BUY:
            if intent.sl is not None and float(intent.sl) >= price:
                return OrderResult(ok=False, status="CHECK_REJECTED", message="invalid_sl", retcode=10016)
            if intent.tp is not None and float(intent.tp) <= price:
                return OrderResult(ok=False, status="CHECK_REJECTED", message="invalid_tp", retcode=10016)
        else:
            if intent.sl is not None and float(intent.sl) <= price:
                return OrderResult(ok=False, status="CHECK_REJECTED", message="invalid_sl", retcode=10016)
            if intent.tp is not None and float(intent.tp) >= price:
                return OrderResult(ok=False, status="CHECK_REJECTED", message="invalid_tp", retcode=10016)

        return OrderResult(ok=True, status="CHECK_OK", message="backtest precheck ok", retcode=0)

    def send_order(self, intent: OrderIntent) -> OrderResult:
        check = self.precheck_order(intent)
        if not check.ok:
            return check

        quote = self._quote(intent.symbol)
        if quote is None:
            return OrderResult(ok=False, status="NO_QUOTE", message="Missing quote in backtest.")
        bid, ask = quote
        price = ask if intent.side == Side.BUY else bid

        ticket = self._ticket_counter
        self._ticket_counter += 1

        position = Position(
            ticket=ticket,
            symbol=intent.symbol,
            side=intent.side,
            volume=float(intent.volume),
            price_open=float(price),
            sl=intent.sl,
            tp=intent.tp,
            comment=intent.comment,
            magic=int(self.execution_cfg.get("magic", 0)),
            time_open_utc=datetime.now(timezone.utc),
            metadata={
                "strategy": intent.strategy,
                "reason": intent.reason,
                "external_signal_id": intent.external_signal_id,
            },
        )
        self._positions[ticket] = position
        return OrderResult(
            ok=True,
            status="FILLED_SIM",
            ticket=ticket,
            filled_price=price,
            message="Simulated backtest fill.",
        )

    def modify_position_sl_tp(
        self,
        position: Position,
        sl: Optional[float],
        tp: Optional[float],
        reason: str,
    ) -> OrderResult:
        current = self._positions.get(position.ticket)
        if current is None:
            return OrderResult(ok=False, status="MODIFY_NOT_FOUND", message="Position not found.")

        if sl is not None:
            current.sl = float(sl)
        if tp is not None:
            current.tp = float(tp)
        return OrderResult(ok=True, status="MODIFIED_SIM", ticket=current.ticket, message=reason)

    def close_position(self, position: Position, reason: str) -> OrderResult:
        current = self._positions.get(position.ticket)
        if current is None:
            return OrderResult(ok=False, status="POSITION_NOT_FOUND", message="Position not found for close.")

        quote = self._quote(position.symbol)
        if quote is None:
            return OrderResult(ok=False, status="NO_QUOTE", message="Missing quote for close.")
        bid, ask = quote
        close_price = bid if current.side == Side.BUY else ask

        close_volume = min(float(position.volume), float(current.volume))
        if close_volume <= 0:
            return OrderResult(ok=False, status="INVALID_CLOSE_VOLUME", message="Close volume must be positive.")

        direction = 1.0 if current.side == Side.BUY else -1.0
        pnl = (close_price - current.price_open) * direction * close_volume * self.contract_size
        commission = self.commission_per_lot * close_volume
        net_pnl = pnl - commission
        self.balance += net_pnl

        tolerance = 1e-9
        remaining_volume = float(current.volume) - close_volume
        if remaining_volume > tolerance:
            current.volume = max(0.0, remaining_volume)
            status = "CLOSED_PARTIAL_SIM"
        else:
            self._positions.pop(position.ticket, None)
            status = "CLOSED_SIM"

        return OrderResult(
            ok=True,
            status=status,
            ticket=position.ticket,
            filled_price=close_price,
            pnl=net_pnl,
            message=reason,
        )

    def close_all_positions(self, reason: str) -> List[OrderResult]:
        results: List[OrderResult] = []
        for position in list(self._positions.values()):
            results.append(self.close_position(position, reason=reason))
        return results

    def account_info(self) -> Dict[str, Any]:
        floating = 0.0
        for position in self._positions.values():
            quote = self._quote(position.symbol)
            if quote is None:
                continue
            bid, ask = quote
            mark = bid if position.side == Side.BUY else ask
            direction = 1.0 if position.side == Side.BUY else -1.0
            floating += (mark - position.price_open) * direction * float(position.volume) * self.contract_size
        equity = self.balance + floating
        return {
            "balance": self.balance,
            "equity": equity,
            "floating_pnl": floating,
            "open_positions": len(self._positions),
            "index": self.current_index,
            "max_index": self.max_index,
            "currency": self.account_currency,
        }

    def get_position_close_info(self, ticket: int) -> Optional[Dict[str, Any]]:
        # Backtest gateway already returns realized PnL on close_position(), and the runtime/order manager
        # emits trade_ledger/position_exit events for those closes.
        return None

    def step(self) -> bool:
        if self.current_index >= self.max_index:
            return False
        self.current_index += 1
        return self.current_index <= self.max_index
