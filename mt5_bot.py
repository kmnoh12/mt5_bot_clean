"""
MT5 volatility breakout bot.
"""

from __future__ import annotations

import getpass
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

try:
    import MetaTrader5 as mt5
except ImportError:  # pragma: no cover
    mt5 = None


CONFIG_PATH = Path(__file__).with_name("config.json")
DEFAULT_SYMBOLS = ["BTCUSD", "GOLD", "US100Cash"]
DEFAULT_TIMEFRAME = "TIMEFRAME_M15"


def log(level: str, message: str) -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{level}] {message}")


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


class MT5Client:
    def __init__(self, mt5_config: Dict[str, Any]) -> None:
        self.mt5_config = mt5_config
        self.connected = False
        self.initialized = False

    def _try_initialize(self, path: Optional[Path]) -> bool:
        if mt5 is None:
            return False
        timeout_ms = int(self.mt5_config.get("init_timeout_ms", 120000))
        mt5.shutdown()
        if path is None:
            try:
                ok = mt5.initialize(timeout=timeout_ms)
            except TypeError:
                ok = mt5.initialize()
            label = "default auto-detect"
        else:
            try:
                ok = mt5.initialize(path=str(path), timeout=timeout_ms)
            except TypeError:
                ok = mt5.initialize(path=str(path))
            label = str(path)
        if ok:
            log("INFO", f"MT5 initialize success ({label})")
            return True
        log("WARN", f"MT5 initialize failed ({label}): {mt5.last_error()}")
        return False

    @staticmethod
    def _unique_existing_paths(paths: Iterable[Path]) -> List[Path]:
        seen = set()
        existing: List[Path] = []
        for path in paths:
            key = str(path).lower()
            if key in seen:
                continue
            seen.add(key)
            if path.exists():
                existing.append(path)
        return existing

    def _scan_terminal_paths(self) -> List[Path]:
        static_candidates = [
            Path(r"C:\Program Files\MetaTrader 5\terminal64.exe"),
            Path(r"C:\Program Files\MetaTrader 5\terminal.exe"),
            Path(r"C:\Program Files (x86)\MetaTrader 5\terminal64.exe"),
            Path(r"C:\Program Files (x86)\MetaTrader 5\terminal.exe"),
        ]

        search_roots: List[Path] = []
        for env_key in ("PROGRAMFILES", "PROGRAMFILES(X86)", "LOCALAPPDATA"):
            value = os.environ.get(env_key)
            if value:
                search_roots.append(Path(value))

        roaming_root = Path.home() / "AppData" / "Roaming" / "MetaQuotes" / "Terminal"
        if roaming_root.exists():
            search_roots.append(roaming_root)

        discovered: List[Path] = []
        for root in search_roots:
            if not root.exists():
                continue
            base_depth = len(root.parts)
            for current, dirs, files in os.walk(root):
                current_path = Path(current)
                depth = len(current_path.parts) - base_depth
                if depth > 6:
                    dirs[:] = []
                    continue
                lower_files = {name.lower(): name for name in files}
                for exe_name in ("terminal64.exe", "terminal.exe"):
                    if exe_name in lower_files:
                        discovered.append(current_path / lower_files[exe_name])

        return self._unique_existing_paths([*static_candidates, *discovered])

    def _initialize_terminal(self) -> bool:
        configured_path = str(self.mt5_config.get("path", "")).strip()
        if configured_path:
            configured = Path(configured_path)
            if self._try_initialize(configured):
                self.initialized = True
                return True
            log("WARN", "Configured MT5 path failed. Falling back to auto-scan.")

        if self._try_initialize(None):
            self.initialized = True
            return True

        log("INFO", "Auto-scanning local MT5 terminal paths...")
        candidates = self._scan_terminal_paths()
        log("INFO", f"Auto-scan found {len(candidates)} terminal candidate(s).")
        for candidate in candidates:
            if self._try_initialize(candidate):
                self.initialized = True
                return True

        if sys.stdin.isatty():
            try:
                manual = input("MT5 terminal not found. Enter terminal64.exe path to retry (blank to skip): ").strip()
            except EOFError:
                log("INFO", "No interactive input available for manual terminal path retry.")
                manual = ""
            if manual and self._try_initialize(Path(manual)):
                self.mt5_config["path"] = manual
                self.initialized = True
                return True
        return False

    @staticmethod
    def _parse_login_id(login: str) -> Optional[int]:
        try:
            return int(login)
        except (TypeError, ValueError):
            return None

    def _login(self, login: str, password: str, server: str) -> bool:
        if mt5 is None:
            return False
        login_id = self._parse_login_id(login)
        if login_id is None:
            log("ERROR", f"Invalid mt5.login value: {login!r}")
            return False
        if mt5.login(login_id, password=password, server=server):
            log("INFO", f"MT5 login success ({login}@{server})")
            return True
        log("WARN", f"MT5 login failed: {mt5.last_error()}")
        return False

    def connect(self) -> bool:
        if mt5 is None:
            raise RuntimeError("MetaTrader5 package is not installed.")

        if not self._initialize_terminal():
            log("ERROR", "Unable to initialize MT5 terminal.")
            return False

        login = str(self.mt5_config.get("login", "")).strip()
        password = str(self.mt5_config.get("password", ""))
        server = str(self.mt5_config.get("server", "")).strip()

        if login and server:
            if self._login(login, password, server):
                self.connected = True
                return True

            if password.strip() == "":
                prompt = "MT5 password is blank and login failed. Enter password to retry: "
                try:
                    retry_password = getpass.getpass(prompt=prompt)
                except (EOFError, KeyboardInterrupt):
                    log("ERROR", "Password prompt aborted. MT5 login retry canceled.")
                    mt5.shutdown()
                    return False
                if retry_password and self._login(login, retry_password, server):
                    self.mt5_config["password"] = retry_password
                    self.connected = True
                    return True
                log("ERROR", "Retry login failed.")
                mt5.shutdown()
                return False

            log("ERROR", "MT5 login failed with configured credentials.")
            mt5.shutdown()
            return False

        log("INFO", "No explicit login/server configured. Using current terminal session.")
        self.connected = True
        return True

    def disconnect(self) -> None:
        if mt5 is not None:
            mt5.shutdown()
        self.connected = False
        self.initialized = False


@dataclass
class BreakoutSignal:
    side: str
    close: float
    atr: float
    breakout_high: float
    breakout_low: float


class VolatilityBreakoutStrategy:
    def __init__(self, lookback: int = 20, atr_period: int = 14, atr_multiplier: float = 1.0) -> None:
        self.lookback = lookback
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier

    def _compute_atr(self, candles: Sequence[Mapping[str, float]]) -> Optional[float]:
        if len(candles) < self.atr_period + 1:
            return None

        true_ranges: List[float] = []
        for idx, candle in enumerate(candles):
            high = candle["high"]
            low = candle["low"]
            if idx == 0:
                tr = high - low
            else:
                prev_close = candles[idx - 1]["close"]
                tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
            true_ranges.append(tr)

        window = true_ranges[-self.atr_period :]
        return sum(window) / float(self.atr_period)

    def generate_signal(self, candles: Sequence[Mapping[str, float]]) -> Optional[BreakoutSignal]:
        if len(candles) < max(self.lookback + 3, self.atr_period + 2):
            return None

        # Use only closed candles (skip current potentially in-progress bar).
        closed = candles[:-1]
        if len(closed) < max(self.lookback + 2, self.atr_period + 1):
            return None

        atr = self._compute_atr(closed)
        if atr is None:
            return None

        signal_candle = closed[-1]
        lookback_candles = closed[-(self.lookback + 1) : -1]
        range_high = max(candle["high"] for candle in lookback_candles)
        range_low = min(candle["low"] for candle in lookback_candles)
        breakout_high = range_high + (atr * self.atr_multiplier)
        breakout_low = range_low - (atr * self.atr_multiplier)
        close_price = signal_candle["close"]

        if close_price > breakout_high:
            side = "BUY"
        elif close_price < breakout_low:
            side = "SELL"
        else:
            side = "HOLD"

        return BreakoutSignal(
            side=side,
            close=close_price,
            atr=atr,
            breakout_high=breakout_high,
            breakout_low=breakout_low,
        )


@dataclass
class RiskManager:
    stop_loss_percent: float

    def stop_loss_price(self, entry_price: float, side: str, digits: int) -> float:
        ratio = self.stop_loss_percent / 100.0
        if side == "BUY":
            sl = entry_price * (1.0 - ratio)
        elif side == "SELL":
            sl = entry_price * (1.0 + ratio)
        else:
            raise ValueError("side must be BUY or SELL")
        return round(sl, digits)


class MT5Trader:
    def __init__(self, client: MT5Client, risk: RiskManager, lot: float, dry_run: bool = True) -> None:
        self.client = client
        self.risk = risk
        self.lot = lot
        self.dry_run = dry_run

    @staticmethod
    def ensure_symbol_ready(symbol: str) -> bool:
        if mt5 is None:
            return False
        info = mt5.symbol_info(symbol)
        if info is None:
            log("WARN", f"{symbol}: symbol not found in MT5 terminal.")
            return False
        if not info.visible:
            if not mt5.symbol_select(symbol, True):
                log("WARN", f"{symbol}: symbol_select failed, cannot trade.")
                return False
            log("INFO", f"{symbol}: symbol was hidden and is now enabled.")
        return True

    @staticmethod
    def fetch_candles(symbol: str, timeframe: int, bars: int) -> List[Dict[str, float]]:
        if mt5 is None:
            return []
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars)
        if rates is None or len(rates) == 0:
            log("WARN", f"{symbol}: failed to fetch OHLC rates.")
            return []
        candles: List[Dict[str, float]] = []
        for row in rates:
            candles.append(
                {
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                }
            )
        return candles

    def place_order(self, symbol: str, side: str) -> Dict[str, Any]:
        if mt5 is None:
            return {"status": "MT5_NOT_AVAILABLE"}
        if not self.client.connected:
            return {"status": "NOT_CONNECTED"}

        info = mt5.symbol_info(symbol)
        tick = mt5.symbol_info_tick(symbol)
        if info is None or tick is None:
            return {"status": "NO_SYMBOL_DATA", "symbol": symbol}

        if side == "BUY":
            order_type = mt5.ORDER_TYPE_BUY
            price = float(tick.ask)
        elif side == "SELL":
            order_type = mt5.ORDER_TYPE_SELL
            price = float(tick.bid)
        else:
            return {"status": "INVALID_SIDE", "symbol": symbol, "side": side}

        digits = int(getattr(info, "digits", 5))
        stop_loss = self.risk.stop_loss_price(price, side, digits)
        filling = int(getattr(info, "filling_mode", mt5.ORDER_FILLING_IOC))
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": self.lot,
            "type": order_type,
            "price": price,
            "sl": stop_loss,
            "deviation": 20,
            "magic": 20260206,
            "comment": "volatility_breakout",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": filling,
        }

        if self.dry_run:
            return {"status": "DRY_RUN", "request": request}

        result = mt5.order_send(request)
        if result is None:
            return {"status": "SEND_FAILED", "error": mt5.last_error(), "request": request}

        retcode = int(getattr(result, "retcode", -1))
        if retcode == mt5.TRADE_RETCODE_DONE:
            return {"status": "FILLED", "retcode": retcode, "request": request}
        return {"status": "REJECTED", "retcode": retcode, "request": request}

    @staticmethod
    def has_open_position(symbol: str) -> bool:
        if mt5 is None:
            return False
        positions = mt5.positions_get(symbol=symbol)
        if positions is None:
            return False
        return len(positions) > 0


def resolve_symbols(trading: Dict[str, Any]) -> List[str]:
    symbols = trading.get("symbols")
    if isinstance(symbols, list) and symbols:
        return [str(item).strip() for item in symbols if str(item).strip()]
    legacy = str(trading.get("symbol", "")).strip()
    return [legacy] if legacy else DEFAULT_SYMBOLS[:]


def resolve_timeframe(name: str) -> int:
    if mt5 is None:
        raise RuntimeError("MetaTrader5 package is not installed.")
    value = getattr(mt5, name, None)
    if value is None:
        log("WARN", f"Unknown timeframe '{name}'. Falling back to {DEFAULT_TIMEFRAME}.")
        value = getattr(mt5, DEFAULT_TIMEFRAME)
    return int(value)


def run_bot(config: Dict[str, Any]) -> None:
    mt5_config = config.get("mt5", {})
    trading = config.get("trading", {})

    symbols = resolve_symbols(trading)
    timeframe_name = str(trading.get("timeframe", DEFAULT_TIMEFRAME))
    timeframe = resolve_timeframe(timeframe_name)
    lot = float(trading.get("lot", 0.01))
    stop_loss_percent = float(trading.get("stop_loss_percent", 2.0))
    dry_run = bool(trading.get("dry_run", True))
    atr_period = int(trading.get("atr_period", 14))
    lookback = int(trading.get("breakout_lookback", 20))
    atr_multiplier = float(trading.get("atr_multiplier", 1.0))
    poll_seconds = int(trading.get("poll_seconds", 30))
    bars_to_fetch = max(atr_period + 10, lookback + 10, 60)

    client = MT5Client(mt5_config)
    if not client.connect():
        return

    strategy = VolatilityBreakoutStrategy(
        lookback=lookback,
        atr_period=atr_period,
        atr_multiplier=atr_multiplier,
    )
    risk = RiskManager(stop_loss_percent=stop_loss_percent)
    trader = MT5Trader(client=client, risk=risk, lot=lot, dry_run=dry_run)

    log(
        "INFO",
        f"Bot started. symbols={symbols}, timeframe={timeframe_name}, lot={lot}, "
        f"stop_loss={stop_loss_percent}%, dry_run={dry_run}",
    )

    try:
        while True:
            print("Bot is running...")
            for symbol in symbols:
                if not trader.ensure_symbol_ready(symbol):
                    continue

                candles = trader.fetch_candles(symbol=symbol, timeframe=timeframe, bars=bars_to_fetch)
                if not candles:
                    continue

                signal = strategy.generate_signal(candles)
                if signal is None:
                    log("WARN", f"{symbol}: not enough OHLC data yet.")
                    continue

                log(
                    "INFO",
                    f"{symbol}: signal={signal.side}, close={signal.close:.5f}, "
                    f"atr={signal.atr:.5f}, upper={signal.breakout_high:.5f}, "
                    f"lower={signal.breakout_low:.5f}",
                )

                if signal.side in {"BUY", "SELL"}:
                    if trader.has_open_position(symbol):
                        log("INFO", f"{symbol}: open position exists, skipping new {signal.side} order.")
                        continue
                    order_result = trader.place_order(symbol=symbol, side=signal.side)
                    log("INFO", f"{symbol}: order_status={order_result.get('status')}")
                    if "request" in order_result:
                        log("INFO", f"{symbol}: order_request={order_result['request']}")

            time.sleep(max(1, poll_seconds))
    except KeyboardInterrupt:
        log("INFO", "Bot stopped by user.")
    finally:
        client.disconnect()


def main() -> None:
    config = load_config(CONFIG_PATH)
    run_bot(config)


if __name__ == "__main__":
    main()
