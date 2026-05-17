from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List

try:
    import MetaTrader5 as mt5
except ImportError:  # pragma: no cover
    mt5 = None

from mt5_gateway import MT5Gateway
from risk_manager import BucketRiskConfig, RiskManager
from storage import JsonStorage
from strategies.mean_reversion import MeanReversionStrategy
from strategies.vol_breakout import VolBreakoutStrategy


LOGGER = logging.getLogger(__name__)


@dataclass
class BucketConfig:
    name: str
    enabled: bool
    strategy: str
    timeframe: str
    symbols: List[str]
    risk_per_trade_pct: float
    max_positions: int
    min_lot: float
    max_lot: float


class PortfolioEngine:
    def __init__(
        self,
        config: Dict[str, Any],
        gateway: MT5Gateway,
        risk_manager: RiskManager,
        storage: JsonStorage,
    ) -> None:
        self.config = config or {}
        self.gateway = gateway
        self.risk_manager = risk_manager
        self.storage = storage

        self.general_cfg = self.config.get("general", {})
        self.execution_cfg = self.config.get("execution", {})
        self.portfolio_cfg = self.config.get("portfolio", {})
        self.strategy_cfg = self.config.get("strategies", {})

        self.dry_run = bool(self.general_cfg.get("dry_run", True))
        self.bars_per_request = max(60, int(self.general_cfg.get("bars_per_request", 250)))
        self.max_total_positions = max(1, int(self.portfolio_cfg.get("max_total_positions", 4)))
        self.allow_opposite_position = bool(self.execution_cfg.get("allow_opposite_position", False))
        self.deviation = max(1, int(self.execution_cfg.get("deviation", 20)))
        self.magic = int(self.execution_cfg.get("magic", 20260206))
        self.comment_prefix = str(self.execution_cfg.get("comment_prefix", "dual_bucket")).strip() or "dual_bucket"

        self.strategies = {
            "mean_reversion": MeanReversionStrategy(self.strategy_cfg.get("mean_reversion", {})),
            "vol_breakout": VolBreakoutStrategy(self.strategy_cfg.get("vol_breakout", {})),
        }
        self.buckets = self._parse_buckets(self.portfolio_cfg.get("buckets", {}))

    @staticmethod
    def _normalize_symbols(raw_symbols: Any) -> List[str]:
        if not isinstance(raw_symbols, list):
            return []
        symbols: List[str] = []
        for value in raw_symbols:
            symbol = str(value).strip()
            if symbol:
                symbols.append(symbol)
        return symbols

    def _parse_buckets(self, buckets_map: Dict[str, Any]) -> List[BucketConfig]:
        if not isinstance(buckets_map, dict):
            LOGGER.error("portfolio.buckets must be a dictionary.")
            return []

        parsed: List[BucketConfig] = []
        for bucket_name, raw_cfg in buckets_map.items():
            if not isinstance(raw_cfg, dict):
                LOGGER.warning("Bucket '%s' config is invalid. Skipping.", bucket_name)
                continue

            symbols = self._normalize_symbols(raw_cfg.get("symbols", []))
            parsed.append(
                BucketConfig(
                    name=str(bucket_name),
                    enabled=bool(raw_cfg.get("enabled", True)),
                    strategy=str(raw_cfg.get("strategy", "")).strip(),
                    timeframe=str(raw_cfg.get("timeframe", "TIMEFRAME_M15")).strip(),
                    symbols=symbols,
                    risk_per_trade_pct=float(raw_cfg.get("risk_per_trade_pct", 0.5)),
                    max_positions=max(1, int(raw_cfg.get("max_positions", 1))),
                    min_lot=max(0.0, float(raw_cfg.get("min_lot", 0.01))),
                    max_lot=max(0.0, float(raw_cfg.get("max_lot", 0.1))),
                )
            )
        return parsed

    def _bucket_position_count(self, bucket: BucketConfig, positions: List[Any]) -> int:
        count = 0
        for position in positions:
            symbol = str(getattr(position, "symbol", "") or "")
            comment = str(getattr(position, "comment", "") or "")
            if symbol in bucket.symbols:
                count += 1
                continue
            if f"{self.comment_prefix}:{bucket.name}" in comment:
                count += 1
        return count

    @staticmethod
    def _position_side(position: Any) -> str:
        ptype = int(getattr(position, "type", -1))
        buy_type = int(getattr(mt5, "POSITION_TYPE_BUY", 0)) if mt5 is not None else 0
        return "BUY" if ptype == buy_type else "SELL"

    def _build_order_request(
        self,
        symbol: str,
        side: str,
        price: float,
        volume: float,
        sl: float,
        tp: float,
        filling_mode: int,
        bucket_name: str,
    ) -> Dict[str, Any]:
        side = side.upper()
        if mt5 is None:
            return {
                "symbol": symbol,
                "side": side,
                "volume": volume,
                "price": price,
                "sl": sl,
                "tp": tp,
                "deviation": self.deviation,
                "magic": self.magic,
                "comment": f"{self.comment_prefix}:{bucket_name}",
            }

        order_type_buy = int(getattr(mt5, "ORDER_TYPE_BUY", 0))
        order_type_sell = int(getattr(mt5, "ORDER_TYPE_SELL", 1))
        order_type = order_type_buy if side == "BUY" else order_type_sell

        return {
            "action": int(getattr(mt5, "TRADE_ACTION_DEAL", 1)),
            "symbol": symbol,
            "volume": float(volume),
            "type": order_type,
            "price": float(price),
            "sl": float(sl),
            "tp": float(tp),
            "deviation": self.deviation,
            "magic": self.magic,
            "comment": f"{self.comment_prefix}:{bucket_name}",
            "type_time": int(getattr(mt5, "ORDER_TIME_GTC", 0)),
            "type_filling": int(filling_mode),
        }

    def run_once(self) -> None:
        if not self.buckets:
            LOGGER.error("No valid bucket configuration available.")
            return

        equity = self.gateway.account_equity()
        if equity is None or equity <= 0:
            LOGGER.warning("Failed to get account equity. Using fallback value=1000.0")
            equity = 1000.0

        current_positions = self.gateway.positions_get()
        total_open_positions = len(current_positions)

        for bucket in self.buckets:
            if not bucket.enabled:
                continue
            if not bucket.symbols:
                LOGGER.warning("Bucket '%s' has no symbols. Skipping.", bucket.name)
                continue

            strategy = self.strategies.get(bucket.strategy)
            if strategy is None:
                LOGGER.error("Unknown strategy '%s' in bucket '%s'.", bucket.strategy, bucket.name)
                continue

            timeframe = self.gateway.resolve_timeframe(bucket.timeframe)
            if timeframe is None:
                LOGGER.error("Bucket '%s': invalid timeframe '%s'.", bucket.name, bucket.timeframe)
                continue

            bucket_open_positions = self._bucket_position_count(bucket, current_positions)
            if bucket_open_positions >= bucket.max_positions:
                LOGGER.info(
                    "Bucket '%s' limit reached (%s/%s).",
                    bucket.name,
                    bucket_open_positions,
                    bucket.max_positions,
                )
                continue

            for symbol in bucket.symbols:
                if total_open_positions >= self.max_total_positions:
                    LOGGER.info(
                        "Global position limit reached (%s/%s).",
                        total_open_positions,
                        self.max_total_positions,
                    )
                    break
                if bucket_open_positions >= bucket.max_positions:
                    break

                try:
                    placed = self._process_symbol(
                        bucket=bucket,
                        symbol=symbol,
                        strategy=strategy,
                        timeframe=timeframe,
                        equity=equity,
                    )
                except Exception:
                    LOGGER.exception("Bucket '%s' symbol '%s' processing failed.", bucket.name, symbol)
                    continue

                if placed:
                    bucket_open_positions += 1
                    total_open_positions += 1

        state = self.storage.load_state()
        state.update(
            {
                "last_run_utc": datetime.now(timezone.utc).isoformat(),
                "dry_run": self.dry_run,
                "open_positions": len(self.gateway.positions_get()),
                "equity": equity,
            }
        )
        self.storage.save_state(state)

    def _process_symbol(
        self,
        bucket: BucketConfig,
        symbol: str,
        strategy: Any,
        timeframe: int,
        equity: float,
    ) -> bool:
        symbol = str(symbol).strip()
        if not symbol:
            return False

        if not self.gateway.ensure_symbol(symbol):
            LOGGER.warning("%s (%s): symbol not available.", symbol, bucket.name)
            return False

        rates = self.gateway.copy_rates(symbol=symbol, timeframe=timeframe, bars=self.bars_per_request)
        if rates is None or rates.empty:
            LOGGER.warning("%s (%s): no rate data.", symbol, bucket.name)
            return False

        signal = strategy.generate(rates)
        side = str(signal.get("side", "HOLD")).upper()
        reason = str(signal.get("reason", "NO_REASON"))
        atr_value = signal.get("atr")
        metrics = signal.get("metrics", {})

        self.storage.append_event(
            {
                "event": "signal",
                "bucket": bucket.name,
                "symbol": symbol,
                "strategy": bucket.strategy,
                "side": side,
                "reason": reason,
                "atr": atr_value,
                "metrics": metrics,
            }
        )

        LOGGER.info(
            "[%s] %s %s -> %s (%s)",
            bucket.name,
            symbol,
            bucket.strategy,
            side,
            reason,
        )

        if side not in {"BUY", "SELL"}:
            return False

        symbol_positions = self.gateway.positions_get(symbol=symbol)
        if symbol_positions:
            if not self.allow_opposite_position:
                LOGGER.info("%s (%s): open position exists, skipping.", symbol, bucket.name)
                return False
            same_side_exists = any(self._position_side(p) == side for p in symbol_positions)
            if same_side_exists:
                LOGGER.info("%s (%s): same-side position exists, skipping duplicate.", symbol, bucket.name)
                return False

        symbol_info = self.gateway.symbol_info(symbol)
        tick = self.gateway.symbol_tick(symbol)
        if symbol_info is None or tick is None:
            LOGGER.warning("%s (%s): missing symbol_info/tick.", symbol, bucket.name)
            return False

        if side == "BUY":
            price = float(getattr(tick, "ask", 0.0) or 0.0)
        else:
            price = float(getattr(tick, "bid", 0.0) or 0.0)
        if price <= 0:
            LOGGER.warning("%s (%s): invalid tick price=%s.", symbol, bucket.name, price)
            return False

        risk_cfg = BucketRiskConfig(
            name=bucket.name,
            risk_per_trade_pct=bucket.risk_per_trade_pct,
            min_lot=bucket.min_lot,
            max_lot=bucket.max_lot,
        )
        plan = self.risk_manager.build_order_plan(
            symbol_info=symbol_info,
            side=side,
            price=price,
            atr=atr_value,
            equity=equity,
            bucket_risk=risk_cfg,
        )
        if plan is None:
            LOGGER.warning("%s (%s): risk plan unavailable.", symbol, bucket.name)
            return False

        filling_mode = int(
            getattr(symbol_info, "filling_mode", int(getattr(mt5, "ORDER_FILLING_IOC", 1) if mt5 else 1))
        )
        request = self._build_order_request(
            symbol=symbol,
            side=side,
            price=price,
            volume=float(plan["volume"]),
            sl=float(plan["sl"]),
            tp=float(plan["tp"]),
            filling_mode=filling_mode,
            bucket_name=bucket.name,
        )

        if self.dry_run:
            LOGGER.info("DRY_RUN order: %s", request)
            self.storage.append_event(
                {
                    "event": "order_dry_run",
                    "bucket": bucket.name,
                    "symbol": symbol,
                    "side": side,
                    "request": request,
                    "risk_plan": plan,
                }
            )
            return True

        result = self.gateway.send_order(request)
        LOGGER.info("%s (%s): order result=%s", symbol, bucket.name, result.get("status"))
        self.storage.append_event(
            {
                "event": "order_result",
                "bucket": bucket.name,
                "symbol": symbol,
                "side": side,
                "request": request,
                "risk_plan": plan,
                "result": result,
            }
        )
        return bool(result.get("ok"))
