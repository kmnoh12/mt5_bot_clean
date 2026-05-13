from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping


@dataclass(frozen=True)
class LeverageMarginInput:
    symbol: str
    entry_price: float
    lot: float
    contract_size: float
    account_equity: float
    account_leverage: float = 500.0


def position_notional(entry_price: float, lot: float, contract_size: float) -> float:
    return abs(float(entry_price) * float(lot) * float(contract_size))


def effective_leverage(entry_price: float, lot: float, contract_size: float, account_equity: float) -> float:
    equity = max(float(account_equity), 1e-12)
    return position_notional(entry_price, lot, contract_size) / equity


def required_margin(entry_price: float, lot: float, contract_size: float, account_leverage: float) -> float:
    leverage = max(float(account_leverage), 1e-12)
    return position_notional(entry_price, lot, contract_size) / leverage


def margin_used_pct(entry_price: float, lot: float, contract_size: float, account_equity: float, account_leverage: float) -> float:
    equity = max(float(account_equity), 1e-12)
    return required_margin(entry_price, lot, contract_size, account_leverage) / equity * 100.0


def analyze_trades(
    trades: Iterable[Mapping[str, Any]],
    *,
    account_equity: float,
    account_leverage: float = 500.0,
    contract_size: float = 1.0,
) -> Dict[str, Any]:
    rows = []
    for trade in trades:
        entry = float(trade.get("entry_price", 0.0) or 0.0)
        lot = float(trade.get("lot", 0.0) or 0.0)
        symbol = str(trade.get("symbol", ""))
        eff = effective_leverage(entry, lot, contract_size, account_equity)
        margin_pct = margin_used_pct(entry, lot, contract_size, account_equity, account_leverage)
        rows.append(
            {
                "symbol": symbol,
                "entry_price": entry,
                "lot": lot,
                "notional": position_notional(entry, lot, contract_size),
                "effective_leverage": eff,
                "margin_used_pct": margin_pct,
            }
        )
    return {
        "account_equity": float(account_equity),
        "account_leverage": float(account_leverage),
        "trade_count": len(rows),
        "effective_leverage_max": max((row["effective_leverage"] for row in rows), default=0.0),
        "margin_used_pct_max": max((row["margin_used_pct"] for row in rows), default=0.0),
        "trades": rows,
    }


def config_margin_verdict(metrics: Mapping[str, Any], config: Mapping[str, Any]) -> Dict[str, Any]:
    risk = dict(config.get("risk", {}) or {})
    max_eff = float(risk.get("max_effective_leverage", 0.0) or 0.0)
    max_margin = float(risk.get("max_margin_used_pct", 0.0) or 0.0)
    eff = float(metrics.get("effective_leverage_max", 0.0) or 0.0)
    margin = float(metrics.get("margin_used_pct_max", 0.0) or 0.0)
    return {
        "max_effective_leverage": max_eff,
        "observed_effective_leverage_max": eff,
        "effective_leverage_pass": eff <= max_eff + 1e-12,
        "max_margin_used_pct": max_margin,
        "observed_margin_used_pct_max": margin,
        "margin_used_pct_pass": margin <= max_margin + 1e-12,
    }

