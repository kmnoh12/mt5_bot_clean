from core.optimization.leverage_margin import analyze_trades, effective_leverage, margin_used_pct, required_margin


def test_leverage_margin_formulas() -> None:
    assert effective_leverage(100.0, 0.5, 2.0, 1000.0) == 0.1
    assert required_margin(100.0, 0.5, 2.0, 10.0) == 10.0
    assert margin_used_pct(100.0, 0.5, 2.0, 1000.0, 10.0) == 1.0


def test_analyze_trades_reports_maxima() -> None:
    out = analyze_trades(
        [{"symbol": "BTCUSD", "entry_price": 100.0, "lot": 1.0}, {"symbol": "BTCUSD", "entry_price": 200.0, "lot": 1.0}],
        account_equity=100.0,
        account_leverage=100.0,
        contract_size=1.0,
    )

    assert out["effective_leverage_max"] == 2.0
    assert out["margin_used_pct_max"] == 2.0

