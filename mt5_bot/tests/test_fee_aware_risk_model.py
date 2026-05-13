import sys
from pathlib import Path

import pytest

TEST_ROOT = Path(__file__).resolve().parents[1]
if str(TEST_ROOT) not in sys.path:
    sys.path.insert(0, str(TEST_ROOT))

from core.risk_model import FeeAwareRiskInput, FeeAwareRiskModel


def _estimate(**overrides):
    values = {
        "symbol": "XAUUSD",
        "entry_price": 100.0,
        "stop_price": 99.0,
        "direction": "long",
        "lot": 0.1,
        "spread": 0.02,
        "commission_per_lot": 3.0,
        "expected_slippage_points": 2.0,
        "tick_size": 0.01,
        "tick_value": 1.0,
        "contract_size": 100.0,
    }
    values.update(overrides)
    return FeeAwareRiskModel().estimate(FeeAwareRiskInput(**values))


def test_long_loss_includes_gross_spread_slippage_and_round_turn_commission():
    result = _estimate()

    assert result.estimated_gross_loss_usd == pytest.approx(10.0)
    assert result.estimated_cost_usd == pytest.approx(1.0)
    assert result.estimated_net_loss_usd == pytest.approx(11.0)


def test_short_loss_uses_stop_above_entry():
    result = _estimate(direction="short", stop_price=101.0)

    assert result.estimated_gross_loss_usd == pytest.approx(10.0)
    assert result.estimated_net_loss_usd == pytest.approx(11.0)


def test_buy_alias_matches_long():
    result = _estimate(direction="buy")

    assert result.estimated_net_loss_usd == pytest.approx(11.0)


def test_sell_alias_matches_short():
    result = _estimate(direction="sell", stop_price=101.0)

    assert result.estimated_net_loss_usd == pytest.approx(11.0)


def test_take_profit_returns_net_profit_and_fee_adjusted_rr():
    result = _estimate(take_profit_price=102.0)

    assert result.estimated_net_profit_at_tp_usd == pytest.approx(19.0)
    assert result.fee_adjusted_rr == pytest.approx(19.0 / 11.0)


def test_short_take_profit_returns_net_profit_and_fee_adjusted_rr():
    result = _estimate(direction="short", stop_price=101.0, take_profit_price=98.0)

    assert result.estimated_net_profit_at_tp_usd == pytest.approx(19.0)
    assert result.fee_adjusted_rr == pytest.approx(19.0 / 11.0)


def test_hard_max_loss_passes_when_net_loss_equal_to_limit():
    result = _estimate(hard_max_net_loss_usd=11.0)

    assert result.hard_max_loss_pass is True


def test_hard_max_loss_fails_when_net_loss_exceeds_limit():
    result = _estimate(hard_max_net_loss_usd=10.99)

    assert result.hard_max_loss_pass is False


def test_zero_cost_assumption_returns_gross_as_net():
    result = _estimate(spread=0.0, commission_per_lot=0.0, expected_slippage_points=0.0)

    assert result.estimated_cost_usd == pytest.approx(0.0)
    assert result.estimated_net_loss_usd == pytest.approx(result.estimated_gross_loss_usd)


def test_invalid_stop_direction_raises():
    with pytest.raises(ValueError, match="stop_price"):
        _estimate(stop_price=101.0)


def test_invalid_take_profit_direction_raises():
    with pytest.raises(ValueError, match="take_profit_price"):
        _estimate(take_profit_price=99.5)


def test_invalid_direction_raises():
    with pytest.raises(ValueError, match="unsupported direction"):
        _estimate(direction="flat")
