import sys
from pathlib import Path

import pytest

TEST_ROOT = Path(__file__).resolve().parents[1]
if str(TEST_ROOT) not in sys.path:
    sys.path.insert(0, str(TEST_ROOT))

from execution.position_sizer import NetRiskPositionSizeInput, NetRiskPositionSizer, SymbolVolumeSpec


def _spec(**overrides):
    values = {
        "volume_min": 0.01,
        "volume_step": 0.01,
        "volume_max": 10.0,
        "tick_size": 0.01,
        "tick_value": 1.0,
        "contract_size": 100.0,
    }
    values.update(overrides)
    return SymbolVolumeSpec(**values)


def _size(**overrides):
    values = {
        "symbol": "XAUUSD",
        "target_net_loss_usd": 11.0,
        "hard_max_net_loss_usd": 12.0,
        "entry_price": 100.0,
        "stop_price": 99.0,
        "direction": "long",
        "symbol_spec": _spec(),
        "spread": 0.02,
        "commission_per_lot": 3.0,
        "expected_slippage_points": 2.0,
    }
    values.update(overrides)
    return NetRiskPositionSizer().size(NetRiskPositionSizeInput(**values))


def test_sizes_long_lot_to_target_net_loss():
    result = _size()

    assert result.passed is True
    assert result.recommended_lot == pytest.approx(0.1)
    assert result.estimated_net_loss == pytest.approx(11.0)


def test_sizes_short_lot_to_target_net_loss():
    result = _size(direction="short", stop_price=101.0)

    assert result.passed is True
    assert result.recommended_lot == pytest.approx(0.1)
    assert result.estimated_net_loss == pytest.approx(11.0)


def test_rounds_to_nearest_volume_step_when_under_hard_max():
    result = _size(target_net_loss_usd=11.7, hard_max_net_loss_usd=13.0)

    assert result.passed is True
    assert result.recommended_lot == pytest.approx(0.11)
    assert result.estimated_net_loss == pytest.approx(12.1)


def test_revalidates_and_floors_when_nearest_rounding_would_exceed_hard_max():
    result = _size(target_net_loss_usd=11.55, hard_max_net_loss_usd=11.8)

    assert result.passed is True
    assert result.recommended_lot == pytest.approx(0.1)
    assert result.estimated_net_loss == pytest.approx(11.0)


def test_min_lot_risk_exceeds_hard_max_fails_with_required_reason():
    result = _size(
        target_net_loss_usd=1.0,
        hard_max_net_loss_usd=5.0,
        symbol_spec=_spec(volume_min=0.1, volume_step=0.01),
    )

    assert result.passed is False
    assert result.recommended_lot is None
    assert result.failure_reason == "min_lot_risk_exceeds_hard_max"


def test_target_below_min_lot_uses_min_when_hard_max_allows():
    result = _size(target_net_loss_usd=0.5, hard_max_net_loss_usd=2.0)

    assert result.passed is True
    assert result.recommended_lot == pytest.approx(0.01)
    assert result.estimated_net_loss == pytest.approx(1.1)


def test_respects_volume_max_when_target_is_large():
    result = _size(target_net_loss_usd=1_000.0, hard_max_net_loss_usd=2_000.0, symbol_spec=_spec(volume_max=0.5))

    assert result.passed is True
    assert result.recommended_lot == pytest.approx(0.5)
    assert result.estimated_net_loss == pytest.approx(55.0)


def test_commission_changes_selected_lot():
    no_commission = _size(target_net_loss_usd=10.0, hard_max_net_loss_usd=10.5, commission_per_lot=0.0)
    with_commission = _size(target_net_loss_usd=10.0, hard_max_net_loss_usd=10.5, commission_per_lot=25.0)

    assert no_commission.passed is True
    assert with_commission.passed is True
    assert no_commission.recommended_lot > with_commission.recommended_lot


def test_spread_changes_estimated_net_loss():
    tight = _size(spread=0.0)
    wide = _size(spread=0.05)

    assert tight.passed is True
    assert wide.passed is True
    assert wide.estimated_net_loss > tight.estimated_net_loss


def test_slippage_changes_estimated_net_loss():
    none = _size(expected_slippage_points=0.0)
    slippage = _size(expected_slippage_points=5.0)

    assert none.passed is True
    assert slippage.passed is True
    assert slippage.estimated_net_loss > none.estimated_net_loss


def test_invalid_volume_step_raises():
    with pytest.raises(ValueError, match="volume_step"):
        _size(symbol_spec=_spec(volume_step=0.0))


def test_invalid_volume_bounds_raises():
    with pytest.raises(ValueError, match="volume_max"):
        _size(symbol_spec=_spec(volume_min=1.0, volume_max=0.5))


def test_invalid_short_stop_raises_from_risk_model():
    with pytest.raises(ValueError, match="stop_price"):
        _size(direction="short", stop_price=99.0)
