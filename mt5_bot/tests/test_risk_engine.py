import math
import sys
import unittest
from pathlib import Path

TEST_ROOT = Path(__file__).resolve().parents[1]
if str(TEST_ROOT) not in sys.path:
    sys.path.insert(0, str(TEST_ROOT))

from core.models import SymbolConstraints
from execution import risk_manager
from execution.risk_engine import RiskEngine

class RiskEngineTests(unittest.TestCase):
    def test_legacy_risk_engine_import_is_compatible(self) -> None:
        self.assertIs(RiskEngine, risk_manager.RiskEngine)

    def test_plan_entry_volume_quantized(self) -> None:
        engine = RiskEngine(
            {
                "risk_per_trade_pct": 0.015,
                "daily_loss_limit_pct": 0.06,
                "session_loss_limit_pct": 0.12,
                "max_consecutive_losses": 5,
            }
        )
        constraints = SymbolConstraints(min_volume=0.01, max_volume=1.0, volume_step=0.01, contract_size=1.0)

        volume, err, meta = engine.plan_entry_volume(
            constraints=constraints,
            equity=1000.0,
            entry_price=100.0,
            sl_price=99.0,
            requested_volume=0.2,
            volume_scale=1.0,
        )

        self.assertIsNone(err)
        self.assertIsNotNone(volume)
        self.assertAlmostEqual(volume, 0.2)
        self.assertGreater(meta.get("risk_amount", 0.0), 0.0)

    def test_plan_entry_volume_scales_with_kelly_drawdown_and_streak(self) -> None:
        engine = RiskEngine(
            {
                "risk_per_trade_pct": 0.02,
                "max_risk_per_trade_pct": 0.08,
                "daily_loss_limit_pct": 0.90,
                "session_loss_limit_pct": 0.90,
                "max_consecutive_losses": 8,
            }
        )
        constraints = SymbolConstraints(min_volume=0.01, max_volume=100.0, volume_step=0.01, contract_size=1.0)

        engine.sync_account({"equity": 1000.0})
        engine.on_trade_close(-1.0)
        engine.on_trade_close(-1.0)

        volume, err, meta = engine.plan_entry_volume(
            constraints=constraints,
            equity=900.0,
            entry_price=100.0,
            sl_price=99.0,
            requested_volume=100.0,
            volume_scale=1.0,
            win_probability=0.60,
            payoff_ratio=1.0,
        )

        self.assertIsNone(err)
        self.assertAlmostEqual(float(volume or 0.0), 61.2, places=2)
        self.assertAlmostEqual(float(meta.get("kelly_multiplier", 0.0)), 4.0, places=6)
        self.assertAlmostEqual(float(meta.get("dd_multiplier", 0.0)), 1.0, places=6)
        self.assertAlmostEqual(float(meta.get("streak_multiplier", 0.0)), 0.85, places=6)
        self.assertAlmostEqual(float(meta.get("risk_pct_effective", 0.0)), 0.068, places=6)

    def test_effective_risk_is_capped_by_max_risk_per_trade_pct(self) -> None:
        engine = RiskEngine(
            {
                "risk_per_trade_pct": 0.05,
                "max_risk_per_trade_pct": 0.05,
                "daily_loss_limit_pct": 0.90,
                "session_loss_limit_pct": 0.90,
            }
        )
        constraints = SymbolConstraints(min_volume=0.01, max_volume=100.0, volume_step=0.01, contract_size=1.0)
        volume, err, meta = engine.plan_entry_volume(
            constraints=constraints,
            equity=1000.0,
            entry_price=100.0,
            sl_price=99.0,
            requested_volume=100.0,
            volume_scale=1.0,
        )
        self.assertIsNone(err)
        self.assertIsNotNone(volume)
        self.assertLessEqual(float(meta.get("risk_pct_effective", 1.0)), 0.05 + 1e-12)

    def test_dynamic_risk_cap_uses_recent_24h_winrate(self) -> None:
        engine = RiskEngine(
            {
                "risk_per_trade_pct": 0.007,
                "dynamic_risk_cap_enabled": True,
                "dynamic_risk_min_pct": 0.003,
                "dynamic_risk_max_pct": 0.012,
                "dynamic_risk_lookback_hours": 24,
                "dynamic_risk_min_samples": 3,
                "dynamic_risk_winrate_floor": 0.0,
                "dynamic_risk_winrate_ceiling": 1.0,
            }
        )
        constraints = SymbolConstraints(min_volume=0.01, max_volume=100.0, volume_step=0.01, contract_size=1.0)
        engine.on_trade_close(+1.0, symbol="GOLD")
        engine.on_trade_close(-1.0, symbol="GOLD")
        engine.on_trade_close(-1.0, symbol="GOLD")

        volume, err, meta = engine.plan_entry_volume(
            constraints=constraints,
            equity=1000.0,
            entry_price=100.0,
            sl_price=99.0,
            requested_volume=100.0,
            volume_scale=1.0,
            symbol="GOLD",
        )

        self.assertIsNone(err)
        self.assertIsNotNone(volume)
        self.assertAlmostEqual(float(meta.get("risk_pct_base", 0.0)), 0.006, places=6)

    def test_dynamic_risk_cap_falls_back_to_static_when_sample_small(self) -> None:
        engine = RiskEngine(
            {
                "risk_per_trade_pct": 0.007,
                "dynamic_risk_cap_enabled": True,
                "dynamic_risk_min_pct": 0.003,
                "dynamic_risk_max_pct": 0.012,
                "dynamic_risk_lookback_hours": 24,
                "dynamic_risk_min_samples": 5,
                "dynamic_risk_winrate_floor": 0.0,
                "dynamic_risk_winrate_ceiling": 1.0,
            }
        )
        constraints = SymbolConstraints(min_volume=0.01, max_volume=100.0, volume_step=0.01, contract_size=1.0)
        engine.on_trade_close(+1.0, symbol="GOLD")
        engine.on_trade_close(-1.0, symbol="GOLD")

        volume, err, meta = engine.plan_entry_volume(
            constraints=constraints,
            equity=1000.0,
            entry_price=100.0,
            sl_price=99.0,
            requested_volume=100.0,
            volume_scale=1.0,
            symbol="GOLD",
        )

        self.assertIsNone(err)
        self.assertIsNotNone(volume)
        self.assertAlmostEqual(float(meta.get("risk_pct_base", 0.0)), 0.007, places=6)

    def test_session_drawdown_halts(self) -> None:
        engine = RiskEngine(
            {
                "risk_per_trade_pct": 0.015,
                "daily_loss_limit_pct": 0.06,
                "session_loss_limit_pct": 0.12,
                "max_consecutive_losses": 5,
            }
        )
        self.assertIsNone(engine.evaluate_limits({"equity": 100.0}))
        reason = engine.evaluate_limits({"equity": 85.0})
        self.assertIsNotNone(reason)
        self.assertTrue(engine.status().halted)

    def test_drawdown_hard_stop_halts(self) -> None:
        engine = RiskEngine(
            {
                "risk_per_trade_pct": 0.015,
                "daily_loss_limit_pct": 0.90,
                "session_loss_limit_pct": 0.90,
                "max_consecutive_losses": 5,
            }
        )
        self.assertIsNone(engine.evaluate_limits({"equity": 100.0}))
        reason = engine.evaluate_limits({"equity": 69.0})
        self.assertIsNotNone(reason)
        self.assertTrue(str(reason).startswith("EQUITY_DRAWDOWN_HARD_STOP_"))
        self.assertTrue(engine.status().halted)

    def test_plan_entry_volume_blocks_on_drawdown_hard_stop(self) -> None:
        engine = RiskEngine({"risk_per_trade_pct": 0.015})
        constraints = SymbolConstraints(min_volume=0.01, max_volume=10.0, volume_step=0.01, contract_size=1.0)
        engine.sync_account({"equity": 1000.0})

        volume, err, meta = engine.plan_entry_volume(
            constraints=constraints,
            equity=690.0,
            entry_price=100.0,
            sl_price=99.0,
            requested_volume=1.0,
            volume_scale=1.0,
        )

        self.assertIsNone(volume)
        self.assertEqual(err, "DRAWDOWN_HARD_STOP")
        self.assertGreaterEqual(float(meta.get("drawdown_pct", 0.0)), 0.30)

    def test_min_volume_exceeds_risk_limit_fails(self) -> None:
        engine = RiskEngine({"risk_per_trade_pct": 0.015})
        constraints = SymbolConstraints(min_volume=0.01, max_volume=10.0, volume_step=0.01, contract_size=5000.0)

        volume, err, meta = engine.plan_entry_volume(
            constraints=constraints,
            equity=100.0,
            entry_price=85.7,
            sl_price=85.6,
            requested_volume=1.0,
            volume_scale=1.0,
        )

        self.assertIsNone(volume)
        self.assertEqual(err, "MIN_VOLUME_EXCEEDS_RISK_LIMIT")
        self.assertGreater(float(meta.get("min_required_risk_amount", 0.0)), float(meta.get("risk_amount", 0.0)))

    def test_min_volume_risk_floor_is_used_when_budget_allows(self) -> None:
        engine = RiskEngine({"risk_per_trade_pct": 0.015})
        constraints = SymbolConstraints(min_volume=0.02, max_volume=10.0, volume_step=0.01, contract_size=1.0)

        volume, err, meta = engine.plan_entry_volume(
            constraints=constraints,
            equity=20000.0,
            entry_price=100.0,
            sl_price=99.0,
            requested_volume=0.01,
            volume_scale=0.5,
        )

        self.assertIsNotNone(volume)
        self.assertIsNone(err)
        self.assertAlmostEqual(volume, 0.02)
        self.assertEqual(meta.get("volume_source"), "min_volume_risk_floor")

    def test_plan_entry_volume_uses_fx_rate(self) -> None:
        engine = RiskEngine({"risk_per_trade_pct": 0.015})
        constraints = SymbolConstraints(min_volume=0.01, max_volume=10.0, volume_step=0.01, contract_size=5000.0)

        volume, err, meta = engine.plan_entry_volume(
            constraints=constraints,
            equity=10_000_000.0,  # KRW
            entry_price=85.7,
            sl_price=85.2,
            requested_volume=1.0,
            volume_scale=1.0,
            quote_to_account_rate=1350.0,  # USD -> KRW
            require_fx_rate=True,
        )

        self.assertIsNone(err)
        self.assertEqual(volume, 0.07)
        self.assertAlmostEqual(float(meta.get("risk_amount_quote", 0.0)), 166.6666666667, places=6)

    def test_plan_entry_volume_requires_fx_rate_when_requested(self) -> None:
        engine = RiskEngine({"risk_per_trade_pct": 0.015})
        constraints = SymbolConstraints(min_volume=0.01, max_volume=10.0, volume_step=0.01, contract_size=5000.0)

        volume, err, _ = engine.plan_entry_volume(
            constraints=constraints,
            equity=10_000_000.0,
            entry_price=85.7,
            sl_price=85.2,
            requested_volume=1.0,
            volume_scale=1.0,
            quote_to_account_rate=None,
            require_fx_rate=True,
        )

        self.assertIsNone(volume)
        self.assertEqual(err, "MISSING_FX_RATE")

    def test_malformed_config_and_snapshot_do_not_raise(self) -> None:
        engine = RiskEngine(
            config={
                "risk_per_trade_pct": "oops",
                "max_risk_per_trade_pct": float("nan"),
                "daily_loss_limit_pct": "bad",
                "session_loss_limit_pct": float("inf"),
                "max_consecutive_losses": "wrong",
                "kelly_fraction": "not-a-number",
                "dd_hard_stop_pct": object(),
            },
            snapshot={
                "session_start_equity": "n/a",
                "daily_start_equity": float("nan"),
                "consecutive_losses": "broken",
                "equity_peak": float("-inf"),
            },
        )

        self.assertAlmostEqual(engine.risk_per_trade_pct, 0.05)
        self.assertAlmostEqual(engine.max_risk_per_trade_pct, 0.15)
        self.assertAlmostEqual(engine.daily_loss_limit_pct, 0.12)
        self.assertAlmostEqual(engine.session_loss_limit_pct, 0.25)
        self.assertEqual(engine.max_consecutive_losses, 5)
        self.assertEqual(engine.status().consecutive_losses, 0)
        self.assertIsNone(engine.status().session_start_equity)
        self.assertIsNone(engine.status().equity_peak)

    def test_plan_entry_volume_invalid_constraints_returns_invalid_constraints_or_scale(self) -> None:
        engine = RiskEngine({"risk_per_trade_pct": 0.015})
        constraints = SymbolConstraints(
            min_volume=0.01,
            max_volume=1.0,
            volume_step=0.01,
            contract_size=float("nan"),
        )

        volume, err, meta = engine.plan_entry_volume(
            constraints=constraints,
            equity=1000.0,
            entry_price=100.0,
            sl_price=99.0,
            requested_volume=0.2,
            volume_scale=1.0,
        )

        self.assertIsNone(volume)
        self.assertEqual(err, "INVALID_CONSTRAINTS_OR_SCALE")
        self.assertEqual(float(meta.get("constraints_valid", 1.0)), 0.0)

    def test_plan_entry_volume_invalid_scale_returns_invalid_constraints_or_scale(self) -> None:
        engine = RiskEngine({"risk_per_trade_pct": 0.015})
        constraints = SymbolConstraints(min_volume=0.01, max_volume=1.0, volume_step=0.01, contract_size=1.0)

        volume, err, meta = engine.plan_entry_volume(
            constraints=constraints,
            equity=1000.0,
            entry_price=100.0,
            sl_price=99.0,
            requested_volume=0.2,
            volume_scale="bad-scale",
        )

        self.assertIsNone(volume)
        self.assertEqual(err, "INVALID_CONSTRAINTS_OR_SCALE")
        self.assertEqual(float(meta.get("scale_valid", 1.0)), 0.0)

    def test_quantize_volume_never_throws_on_invalid_constraints(self) -> None:
        constraints = SymbolConstraints(
            min_volume="bad",
            max_volume="bad",
            volume_step="bad",
            contract_size="bad",
        )

        volume = RiskEngine._quantize_volume(raw_volume=float("nan"), constraints=constraints)
        self.assertTrue(math.isfinite(volume))
        self.assertGreaterEqual(volume, 0.0)

    def test_consecutive_losses_halts(self) -> None:
        engine = RiskEngine({"max_consecutive_losses": 3})
        engine.on_trade_close(-1.0)
        engine.on_trade_close(-0.5)
        self.assertFalse(engine.status().halted)
        engine.on_trade_close(-2.0)
        self.assertTrue(engine.status().halted)


if __name__ == "__main__":
    unittest.main()
