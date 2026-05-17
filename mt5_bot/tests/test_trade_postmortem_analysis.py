from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from tools.trade_postmortem import (
    analyze_trades,
    compute_features,
    extract_closed_trades,
    generate_chart,
    load_bars_from_files,
)


def _fixture_events() -> list[dict]:
    return [
        {
            "event": "decision",
            "symbol": "BTCUSD",
            "strategy": "liquidity_sweep_reversal",
            "action": "SELL",
            "reason": "LSR_SELL_ENTRY",
            "state": "ENTRY_PENDING",
            "metadata": {
                "entry_style": "liquidity_sweep_reversal",
                "signal_close": 101.0,
                "risk_per_unit": 10.0,
                "sweep_level": 99.0,
                "sweep_extreme": 105.0,
                "sweep_event_key": "SELL|2099-05-14T16:00:00+00:00",
                "time_from_sweep_to_reclaim_sec": 45.0,
                "reclaim_quality": {
                    "confirmation_path": "retest",
                    "retest_confirmed": True,
                    "time_from_sweep_to_reclaim_sec": 45.0,
                    "reclaim_distance_atr": 0.25,
                    "sweep_depth_atr": 1.5,
                    "reclaim_to_sweep_depth_ratio": 0.1667,
                },
                "stage_a_target": 90.0,
                "adx_entry": 50.0,
                "displacement_ratio": 2.5,
                "estimated_net_loss": 1.0,
                "estimated_net_profit_at_tp": 3.0,
                "fee_adjusted_rr": 3.0,
            },
            "ts_utc": "2026-05-14T16:10:00+00:00",
        },
        {
            "event": "entry_quality_score",
            "symbol": "BTCUSD",
            "strategy": "liquidity_sweep_reversal",
            "score": 0.2,
            "threshold": 0.2,
            "allow": True,
            "risk_mode": "neutral",
            "features": {"m5_align": 1.0},
            "ts_utc": "2026-05-14T16:10:01+00:00",
        },
        {
            "event": "order_submit",
            "symbol": "BTCUSD",
            "strategy": "liquidity_sweep_reversal",
            "action": "SELL",
            "reason": "LSR_SELL_ENTRY",
            "intent": {
                "symbol": "BTCUSD",
                "side": "SELL",
                "volume": 0.01,
                "sl": 110.0,
                "tp": 70.0,
                "metadata": {
                    "risk_per_unit": 10.0,
                    "sweep_level": 99.0,
                    "sweep_extreme": 105.0,
                    "displacement_ratio": 2.5,
                    "estimated_net_loss": 1.0,
                    "fee_adjusted_rr": 3.0,
                    "risk_model": {"estimated_cost_usd": 0.2},
                },
            },
            "result": {
                "ok": True,
                "status": "FILLED",
                "ticket": 123,
                "filled_price": 100.0,
                "raw": {"deal": 456, "order": 123, "price": 100.0},
            },
            "ts_utc": "2026-05-14T16:11:00+00:00",
        },
        {
            "event": "decision",
            "symbol": "BTCUSD",
            "strategy": "liquidity_sweep_reversal",
            "state": "IN_POSITION",
            "metadata": {"move_rr": -0.5},
            "ts_utc": "2026-05-14T16:12:00+00:00",
        },
        {
            "event": "position_exit",
            "symbol": "BTCUSD",
            "strategy": "liquidity_sweep_reversal",
            "reason": "BROKER_AUTO_CLOSE:4",
            "result": {
                "ok": True,
                "status": "CLOSED_BROKER",
                "ticket": 123,
                "filled_price": 112.0,
                "pnl": -1.3,
            },
            "ts_utc": "2026-05-14T16:13:00+00:00",
        },
        {
            "event": "trade_ledger_normalized",
            "ticket": 123,
            "symbol": "BTCUSD",
            "strategy": "liquidity_sweep_reversal",
            "side": "SELL",
            "entry_price": 100.0,
            "exit_price": 112.0,
            "volume": 0.01,
            "realized_pnl": -1.3,
            "swap": 0.0,
            "commission": -0.08,
            "fee": -0.02,
            "reason": "BROKER_AUTO_CLOSE:4",
            "ts_utc": "2026-05-14T16:13:00+00:00",
        },
    ]


class TradePostmortemAnalysisTests(unittest.TestCase):
    def test_event_parser_pairs_filled_entry_with_closed_trade(self) -> None:
        trades = extract_closed_trades(_fixture_events(), symbol="BTCUSD")
        self.assertEqual(len(trades), 1)
        trade = trades[0]
        self.assertEqual(trade.ticket, 123)
        self.assertEqual(trade.entry_deal, 456)
        self.assertEqual(trade.side, "SELL")
        self.assertIn("BTCUSD", trade.trade_key)
        self.assertEqual(trade.quality_event["score"], 0.2)

    def test_feature_computation_flags_trash_timing(self) -> None:
        trade = extract_closed_trades(_fixture_events(), symbol="BTCUSD")[0]
        bars = {
            "M1": [
                {"time": "2026-05-14T16:09:00+00:00", "open": 95, "high": 96, "low": 94, "close": 96, "spread": 9},
                {"time": "2026-05-14T16:10:00+00:00", "open": 96, "high": 101, "low": 95, "close": 101, "spread": 12},
                {"time": "2026-05-14T16:11:00+00:00", "open": 100, "high": 103, "low": 99, "close": 102, "spread": 18},
                {"time": "2026-05-14T16:12:00+00:00", "open": 102, "high": 112, "low": 101, "close": 111, "spread": 25},
            ]
        }
        features = compute_features(trade, bars)
        self.assertTrue(features["entry_chased_extension"])
        self.assertTrue(features["entered_into_exhaustion"])
        self.assertTrue(features["entered_against_short_term_momentum"])
        self.assertAlmostEqual(features["price_r_multiple"], -1.2)
        self.assertAlmostEqual(features["pnl_r_multiple"], -1.3)
        self.assertAlmostEqual(features["entry_implementation_shortfall_price"], 1.0)
        self.assertAlmostEqual(features["entry_implementation_shortfall_r"], 0.1)
        self.assertAlmostEqual(features["net_execution_drag_r"], 0.1)
        self.assertAlmostEqual(features["estimated_cost_usd"], 0.2)
        self.assertAlmostEqual(features["estimated_cost_to_expected_loss_r"], 0.2)
        self.assertAlmostEqual(features["realized_explicit_cost_usd"], 0.1)
        self.assertAlmostEqual(features["realized_explicit_cost_r"], 0.1)
        self.assertEqual(features["spread_points"], 18)
        self.assertEqual(features["time_from_sweep_to_entry_sec"], 45.0)
        self.assertEqual(features["confirmation_path"], "retest")
        self.assertTrue(features["clean_reclaim"])
        self.assertTrue(features["clean_reclaim_confirmed"])
        self.assertGreater(features["lsr_confirmation_score"], 0.6)
        self.assertEqual(features["lsr_confirmation_band"], "mixed")
        self.assertEqual(features["reclaim_distance_atr"], 0.25)
        self.assertEqual(features["sweep_depth_atr"], 1.5)
        self.assertEqual(features["reclaim_to_sweep_depth_ratio"], 0.1667)
        self.assertEqual(features["bar_analysis_source_timeframe"], "M1")
        self.assertEqual(features["bars_m1_count"], 4)
        self.assertIn(features["last_swing_direction"], {"UP", "DOWN", "FLAT", None})

    def test_unconfirmed_lsr_reclaim_chase_is_flagged_review_only(self) -> None:
        events = _fixture_events()
        events[0]["metadata"]["reclaim_quality"] = {
            "confirmation_path": "reclaim_only",
            "retest_confirmed": False,
            "time_from_sweep_to_reclaim_sec": 15.0,
            "reclaim_distance_atr": 0.2,
            "sweep_depth_atr": 1.0,
            "reclaim_to_sweep_depth_ratio": 0.2,
        }
        events[0]["metadata"]["time_from_sweep_to_reclaim_sec"] = 15.0
        trade = extract_closed_trades(events, symbol="BTCUSD")[0]

        features = compute_features(trade, {})

        self.assertEqual(features["confirmation_path"], "reclaim_only")
        self.assertFalse(features["retest_confirmed"])
        self.assertTrue(features["lsr_unconfirmed_reclaim"])
        self.assertTrue(features["shallow_reclaim_confirmation"])
        self.assertEqual(features["shallow_reclaim_threshold_atr"], 0.25)
        self.assertTrue(features["weak_reclaim_after_deep_sweep"])
        self.assertTrue(features["lsr_unconfirmed_reclaim_chase"])
        self.assertLess(features["lsr_confirmation_score"], 0.4)
        self.assertEqual(features["lsr_confirmation_band"], "weak")

    def test_bar_file_input_parsing_csv_and_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            csv_path = tmp / "bars.csv"
            csv_path.write_text(
                "\n".join(
                    [
                        "time,timeframe,symbol,open,high,low,close,tick_volume",
                        "2026-05-14T16:09:00+00:00,M1,BTCUSD,95,96,94,95.5,10",
                        "2026-05-14T16:10:00+00:00,M1,BTCUSD,95.5,101,95,101,12",
                    ]
                ),
                encoding="utf-8",
            )
            json_path = tmp / "bars.json"
            json_path.write_text(
                json.dumps(
                    {
                        "M5": [
                            {
                                "time": "2026-05-14T16:10:00+00:00",
                                "open": 95,
                                "high": 103,
                                "low": 94,
                                "close": 102,
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            source = load_bars_from_files(
                [str(csv_path)],
                [str(json_path)],
                default_timeframe="M1",
                symbol_filter="BTCUSD",
            )
            self.assertEqual(source.source, "REAL_BARS_FILE")
            self.assertEqual(len(source.bars["M1"]), 2)
            self.assertEqual(len(source.bars["M5"]), 1)
            self.assertEqual(source.bars["M1"][0]["open"], 95.0)

    def test_feature_extraction_from_real_bars(self) -> None:
        trade = extract_closed_trades(_fixture_events(), symbol="BTCUSD")[0]
        bars = {
            "M1": [
                {"time": "2026-05-14T16:06:00+00:00", "open": 90, "high": 92, "low": 89, "close": 91},
                {"time": "2026-05-14T16:07:00+00:00", "open": 91, "high": 94, "low": 90, "close": 93},
                {"time": "2026-05-14T16:08:00+00:00", "open": 93, "high": 97, "low": 92, "close": 96},
                {"time": "2026-05-14T16:09:00+00:00", "open": 96, "high": 100, "low": 95, "close": 99},
                {"time": "2026-05-14T16:10:00+00:00", "open": 99, "high": 102, "low": 98, "close": 101},
                {"time": "2026-05-14T16:11:00+00:00", "open": 100, "high": 104, "low": 99, "close": 103},
                {"time": "2026-05-14T16:12:00+00:00", "open": 103, "high": 106, "low": 97, "close": 98},
                {"time": "2026-05-14T16:13:00+00:00", "open": 98, "high": 113, "low": 97, "close": 112},
            ]
        }
        features = compute_features(trade, bars)
        self.assertEqual(features["recent_high"], 104)
        self.assertEqual(features["recent_low"], 89)
        self.assertAlmostEqual(features["entry_position_in_recent_range"], (100 - 89) / (104 - 89))
        self.assertEqual(features["adverse_excursion_price"], 13)
        self.assertEqual(features["favorable_excursion_price"], 3)
        self.assertIsNotNone(features["ema20"])
        self.assertIsNotNone(features["pullback_depth"])

    def test_chart_render_with_synthetic_ohlc_or_fallback(self) -> None:
        trade = extract_closed_trades(_fixture_events(), symbol="BTCUSD")[0]
        bars = {
            "M1": [
                {"time": "2026-05-14T16:09:00+00:00", "open": 95, "high": 97, "low": 94, "close": 96},
                {"time": "2026-05-14T16:10:00+00:00", "open": 96, "high": 101, "low": 95, "close": 100},
                {"time": "2026-05-14T16:11:00+00:00", "open": 100, "high": 104, "low": 99, "close": 103},
                {"time": "2026-05-14T16:12:00+00:00", "open": 103, "high": 112, "low": 101, "close": 111},
            ]
        }
        features = compute_features(trade, bars)
        with tempfile.TemporaryDirectory() as tmpdir:
            chart_path = Path(tmpdir) / "chart.png"
            warning = generate_chart(trade, bars, features, chart_path)
            self.assertTrue(chart_path.exists())
            self.assertGreater(chart_path.stat().st_size, 100)
            if warning is not None:
                self.assertIn("matplotlib unavailable", warning)

    def test_cli_writes_reports_and_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            events_path = tmp / "events.jsonl"
            with events_path.open("w", encoding="utf-8") as handle:
                for event in _fixture_events():
                    handle.write(json.dumps(event, ensure_ascii=False) + "\n")
            out_dir = tmp / "postmortems"
            args = argparse.Namespace(
                events=str(events_path),
                output_dir=str(out_dir),
                symbol="BTCUSD",
                limit=10,
                trade_key=None,
                force=False,
                mt5=False,
                no_mt5=True,
                bars_csv=[],
                bars_json=[],
                bars_timeframe="M1",
            )
            first = analyze_trades(args)
            self.assertEqual(len(first["analyzed"]), 1)
            trade_key = first["analyzed"][0]["trade_key"]
            self.assertEqual(first["analyzed"][0]["bar_source"], "FALLBACK_EVENT_PATH")
            self.assertTrue((out_dir / f"{trade_key}.json").exists())
            self.assertTrue((out_dir / f"{trade_key}.md").exists())
            self.assertTrue((out_dir / "assets" / f"{trade_key}.png").exists())

            second = analyze_trades(args)
            self.assertEqual(second["analyzed"], [])
            self.assertEqual(second["skipped_indexed"], [trade_key])

            index_rows = (out_dir / "postmortem_index.jsonl").read_text(encoding="utf-8").strip().splitlines()
            learning_rows = (out_dir / "learning_samples.jsonl").read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(index_rows), 1)
            self.assertEqual(len(learning_rows), 1)


if __name__ == "__main__":
    unittest.main()
