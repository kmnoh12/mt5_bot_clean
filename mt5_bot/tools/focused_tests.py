from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

SUITES: dict[str, list[str]] = {
    "no-pandas": [
        "tests/test_blocker_chain_fixes_no_pandas.py",
        "tests/test_conservative_profile_defaults.py",
        "tests/test_dashboard_settings.py",
    ],
    "lsr": [
        "tests/test_lsr_strategy.py",
        "tests/test_lsr_confirmation_quality.py",
        "tests/test_lsr_tick_sweep_reclaim.py",
    ],
    "runtime": [
        "tests/test_runtime_entry_skip_state.py",
        "tests/test_runtime_quality_first_no_trade_day.py",
        "tests/test_runtime_daily_reference_levels.py",
        "tests/test_runtime_manual_protection.py",
        "tests/test_runtime_no_exit_spam.py",
        "tests/test_runtime_v4_opportunity_integration.py",
        "tests/test_runtime_winner_profile_update.py",
    ],
    "postmortem": [
        "tests/test_trade_postmortem_analysis.py",
        "tests/test_trade_postmortem_learning.py",
        "tests/test_trade_ledger_normalization.py",
        "tests/test_summarize_recent_events.py",
    ],
    "learning": [
        "tests/test_trade_postmortem_learning.py",
        "tests/test_auto_tuning_loop.py",
        "tests/test_runtime_winner_profile_update.py",
        "tests/test_active_opportunity_report.py",
    ],
    "order": [
        "tests/test_order_manager.py",
        "tests/test_mt5_request_guard.py",
        "tests/test_order_permission_state.py",
        "tests/test_stop_tp_clamp.py",
        "tests/test_initial_exit_planner.py",
        "tests/test_exit_retry_backoff.py",
    ],
    "risk": [
        "tests/test_risk_engine.py",
        "tests/test_fee_aware_risk_model.py",
        "tests/test_dynamic_lot_risk_manager.py",
        "tests/test_net_risk_position_sizer.py",
        "tests/test_daily_bleed_guard.py",
        "tests/test_cost_edge_guard.py",
    ],
    "order-risk": [
        "tests/test_order_manager.py",
        "tests/test_mt5_request_guard.py",
        "tests/test_order_permission_state.py",
        "tests/test_risk_engine.py",
        "tests/test_fee_aware_risk_model.py",
        "tests/test_dynamic_lot_risk_manager.py",
        "tests/test_net_risk_position_sizer.py",
    ],
}

ALIASES = {
    "orders": "order",
    "order/risk": "order-risk",
    "risk-order": "order-risk",
}


def existing_files(paths: list[str]) -> list[str]:
    return [p for p in paths if (ROOT / p).exists()]


def suite_paths(names: list[str]) -> list[str]:
    selected: list[str] = []
    seen: set[str] = set()
    for raw in names:
        name = ALIASES.get(raw, raw)
        if name not in SUITES:
            raise SystemExit(f"unknown suite: {raw}. Use --list to see available suites.")
        for path in existing_files(SUITES[name]):
            if path not in seen:
                selected.append(path)
                seen.add(path)
    return selected


def print_suites() -> None:
    print("Available focused test suites:")
    for name in sorted(SUITES):
        paths = existing_files(SUITES[name])
        print(f"- {name}: {len(paths)} files")
        for path in paths:
            print(f"  {path}")
    print("\nAliases: " + ", ".join(f"{k}={v}" for k, v in sorted(ALIASES.items())))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run dependency-light focused pytest suites by area. "
            "This script only invokes the current Python interpreter; it never installs packages."
        )
    )
    parser.add_argument("areas", nargs="*", help="Suite names, e.g. no-pandas lsr runtime order/risk")
    parser.add_argument("--list", action="store_true", help="List suite mappings and exit")
    parser.add_argument("--dry-run", action="store_true", help="Print the pytest command without running it")
    parser.add_argument(
        "--pytest-arg",
        action="append",
        default=[],
        help="Extra argument passed to pytest. Repeat for multiple args, e.g. --pytest-arg=-q",
    )
    args = parser.parse_args(argv)

    if args.list:
        print_suites()
        return 0
    if not args.areas:
        parser.error("provide at least one area or use --list")

    paths = suite_paths(args.areas)
    if not paths:
        raise SystemExit("selected suites have no existing test files")
    cmd = [sys.executable, "-m", "pytest", *args.pytest_arg, *paths]
    print("Command:")
    print(" ".join(cmd))
    if args.dry_run:
        return 0
    return subprocess.run(cmd, cwd=ROOT, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
