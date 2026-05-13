from __future__ import annotations

import json
from typing import Any, Dict


def build_no_trade_report_json(snapshot: Dict[str, Any], *, indent: int = 2) -> str:
    return json.dumps(_plain_snapshot(snapshot), indent=indent, sort_keys=True)


def build_no_trade_report_markdown(snapshot: Dict[str, Any]) -> str:
    data = _plain_snapshot(snapshot)
    lines = [
        "# No-Trade Bias Report",
        "",
        f"- Status: {data.get('status', 'unknown')}",
        f"- Raw signals: {int(data.get('raw_signal_count', 0))}",
        f"- Scored signals: {int(data.get('scored_signal_count', 0))}",
        f"- Eligible signals: {int(data.get('eligible_signal_count', 0))}",
        f"- Executed trades: {int(data.get('executed_trade_count', 0))}",
        f"- No-trade hours: {float(data.get('no_trade_hours', 0.0)):.2f}",
        f"- No-trade days: {int(data.get('no_trade_days_count', 0))}",
        f"- Zero trade success: {bool(data.get('zero_trade_success', False))}",
        "",
        "## Warnings",
    ]
    warnings = data.get("warnings") or []
    lines.extend([f"- {item}" for item in warnings] or ["- none"])
    lines.extend(["", "## Failures"])
    failures = data.get("failures") or []
    lines.extend([f"- {item}" for item in failures] or ["- none"])
    lines.extend(["", "## Block Rate By Reason"])
    block_rates = data.get("block_rate_by_reason") or {}
    if block_rates:
        for reason, rate in sorted(block_rates.items()):
            lines.append(f"- {reason}: {float(rate):.2%}")
    else:
        lines.append("- none")

    lines.extend(["", "## Top Rejected Opportunities"])
    rejected = data.get("top_rejected_opportunities") or []
    if rejected:
        for item in rejected:
            lines.append(
                "- {id} {symbol} score={score:.3f} rr={rr:.3f} reason={reason}".format(
                    id=item.get("opportunity_id", ""),
                    symbol=item.get("symbol", ""),
                    score=float(item.get("score", 0.0)),
                    rr=float(item.get("fee_adjusted_rr", 0.0)),
                    reason=item.get("reason", ""),
                ).strip()
            )
    else:
        lines.append("- none")

    lines.extend(["", "## Best Missed Opportunity"])
    best = data.get("best_missed_opportunity")
    if best:
        lines.append(
            "- {id} {symbol} score={score:.3f} rr={rr:.3f} reason={reason}".format(
                id=best.get("opportunity_id", ""),
                symbol=best.get("symbol", ""),
                score=float(best.get("score", 0.0)),
                rr=float(best.get("fee_adjusted_rr", 0.0)),
                reason=best.get("reason", ""),
            ).strip()
        )
    else:
        lines.append("- none")
    return "\n".join(lines) + "\n"


def _plain_snapshot(snapshot: Any) -> Dict[str, Any]:
    if hasattr(snapshot, "to_dict"):
        snapshot = snapshot.to_dict()
    return dict(snapshot or {})
