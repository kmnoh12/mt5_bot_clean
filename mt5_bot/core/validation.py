from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple


def load_validation_report(path: str) -> Dict:
    report_path = Path(path)
    if not report_path.exists():
        return {}
    try:
        with report_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, dict):
            return payload
    except Exception:
        return {}
    return {}


def check_live_readiness(report_path: str) -> Tuple[bool, str]:
    payload = load_validation_report(report_path)
    if not payload:
        return False, "validation_report_missing"
    if bool(payload.get("oos_pass", False)):
        return True, "oos_pass"
    return False, str(payload.get("reason", "oos_failed"))
