from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class TrialResult:
    trial_id: int
    seed: int
    config: Dict[str, Any]
    metrics: Dict[str, Any]
    train_metrics: Dict[str, Any] = field(default_factory=dict)
    oos_metrics: Dict[str, Any] = field(default_factory=dict)
    symbol_metrics: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    rejected: bool = False
    reject_reasons: List[str] = field(default_factory=list)
    robust_score: float = 0.0
    rank_bucket: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trial_id": int(self.trial_id),
            "seed": int(self.seed),
            "config": dict(self.config),
            "metrics": dict(self.metrics),
            "train_metrics": dict(self.train_metrics),
            "oos_metrics": dict(self.oos_metrics),
            "symbol_metrics": {str(k): dict(v) for k, v in self.symbol_metrics.items()},
            "rejected": bool(self.rejected),
            "reject_reasons": list(self.reject_reasons),
            "robust_score": float(self.robust_score),
            "rank_bucket": self.rank_bucket,
        }


def flatten_trial_for_csv(trial: TrialResult) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "trial_id": trial.trial_id,
        "seed": trial.seed,
        "rejected": trial.rejected,
        "reject_reasons": "|".join(trial.reject_reasons),
        "robust_score": trial.robust_score,
    }
    for prefix, values in (("config", trial.config), ("metrics", trial.metrics), ("oos", trial.oos_metrics)):
        for key, value in values.items():
            if isinstance(value, (dict, list, tuple)):
                continue
            row[f"{prefix}.{key}"] = value
    return row

