"""Pluggable TARC scoring rules (§4.3 step 4)."""

from __future__ import annotations

from typing import Protocol


class ScoringRule(Protocol):
    def score(
        self,
        predicted_apogee_ft: float,
        target_apogee_ft: float,
        predicted_time_s: float,
        target_time_s: float,
    ) -> float: ...


def load_scoring_rule(version: str = "tarc_2026") -> ScoringRule:
    if version in ("tarc_2026", "tarc_default", "default"):
        from optimizer.scoring_rules.tarc_default import TarcDefaultScoring

        return TarcDefaultScoring()
    raise ValueError(f"Unknown scoring rule version: {version}")
