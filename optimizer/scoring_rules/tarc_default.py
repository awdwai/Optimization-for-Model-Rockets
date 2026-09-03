"""Default TARC-style penalty: |Δapogee| + 4*|Δtime| (common scoring form).

Swap this module per season without touching the optimizer.
"""

from __future__ import annotations


class TarcDefaultScoring:
    """Penalty = |apogee_error_ft| + 4 * |time_error_s|."""

    TIME_WEIGHT = 4.0

    def score(
        self,
        predicted_apogee_ft: float,
        target_apogee_ft: float,
        predicted_time_s: float,
        target_time_s: float,
    ) -> float:
        return abs(predicted_apogee_ft - target_apogee_ft) + self.TIME_WEIGHT * abs(
            predicted_time_s - target_time_s
        )
