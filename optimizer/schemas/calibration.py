"""Calibration model artifact (§3.5) — consumed by the Predictor."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field

from optimizer.schemas.environment import ThermalActivity


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class FitMethod(str, Enum):
    least_squares = "least_squares"
    bayesian = "bayesian"
    grid_search = "grid_search"


class CpShiftAppliedTo(str, Enum):
    stability_margin_only = "stability_margin_only"
    full_trajectory = "full_trajectory"


class ChuteCdEntry(BaseModel):
    diam_in: float = Field(gt=0)
    cd: float = Field(gt=0)


class ThermalDescentMultiplier(BaseModel):
    category: ThermalActivity
    mean: float = Field(ge=1.0)
    std: float = Field(ge=0)


class ResidualStats(BaseModel):
    apogee_mean_error_ft: float = 0.0
    apogee_std_error_ft: float = 15.0
    descent_time_mean_error_s: float = 0.0
    descent_time_std_error_s: float = 1.5


class FitMetadata(BaseModel):
    method: FitMethod = FitMethod.bayesian
    fitted_at: datetime = Field(default_factory=_utc_now)
    cp_shift_applied_to: CpShiftAppliedTo = CpShiftAppliedTo.stability_margin_only


class Corrections(BaseModel):
    cd_body_multiplier: float = Field(default=1.0, ge=0.7, le=1.3)
    cd_chute_by_diam: list[ChuteCdEntry] = Field(default_factory=list)
    thrust_scale_factor: float = Field(default=1.0, ge=0.9, le=1.1)
    thrust_scale_std: float = Field(
        default=0.03, ge=0.0, description="Posterior / motor-to-motor std"
    )
    burn_time_scale_factor: float = Field(default=1.0, ge=0.85, le=1.15)
    cp_shift_in: float = 0.0
    thermal_descent_multiplier_distribution: list[ThermalDescentMultiplier] = Field(
        default_factory=lambda: [
            ThermalDescentMultiplier(category=ThermalActivity.none, mean=1.02, std=0.01),
            ThermalDescentMultiplier(category=ThermalActivity.light, mean=1.08, std=0.04),
            ThermalDescentMultiplier(
                category=ThermalActivity.moderate, mean=1.18, std=0.08
            ),
            ThermalDescentMultiplier(
                category=ThermalActivity.strong, mean=1.30, std=0.12
            ),
        ]
    )

    def chute_cd(self, diam_in: float, default: float = 1.5) -> float:
        if not self.cd_chute_by_diam:
            return default
        best = min(self.cd_chute_by_diam, key=lambda e: abs(e.diam_in - diam_in))
        return best.cd

    def thermal_params(self, category: ThermalActivity) -> ThermalDescentMultiplier:
        for entry in self.thermal_descent_multiplier_distribution:
            if entry.category == category:
                return entry
        return ThermalDescentMultiplier(category=category, mean=1.0, std=0.0)


class CalibrationModel(BaseModel):
    rocket_id: str
    version: str = "identity"
    fitted_from_flights: list[str] = Field(default_factory=list)
    corrections: Corrections = Field(default_factory=Corrections)
    residual_stats: ResidualStats = Field(default_factory=ResidualStats)
    fit_metadata: FitMetadata = Field(default_factory=FitMetadata)

    @classmethod
    def identity(cls, rocket_id: str = "unknown") -> CalibrationModel:
        return cls(rocket_id=rocket_id, version="identity")
