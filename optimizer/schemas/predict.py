"""Predictor I/O schemas (§3.2 End User Input, §3.4 Predictor Output)."""

from __future__ import annotations

from pydantic import BaseModel, Field

from optimizer.schemas.environment import EnvironmentalState


class ContestObjectives(BaseModel):
    desired_apogee_ft: float = Field(gt=0)
    desired_apogee_tolerance_ft: float = Field(gt=0)
    desired_time_s: float = Field(gt=0)
    desired_time_tolerance_s: float = Field(gt=0)


class Payload(BaseModel):
    egg_mass_g: float = Field(gt=0)


class PredictInput(BaseModel):
    environment: EnvironmentalState
    payload: Payload
    contest_objectives: ContestObjectives
    motor_designation: str
    chute_inventory_in: list[float] = Field(min_length=1)
    ballast_range_g: tuple[float, float]
    rocket_id: str = "default"

    @property
    def ballast_min(self) -> float:
        return min(self.ballast_range_g)

    @property
    def ballast_max(self) -> float:
        return max(self.ballast_range_g)


class Confidence(BaseModel):
    apogee_std_ft: float
    descent_time_std_s: float
    hit_probability: float = Field(ge=0, le=1)
    sample_count: int = Field(ge=0)


class Alternate(BaseModel):
    ballast_g: float
    chute_diam_in: float
    predicted_score_penalty: float
    predicted_apogee_ft: float | None = None
    predicted_descent_time_s: float | None = None
    hit_probability: float | None = None


class Recommendation(BaseModel):
    ballast_g: float
    chute_diam_in: float
    predicted_apogee_ft: float
    predicted_descent_time_s: float
    predicted_score_penalty: float
    feasible: bool
    closest_miss_ft: float = 0.0
    closest_miss_s: float = 0.0
    confidence: Confidence
    alternates: list[Alternate] = Field(default_factory=list)
    risk_averse_penalty_p90: float | None = None
