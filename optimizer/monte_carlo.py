"""Monte Carlo / uncertainty layer (§5)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from optimizer.schemas.calibration import CalibrationModel
from optimizer.schemas.environment import (
    EnvironmentalState,
    ThermalActivity,
    WindLayer,
)
from optimizer.schemas.predict import ContestObjectives
from optimizer.scoring_rules import ScoringRule
from optimizer.sim_core import SimResult, SimulationCore

# Provisional encounter probabilities until Trainer fits them from logs (§5)
PROVISIONAL_THERMAL_P: dict[ThermalActivity, float] = {
    ThermalActivity.none: 0.02,
    ThermalActivity.light: 0.10,
    ThermalActivity.moderate: 0.30,
    ThermalActivity.strong: 0.55,
}


@dataclass
class MCSampleOutcome:
    apogee_ft: float
    descent_time_s: float
    penalty: float
    stability_calibers: float
    unstable: bool


@dataclass
class MCResult:
    mean_penalty: float
    p90_penalty: float
    mean_apogee_ft: float
    mean_descent_time_s: float
    apogee_std_ft: float
    descent_time_std_s: float
    hit_probability: float
    sample_count: int
    outcomes: list[MCSampleOutcome]


def _perturb_environment(
    base: EnvironmentalState,
    rng: np.random.Generator,
) -> EnvironmentalState:
    wind = base.wind
    gust_spread = max(wind.wind_gust_mph - wind.ground_wind_speed_mph, 0.5)
    # Treat gust as ~2.5σ above mean
    sigma_speed = gust_spread / 2.5
    speed = max(0.0, rng.normal(wind.ground_wind_speed_mph, sigma_speed))
    direction = float(rng.normal(wind.wind_direction_deg, 8.0)) % 360.0

    gust_factor = speed / max(wind.ground_wind_speed_mph, 1e-3)

    new_layers: list[WindLayer] = []
    for layer in wind.wind_gradient:
        new_layers.append(
            WindLayer(
                layer_ft=layer.layer_ft,
                speed_mph=max(0.0, layer.speed_mph * gust_factor),
                direction_deg=float(rng.normal(layer.direction_deg, 8.0)) % 360.0,
            )
        )

    data = base.model_dump()
    data["wind"]["ground_wind_speed_mph"] = speed
    data["wind"]["wind_direction_deg"] = direction
    data["wind"]["wind_gradient"] = new_layers
    return EnvironmentalState.model_validate(data)


def _apply_thermal(
    descent_s: float,
    category: ThermalActivity,
    calibration: CalibrationModel,
    rng: np.random.Generator,
) -> float:
    p = PROVISIONAL_THERMAL_P.get(category, 0.1)
    if rng.random() > p:
        return descent_s
    params = calibration.corrections.thermal_params(category)
    multiplier = max(1.0, rng.normal(params.mean, params.std))
    return descent_s * multiplier


class MonteCarloRunner:
    def __init__(
        self,
        sim: SimulationCore,
        scoring: ScoringRule,
        *,
        min_stability_calibers: float = 1.0,
        batch_size: int = 50,
        se_threshold: float = 0.5,
        max_samples: int = 200,
        min_samples: int = 50,
        unstable_penalty: float = 1e6,
        seed: int | None = 42,
    ) -> None:
        self.sim = sim
        self.scoring = scoring
        self.min_stability_calibers = min_stability_calibers
        self.batch_size = batch_size
        self.se_threshold = se_threshold
        self.max_samples = max_samples
        self.min_samples = min_samples
        self.unstable_penalty = unstable_penalty
        self.rng = np.random.default_rng(seed)

    def evaluate(
        self,
        environment: EnvironmentalState,
        *,
        egg_mass_g: float,
        ballast_g: float,
        chute_diam_in: float,
        motor_designation: str,
        calibration: CalibrationModel,
        objectives: ContestObjectives,
    ) -> MCResult:
        outcomes: list[MCSampleOutcome] = []
        thrust_mu = calibration.corrections.thrust_scale_factor
        thrust_std = calibration.corrections.thrust_scale_std

        while True:
            batch_n = min(self.batch_size, self.max_samples - len(outcomes))
            for _ in range(batch_n):
                env = _perturb_environment(environment, self.rng)
                # Sample thrust for this draw by temporarily cloning corrections
                thrust_draw = float(self.rng.normal(thrust_mu, thrust_std))
                cal_data = calibration.model_dump()
                cal_data["corrections"]["thrust_scale_factor"] = float(
                    np.clip(thrust_draw, 0.9, 1.1)
                )
                cal = CalibrationModel.model_validate(cal_data)

                result: SimResult = self.sim.simulate(
                    env,
                    egg_mass_g=egg_mass_g,
                    ballast_g=ballast_g,
                    chute_diam_in=chute_diam_in,
                    motor_designation=motor_designation,
                    calibration=cal,
                )

                unstable = (
                    result.rail_exit_stability_calibers < self.min_stability_calibers
                )
                thermal_cat = env.surface.thermal_activity_estimate or ThermalActivity.none
                descent = _apply_thermal(
                    result.descent_time_s, thermal_cat, calibration, self.rng
                )

                if unstable:
                    penalty = self.unstable_penalty
                else:
                    penalty = self.scoring.score(
                        result.apogee_ft,
                        objectives.desired_apogee_ft,
                        descent,
                        objectives.desired_time_s,
                    )

                outcomes.append(
                    MCSampleOutcome(
                        apogee_ft=result.apogee_ft,
                        descent_time_s=descent,
                        penalty=penalty,
                        stability_calibers=result.rail_exit_stability_calibers,
                        unstable=unstable,
                    )
                )

            n = len(outcomes)
            penalties = np.array([o.penalty for o in outcomes], dtype=float)
            se = float(penalties.std(ddof=1) / np.sqrt(n)) if n > 1 else np.inf
            if n >= self.min_samples and (se < self.se_threshold or n >= self.max_samples):
                break
            if n >= self.max_samples:
                break

        penalties = np.array([o.penalty for o in outcomes], dtype=float)
        apogees = np.array([o.apogee_ft for o in outcomes], dtype=float)
        times = np.array([o.descent_time_s for o in outcomes], dtype=float)

        hits = sum(
            1
            for o in outcomes
            if not o.unstable
            and abs(o.apogee_ft - objectives.desired_apogee_ft)
            <= objectives.desired_apogee_tolerance_ft
            and abs(o.descent_time_s - objectives.desired_time_s)
            <= objectives.desired_time_tolerance_s
        )

        return MCResult(
            mean_penalty=float(penalties.mean()),
            p90_penalty=float(np.percentile(penalties, 90)),
            mean_apogee_ft=float(apogees.mean()),
            mean_descent_time_s=float(times.mean()),
            apogee_std_ft=float(apogees.std(ddof=1)) if len(apogees) > 1 else 0.0,
            descent_time_std_s=float(times.std(ddof=1)) if len(times) > 1 else 0.0,
            hit_probability=hits / max(len(outcomes), 1),
            sample_count=len(outcomes),
            outcomes=outcomes,
        )
