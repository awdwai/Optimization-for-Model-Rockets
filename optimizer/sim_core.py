"""Simulation Core interface + analytic stub for optimizer development (§4.1).

The real OpenRocket bridge plugs in later behind the same protocol. The stub
gives the Predictor a deterministic, physically-plausible response surface so
chute/ballast search and Monte Carlo can be developed independently.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from optimizer.schemas.calibration import CalibrationModel
from optimizer.schemas.environment import EnvironmentalState


@dataclass(frozen=True)
class SimResult:
    apogee_ft: float
    descent_time_s: float
    max_velocity_fps: float
    rail_exit_stability_calibers: float


class SimulationCore(Protocol):
    def simulate(
        self,
        environment: EnvironmentalState,
        *,
        egg_mass_g: float,
        ballast_g: float,
        chute_diam_in: float,
        motor_designation: str,
        calibration: CalibrationModel,
    ) -> SimResult: ...


class AnalyticStubSim:
    """Closed-form stand-in until the OpenRocket bridge exists.

    Rough physics:
    - Extra mass lowers apogee ~linearly
    - Higher air density lowers apogee
    - Larger chute / higher Cd lengthens descent
    - Thermal multiplier stretches descent only
    - Body Cd / thrust scales from calibration
    """

    def __init__(
        self,
        *,
        base_apogee_ft: float = 980.0,
        base_descent_s: float = 42.0,
        reference_mass_g: float = 400.0,
        reference_chute_in: float = 15.0,
        min_stability_calibers: float = 1.5,
    ) -> None:
        self.base_apogee_ft = base_apogee_ft
        self.base_descent_s = base_descent_s
        self.reference_mass_g = reference_mass_g
        self.reference_chute_in = reference_chute_in
        self.min_stability_calibers = min_stability_calibers

    def simulate(
        self,
        environment: EnvironmentalState,
        *,
        egg_mass_g: float,
        ballast_g: float,
        chute_diam_in: float,
        motor_designation: str,
        calibration: CalibrationModel,
    ) -> SimResult:
        del motor_designation  # stub ignores specific motor curve
        corr = calibration.corrections
        total_mass = self.reference_mass_g + egg_mass_g + ballast_g

        rho = environment.atmosphere.air_density_slug_ft3 or 0.00237
        density_factor = 0.00237 / rho

        mass_factor = self.reference_mass_g / max(total_mass, 1.0)
        thrust = corr.thrust_scale_factor
        cd_body = corr.cd_body_multiplier

        apogee = (
            self.base_apogee_ft
            * (mass_factor**0.85)
            * (density_factor**0.4)
            * (thrust**1.1)
            / (cd_body**0.5)
        )
        # Mild wind penalty using option-(a) collapsed gradient when present
        wind = collapse_wind_to_average(environment)
        apogee *= max(0.85, 1.0 - 0.004 * wind)

        chute_cd = corr.chute_cd(chute_diam_in)
        area_ratio = (chute_diam_in / self.reference_chute_in) ** 2
        descent = (
            self.base_descent_s
            * (area_ratio**0.5)
            * (chute_cd / 1.5)
            * (rho / 0.00237) ** 0.5
            * (total_mass / self.reference_mass_g) ** 0.5
        )

        # Stability: more ballast aft assumed → slightly higher margin in stub
        stability = self.min_stability_calibers + 0.002 * ballast_g - 0.01 * (
            chute_diam_in - self.reference_chute_in
        )
        # CP shift only affects reported margin in v1
        stability += corr.cp_shift_in / 12.0

        max_v = 180.0 * thrust * mass_factor**0.5

        return SimResult(
            apogee_ft=float(apogee),
            descent_time_s=float(descent),
            max_velocity_fps=float(max_v),
            rail_exit_stability_calibers=float(stability),
        )


def collapse_wind_to_average(environment: EnvironmentalState) -> float:
    """§4.1 option (a): altitude-weighted average of wind_gradient layers."""
    layers = environment.wind.wind_gradient
    if not layers:
        return environment.wind.ground_wind_speed_mph
    weights = []
    speeds = []
    for layer in layers:
        lo, hi = layer.layer_ft
        w = max(hi - lo, 1.0)
        weights.append(w)
        speeds.append(layer.speed_mph)
    return float(np.average(speeds, weights=weights))
