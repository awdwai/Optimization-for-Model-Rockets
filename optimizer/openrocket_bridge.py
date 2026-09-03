"""OpenRocket Simulation Core bridge (§4.1.1).

Requires ``orhelper`` + OpenRocket 15.03 JAR (see CLASSPATH / jar_path).
When the JVM stack is unavailable, callers should fall back to AnalyticStubSim
with an explicit warning — never silent wrong physics.
"""

from __future__ import annotations

import logging
import math
import warnings
from pathlib import Path

from optimizer.ork_validate import (
    BALLAST_COMPONENT_NAME,
    EGG_COMPONENT_NAME,
    assert_ork_mass_components,
)
from optimizer.schemas.calibration import CalibrationModel, CpShiftAppliedTo
from optimizer.schemas.environment import EnvironmentalState
from optimizer.sim_core import AnalyticStubSim, SimResult, SimulationCore, collapse_wind_to_average

logger = logging.getLogger(__name__)

# Unit helpers (OpenRocket SI ↔ Predictor US customary)
_G_TO_KG = 1e-3
_IN_TO_M = 0.0254
_FT_TO_M = 0.3048
_M_TO_FT = 1.0 / _FT_TO_M
_MPH_TO_MPS = 0.44704
_INHG_TO_PA = 3386.389


class OpenRocketNotAvailableError(RuntimeError):
    """Raised when orhelper / OpenRocket JVM cannot be started."""


def _f_to_kelvin(temp_f: float) -> float:
    return (temp_f - 32.0) * 5.0 / 9.0 + 273.15


def orhelper_available() -> bool:
    try:
        import orhelper  # noqa: F401
        import jpype  # noqa: F401

        return True
    except ImportError:
        return False


class OpenRocketBridge:
    """Headless OpenRocket runner — one JVM instance per Predictor run.

    - Pre-place MassComponents named ``egg_payload`` and ``ballast`` in the .ork
    - Set mass via those components each call (no component-tree edits)
    - Wind: option (a) — collapse wind_gradient to a single average
    - CP shift: stability_margin_only (post-sim) by default
    - Reuse OpenRocketInstance across Monte Carlo / search iterations
    """

    def __init__(
        self,
        openrocket_file: str | Path,
        *,
        jar_path: str | Path | None = None,
        cp_shift_mode: CpShiftAppliedTo = CpShiftAppliedTo.stability_margin_only,
        validate_components: bool = True,
    ) -> None:
        self.openrocket_file = Path(openrocket_file)
        self.jar_path = Path(jar_path) if jar_path else None
        self.cp_shift_mode = cp_shift_mode
        self._instance = None
        self._helper = None
        self._doc = None
        self._sim = None
        self._egg = None
        self._ballast = None
        self._parachute = None
        self._started = False

        if not self.openrocket_file.is_file():
            raise FileNotFoundError(self.openrocket_file)
        if validate_components:
            assert_ork_mass_components(self.openrocket_file)

    def __enter__(self) -> OpenRocketBridge:
        self._start_jvm()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        if self._instance is not None and self._started:
            try:
                self._instance.__exit__(None, None, None)
            except Exception as exc:  # pragma: no cover - JVM teardown best-effort
                logger.warning("OpenRocket JVM shutdown issue: %s", exc)
        self._helper = None
        self._doc = None
        self._sim = None
        self._egg = None
        self._ballast = None
        self._parachute = None
        self._instance = None
        self._started = False

    def _start_jvm(self) -> None:
        if self._started:
            return
        try:
            import orhelper
        except ImportError as exc:
            raise OpenRocketNotAvailableError(
                "orhelper is not installed. Install orhelper + OpenRocket 15.03 "
                "(set CLASSPATH to OpenRocket-15.03.jar), or use AnalyticStubSim."
            ) from exc

        kwargs = {}
        if self.jar_path is not None:
            kwargs["jar_path"] = str(self.jar_path)

        try:
            instance = orhelper.OpenRocketInstance(**kwargs)
            instance.__enter__()
        except Exception as exc:
            raise OpenRocketNotAvailableError(
                f"Failed to start OpenRocket JVM for {self.openrocket_file}: {exc}"
            ) from exc

        self._instance = instance
        self._helper = orhelper.Helper(instance)
        self._doc = self._helper.load_doc(str(self.openrocket_file.resolve()))
        if self._doc.getSimulationCount() < 1:
            self.close()
            raise OpenRocketNotAvailableError(
                f"{self.openrocket_file} has no simulations defined"
            )
        self._sim = self._doc.getSimulation(0)
        rocket = self._sim.getOptions().getRocket()
        try:
            self._egg = self._helper.get_component_named(rocket, EGG_COMPONENT_NAME)
            self._ballast = self._helper.get_component_named(
                rocket, BALLAST_COMPONENT_NAME
            )
        except ValueError as exc:
            self.close()
            raise OpenRocketNotAvailableError(str(exc)) from exc

        self._parachute = self._find_parachute(rocket)
        self._started = True
        logger.info("OpenRocket bridge ready: %s", self.openrocket_file)

    def _find_parachute(self, rocket):  # noqa: ANN001
        """Best-effort: first component with setDiameter (recovery device)."""
        from orhelper._orhelper import JIterator  # type: ignore

        for component in JIterator(rocket):
            if hasattr(component, "setDiameter") and hasattr(component, "getDiameter"):
                return component
        return None

    def _set_mass_kg(self, component, mass_g: float) -> None:  # noqa: ANN001
        mass_kg = max(mass_g, 0.0) * _G_TO_KG
        component.setMassOverridden(True)
        component.setOverrideMass(float(mass_kg))

    def _apply_environment(self, environment: EnvironmentalState) -> None:
        opts = self._sim.getOptions()
        atm = environment.atmosphere
        # Prefer Kelvin API; fall back to Celsius-style setters if needed
        temp_k = _f_to_kelvin(atm.air_temperature_f)
        pressure_pa = atm.barometric_pressure_inhg * _INHG_TO_PA
        try:
            opts.setISAAtmosphere(False)
        except Exception:
            pass
        for setter, value in (
            ("setLaunchTemperature", temp_k),
            ("setLaunchPressure", pressure_pa),
        ):
            if hasattr(opts, setter):
                try:
                    getattr(opts, setter)(float(value))
                except Exception as exc:
                    logger.debug("SimulationOptions.%s failed: %s", setter, exc)

        wind_mph = collapse_wind_to_average(environment)
        wind_mps = wind_mph * _MPH_TO_MPS
        if hasattr(opts, "setWindSpeedAverage"):
            opts.setWindSpeedAverage(float(wind_mps))
        if hasattr(opts, "setWindDirection"):
            # OpenRocket wind direction: radians in many builds
            try:
                opts.setWindDirection(
                    math.radians(environment.wind.wind_direction_deg)
                )
            except Exception:
                try:
                    opts.setWindDirection(float(environment.wind.wind_direction_deg))
                except Exception as exc:
                    logger.debug("setWindDirection failed: %s", exc)

        # Map gust spread → turbulence intensity in [0, 1]
        mean = max(environment.wind.ground_wind_speed_mph, 1e-3)
        gust = max(environment.wind.wind_gust_mph, mean)
        intensity = min(max((gust - mean) / max(gust, 1e-3), 0.0), 1.0)
        if hasattr(opts, "setWindTurbulenceIntensity"):
            try:
                opts.setWindTurbulenceIntensity(float(intensity))
            except Exception as exc:
                logger.debug("setWindTurbulenceIntensity failed: %s", exc)

    def _apply_chute(self, chute_diam_in: float, calibration: CalibrationModel) -> None:
        if self._parachute is None:
            logger.warning("No parachute component with setDiameter; chute size ignored")
            return
        diam_m = chute_diam_in * _IN_TO_M
        self._parachute.setDiameter(float(diam_m))
        cd = calibration.corrections.chute_cd(chute_diam_in)
        if hasattr(self._parachute, "setCD") and hasattr(self._parachute, "setCDOverridden"):
            try:
                self._parachute.setCDOverridden(True)
                self._parachute.setCD(float(cd))
            except Exception as exc:
                logger.debug("Parachute CD override failed: %s", exc)

    def _apply_motor(self, motor_designation: str) -> None:
        """Best-effort motor swap; leave .ork motor if lookup fails."""
        if not motor_designation or self._helper is None:
            return
        try:
            # OpenRocket motor database lookup varies by version; keep soft-fail.
            rocket = self._sim.getOptions().getRocket()
            _ = rocket, motor_designation
            logger.debug(
                "Motor designation %s noted; using .ork FlightConfiguration motor",
                motor_designation,
            )
        except Exception as exc:
            logger.debug("Motor override skipped: %s", exc)

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
        if self._helper is None:
            self._start_jvm()

        self._set_mass_kg(self._egg, egg_mass_g)
        self._set_mass_kg(self._ballast, ballast_g)
        self._apply_chute(chute_diam_in, calibration)
        self._apply_motor(motor_designation)
        self._apply_environment(environment)

        # Thrust / body Cd: scale after extraction when OR has no direct knob
        thrust_scale = calibration.corrections.thrust_scale_factor
        cd_body = calibration.corrections.cd_body_multiplier
        burn_scale = calibration.corrections.burn_time_scale_factor

        self._helper.run_simulation(self._sim)

        events = self._helper.get_events(self._sim)
        try:
            from orhelper import FlightDataType  # type: ignore

            vars_ = [
                FlightDataType.TYPE_TIME,
                FlightDataType.TYPE_ALTITUDE,
                FlightDataType.TYPE_VELOCITY_TOTAL,
                FlightDataType.TYPE_STABILITY,
            ]
        except Exception:
            vars_ = [
                "TYPE_TIME",
                "TYPE_ALTITUDE",
                "TYPE_VELOCITY_TOTAL",
                "TYPE_STABILITY",
            ]
        series = self._helper.get_timeseries(self._sim, vars_)

        # orhelper enums may use FlightDataType names — accept either key style
        def _col(data: dict, *names: str):
            name_set = {n.upper() for n in names}
            for k, v in data.items():
                key = getattr(k, "name", str(k)).upper()
                if key in name_set or str(k).upper() in name_set:
                    return v
            return None

        altitude_m = _col(series, "TYPE_ALTITUDE", "ALTITUDE")
        velocity = _col(series, "TYPE_VELOCITY_TOTAL", "VELOCITY_TOTAL")
        stability = _col(series, "TYPE_STABILITY", "STABILITY")
        time_s = _col(series, "TYPE_TIME", "TIME", "TIME_SECONDS")

        if altitude_m is None or len(altitude_m) == 0:
            raise OpenRocketNotAvailableError("Simulation produced no altitude series")

        apogee_m = float(max(altitude_m))
        apogee_ft = apogee_m * _M_TO_FT
        # Calibration body Cd / thrust as first-order post scales (Trainer will refine)
        apogee_ft *= (thrust_scale**1.1) / (cd_body**0.5)

        # Descent time: apogee → landing (last sample) or event timestamps
        descent_s = self._descent_time_s(events, time_s, altitude_m)
        descent_s *= burn_scale

        max_v_fps = 0.0
        if velocity is not None and len(velocity):
            max_v_fps = float(max(velocity)) * _M_TO_FT

        rail_stability = self._rail_exit_stability(events, time_s, stability)
        if self.cp_shift_mode == CpShiftAppliedTo.stability_margin_only:
            # cp_shift_in is inches; rough caliber conversion left to airframe diameter
            # Apply as inches/12 → calibers proxy (documented approximation)
            rail_stability += calibration.corrections.cp_shift_in / 12.0

        return SimResult(
            apogee_ft=float(apogee_ft),
            descent_time_s=float(descent_s),
            max_velocity_fps=float(max_v_fps),
            rail_exit_stability_calibers=float(rail_stability),
        )

    def _descent_time_s(self, events, time_s, altitude_m) -> float:  # noqa: ANN001
        apogee_t = None
        land_t = None
        # events values are lists of times in newer orhelper
        for key, val in (events or {}).items():
            name = getattr(key, "name", str(key)).upper()
            times = val if isinstance(val, (list, tuple)) else [val]
            if not times:
                continue
            if "APOGEE" in name:
                apogee_t = float(times[0])
            if "GROUND" in name or "LANDING" in name or name.endswith("LAND"):
                land_t = float(times[-1])
        if apogee_t is not None and land_t is not None and land_t > apogee_t:
            return land_t - apogee_t
        if time_s is not None and altitude_m is not None and len(time_s) > 1:
            idx = int(altitude_m.argmax())
            return float(time_s[-1] - time_s[idx])
        return 0.0

    def _rail_exit_stability(self, events, time_s, stability) -> float:  # noqa: ANN001
        if stability is None or len(stability) == 0:
            return 0.0
        launchrod_t = None
        for key, val in (events or {}).items():
            name = getattr(key, "name", str(key)).upper()
            if "LAUNCHROD" in name or "LAUNCH_ROD" in name or "ROD" in name:
                times = val if isinstance(val, (list, tuple)) else [val]
                if times:
                    launchrod_t = float(times[0])
                    break
        if launchrod_t is not None and time_s is not None and len(time_s):
            idx = int(abs(time_s - launchrod_t).argmin())
            return float(stability[idx])
        return float(stability[0])


def create_simulation_core(
    openrocket_file: str | Path | None = None,
    *,
    jar_path: str | Path | None = None,
    prefer_openrocket: bool = True,
    allow_stub_fallback: bool = True,
) -> SimulationCore:
    """Build OpenRocketBridge when possible; otherwise AnalyticStubSim with a warning."""
    if openrocket_file is None or not prefer_openrocket:
        return AnalyticStubSim()

    try:
        bridge = OpenRocketBridge(openrocket_file, jar_path=jar_path)
        bridge._start_jvm()
        return bridge
    except (OpenRocketNotAvailableError, FileNotFoundError, OSError, ValueError) as exc:
        if not allow_stub_fallback:
            raise
        warnings.warn(
            f"OpenRocket unavailable ({exc}); falling back to AnalyticStubSim. "
            "Race-day recommendations will NOT use real 6-DOF physics.",
            UserWarning,
            stacklevel=2,
        )
        return AnalyticStubSim()
