from __future__ import annotations

import copy
import os
import zipfile
from pathlib import Path

import pytest

from optimizer.calibration_loader import (
    load_calibration_for_rocket,
    resolve_latest_calibration,
)
from optimizer.monte_carlo import MonteCarloRunner, PROVISIONAL_THERMAL_P
from optimizer.openrocket_bridge import (
    OpenRocketBridge,
    create_simulation_core,
    orhelper_available,
)
from optimizer.ork_validate import (
    OrkValidationError,
    assert_ork_mass_components,
    validate_ork_mass_components,
)
from optimizer.predictor import Predictor, PredictorConfig
from optimizer.schemas.calibration import CalibrationModel
from optimizer.schemas.environment import EnvironmentalState, ThermalActivity
from optimizer.schemas.immutable import ImmutableParameters
from optimizer.schemas.predict import PredictInput
from optimizer.scoring_rules.tarc_default import TarcDefaultScoring
from optimizer.sim_core import AnalyticStubSim, collapse_wind_to_average


SAMPLE = {
    "rocket_id": "demo_rocket",
    "motor_designation": "F50-4T",
    "chute_inventory_in": [12, 15, 18],
    "ballast_range_g": [0, 60],
    "payload": {"egg_mass_g": 57.0},
    "contest_objectives": {
        "desired_apogee_ft": 850,
        "desired_apogee_tolerance_ft": 25,
        "desired_time_s": 44,
        "desired_time_tolerance_s": 4,
    },
    "environment": {
        "atmosphere": {
            "air_temperature_f": 75,
            "barometric_pressure_inhg": 29.9,
            "relative_humidity_pct": 40,
        },
        "wind": {
            "ground_wind_speed_mph": 5,
            "wind_direction_deg": 90,
            "wind_gust_mph": 9,
            "wind_gradient": [],
        },
        "surface": {"ground_temperature_f": 80},
    },
}


def test_predictor_returns_recommendation():
    predict_input = PredictInput.model_validate(SAMPLE)
    predictor = Predictor(
        config=PredictorConfig(
            monte_carlo_samples=30,
            monte_carlo_min_samples=20,
            ballast_coarse_steps=8,
            ballast_fine_step_g=2.0,
            seed=1,
        )
    )
    rec = predictor.recommend(predict_input)
    assert rec.chute_diam_in in predict_input.chute_inventory_in
    assert predict_input.ballast_min <= rec.ballast_g <= predict_input.ballast_max
    assert 0.0 <= rec.confidence.hit_probability <= 1.0
    assert len(rec.alternates) == len(predict_input.chute_inventory_in) - 1
    assert rec.confidence.sample_count >= 20


def test_density_precedence_from_da():
    from optimizer.schemas.environment import Atmosphere

    atm = Atmosphere(
        air_temperature_f=70,
        barometric_pressure_inhg=29.92,
        relative_humidity_pct=50,
        density_altitude_ft=2000,
        air_density_slug_ft3=0.0020,
    )
    assert atm.air_density_slug_ft3 is not None
    assert atm.density_altitude_ft == 2000
    assert abs(atm.air_density_slug_ft3 - 0.0020) > 1e-5


def test_thermal_explicit_none_override():
    data = copy.deepcopy(SAMPLE)
    data["environment"]["surface"]["ground_temperature_f"] = 120
    data["environment"]["surface"]["thermal_activity_estimate"] = "none"
    env = PredictInput.model_validate(data).environment
    assert env.surface.thermal_activity_estimate == ThermalActivity.none
    assert env.surface.thermal_was_explicit


def test_thermal_auto_derive_from_delta():
    data = copy.deepcopy(SAMPLE)
    data["environment"]["atmosphere"]["air_temperature_f"] = 70
    data["environment"]["surface"]["ground_temperature_f"] = 95
    env = PredictInput.model_validate(data).environment
    assert env.surface.thermal_activity_estimate == ThermalActivity.strong


def test_unstable_samples_get_large_penalty():
    predict_input = PredictInput.model_validate(SAMPLE)
    sim = AnalyticStubSim(min_stability_calibers=0.0)
    scoring = TarcDefaultScoring()
    runner = MonteCarloRunner(
        sim,
        scoring,
        min_stability_calibers=100.0,
        batch_size=10,
        min_samples=10,
        max_samples=10,
        unstable_penalty=1e6,
        seed=0,
    )
    result = runner.evaluate(
        predict_input.environment,
        egg_mass_g=57.0,
        ballast_g=0.0,
        chute_diam_in=15.0,
        motor_designation="F50-4T",
        calibration=CalibrationModel.identity(),
        objectives=predict_input.contest_objectives,
    )
    assert all(o.unstable for o in result.outcomes)
    assert result.mean_penalty >= 1e6


def test_risk_aversion_exposed_in_recommendation():
    predict_input = PredictInput.model_validate(SAMPLE)
    cfg = PredictorConfig(
        monte_carlo_samples=24,
        monte_carlo_min_samples=16,
        ballast_coarse_steps=6,
        ballast_fine_step_g=5.0,
        seed=2,
        risk_aversion=1.0,
    )
    rec = Predictor(config=cfg).recommend(predict_input)
    assert rec.risk_averse_penalty_p90 is not None
    assert rec.predicted_score_penalty == pytest.approx(rec.risk_averse_penalty_p90)


def test_collapse_wind_gradient():
    env = PredictInput.model_validate(SAMPLE).environment
    data = env.model_dump()
    data["wind"]["wind_gradient"] = [
        {"layer_ft": [0, 50], "speed_mph": 4.0, "direction_deg": 90},
        {"layer_ft": [50, 150], "speed_mph": 10.0, "direction_deg": 90},
    ]
    env2 = EnvironmentalState.model_validate(data)
    avg = collapse_wind_to_average(env2)
    assert avg == pytest.approx(8.0)


def test_openrocket_bridge_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        OpenRocketBridge("definitely_missing_rocket.ork")


def test_create_simulation_core_falls_back_to_stub(tmp_path, recwarn):
    # Tiny fake .ork containing required names but no real OR runtime
    fake = tmp_path / "fake.ork"
    fake.write_text(
        '<?xml version="1.0"?><rocket><component name="egg_payload"/>'
        '<component name="ballast"/></rocket>',
        encoding="utf-8",
    )
    sim = create_simulation_core(fake, allow_stub_fallback=True)
    assert isinstance(sim, AnalyticStubSim)
    assert any(issubclass(w.category, UserWarning) for w in recwarn)


def test_provisional_thermal_probabilities_cover_categories():
    assert set(PROVISIONAL_THERMAL_P) == set(ThermalActivity)


def test_immutable_parameters_roundtrip():
    sample = Path("samples/immutable_demo.json")
    imm = ImmutableParameters.model_validate_json(sample.read_text(encoding="utf-8"))
    assert imm.rocket_id == "demo_rocket"
    assert imm.openrocket_file.endswith(".ork")


def test_ork_validate_detects_missing_components(tmp_path):
    bad = tmp_path / "bad.ork"
    bad.write_text("<rocket><component name='nose'/></rocket>", encoding="utf-8")
    missing = validate_ork_mass_components(bad)
    assert "egg_payload" in missing and "ballast" in missing
    with pytest.raises(OrkValidationError):
        assert_ork_mass_components(bad)


def test_ork_validate_zip_wrapped(tmp_path):
    zpath = tmp_path / "wrapped.ork"
    with zipfile.ZipFile(zpath, "w") as zf:
        zf.writestr(
            "rocket.ork",
            '<rocket name="egg_payload"><x>ballast</x></rocket>',
        )
    assert validate_ork_mass_components(zpath) == []


def test_calibration_latest_loader():
    root = Path("samples/calibration")
    path = resolve_latest_calibration(root, "demo_rocket")
    cal = load_calibration_for_rocket(
        "demo_rocket", calibration_dir=root
    )
    assert path.name == "latest.json"
    assert cal.rocket_id == "demo_rocket"
    assert cal.corrections.thrust_scale_factor == 1.0


@pytest.mark.orhelper
def test_openrocket_integration_optional():
    """Runs only when orhelper + a real .ork are available.

    Set OPENROCKET_ORK to a .ork that contains egg_payload and ballast.
    """
    if not orhelper_available():
        pytest.skip("orhelper not installed")
    ork = Path(__file__).resolve().parents[1] / "samples" / "demo_rocket.ork"
    env_ork = os.environ.get("OPENROCKET_ORK")
    if env_ork:
        ork = Path(env_ork)
    if not ork.is_file():
        pytest.skip("No OpenRocket .ork fixture (set OPENROCKET_ORK)")

    predict_input = PredictInput.model_validate(SAMPLE)
    with OpenRocketBridge(ork) as bridge:
        result = bridge.simulate(
            predict_input.environment,
            egg_mass_g=57.0,
            ballast_g=10.0,
            chute_diam_in=15.0,
            motor_designation="F50-4T",
            calibration=CalibrationModel.identity(),
        )
    assert result.apogee_ft > 0
    assert result.descent_time_s > 0
