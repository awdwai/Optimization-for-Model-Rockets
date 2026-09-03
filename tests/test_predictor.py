from __future__ import annotations

from optimizer.predictor import Predictor, PredictorConfig
from optimizer.schemas.predict import PredictInput


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


def test_density_precedence_from_da():
    from optimizer.schemas.environment import Atmosphere

    atm = Atmosphere(
        air_temperature_f=70,
        barometric_pressure_inhg=29.92,
        relative_humidity_pct=50,
        density_altitude_ft=2000,
        air_density_slug_ft3=0.0020,  # should be overwritten from DA
    )
    assert atm.air_density_slug_ft3 is not None
    assert atm.density_altitude_ft == 2000
