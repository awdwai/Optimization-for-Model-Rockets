"""Baked-in demo defaults for the Streamlit race-day UI (no user files required)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

# Repo root is parent of ui/
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SAMPLE_PATH = _REPO_ROOT / "samples" / "predict_input.yaml"

# Fallback when samples/predict_input.yaml is missing (mirrors the sample file).
HARDCODED_PREDICT_INPUT: dict[str, Any] = {
    "rocket_id": "demo_rocket",
    "motor_designation": "F50-4T",
    "chute_inventory_in": [12, 15, 18, 24],
    "ballast_range_g": [0, 80],
    "payload": {"egg_mass_g": 57.5},
    "contest_objectives": {
        "desired_apogee_ft": 850,
        "desired_apogee_tolerance_ft": 20,
        "desired_time_s": 45,
        "desired_time_tolerance_s": 3,
    },
    "environment": {
        "atmosphere": {
            "air_temperature_f": 78,
            "barometric_pressure_inhg": 29.92,
            "relative_humidity_pct": 45,
        },
        "wind": {
            "ground_wind_speed_mph": 6,
            "wind_direction_deg": 180,
            "wind_gust_mph": 11,
            "wind_gradient": [
                {"layer_ft": [0, 50], "speed_mph": 6, "direction_deg": 180},
                {"layer_ft": [50, 100], "speed_mph": 8, "direction_deg": 185},
                {"layer_ft": [100, 150], "speed_mph": 9, "direction_deg": 190},
                {"layer_ft": [150, 200], "speed_mph": 10, "direction_deg": 195},
            ],
        },
        "surface": {"ground_temperature_f": 88},
    },
}


def load_demo_predict_input() -> dict[str, Any]:
    """Load sample PredictInput mapping, or return hardcoded fallback."""
    if _SAMPLE_PATH.is_file():
        data = yaml.safe_load(_SAMPLE_PATH.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return data
    return dict(HARDCODED_PREDICT_INPUT)
