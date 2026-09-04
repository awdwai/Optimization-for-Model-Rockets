# Run: streamlit run ui/app.py
"""Streamlit demo: race-day Predictor with baked-in samples + AnalyticStubSim."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Ensure repo root is on sys.path when launched as `streamlit run ui/app.py`
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from optimizer.predictor import Predictor, PredictorConfig
from optimizer.schemas.calibration import CalibrationModel
from optimizer.schemas.predict import PredictInput, Recommendation
from optimizer.sim_core import AnalyticStubSim
from ui.demo_defaults import load_demo_predict_input

# Fast MC settings so the demo returns in a few seconds
_DEMO_CONFIG = PredictorConfig(
    monte_carlo_samples=30,
    monte_carlo_min_samples=20,
    monte_carlo_batch_size=20,
    ballast_coarse_steps=8,
    ballast_fine_step_g=2.0,
    ballast_refine_halfwidth_g=10.0,
    seed=42,
)


def _parse_chute_inventory(text: str) -> list[float]:
    parts = [p.strip() for p in text.replace(";", ",").split(",") if p.strip()]
    return [float(p) for p in parts]


def _build_predict_input_from_form(defaults: dict) -> PredictInput:
    atm = defaults["environment"]["atmosphere"]
    wind = defaults["environment"]["wind"]
    surface = defaults["environment"]["surface"]
    objs = defaults["contest_objectives"]
    payload = defaults["payload"]
    lo, hi = defaults["ballast_range_g"]

    chute_text = st.sidebar.text_input(
        "Chute inventory (in, comma-separated)",
        value=", ".join(str(int(c) if float(c).is_integer() else c) for c in defaults["chute_inventory_in"]),
    )
    ballast_lo = st.sidebar.number_input("Ballast min (g)", value=float(lo), min_value=0.0, step=1.0)
    ballast_hi = st.sidebar.number_input("Ballast max (g)", value=float(hi), min_value=0.0, step=1.0)

    st.sidebar.subheader("Targets")
    apogee = st.sidebar.number_input("Desired apogee (ft)", value=float(objs["desired_apogee_ft"]), min_value=1.0)
    apogee_tol = st.sidebar.number_input(
        "Apogee tolerance (ft)", value=float(objs["desired_apogee_tolerance_ft"]), min_value=0.1
    )
    time_s = st.sidebar.number_input("Desired time (s)", value=float(objs["desired_time_s"]), min_value=1.0)
    time_tol = st.sidebar.number_input(
        "Time tolerance (s)", value=float(objs["desired_time_tolerance_s"]), min_value=0.1
    )

    st.sidebar.subheader("Environment")
    air_temp = st.sidebar.number_input("Air temp (°F)", value=float(atm["air_temperature_f"]))
    pressure = st.sidebar.number_input(
        "Pressure (inHg)", value=float(atm["barometric_pressure_inhg"]), min_value=0.1
    )
    humidity = st.sidebar.number_input(
        "Humidity (%)", value=float(atm["relative_humidity_pct"]), min_value=0.0, max_value=100.0
    )
    wind_speed = st.sidebar.number_input(
        "Ground wind (mph)", value=float(wind["ground_wind_speed_mph"]), min_value=0.0
    )
    wind_dir = st.sidebar.number_input(
        "Wind direction (°)", value=float(wind["wind_direction_deg"]), min_value=0.0, max_value=359.9
    )
    wind_gust = st.sidebar.number_input("Wind gust (mph)", value=float(wind["wind_gust_mph"]), min_value=0.0)
    ground_temp = st.sidebar.number_input("Ground temp (°F)", value=float(surface["ground_temperature_f"]))

    egg_mass = st.sidebar.number_input("Egg mass (g)", value=float(payload["egg_mass_g"]), min_value=0.1)
    motor = st.sidebar.text_input("Motor designation", value=str(defaults["motor_designation"]))
    rocket_id = st.sidebar.text_input("Rocket ID", value=str(defaults.get("rocket_id", "demo_rocket")))

    return PredictInput.model_validate(
        {
            "rocket_id": rocket_id,
            "motor_designation": motor,
            "chute_inventory_in": _parse_chute_inventory(chute_text),
            "ballast_range_g": [ballast_lo, ballast_hi],
            "payload": {"egg_mass_g": egg_mass},
            "contest_objectives": {
                "desired_apogee_ft": apogee,
                "desired_apogee_tolerance_ft": apogee_tol,
                "desired_time_s": time_s,
                "desired_time_tolerance_s": time_tol,
            },
            "environment": {
                "atmosphere": {
                    "air_temperature_f": air_temp,
                    "barometric_pressure_inhg": pressure,
                    "relative_humidity_pct": humidity,
                },
                "wind": {
                    "ground_wind_speed_mph": wind_speed,
                    "wind_direction_deg": wind_dir,
                    "wind_gust_mph": wind_gust,
                    # Keep sample wind layers when present; form edits ground wind only
                    "wind_gradient": wind.get("wind_gradient", []),
                },
                "surface": {"ground_temperature_f": ground_temp},
            },
        }
    )


def _render_recommendation(rec: Recommendation) -> None:
    st.success("Recommendation ready (AnalyticStubSim + identity calibration)")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Ballast", f"{rec.ballast_g:.1f} g")
    c2.metric("Chute", f"{rec.chute_diam_in:.0f} in")
    c3.metric("Predicted apogee", f"{rec.predicted_apogee_ft:.1f} ft")
    c4.metric("Predicted time", f"{rec.predicted_descent_time_s:.1f} s")

    c5, c6, c7 = st.columns(3)
    c5.metric("Penalty", f"{rec.predicted_score_penalty:.2f}")
    c6.metric("Feasible", "Yes" if rec.feasible else "No")
    c7.metric("Hit probability", f"{rec.confidence.hit_probability:.0%}")

    st.subheader("Confidence")
    conf = rec.confidence
    st.write(
        f"Apogee σ = **{conf.apogee_std_ft:.1f} ft**, "
        f"descent-time σ = **{conf.descent_time_std_s:.2f} s**, "
        f"MC samples = **{conf.sample_count}**"
    )

    if rec.alternates:
        st.subheader("Alternates")
        rows = [
            {
                "ballast_g": a.ballast_g,
                "chute_diam_in": a.chute_diam_in,
                "penalty": a.predicted_score_penalty,
                "apogee_ft": a.predicted_apogee_ft,
                "time_s": a.predicted_descent_time_s,
                "hit_prob": a.hit_probability,
            }
            for a in rec.alternates
        ]
        st.dataframe(rows, use_container_width=True)


def main() -> None:
    st.set_page_config(page_title="TARC Predictor Demo", layout="wide")
    st.title("TARC race-day Predictor")
    st.caption(
        "Demo mode: prefilled from samples/predict_input.yaml, identity calibration, "
        "AnalyticStubSim only — no .ork or Flight Logs required."
    )

    defaults = load_demo_predict_input()
    predict_input = _build_predict_input_from_form(defaults)

    st.sidebar.markdown("---")
    run = st.sidebar.button("Run demo", type="primary", use_container_width=True)

    if run:
        with st.spinner("Running Predictor (fast MC)…"):
            predictor = Predictor(
                sim=AnalyticStubSim(),
                calibration=CalibrationModel.identity(predict_input.rocket_id),
                config=_DEMO_CONFIG,
            )
            rec = predictor.recommend(predict_input)
        st.session_state["last_recommendation"] = rec

    if "last_recommendation" in st.session_state:
        _render_recommendation(st.session_state["last_recommendation"])
    else:
        st.info("Adjust optional inputs in the sidebar, then click **Run demo**.")


if __name__ == "__main__":
    main()
