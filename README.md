# Optimization for Model Rockets

TARC race-day **Predictor**: recommend ballast + chute from weather and contest targets.

## Layout

```
optimizer/          # Core package (python -m optimizer)
  schemas/          # Pydantic inputs / outputs
  scoring_rules/    # Contest scoring (e.g. TARC default)
  predictor.py      # Search + recommendation
  sim_core.py       # Simulation interface + AnalyticStubSim
  cli.py            # YAML/JSON CLI
ui/                 # Streamlit demo
samples/            # Demo YAML/JSON + calibration fixtures
tests/              # pytest
run_demo.bat        # Windows: install deps + launch UI
requirements.txt
```

## Quick start

```bash
pip install -r requirements.txt
```

**Demo UI (Windows):** double-click `run_demo.bat`, or:

```bash
python -m streamlit run ui/app.py
```

**CLI (stub simulator, no OpenRocket):**

```bash
python -m optimizer --stub -i samples/predict_input.yaml
```

**Tests:**

```bash
python -m pytest tests/ -q
```

Optional OpenRocket bridge: install `orhelper` / `jpype1` and place an OpenRocket 15.03 JAR on `CLASSPATH` (see comments in `requirements.txt`).
