"""CLI for the Predictor: YAML/JSON End User Input → recommendation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

from optimizer.predictor import Predictor, PredictorConfig
from optimizer.schemas.calibration import CalibrationModel
from optimizer.schemas.predict import PredictInput


def _load_mapping(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in {path}")
    return data


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="TARC Predictor — recommend ballast + chute for race day."
    )
    p.add_argument(
        "--input",
        "-i",
        type=Path,
        required=True,
        help="PredictInput YAML/JSON (§3.2 End User Input)",
    )
    p.add_argument(
        "--calibration",
        "-c",
        type=Path,
        default=None,
        help="Calibration model JSON (§3.5). Defaults to identity corrections.",
    )
    p.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Write recommendation JSON here (also printed).",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mc-samples", type=int, default=80)
    p.add_argument(
        "--risk-aversion",
        type=float,
        default=0.0,
        help="0 = optimize mean penalty; 1 = optimize 90th-percentile penalty",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    predict_input = PredictInput.model_validate(_load_mapping(args.input))

    if args.calibration:
        calibration = CalibrationModel.model_validate(_load_mapping(args.calibration))
    else:
        calibration = CalibrationModel.identity(predict_input.rocket_id)

    config = PredictorConfig(
        seed=args.seed,
        monte_carlo_samples=args.mc_samples,
        risk_aversion=args.risk_aversion,
    )
    predictor = Predictor(calibration=calibration, config=config)
    recommendation = predictor.recommend(predict_input)
    payload = recommendation.model_dump()
    text = json.dumps(payload, indent=2)
    print(text)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
