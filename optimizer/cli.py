"""CLI for the Predictor: YAML/JSON End User Input → recommendation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

from optimizer.calibration_loader import load_calibration_for_rocket
from optimizer.openrocket_bridge import create_simulation_core
from optimizer.predictor import Predictor, PredictorConfig
from optimizer.schemas.immutable import ImmutableParameters
from optimizer.schemas.predict import PredictInput
from optimizer.sim_core import AnalyticStubSim


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
        "--calibration-dir",
        type=Path,
        default=None,
        help="Calibration registry root; loads <rocket_id>/latest.json when -c omitted.",
    )
    p.add_argument(
        "--immutable",
        type=Path,
        default=None,
        help="ImmutableParameters YAML/JSON (§3.1) including openrocket_file.",
    )
    p.add_argument(
        "--ork",
        type=Path,
        default=None,
        help="Path to .ork file (overrides immutable.openrocket_file). Enables OpenRocket.",
    )
    p.add_argument(
        "--jar",
        type=Path,
        default=None,
        help="OpenRocket JAR path (default: CLASSPATH / orhelper default).",
    )
    p.add_argument(
        "--stub",
        action="store_true",
        help="Force AnalyticStubSim even if --ork is provided.",
    )
    p.add_argument(
        "--no-stub-fallback",
        action="store_true",
        help="Fail if OpenRocket cannot start (do not fall back to stub).",
    )
    p.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Write recommendation JSON here (also printed).",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--mc-samples",
        type=int,
        default=80,
        help="Max Monte Carlo samples per candidate (speed/accuracy knob).",
    )
    p.add_argument(
        "--ballast-coarse-steps",
        type=int,
        default=30,
        help="Coarse ballast grid resolution across ballast_range_g.",
    )
    p.add_argument(
        "--ballast-fine-step",
        type=float,
        default=1.0,
        help="Fine ballast step in grams near the coarse optimum.",
    )
    p.add_argument(
        "--risk-aversion",
        type=float,
        default=0.0,
        help="0 = optimize mean penalty; 1 = optimize 90th-percentile penalty",
    )
    p.add_argument(
        "--min-stability",
        type=float,
        default=1.0,
        help="Rail-exit stability floor in calibers; below → unstable penalty.",
    )
    return p


def _resolve_ork_path(args: argparse.Namespace, input_path: Path) -> Path | None:
    if args.ork is not None:
        return args.ork
    if args.immutable is not None:
        imm = ImmutableParameters.model_validate(_load_mapping(args.immutable))
        return imm.resolved_ork_path(args.immutable.parent)
    return None


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    predict_input = PredictInput.model_validate(_load_mapping(args.input))

    calibration = load_calibration_for_rocket(
        predict_input.rocket_id,
        calibration_file=args.calibration,
        calibration_dir=args.calibration_dir,
    )

    config = PredictorConfig(
        seed=args.seed,
        monte_carlo_samples=args.mc_samples,
        ballast_coarse_steps=args.ballast_coarse_steps,
        ballast_fine_step_g=args.ballast_fine_step,
        risk_aversion=args.risk_aversion,
        min_stability_calibers=args.min_stability,
    )

    ork_path = _resolve_ork_path(args, args.input)
    sim = None
    bridge_to_close = None
    try:
        if args.stub or ork_path is None:
            sim = AnalyticStubSim()
        else:
            sim = create_simulation_core(
                ork_path,
                jar_path=args.jar,
                prefer_openrocket=True,
                allow_stub_fallback=not args.no_stub_fallback,
            )
            # OpenRocketBridge needs close(); stub does not
            if hasattr(sim, "close"):
                bridge_to_close = sim

        predictor = Predictor(sim=sim, calibration=calibration, config=config)
        recommendation = predictor.recommend(predict_input)
        payload = recommendation.model_dump(mode="json")
        text = json.dumps(payload, indent=2)
        print(text)
        if args.output:
            args.output.write_text(text + "\n", encoding="utf-8")
        return 0
    finally:
        if bridge_to_close is not None:
            bridge_to_close.close()


if __name__ == "__main__":
    sys.exit(main())
