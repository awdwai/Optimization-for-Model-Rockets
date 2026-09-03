"""Predictor — race-day ballast/chute optimizer (§4.3)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from optimizer.monte_carlo import MCResult, MonteCarloRunner
from optimizer.schemas.calibration import CalibrationModel
from optimizer.schemas.predict import (
    Alternate,
    Confidence,
    PredictInput,
    Recommendation,
)
from optimizer.scoring_rules import ScoringRule, load_scoring_rule
from optimizer.sim_core import AnalyticStubSim, SimulationCore


@dataclass
class PredictorConfig:
    scoring_version: str = "tarc_default"
    min_stability_calibers: float = 1.0
    monte_carlo_samples: int = 100
    monte_carlo_min_samples: int = 40
    monte_carlo_batch_size: int = 40
    monte_carlo_se_threshold: float = 1.0
    ballast_coarse_steps: int = 30
    ballast_refine_halfwidth_g: float = 15.0
    ballast_fine_step_g: float = 1.0
    risk_aversion: float = 0.0  # 0 = mean penalty, 1 = p90 penalty
    seed: int = 42


@dataclass
class _CandidateEval:
    chute_diam_in: float
    ballast_g: float
    mc: MCResult

    def score(self, risk_aversion: float) -> float:
        ra = min(max(risk_aversion, 0.0), 1.0)
        return (1.0 - ra) * self.mc.mean_penalty + ra * self.mc.p90_penalty


class Predictor:
    def __init__(
        self,
        sim: SimulationCore | None = None,
        calibration: CalibrationModel | None = None,
        scoring: ScoringRule | None = None,
        config: PredictorConfig | None = None,
    ) -> None:
        self.config = config or PredictorConfig()
        self.sim = sim or AnalyticStubSim()
        self.calibration = calibration or CalibrationModel.identity()
        self.scoring = scoring or load_scoring_rule(self.config.scoring_version)
        self.mc = MonteCarloRunner(
            self.sim,
            self.scoring,
            min_stability_calibers=self.config.min_stability_calibers,
            batch_size=self.config.monte_carlo_batch_size,
            se_threshold=self.config.monte_carlo_se_threshold,
            max_samples=self.config.monte_carlo_samples,
            min_samples=self.config.monte_carlo_min_samples,
            seed=self.config.seed,
        )

    def _ballast_grid(self, lo: float, hi: float, steps: int) -> np.ndarray:
        if hi <= lo:
            return np.array([lo], dtype=float)
        return np.linspace(lo, hi, num=max(steps, 2))

    def _unique_ballast(self, values: np.ndarray) -> list[float]:
        # Round to 0.1 g so coarse/fine overlap does not double-evaluate
        rounded = np.round(values.astype(float), 1)
        return [float(v) for v in dict.fromkeys(rounded.tolist())]

    def _evaluate(
        self, predict_input: PredictInput, chute: float, ballast: float
    ) -> _CandidateEval:
        mc = self.mc.evaluate(
            predict_input.environment,
            egg_mass_g=predict_input.payload.egg_mass_g,
            ballast_g=ballast,
            chute_diam_in=chute,
            motor_designation=predict_input.motor_designation,
            calibration=self.calibration,
            objectives=predict_input.contest_objectives,
        )
        return _CandidateEval(chute_diam_in=chute, ballast_g=ballast, mc=mc)

    def recommend(self, predict_input: PredictInput) -> Recommendation:
        cfg = self.config
        lo, hi = predict_input.ballast_min, predict_input.ballast_max
        best_by_chute: dict[float, _CandidateEval] = {}

        for chute in predict_input.chute_inventory_in:
            coarse = self._unique_ballast(self._ballast_grid(lo, hi, cfg.ballast_coarse_steps))
            coarse_evals = [
                self._evaluate(predict_input, chute, b) for b in coarse
            ]
            coarse_best = min(coarse_evals, key=lambda e: e.score(cfg.risk_aversion))

            refine_lo = max(lo, coarse_best.ballast_g - cfg.ballast_refine_halfwidth_g)
            refine_hi = min(hi, coarse_best.ballast_g + cfg.ballast_refine_halfwidth_g)
            n_fine = max(
                2, int(round((refine_hi - refine_lo) / cfg.ballast_fine_step_g)) + 1
            )
            fine = self._unique_ballast(self._ballast_grid(refine_lo, refine_hi, n_fine))
            # Skip ballast already evaluated on the coarse pass
            coarse_set = set(coarse)
            fine_only = [b for b in fine if b not in coarse_set]
            fine_evals = [
                self._evaluate(predict_input, chute, b) for b in fine_only
            ]
            chute_best = min(
                coarse_evals + fine_evals, key=lambda e: e.score(cfg.risk_aversion)
            )
            best_by_chute[chute] = chute_best

        overall = min(best_by_chute.values(), key=lambda e: e.score(cfg.risk_aversion))
        obj = predict_input.contest_objectives
        within_apogee = (
            abs(overall.mc.mean_apogee_ft - obj.desired_apogee_ft)
            <= obj.desired_apogee_tolerance_ft
        )
        within_time = (
            abs(overall.mc.mean_descent_time_s - obj.desired_time_s)
            <= obj.desired_time_tolerance_s
        )
        feasible = within_apogee and within_time

        alternates = [
            Alternate(
                ballast_g=round(ev.ballast_g, 1),
                chute_diam_in=chute,
                predicted_score_penalty=round(ev.score(cfg.risk_aversion), 3),
                predicted_apogee_ft=round(ev.mc.mean_apogee_ft, 2),
                predicted_descent_time_s=round(ev.mc.mean_descent_time_s, 2),
                hit_probability=round(ev.mc.hit_probability, 4),
            )
            for chute, ev in sorted(best_by_chute.items())
            if chute != overall.chute_diam_in
        ]

        return Recommendation(
            ballast_g=round(overall.ballast_g, 1),
            chute_diam_in=overall.chute_diam_in,
            predicted_apogee_ft=round(overall.mc.mean_apogee_ft, 2),
            predicted_descent_time_s=round(overall.mc.mean_descent_time_s, 2),
            predicted_score_penalty=round(overall.score(cfg.risk_aversion), 3),
            feasible=feasible,
            closest_miss_ft=round(
                abs(overall.mc.mean_apogee_ft - obj.desired_apogee_ft), 2
            ),
            closest_miss_s=round(
                abs(overall.mc.mean_descent_time_s - obj.desired_time_s), 2
            ),
            confidence=Confidence(
                apogee_std_ft=round(overall.mc.apogee_std_ft, 2),
                descent_time_std_s=round(overall.mc.descent_time_std_s, 2),
                hit_probability=round(overall.mc.hit_probability, 4),
                sample_count=overall.mc.sample_count,
            ),
            alternates=alternates,
            risk_averse_penalty_p90=round(overall.mc.p90_penalty, 3),
        )
