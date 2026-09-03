"""TARC Predictor (race-day optimizer) — ballast + chute search."""

from optimizer.openrocket_bridge import OpenRocketBridge, create_simulation_core
from optimizer.predictor import Predictor, PredictorConfig
from optimizer.schemas.immutable import ImmutableParameters
from optimizer.schemas.predict import PredictInput, Recommendation
from optimizer.sim_core import AnalyticStubSim, SimulationCore

__all__ = [
    "AnalyticStubSim",
    "ImmutableParameters",
    "OpenRocketBridge",
    "Predictor",
    "PredictorConfig",
    "PredictInput",
    "Recommendation",
    "SimulationCore",
    "create_simulation_core",
]
