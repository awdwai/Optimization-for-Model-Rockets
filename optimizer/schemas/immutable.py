"""Immutable Parameters — Predictor-facing subset (§3.1)."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field, field_validator


class MassProperties(BaseModel):
    """As-built empty/dry reference (no egg, no ballast)."""

    dry_mass_g: float = Field(gt=0, description="Airframe only — NO egg, NO ballast, NO motor")
    wet_mass_g: float | None = Field(
        default=None, description="Airframe + motor mass when known"
    )
    cm_in: float | None = Field(default=None, description="CG from nose, empty condition")
    cp_in: float | None = Field(default=None, description="CP from OpenRocket static analysis")
    moi_axial: float | None = None
    moi_transverse: float | None = None


class MotorProperties(BaseModel):
    """Nominal motor for this airframe design (may be overridden per flight in EUI)."""

    motor_designation: str | None = None
    thrust_curve_path: str | None = None
    motor_wet_mass_g: float | None = None
    motor_dry_mass_g: float | None = None


class ImmutableParameters(BaseModel):
    """Geometry/mass ground truth the Predictor loads before OpenRocket runs."""

    rocket_id: str
    openrocket_file: str
    mass_properties: MassProperties | None = None
    motor_properties: MotorProperties | None = None

    @field_validator("openrocket_file")
    @classmethod
    def _nonempty_path(cls, v: str) -> str:
        if not str(v).strip():
            raise ValueError("openrocket_file must be a non-empty path")
        return v

    def resolved_ork_path(self, base: Path | None = None) -> Path:
        path = Path(self.openrocket_file)
        if not path.is_absolute() and base is not None:
            path = base / path
        return path
