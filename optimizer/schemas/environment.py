"""Shared EnvironmentalState schema (§3.2 / §2.2)."""

from __future__ import annotations

from enum import Enum
from pydantic import BaseModel, Field, model_validator


class ThermalActivity(str, Enum):
    none = "none"
    light = "light"
    moderate = "moderate"
    strong = "strong"


class WindLayer(BaseModel):
    layer_ft: tuple[float, float]
    speed_mph: float = Field(ge=0)
    direction_deg: float = Field(ge=0, lt=360)


class Atmosphere(BaseModel):
    air_temperature_f: float
    barometric_pressure_inhg: float = Field(gt=0, description="Station pressure")
    relative_humidity_pct: float = Field(ge=0, le=100)
    air_density_slug_ft3: float | None = None
    density_altitude_ft: float | None = None

    @model_validator(mode="after")
    def apply_density_precedence(self) -> Atmosphere:
        from optimizer.atmosphere import (
            air_density_from_density_altitude,
            derive_air_density,
            derive_density_altitude,
        )

        da = self.density_altitude_ft
        rho = self.air_density_slug_ft3

        if da is not None and rho is None:
            object.__setattr__(
                self, "air_density_slug_ft3", air_density_from_density_altitude(da)
            )
        elif da is None and rho is None:
            object.__setattr__(
                self,
                "air_density_slug_ft3",
                derive_air_density(
                    self.air_temperature_f,
                    self.barometric_pressure_inhg,
                    self.relative_humidity_pct,
                ),
            )
            object.__setattr__(
                self,
                "density_altitude_ft",
                derive_density_altitude(
                    self.air_temperature_f,
                    self.barometric_pressure_inhg,
                    self.relative_humidity_pct,
                ),
            )
        elif da is None and rho is not None:
            from optimizer.atmosphere import density_altitude_from_air_density

            object.__setattr__(
                self, "density_altitude_ft", density_altitude_from_air_density(rho)
            )
        elif da is not None and rho is not None:
            # Precedence: DA wins — never independently keep a conflicting ρ.
            object.__setattr__(
                self, "air_density_slug_ft3", air_density_from_density_altitude(da)
            )
        return self


class Wind(BaseModel):
    ground_wind_speed_mph: float = Field(ge=0)
    wind_direction_deg: float = Field(ge=0, lt=360)
    wind_gust_mph: float = Field(ge=0)
    wind_gradient: list[WindLayer] = Field(default_factory=list)


class Surface(BaseModel):
    ground_temperature_f: float
    thermal_activity_estimate: ThermalActivity | None = None

    @property
    def thermal_was_explicit(self) -> bool:
        """True when the caller set thermal_activity_estimate (including none)."""
        return "thermal_activity_estimate" in self.model_fields_set


class EnvironmentalState(BaseModel):
    atmosphere: Atmosphere
    wind: Wind
    surface: Surface

    @model_validator(mode="after")
    def refine_thermal_from_delta(self) -> EnvironmentalState:
        # Respect explicit override (including thermal_activity_estimate: none).
        if self.surface.thermal_was_explicit:
            return self
        if self.surface.thermal_activity_estimate is not None:
            return self
        delta = self.surface.ground_temperature_f - self.atmosphere.air_temperature_f
        if delta < 5:
            est = ThermalActivity.none
        elif delta < 12:
            est = ThermalActivity.light
        elif delta < 20:
            est = ThermalActivity.moderate
        else:
            est = ThermalActivity.strong
        object.__setattr__(self.surface, "thermal_activity_estimate", est)
        return self

