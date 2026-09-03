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
            # Keep supplied density; leave DA unset unless caller derives it later.
            pass
        # If both supplied: trust DA for density (precedence rule) only when
        # density was not independently intended — re-derive density from DA.
        elif da is not None and rho is not None:
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


class EnvironmentalState(BaseModel):
    atmosphere: Atmosphere
    wind: Wind
    surface: Surface

    @model_validator(mode="after")
    def refine_thermal_from_delta(self) -> EnvironmentalState:
        # Respect explicit override; otherwise derive from ground − air delta.
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

