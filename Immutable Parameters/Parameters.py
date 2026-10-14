# Immutable Parameters
# Fixed per rocket design; not set by the end user at flight time.
# Set values only at construction; fields cannot be reassigned later.

from dataclasses import dataclass


@dataclass(frozen=True)
class mass_properties:
    dry_mass_g: float | None = None  # airframe only
    wet_mass: float | None = None  # airframe and motor mass
    cm_in: float | None = None  # as-built, empty condition — define reference state
    cp_in: float | None = None  # from OpenRocket static stability analysis
    moi_axial: float | None = None  # pulled from .ork if not supplied
    moi_transverse: float | None = None  # pulled from .ork if not supplied


@dataclass(frozen=True)
class motor_properties:
    motor_designation: str | None = None  # e.g. "F50-4T" - nominal motor, immutable per design
    thrust_curve_path: str = "thrust_curve.csv"  # .eng (RASP) when provided; otherwise this CSV
    motor_wet_mass: float | None = None  # used to derive the propellent mass
    motor_dry_mass: float | None = None  # used to derive the propellent mass

    @property
    def propellent_mass(self) -> float | None:
        """Propellent mass used to determine the mass lost as the motor fires."""
        if self.motor_wet_mass is None or self.motor_dry_mass is None:
            return None
        return self.motor_wet_mass - self.motor_dry_mass


@dataclass(frozen=True)
class RocketParameters:
    """Design-fixed rocket parameters."""

    openrocket_file: str  # path (.ork) — required
    rocket_id: str | None = None
    mass_properties: mass_properties | None = None
    motor_properties: motor_properties | None = None
