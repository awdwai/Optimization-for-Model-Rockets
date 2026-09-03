"""Atmosphere / density-altitude helpers (§3.2 precedence rules)."""

from __future__ import annotations

# Standard constants (US customary / slug-ft system used by TARC tooling)
RHO_SL_SLUG_FT3 = 0.0023769  # sea-level standard density
T0_R = 518.67  # sea-level standard temp Rankine
L_R_PER_FT = 0.00356616  # ISA lapse rate °R/ft
G_OVER_R = 0.0341632 / 0.00356616  # ≈5.2561 for DA approx


def _f_to_rankine(temp_f: float) -> float:
    return temp_f + 459.67


def _saturation_vapor_pressure_inhg(temp_f: float) -> float:
    """Magnus approx converted to inHg."""
    temp_c = (temp_f - 32.0) * 5.0 / 9.0
    e_hpa = 6.112 * 10 ** ((7.5 * temp_c) / (237.7 + temp_c))
    return e_hpa * 0.02953  # hPa → inHg


def derive_air_density(
    air_temperature_f: float,
    barometric_pressure_inhg: float,
    relative_humidity_pct: float,
) -> float:
    """Moist-air density in slug/ft³ from station pressure + RH."""
    t_r = _f_to_rankine(air_temperature_f)
    e = _saturation_vapor_pressure_inhg(air_temperature_f) * (
        relative_humidity_pct / 100.0
    )
    # Dry air gas constant in (ft·lbf)/(slug·°R); vapor correction via virtual temp
    p_dry = max(barometric_pressure_inhg - e, 1e-6)
    # ρ = P / (R T) with R_air ≈ 1716 ft·lbf/(slug·°R); convert inHg → psf (*70.726)
    p_psf = p_dry * 70.726182
    e_psf = e * 70.726182
    # Virtual temperature style: ρ = (P_d/(R_d T) + P_v/(R_v T))
    r_d = 1716.49
    r_v = 2765.0  # approx water vapor gas constant in same units
    rho = p_psf / (r_d * t_r) + e_psf / (r_v * t_r)
    return float(rho)


def derive_density_altitude(
    air_temperature_f: float,
    barometric_pressure_inhg: float,
    relative_humidity_pct: float,
) -> float:
    """Approximate density altitude (ft) from temp/pressure/humidity."""
    rho = derive_air_density(
        air_temperature_f, barometric_pressure_inhg, relative_humidity_pct
    )
    return density_altitude_from_air_density(rho)


def density_altitude_from_air_density(rho_slug_ft3: float) -> float:
    ratio = max(min(rho_slug_ft3 / RHO_SL_SLUG_FT3, 1.5), 0.3)
    # σ = (1 - 6.87535e-6 * DA)^4.2561  → invert
    sigma = ratio
    da = (1.0 - sigma ** (1.0 / 4.2561)) / 6.87535e-6
    return float(da)


def air_density_from_density_altitude(density_altitude_ft: float) -> float:
    """ISA-style density from density altitude (precedence: DA → ρ)."""
    sigma = (1.0 - 6.87535e-6 * density_altitude_ft) ** 4.2561
    return float(RHO_SL_SLUG_FT3 * max(sigma, 0.05))
