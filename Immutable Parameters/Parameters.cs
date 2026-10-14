// Immutable Parameters
// Fixed per rocket design; not set by the end user at flight time.
// Set values only in an object initializer; properties cannot be reassigned later.

namespace Rocket.ImmutableParameters;

/// <summary>
/// Design-fixed rocket parameters/// </summary>
public sealed class RocketParameters
{
    public required string openrocket_file { get; init; } // path (.ork) — required
    public string? rocket_id { get; init; } = null;
    public mass_properties? mass_properties { get; init; } = null;
    public motor_properties? motor_properties { get; init; } = null;
}

public sealed class mass_properties
{
    public float? dry_mass_g { get; init; } = null;      // airframe only 
    public float? wet_mass { get; init; } = null;        // airframe and motor mass 
    public float? cm_in { get; init; } = null;           // as-built, empty condition — define reference state
    public float? cp_in { get; init; } = null;           // from OpenRocket static stability analysis
    public float? moi_axial { get; init; } = null;       // pulled from .ork if not supplied
    public float? moi_transverse { get; init; } = null;  // pulled from .ork if not supplied
}

public sealed class motor_properties
{
    public string? motor_designation { get; init; } = null; // e.g. "F50-4T" - nominal motor, immutable per design
    public string? thrust_curve_path { get; init; } = null; // .eng (RASP Format) — thrust vs. time
    public float? motor_wet_mass { get; init; } = null;     // used to derive the propellent mass
    public float? motor_dry_mass { get; init; } = null;     // used to derive the propellent mass

    /// <summary>
    /// Propellent mass used to determine the mass lost as the motor fires.
    /// </summary>
    public float? propellent_mass =>
        motor_wet_mass is { } wet && motor_dry_mass is { } dry
            ? wet - dry
            : null;
}
