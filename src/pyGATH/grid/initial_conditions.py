"""Built-in grid initial conditions."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import Any

import jax.numpy as jnp
import numpy as np

from .geometry import Geometry
from .hydro import HydroFields


def uniform(
    grid,
    *,
    ne: float = 1.0,
    Te: float = 1.0,
    Ti: float = 1.0,
    velocity=(0.0, 0.0, 0.0),
) -> HydroFields:
    """Return spatially uniform vertex fields.

    Densities are in m\N{SUPERSCRIPT MINUS}\N{SUPERSCRIPT THREE}, temperatures in eV, and velocity in
    m/s. Velocity components use the grid's local orthonormal basis.
    """
    shape = grid.vertex_shape
    dtype = grid.xb.dtype
    return HydroFields(
        ne=jnp.full(shape, ne, dtype=dtype),
        Te=jnp.full(shape, Te, dtype=dtype),
        Ti=jnp.full(shape, Ti, dtype=dtype),
        velocity=jnp.broadcast_to(jnp.asarray(velocity, dtype=dtype), (*shape, 3)),
    )


def linear_density_x(
    grid,
    *,
    dne_dx: float,
    ne_at_x_min: float = 0.0,
    Te: float = 1.0,
    Ti: float = 1.0,
    velocity=(0.0, 0.0, 0.0),
) -> HydroFields:
    """Return a Cartesian plasma with electron density linear in ``x``.

    ``dne_dx`` is in m\N{SUPERSCRIPT MINUS}\N{SUPERSCRIPT FOUR}; densities are
    in m\N{SUPERSCRIPT MINUS}\N{SUPERSCRIPT THREE}, temperatures in eV, and
    velocity in m/s. ``ne_at_x_min`` is the density on the lower x boundary.
    """
    if grid.geom is not Geometry.CARTESIAN:
        raise ValueError("linear_density_x requires a Cartesian grid")
    x, _y, _z = grid.vertex_mesh()
    shape = grid.vertex_shape
    dtype = grid.xb.dtype
    ne = ne_at_x_min + dne_dx * (x - grid.xb[0])
    return HydroFields(
        ne=ne,
        Te=jnp.full(shape, Te, dtype=dtype),
        Ti=jnp.full(shape, Ti, dtype=dtype),
        velocity=jnp.broadcast_to(jnp.asarray(velocity, dtype=dtype), (*shape, 3)),
    )


def omega_lilac_reflection_1d(
    grid,
    *,
    omega_rad_s: float = 5.366_528_681_791_604e15,
    scale: float = 1.0 / 16.0,
    coordinate_offset_m: float = 36.42e-6,
    density_at_scale_over_ncritical: float = 1.165,
    density_scale_radius_m: float | None = None,
    density_power: float = 3.78,
    electron_temperature_ev: float = 1.0,
    ion_temperature_ev: float = 1.0,
) -> HydroFields:
    """Return the 1-D reflection profile from Follett et al. (2022), Sec. III A.

    The paper's spherical LILAC fit is mapped onto Cartesian ``x`` through
    ``r = coordinate_offset_m - x``. Its default ``S=1/16`` parameters put
    the critical surface near ``x=14.1 um`` for a 351 nm laser. Temperatures
    are numerically required hydro fields but do not affect the intended
    field-only example when inverse bremsstrahlung is disabled.
    """
    if grid.geom is not Geometry.CARTESIAN or grid.dimensions != 1:
        raise ValueError(
            "omega_lilac_reflection_1d requires a one-dimensional Cartesian grid"
        )
    positive_parameters = {
        "omega_rad_s": omega_rad_s,
        "scale": scale,
        "coordinate_offset_m": coordinate_offset_m,
        "density_at_scale_over_ncritical": density_at_scale_over_ncritical,
        "density_power": density_power,
        "electron_temperature_ev": electron_temperature_ev,
        "ion_temperature_ev": ion_temperature_ev,
    }
    for name, value in positive_parameters.items():
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
    density_radius = (
        343.0e-6 * scale if density_scale_radius_m is None else density_scale_radius_m
    )
    if not np.isfinite(density_radius) or density_radius <= 0.0:
        raise ValueError("density_scale_radius_m must be finite and positive")
    if coordinate_offset_m <= float(grid.xb[-1]):
        raise ValueError(
            "coordinate_offset_m must exceed the upper x extent so r stays positive"
        )

    vacuum_permittivity = 8.854_187_812_8e-12
    electron_mass = 9.109_383_713_9e-31
    elementary_charge = 1.602_176_634e-19
    ncritical = (
        vacuum_permittivity * electron_mass * omega_rad_s**2 / elementary_charge**2
    )
    x, _y, _z = grid.vertex_mesh()
    radius = coordinate_offset_m - x
    ne = (
        density_at_scale_over_ncritical
        * ncritical
        * (density_radius / radius) ** density_power
    )
    Te = jnp.full(grid.vertex_shape, electron_temperature_ev, dtype=grid.xb.dtype)
    Ti = jnp.full(grid.vertex_shape, ion_temperature_ev, dtype=grid.xb.dtype)
    velocity = jnp.zeros((*grid.vertex_shape, 3), dtype=grid.xb.dtype)
    return HydroFields(ne=ne, Te=Te, Ti=Ti, velocity=velocity)


def omega_lilac_reflection_2d(
    grid,
    *,
    omega_rad_s: float = 5.366_528_681_791_604e15,
    scale: float = 1.0 / 64.0,
    density_at_scale_over_ncritical: float = 1.165,
    density_scale_radius_m: float | None = None,
    density_power: float = 3.78,
    maximum_density_over_ncritical: float = 4.0,
    electron_temperature_ev: float = 1.0,
    ion_temperature_ev: float = 1.0,
) -> HydroFields:
    """Return the 2-D reflection profile from Follett et al. (2022), Sec. III B.

    The circularly symmetric LILAC fit is evaluated directly in Cartesian
    ``x-y`` or cylindrical ``r-phi`` coordinates. Density is capped well past
    the critical surface to keep the unresolved central singularity finite;
    this does not affect rays, which turn where ``ne/ncritical <= 1``.
    """
    if grid.dimensions != 2 or grid.geom not in (
        Geometry.CARTESIAN,
        Geometry.CYLINDRICAL,
    ):
        raise ValueError(
            "omega_lilac_reflection_2d requires a two-dimensional Cartesian "
            "or cylindrical grid"
        )
    positive_parameters = {
        "omega_rad_s": omega_rad_s,
        "scale": scale,
        "density_at_scale_over_ncritical": density_at_scale_over_ncritical,
        "density_power": density_power,
        "maximum_density_over_ncritical": maximum_density_over_ncritical,
        "electron_temperature_ev": electron_temperature_ev,
        "ion_temperature_ev": ion_temperature_ev,
    }
    for name, value in positive_parameters.items():
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
    if maximum_density_over_ncritical <= 1.0:
        raise ValueError("maximum_density_over_ncritical must exceed one")
    density_radius = (
        343.0e-6 * scale if density_scale_radius_m is None else density_scale_radius_m
    )
    if not np.isfinite(density_radius) or density_radius <= 0.0:
        raise ValueError("density_scale_radius_m must be finite and positive")

    vacuum_permittivity = 8.854_187_812_8e-12
    electron_mass = 9.109_383_713_9e-31
    elementary_charge = 1.602_176_634e-19
    ncritical = (
        vacuum_permittivity * electron_mass * omega_rad_s**2 / elementary_charge**2
    )
    first, second, _z = grid.vertex_mesh()
    radius = jnp.hypot(first, second) if grid.geom is Geometry.CARTESIAN else first
    cap_radius = density_radius * (
        density_at_scale_over_ncritical / maximum_density_over_ncritical
    ) ** (1.0 / density_power)
    density_ratio = (
        density_at_scale_over_ncritical
        * (density_radius / jnp.maximum(radius, cap_radius)) ** density_power
    )
    ne = ncritical * jnp.minimum(density_ratio, maximum_density_over_ncritical)
    Te = jnp.full(grid.vertex_shape, electron_temperature_ev, dtype=grid.xb.dtype)
    Ti = jnp.full(grid.vertex_shape, ion_temperature_ev, dtype=grid.xb.dtype)
    velocity = jnp.zeros((*grid.vertex_shape, 3), dtype=grid.xb.dtype)
    return HydroFields(ne=ne, Te=Te, Ti=Ti, velocity=velocity)


def omega_lilac_azimuthal_2d(
    grid,
    *,
    omega_rad_s: float = 5.366_528_681_791_604e15,
    scale: float = 1.0 / 64.0,
    density_at_scale_over_ncritical: float = 1.165,
    density_scale_radius_m: float | None = None,
    density_power: float = 3.78,
    maximum_density_over_ncritical: float = 4.0,
    electron_temperature_ev: float = 70.0,
    ion_temperature_over_electron: float = 0.5,
    ion_charge: float = 3.1,
    ion_mass_over_electron_mass: float = 10_319.2,
    mach_offset: float = 1.41,
    mach_log_coefficient: float = 1.37,
    flow_offset_radius_m: float | None = None,
    flow_scale_radius_m: float | None = None,
) -> HydroFields:
    """Return the azimuthally symmetric 2-D plasma from Follett et al.

    This is the density profile from Eq. (13) and the radial Mach-number fit
    from Eq. (33). It supports Cartesian ``x-y`` and cylindrical ``r-phi``
    representations of the same physical plasma. The central density and
    flow are regularized inside the inaccessible overdense region.
    """
    if grid.dimensions != 2 or grid.geom not in (
        Geometry.CARTESIAN,
        Geometry.CYLINDRICAL,
    ):
        raise ValueError(
            "omega_lilac_azimuthal_2d requires a two-dimensional Cartesian "
            "or cylindrical grid"
        )
    positive_parameters = {
        "omega_rad_s": omega_rad_s,
        "scale": scale,
        "density_at_scale_over_ncritical": density_at_scale_over_ncritical,
        "density_power": density_power,
        "maximum_density_over_ncritical": maximum_density_over_ncritical,
        "electron_temperature_ev": electron_temperature_ev,
        "ion_temperature_over_electron": ion_temperature_over_electron,
        "ion_charge": ion_charge,
        "ion_mass_over_electron_mass": ion_mass_over_electron_mass,
    }
    for name, value in positive_parameters.items():
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
    if maximum_density_over_ncritical <= 1.0:
        raise ValueError("maximum_density_over_ncritical must exceed one")
    if not np.isfinite(mach_offset) or not np.isfinite(mach_log_coefficient):
        raise ValueError("Mach-fit coefficients must be finite")

    density_radius = (
        343.0e-6 * scale if density_scale_radius_m is None else density_scale_radius_m
    )
    flow_offset = (
        204.0e-6 * scale if flow_offset_radius_m is None else flow_offset_radius_m
    )
    flow_radius = (
        343.0e-6 * scale if flow_scale_radius_m is None else flow_scale_radius_m
    )
    if density_radius <= 0.0 or flow_radius <= 0.0 or flow_offset < 0.0:
        raise ValueError("LILAC-fit radii must be positive (offset nonnegative)")

    vacuum_permittivity = 8.854_187_812_8e-12
    electron_mass = 9.109_383_713_9e-31
    elementary_charge = 1.602_176_634e-19
    ncritical = (
        vacuum_permittivity * electron_mass * omega_rad_s**2 / elementary_charge**2
    )
    first, second, _z = grid.vertex_mesh()
    radius = jnp.hypot(first, second) if grid.geom is Geometry.CARTESIAN else first
    cap_radius = density_radius * (
        density_at_scale_over_ncritical / maximum_density_over_ncritical
    ) ** (1.0 / density_power)
    evaluation_radius = jnp.maximum(radius, cap_radius)
    density_ratio = (
        density_at_scale_over_ncritical
        * (density_radius / evaluation_radius) ** density_power
    )
    ne = ncritical * jnp.minimum(density_ratio, maximum_density_over_ncritical)
    Te = jnp.full(grid.vertex_shape, electron_temperature_ev, dtype=grid.xb.dtype)
    Ti = ion_temperature_over_electron * Te

    mach_number = mach_offset + mach_log_coefficient * jnp.log(
        (evaluation_radius - flow_offset) / flow_radius
    )
    ion_mass = ion_mass_over_electron_mass * electron_mass
    sound_speed = jnp.sqrt(
        (ion_charge * electron_temperature_ev + 3.0 * Ti) * elementary_charge / ion_mass
    )
    radial_speed = jnp.where(radius > 0.0, mach_number * sound_speed, 0.0)
    if grid.geom is Geometry.CARTESIAN:
        safe_radius = jnp.where(radius > 0.0, radius, 1.0)
        velocity = jnp.stack(
            (
                radial_speed * first / safe_radius,
                radial_speed * second / safe_radius,
                jnp.zeros_like(radial_speed),
            ),
            axis=-1,
        )
    else:
        velocity = jnp.stack(
            (radial_speed, jnp.zeros_like(radial_speed), jnp.zeros_like(radial_speed)),
            axis=-1,
        )
    return HydroFields(ne=ne, Te=Te, Ti=Ti, velocity=velocity)


def omega_lilac_fit(
    grid,
    *,
    omega_rad_s: float = 5.36e15,
    scale: float = 1.0,
    density_at_scale_over_ncritical: float = 1.165,
    density_scale_radius_m: float | None = None,
    density_power: float = 3.78,
    electron_temperature_ev: float = 2.0e3,
    ion_temperature_over_electron: float = 0.5,
    ion_charge: float = 3.1,
    ion_mass_over_electron_mass: float = 10_319.2,
    mach_offset: float = 1.41,
    mach_log_coefficient: float = 1.37,
    flow_offset_radius_m: float | None = None,
    flow_scale_radius_m: float | None = None,
) -> HydroFields:
    """Return the spherical OMEGA/LILAC fit used by Follett et al. (2022).

    The electron density follows Eq. (13),
    ``ne/ncritical = 1.165 * (343 um * scale / r)**3.78``, and the
    radially outward Mach-number fit follows Eq. (33). The defaults reproduce
    the paper's full-scale 60-beam case: ``Te=2 keV``, ``Ti=Te/2``,
    ``Z=3.1``, and ``mi/me=10319.2``. Velocities are returned as physical
    orthonormal spherical components ``(v_r, v_phi, v_theta)``.
    """
    if grid.geom is not Geometry.SPHERICAL:
        raise ValueError("omega_lilac_fit requires a spherical grid")
    positive_parameters = {
        "omega_rad_s": omega_rad_s,
        "scale": scale,
        "density_at_scale_over_ncritical": density_at_scale_over_ncritical,
        "density_power": density_power,
        "electron_temperature_ev": electron_temperature_ev,
        "ion_temperature_over_electron": ion_temperature_over_electron,
        "ion_charge": ion_charge,
        "ion_mass_over_electron_mass": ion_mass_over_electron_mass,
    }
    for name, value in positive_parameters.items():
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
    if not np.isfinite(mach_offset) or not np.isfinite(mach_log_coefficient):
        raise ValueError("Mach-fit coefficients must be finite")

    density_radius = (
        343.0e-6 * scale if density_scale_radius_m is None else density_scale_radius_m
    )
    flow_offset = (
        204.0e-6 * scale if flow_offset_radius_m is None else flow_offset_radius_m
    )
    flow_radius = (
        343.0e-6 * scale if flow_scale_radius_m is None else flow_scale_radius_m
    )
    if density_radius <= 0.0 or flow_radius <= 0.0 or flow_offset < 0.0:
        raise ValueError("OMEGA fit radii must be positive (offset nonnegative)")
    if float(grid.xb[0]) <= flow_offset:
        raise ValueError(
            "omega_lilac_fit requires the radial grid minimum to exceed "
            "flow_offset_radius_m so the logarithmic Mach fit is defined"
        )

    vacuum_permittivity = 8.854_187_812_8e-12
    electron_mass = 9.109_383_713_9e-31
    elementary_charge = 1.602_176_634e-19
    ncritical = (
        vacuum_permittivity * electron_mass * omega_rad_s**2 / elementary_charge**2
    )
    radius, _phi, _theta = grid.vertex_mesh()
    ne = (
        density_at_scale_over_ncritical
        * ncritical
        * (density_radius / radius) ** density_power
    )
    Te = jnp.full(grid.vertex_shape, electron_temperature_ev, dtype=grid.xb.dtype)
    Ti = ion_temperature_over_electron * Te
    mach_number = mach_offset + mach_log_coefficient * jnp.log(
        (radius - flow_offset) / flow_radius
    )
    ion_mass = ion_mass_over_electron_mass * electron_mass
    sound_speed = jnp.sqrt(
        (ion_charge * electron_temperature_ev + 3.0 * Ti) * elementary_charge / ion_mass
    )
    velocity = jnp.stack(
        (mach_number * sound_speed, jnp.zeros_like(ne), jnp.zeros_like(ne)),
        axis=-1,
    )
    return HydroFields(ne=ne, Te=Te, Ti=Ti, velocity=velocity)


INITIAL_CONDITIONS: dict[str, Callable[..., HydroFields]] = {
    "linear_density_x": linear_density_x,
    "omega_lilac_azimuthal_2d": omega_lilac_azimuthal_2d,
    "omega_lilac_fit": omega_lilac_fit,
    "omega_lilac_reflection_1d": omega_lilac_reflection_1d,
    "omega_lilac_reflection_2d": omega_lilac_reflection_2d,
    "uniform": uniform,
}


def resolve(initial_condition: str | Callable[[Any], HydroFields], **parameters):
    """Resolve a built-in name or user callable into a one-argument callable."""
    if isinstance(initial_condition, str):
        try:
            function = INITIAL_CONDITIONS[initial_condition.lower()]
        except KeyError as error:
            choices = ", ".join(sorted(INITIAL_CONDITIONS))
            raise ValueError(
                f"Unknown initial condition {initial_condition!r}; expected {choices}"
            ) from error
    elif callable(initial_condition):
        function = initial_condition
    else:
        raise TypeError("initial_condition must be a registered name or callable")
    return partial(function, **parameters) if parameters else function
