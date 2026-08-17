"""Cold-plasma quantities used by ray initialization and evolution."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp

VACUUM_PERMITTIVITY = 8.854_187_812_8e-12
ELECTRON_MASS = 9.109_383_713_9e-31
ELEMENTARY_CHARGE = 1.602_176_634e-19
SPEED_OF_LIGHT = 299_792_458.0
REDUCED_PLANCK_CONSTANT = 1.054_571_817e-34
DEFAULT_DENSITY_RATIO_THRESHOLD = 0.999


@dataclass(frozen=True)
class InverseBremsstrahlungOptions:
    """Controls for inverse-bremsstrahlung optical depth."""

    enabled: bool = False
    minimum_coulomb_log: float = 2.0
    coulomb_log_override: float | None = None
    critical_collision_frequency_hz: float | None = None

    def validate(self) -> None:
        if self.minimum_coulomb_log <= 0.0:
            raise ValueError("minimum_coulomb_log must be positive")
        if (
            self.coulomb_log_override is not None
            and self.coulomb_log_override < self.minimum_coulomb_log
        ):
            raise ValueError("coulomb_log_override cannot be below minimum_coulomb_log")
        if (
            self.critical_collision_frequency_hz is not None
            and self.critical_collision_frequency_hz <= 0.0
        ):
            raise ValueError("critical_collision_frequency_hz must be positive")
        if (
            self.critical_collision_frequency_hz is not None
            and self.coulomb_log_override is not None
        ):
            raise ValueError(
                "critical_collision_frequency_hz and coulomb_log_override are "
                "mutually exclusive"
            )


def critical_density(omega):
    """Return critical electron density in m^-3 for angular frequency rad/s."""
    omega = jnp.asarray(omega, dtype=jnp.float64)
    return VACUUM_PERMITTIVITY * ELECTRON_MASS * omega**2 / ELEMENTARY_CHARGE**2


def safe_density_ratio(
    ne_over_ncrit,
    threshold: float = DEFAULT_DENSITY_RATIO_THRESHOLD,
):
    """Smoothly cap ``ne/ncrit`` below one without modifying density gradients.

    The upper branch matches both value and first derivative at ``threshold``
    and approaches ``1 - (1 - threshold)^2`` asymptotically. This function is
    only for density values used in permittivity-like expressions; derivatives
    of electron density must remain the raw physical derivatives.
    """
    ratio = jnp.asarray(ne_over_ncrit, dtype=jnp.float64)
    threshold = jnp.asarray(threshold, dtype=jnp.float64)
    amplitude = 1.0 - threshold
    cap = 1.0 - amplitude**2
    transition_scale = cap - threshold
    excess = jnp.maximum(ratio - threshold, 0.0)
    upper = cap - transition_scale * jnp.exp(-excess / transition_scale)
    return jnp.where(ratio <= threshold, ratio, upper)


def safe_permittivity(
    ne_over_ncrit,
    threshold: float = DEFAULT_DENSITY_RATIO_THRESHOLD,
):
    """Return positive ``1 - safe_density_ratio`` without cancellation."""
    ratio = jnp.asarray(ne_over_ncrit, dtype=jnp.float64)
    threshold = jnp.asarray(threshold, dtype=jnp.float64)
    amplitude = 1.0 - threshold
    minimum = amplitude**2
    transition_scale = amplitude - minimum
    excess = jnp.maximum(ratio - threshold, 0.0)
    upper = minimum + transition_scale * jnp.exp(-excess / transition_scale)
    return jnp.where(ratio <= threshold, 1.0 - ratio, upper)


def nrl_coulomb_logarithm(
    ne,
    Te,
    omega,
    effective_charge,
    *,
    minimum=2.0,
    override=None,
):
    """Return the NRL inverse-bremsstrahlung Coulomb logarithm.

    Densities are SI, electron temperature is eV, and angular frequency is
    rad/s. The classical closest-approach scale includes the SI Coulomb
    constant; the competing quantum scale is the electron de Broglie length.
    """
    ne = jnp.asarray(ne, dtype=jnp.float64)
    Te = jnp.asarray(Te, dtype=jnp.float64)
    omega = jnp.asarray(omega, dtype=jnp.float64)
    effective_charge = jnp.asarray(effective_charge, dtype=jnp.float64)
    minimum = jnp.asarray(minimum, dtype=jnp.float64)
    safe_temperature = jnp.where(Te > 0.0, Te, 1.0)
    thermal_energy = safe_temperature * ELEMENTARY_CHARGE
    thermal_velocity = jnp.sqrt(thermal_energy / ELECTRON_MASS)
    plasma_frequency = jnp.sqrt(
        jnp.maximum(ne, 0.0)
        * ELEMENTARY_CHARGE**2
        / (VACUUM_PERMITTIVITY * ELECTRON_MASS)
    )
    classical_distance = (
        effective_charge
        * ELEMENTARY_CHARGE**2
        / (4.0 * jnp.pi * VACUUM_PERMITTIVITY * thermal_energy)
    )
    quantum_distance = REDUCED_PLANCK_CONSTANT / jnp.sqrt(
        ELECTRON_MASS * thermal_energy
    )
    denominator = jnp.maximum(omega, plasma_frequency) * jnp.maximum(
        classical_distance, quantum_distance
    )
    argument = thermal_velocity / jnp.maximum(denominator, jnp.finfo(jnp.float64).tiny)
    calculated = jnp.maximum(jnp.log(argument), minimum)
    if override is not None:
        return jnp.full_like(calculated, override)
    return calculated


def inverse_bremsstrahlung_depth_derivative(
    ne,
    Te,
    omega,
    effective_charge,
    *,
    minimum_coulomb_log=2.0,
    coulomb_log_override=None,
    critical_collision_frequency_hz=None,
    inside=True,
):
    """Return ``dTheta_IB/dtau`` using a power-absorption coefficient.

    ``tau`` is the ray path coordinate defined by ``d tau = ds / n_ref``.
    Multiplying the NRL physical-length coefficient by ``n_ref`` cancels its
    refractive-index denominator. Power and intensity attenuation are
    ``exp(-Theta_IB)``, while electric-field attenuation is
    ``exp(-Theta_IB / 2)``.

    When ``critical_collision_frequency_hz`` is supplied, use
    ``dTheta/dtau = 2 nu_ei,c (ne/ncritical)**2 / c``, the convention used by
    Follett et al. (2022). Otherwise use the NRL coefficient and Coulomb log.
    """
    ne = jnp.asarray(ne, dtype=jnp.float64)
    Te = jnp.asarray(Te, dtype=jnp.float64)
    omega = jnp.asarray(omega, dtype=jnp.float64)
    valid = jnp.asarray(inside) & (ne > 0.0) & (Te > 0.0) & (omega > 0.0)
    safe_ne = jnp.where(valid, ne, 0.0)
    safe_temperature = jnp.where(valid, Te, 1.0)
    if critical_collision_frequency_hz is None:
        coulomb_log = nrl_coulomb_logarithm(
            safe_ne,
            safe_temperature,
            omega,
            effective_charge,
            minimum=minimum_coulomb_log,
            override=coulomb_log_override,
        )
        rate = (
            3.1e-17
            * effective_charge
            * safe_ne**2
            * coulomb_log
            * safe_temperature ** (-1.5)
            * omega ** (-2.0)
        )
    else:
        ncritical = critical_density(omega)
        rate = (
            2.0
            * critical_collision_frequency_hz
            / SPEED_OF_LIGHT
            * (safe_ne / ncritical) ** 2
        )
    return jnp.where(valid, jnp.maximum(rate, 0.0), 0.0)
