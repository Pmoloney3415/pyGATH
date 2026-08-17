"""Batched position and physical-vector coordinate transformations."""

from __future__ import annotations

from enum import Enum

import jax.numpy as jnp


class Geometry(str, Enum):
    """Coordinate systems supported by :mod:`pyGATH.grid`."""

    CARTESIAN = "cartesian"
    CYLINDRICAL = "cylindrical"
    SPHERICAL = "spherical"

    @classmethod
    def parse(cls, value: Geometry | str) -> Geometry:
        if isinstance(value, cls):
            return value
        try:
            return cls(value.lower())
        except (AttributeError, ValueError) as error:
            choices = ", ".join(item.value for item in cls)
            raise ValueError(
                f"Unknown geometry {value!r}; expected one of {choices}"
            ) from error


def _check_last_dimension(values, name: str) -> None:
    if values.ndim < 1 or values.shape[-1] != 3:
        raise ValueError(f"{name} must have shape (..., 3), got {values.shape}")


def _positions_to_cartesian(positions, source: Geometry):
    if source is Geometry.CARTESIAN:
        return positions

    first, phi, third = jnp.moveaxis(positions, -1, 0)
    cos_phi = jnp.cos(phi)
    sin_phi = jnp.sin(phi)

    if source is Geometry.CYLINDRICAL:
        return jnp.stack((first * cos_phi, first * sin_phi, third), axis=-1)

    radius = first
    theta = third
    sin_theta = jnp.sin(theta)
    return jnp.stack(
        (
            radius * sin_theta * cos_phi,
            radius * sin_theta * sin_phi,
            radius * jnp.cos(theta),
        ),
        axis=-1,
    )


def _positions_from_cartesian(positions, target: Geometry):
    if target is Geometry.CARTESIAN:
        return positions

    x, y, z = jnp.moveaxis(positions, -1, 0)
    rho = jnp.hypot(x, y)
    phi = jnp.where(rho == 0, 0.0, jnp.arctan2(y, x))

    if target is Geometry.CYLINDRICAL:
        return jnp.stack((rho, phi, z), axis=-1)

    radius = jnp.sqrt(x * x + y * y + z * z)
    safe_radius = jnp.where(radius == 0, 1.0, radius)
    theta = jnp.where(
        radius == 0,
        0.0,
        jnp.arccos(jnp.clip(z / safe_radius, -1.0, 1.0)),
    )
    return jnp.stack((radius, phi, theta), axis=-1)


def convert_positions(positions, source: Geometry | str, target: Geometry | str):
    """Convert arrays of coordinate triples between supported geometries.

    Inputs may have any leading dimensions but must end in a component axis of
    length three. Cylindrical coordinates are ``(r, phi, z)`` and spherical
    coordinates are ``(r, phi, theta)``, with angles in radians.
    """
    source = Geometry.parse(source)
    target = Geometry.parse(target)
    positions = jnp.asarray(positions, dtype=jnp.float64)
    _check_last_dimension(positions, "positions")
    if source is target:
        return positions
    cartesian = _positions_to_cartesian(positions, source)
    return _positions_from_cartesian(cartesian, target)


def _vectors_to_cartesian(vectors, positions, source: Geometry):
    if source is Geometry.CARTESIAN:
        return vectors

    first, phi, third = jnp.moveaxis(positions, -1, 0)
    del first
    v_first, v_phi, v_third = jnp.moveaxis(vectors, -1, 0)
    cos_phi = jnp.cos(phi)
    sin_phi = jnp.sin(phi)

    if source is Geometry.CYLINDRICAL:
        return jnp.stack(
            (
                v_first * cos_phi - v_phi * sin_phi,
                v_first * sin_phi + v_phi * cos_phi,
                v_third,
            ),
            axis=-1,
        )

    theta = third
    sin_theta = jnp.sin(theta)
    cos_theta = jnp.cos(theta)
    return jnp.stack(
        (
            v_first * sin_theta * cos_phi
            - v_phi * sin_phi
            + v_third * cos_theta * cos_phi,
            v_first * sin_theta * sin_phi
            + v_phi * cos_phi
            + v_third * cos_theta * sin_phi,
            v_first * cos_theta - v_third * sin_theta,
        ),
        axis=-1,
    )


def _vectors_from_cartesian(vectors, positions, target: Geometry):
    if target is Geometry.CARTESIAN:
        return vectors

    x, y, _z = jnp.moveaxis(positions, -1, 0)
    rho = jnp.hypot(x, y)
    phi = jnp.where(rho == 0, 0.0, jnp.arctan2(y, x))
    cos_phi = jnp.cos(phi)
    sin_phi = jnp.sin(phi)
    vx, vy, vz = jnp.moveaxis(vectors, -1, 0)
    v_rho = vx * cos_phi + vy * sin_phi
    v_phi = -vx * sin_phi + vy * cos_phi

    if target is Geometry.CYLINDRICAL:
        return jnp.stack((v_rho, v_phi, vz), axis=-1)

    radius = jnp.sqrt(x * x + y * y + _z * _z)
    safe_radius = jnp.where(radius == 0, 1.0, radius)
    theta = jnp.where(
        radius == 0,
        0.0,
        jnp.arccos(jnp.clip(_z / safe_radius, -1.0, 1.0)),
    )
    sin_theta = jnp.sin(theta)
    cos_theta = jnp.cos(theta)
    v_radius = v_rho * sin_theta + vz * cos_theta
    v_theta = v_rho * cos_theta - vz * sin_theta
    return jnp.stack((v_radius, v_phi, v_theta), axis=-1)


def convert_vectors(
    vectors,
    positions,
    source: Geometry | str,
    target: Geometry | str,
):
    """Convert physical vector components between local orthonormal bases.

    ``positions`` are expressed in ``source`` coordinates and are required
    because cylindrical and spherical bases depend on location. Both inputs
    broadcast over their leading dimensions.
    """
    source = Geometry.parse(source)
    target = Geometry.parse(target)
    vectors = jnp.asarray(vectors, dtype=jnp.float64)
    positions = jnp.asarray(positions, dtype=jnp.float64)
    _check_last_dimension(vectors, "vectors")
    _check_last_dimension(positions, "positions")
    vectors, positions = jnp.broadcast_arrays(vectors, positions)
    if source is target:
        return vectors
    cartesian_positions = _positions_to_cartesian(positions, source)
    cartesian_vectors = _vectors_to_cartesian(vectors, positions, source)
    return _vectors_from_cartesian(cartesian_vectors, cartesian_positions, target)
