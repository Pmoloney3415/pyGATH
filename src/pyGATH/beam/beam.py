"""Beam geometry represented in Cartesian simulation coordinates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp


def spot_basis(direction, rotation_pi=0.0):
    """Construct rotated spot-plane axes for Cartesian unit directions.

    At zero rotation the local ``y`` axis is the projection of global ``z``
    onto the plane normal to ``direction``. For propagation parallel to global
    ``z``, global ``y`` is used as a deterministic fallback. Positive rotation
    follows the right-hand rule about the propagation direction.
    """
    direction = jnp.asarray(direction, dtype=jnp.float64)
    rotation_pi = jnp.asarray(rotation_pi, dtype=jnp.float64)
    global_z = jnp.asarray((0.0, 0.0, 1.0), dtype=jnp.float64)
    global_y = jnp.asarray((0.0, 1.0, 0.0), dtype=jnp.float64)
    projected_z = (
        global_z - jnp.sum(direction * global_z, axis=-1, keepdims=True) * direction
    )
    projected_y = (
        global_y - jnp.sum(direction * global_y, axis=-1, keepdims=True) * direction
    )
    use_fallback = jnp.linalg.norm(projected_z, axis=-1, keepdims=True) < 1.0e-12
    local_y = jnp.where(use_fallback, projected_y, projected_z)
    local_y /= jnp.linalg.norm(local_y, axis=-1, keepdims=True)
    local_x = jnp.cross(local_y, direction)
    angle = jnp.pi * rotation_pi
    cosine = jnp.cos(angle)[..., None]
    sine = jnp.sin(angle)[..., None]
    rotated_x = cosine * local_x + sine * local_y
    rotated_y = cosine * local_y - sine * local_x
    return rotated_x, rotated_y


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class Beam:
    """A single unfocused beam with Cartesian geometry."""

    name: str
    origin: Any
    target: Any
    direction: Any
    axis_x: Any
    axis_y: Any
    width_x: Any
    width_y: Any
    rotation_pi: Any
    supergaussian_index: Any
    omega: Any
    power_fraction: Any
    peak_intensity: Any
    beam_power: Any

    def tree_flatten(self):
        children = (
            self.origin,
            self.target,
            self.direction,
            self.axis_x,
            self.axis_y,
            self.width_x,
            self.width_y,
            self.rotation_pi,
            self.supergaussian_index,
            self.omega,
            self.power_fraction,
            self.peak_intensity,
            self.beam_power,
        )
        return children, self.name

    @classmethod
    def tree_unflatten(cls, name, children):
        return cls(name, *children)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class BeamBatch:
    """A fixed-size collection of beams suitable for JAX transformations."""

    names: tuple[str, ...]
    origin: Any
    target: Any
    direction: Any
    axis_x: Any
    axis_y: Any
    width_x: Any
    width_y: Any
    rotation_pi: Any
    supergaussian_index: Any
    omega: Any
    power_fraction: Any
    peak_intensity: Any
    beam_power: Any
    dimensions: int = 3
    inactive_axis_lengths_m: tuple[float, ...] = ()

    @property
    def nbeams(self) -> int:
        return len(self.names)

    def __len__(self) -> int:
        return self.nbeams

    def __getitem__(self, index: int) -> Beam:
        return Beam(
            name=self.names[index],
            origin=self.origin[index],
            target=self.target[index],
            direction=self.direction[index],
            axis_x=self.axis_x[index],
            axis_y=self.axis_y[index],
            width_x=self.width_x[index],
            width_y=self.width_y[index],
            rotation_pi=self.rotation_pi[index],
            supergaussian_index=self.supergaussian_index[index],
            omega=self.omega[index],
            power_fraction=self.power_fraction[index],
            peak_intensity=self.peak_intensity[index],
            beam_power=self.beam_power[index],
        )

    def tree_flatten(self):
        children = (
            self.origin,
            self.target,
            self.direction,
            self.axis_x,
            self.axis_y,
            self.width_x,
            self.width_y,
            self.rotation_pi,
            self.supergaussian_index,
            self.omega,
            self.power_fraction,
            self.peak_intensity,
            self.beam_power,
        )
        return children, (
            self.names,
            self.dimensions,
            self.inactive_axis_lengths_m,
        )

    @classmethod
    def tree_unflatten(cls, auxiliary, children):
        names, dimensions, inactive_axis_lengths_m = auxiliary
        return cls(
            names,
            *children,
            dimensions=dimensions,
            inactive_axis_lengths_m=inactive_axis_lengths_m,
        )
