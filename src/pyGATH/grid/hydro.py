"""Containers for hydrodynamic data stored on and sampled from a grid."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class HydroFields:
    """Hydrodynamic fields supplied by an initial-condition function.

    Scalar fields have grid-vertex shape ``(nx + 1, ny + 1, nz + 1)``.
    ``velocity`` has one additional final axis of length three and stores
    physical components in the grid's local orthonormal basis.
    """

    ne: Any
    Te: Any
    Ti: Any
    velocity: Any

    def tree_flatten(self):
        return (self.ne, self.Te, self.Ti, self.velocity), None

    @classmethod
    def tree_unflatten(cls, _aux_data, children):
        return cls(*children)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class VertexHydroFields:
    """Complete hydro arrays stored internally on grid vertices."""

    ne: Any
    ni: Any
    Te: Any
    Ti: Any
    velocity: Any

    def tree_flatten(self):
        return (self.ne, self.ni, self.Te, self.Ti, self.velocity), None

    @classmethod
    def tree_unflatten(cls, _aux_data, children):
        return cls(*children)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class SafeHydroState:
    """Values returned for query positions outside the grid.

    Vector values are Cartesian because sampled vectors are always returned in
    Cartesian components.
    """

    ne: Any = 1.0
    Te: Any = 1.0
    Ti: Any = 1.0
    grad_ne: Any = (0.0, 0.0, 0.0)
    velocity: Any = (0.0, 0.0, 0.0)

    def as_arrays(self, dtype) -> SafeHydroState:
        """Return a copy whose values are JAX arrays of a common dtype."""
        return SafeHydroState(
            ne=jnp.asarray(self.ne, dtype=dtype),
            Te=jnp.asarray(self.Te, dtype=dtype),
            Ti=jnp.asarray(self.Ti, dtype=dtype),
            grad_ne=jnp.asarray(self.grad_ne, dtype=dtype),
            velocity=jnp.asarray(self.velocity, dtype=dtype),
        )

    def tree_flatten(self):
        children = (
            self.ne,
            self.Te,
            self.Ti,
            self.grad_ne,
            self.velocity,
        )
        return children, None

    @classmethod
    def tree_unflatten(cls, _aux_data, children):
        return cls(*children)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class HydroState:
    """Hydrodynamic values sampled at one or more Cartesian positions."""

    ne: Any
    ni: Any
    Te: Any
    Ti: Any
    grad_ne: Any
    velocity: Any
    inside: Any

    def tree_flatten(self):
        children = (
            self.ne,
            self.ni,
            self.Te,
            self.Ti,
            self.grad_ne,
            self.velocity,
            self.inside,
        )
        return children, None

    @classmethod
    def tree_unflatten(cls, _aux_data, children):
        return cls(*children)
