"""Constant, fully ionized plasma composition metadata."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class PlasmaComposition:
    """Ion-number fractions and manually supplied mass and charge numbers."""

    names: tuple[str, ...]
    mass_numbers: Any
    charge_numbers: Any
    fractions: Any

    @classmethod
    def create(
        cls,
        elements: Sequence[tuple[str, float, float, float]] | None = None,
    ) -> PlasmaComposition:
        """Validate and normalize ``(name, A, Z, fraction)`` elements."""
        elements = (("hydrogen", 1.0, 1.0, 1.0),) if elements is None else elements
        if not elements:
            raise ValueError("plasma composition must contain at least one element")

        names: list[str] = []
        masses: list[float] = []
        charges: list[float] = []
        fractions: list[float] = []
        for index, (name, mass, charge, fraction) in enumerate(elements):
            if not isinstance(name, str) or not name.strip():
                raise ValueError(f"composition element {index} needs a non-empty name")
            if name.strip() in names:
                raise ValueError(f"duplicate composition element name {name.strip()!r}")
            values = np.asarray((mass, charge, fraction), dtype=np.float64)
            if not np.all(np.isfinite(values)):
                raise ValueError("composition A, Z, and fractions must be finite")
            if mass <= 0.0:
                raise ValueError("composition mass numbers A must be positive")
            if charge <= 0.0:
                raise ValueError("fully ionized charge numbers Z must be positive")
            if fraction < 0.0:
                raise ValueError("composition fractions cannot be negative")
            names.append(name.strip())
            masses.append(float(mass))
            charges.append(float(charge))
            fractions.append(float(fraction))

        fractions_array = np.asarray(fractions, dtype=np.float64)
        total = float(np.sum(fractions_array))
        if total <= 0.0:
            raise ValueError("at least one composition fraction must be positive")
        fractions_array /= total
        return cls(
            names=tuple(names),
            mass_numbers=jnp.asarray(masses, dtype=jnp.float64),
            charge_numbers=jnp.asarray(charges, dtype=jnp.float64),
            fractions=jnp.asarray(fractions_array, dtype=jnp.float64),
        )

    @property
    def mean_charge(self):
        """Return ion-number-weighted mean charge ``sum(f_i Z_i)``."""
        return jnp.sum(self.fractions * self.charge_numbers)

    @property
    def effective_charge(self):
        """Return ``sum(f_i Z_i^2) / sum(f_i Z_i)`` for collisions."""
        return jnp.sum(self.fractions * self.charge_numbers**2) / self.mean_charge

    @property
    def mean_mass_number(self):
        """Return ion-number-weighted mean mass number."""
        return jnp.sum(self.fractions * self.mass_numbers)

    def tree_flatten(self):
        return (
            self.mass_numbers,
            self.charge_numbers,
            self.fractions,
        ), self.names

    @classmethod
    def tree_unflatten(cls, names, children):
        return cls(names, *children)


__all__ = ["PlasmaComposition"]
