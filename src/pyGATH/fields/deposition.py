"""Conservative power-deposition result types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PowerDeposition:
    """Cell-integrated power and conservation diagnostics.

    ``cell_power`` is measured in watts and ``power_density`` in W/m^3.
    ``outside_power`` is the source power not covered by the target mesh.
    """

    power_density: Any
    cell_power: Any
    deposited_power: Any
    outside_power: Any
    source_power: Any

    @property
    def conservation_error(self):
        """Return the signed source-power balance residual in watts."""
        return self.source_power - self.deposited_power - self.outside_power


__all__ = ["PowerDeposition"]
