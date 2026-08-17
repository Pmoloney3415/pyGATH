"""Static indices for the packed global ray state."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RayStateLayout:
    """Indices and slices into ``ray_state[..., attribute]``.

    Neighbour position and momentum slices flatten arrays with shape ``(3, 3)``
    in neighbour-major, Cartesian-component-minor order.
    """

    position: slice = slice(0, 3)
    momentum: slice = slice(3, 6)
    frequency: int = 6
    arc_length: int = 7
    phase_length: int = 8
    path_length: int = 9
    inverse_brems_depth: int = 10
    cbet_depth: int = 11
    srs_depth: int = 12
    sbs_depth: int = 13
    impact_parameter: slice = slice(14, 16)
    impact_parameter_x: int = 14
    impact_parameter_y: int = 15
    neighbour_positions: slice = slice(16, 25)
    neighbour_momenta: slice = slice(25, 34)
    permittivity: int = 34
    ray_power: int = 35
    initial_intensity: int = 36
    initial_electric_field: int = 37
    n_attributes: int = 38

    @property
    def ray_power_fraction(self) -> int:
        """Explicit alias for the dimensionless represented power fraction."""
        return self.ray_power

    @property
    def neighbour_velocities(self) -> slice:
        """Alias matching the physical interpretation of neighbour momenta."""
        return self.neighbour_momenta


RAY_STATE_LAYOUT = RayStateLayout()


@dataclass(frozen=True)
class RaySheetLayout(RayStateLayout):
    """Indices into sheet fields containing ray state and optical fields."""

    uncapped_amplitude: int = 38
    capped_amplitude: int = 39
    electric_field: int = 40
    intensity: int = 41
    inverse_brems_deposition: int = 42
    n_attributes: int = 43


RAY_SHEET_LAYOUT = RaySheetLayout()
