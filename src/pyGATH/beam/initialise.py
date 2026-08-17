"""Sample unfocused beams into the packed global ray state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from pyGATH.grid import Grid
from pyGATH.raytracing.intersections import (
    grid_characteristic_length,
    ray_grid_entry_distance,
)
from pyGATH.raytracing.plasma import (
    SPEED_OF_LIGHT,
    VACUUM_PERMITTIVITY,
    critical_density,
    safe_permittivity,
)
from pyGATH.raytracing.raystatelayout import RAY_STATE_LAYOUT

from .beam import BeamBatch


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class InitializedRays:
    """Packed ray state and per-tube grid-entry information.

    ``beam_entry_distance`` retains its original public name, but has one value
    per primary ray. Each value is the common vacuum distance applied to that
    primary ray and its three neighbours before the plasma solve starts.
    ``neighbour_momentum_tangents`` preserves the uncancelled directional
    derivatives used by the infinitesimal ray-tube solve; ``state`` continues
    to expose absolute neighbour momenta for callers and saved output.
    """

    state: Any
    will_hit_grid: Any
    beam_entry_distance: Any
    beam_power: Any
    total_incident_power: Any
    neighbour_momentum_tangents: Any = None

    def tree_flatten(self):
        return (
            self.state,
            self.will_hit_grid,
            self.beam_entry_distance,
            self.beam_power,
            self.total_incident_power,
            self.neighbour_momentum_tangents,
        ), None

    @classmethod
    def tree_unflatten(cls, _auxiliary, children):
        return cls(*children)


def _sample_impact_parameters(
    width_x: float,
    width_y: float,
    supergaussian_index: float,
    nrays_axis1: int,
    nrays_axis2: int,
    intensity_cutoff: float,
):
    cutoff_radius = (-np.log(intensity_cutoff)) ** (1.0 / supergaussian_index)
    extent_x = width_x * cutoff_radius
    extent_y = width_y * cutoff_radius
    spacing_x = 2.0 * extent_x / nrays_axis1
    spacing_y = 2.0 * extent_y / nrays_axis2
    impact_x = np.linspace(
        -extent_x + 0.5 * spacing_x,
        extent_x - 0.5 * spacing_x,
        nrays_axis1,
        dtype=np.float64,
    )
    impact_y = np.linspace(
        -extent_y + 0.5 * spacing_y,
        extent_y - 0.5 * spacing_y,
        nrays_axis2,
        dtype=np.float64,
    )
    return np.meshgrid(impact_x, impact_y, indexing="ij")


def _sample_impact_parameter_line(
    width: float,
    supergaussian_index: float,
    nrays: int,
    intensity_cutoff: float,
):
    cutoff_radius = (-np.log(intensity_cutoff)) ** (1.0 / supergaussian_index)
    extent = width * cutoff_radius
    spacing = 2.0 * extent / nrays
    impact = np.linspace(
        -extent + 0.5 * spacing,
        extent - 0.5 * spacing,
        nrays,
        dtype=np.float64,
    )[:, None]
    return impact, np.zeros_like(impact)


def _triangle_offsets(axis_x, axis_y, radius: float) -> np.ndarray:
    angles = 2.0 * np.pi * np.arange(3, dtype=np.float64) / 3.0
    return (
        radius * np.cos(angles)[:, None] * axis_x[None, :]
        + radius * np.sin(angles)[:, None] * axis_y[None, :]
    )


def initialize_rays(
    beams: BeamBatch,
    grid: Grid,
    *,
    nrays_axis1: int | None = None,
    nrays_axis2: int | None = None,
    intensity_cutoff: float = 2.0e-4,
    neighbour_spacing_m: float = 1.0e-9,
    launch_padding: float = 10.0,
) -> InitializedRays:
    """Create and advance the packed state for all beams.

    The ray sample is cell-centred over the finite super-Gaussian support
    defined by ``intensity_cutoff``. Each beam plane is first placed upstream
    of its central line's earliest grid intersection by ``launch_padding``
    characteristic grid lengths. Each primary ray and its three neighbours
    are then advanced rigidly and in parallel by the largest of their four
    entry distances. Consequently, every member of a ray tube starts the
    plasma solve on or inside the grid. The common vacuum advance is included
    in the primary ray's arc, phase, and path lengths.
    """
    default_axis1 = 1 if grid.dimensions == 1 else 20
    default_axis2 = 20 if grid.dimensions == 3 else 1
    nrays_axis1 = default_axis1 if nrays_axis1 is None else nrays_axis1
    nrays_axis2 = default_axis2 if nrays_axis2 is None else nrays_axis2
    if (
        isinstance(nrays_axis1, bool)
        or not isinstance(nrays_axis1, int)
        or nrays_axis1 < 1
    ):
        raise ValueError("nrays_axis1 must be a positive integer")
    if (
        isinstance(nrays_axis2, bool)
        or not isinstance(nrays_axis2, int)
        or nrays_axis2 < 1
    ):
        raise ValueError("nrays_axis2 must be a positive integer")
    if not 0.0 < intensity_cutoff < 1.0:
        raise ValueError("intensity_cutoff must lie strictly between zero and one")
    if neighbour_spacing_m <= 0:
        raise ValueError("neighbour_spacing_m must be positive")
    if launch_padding <= 0:
        raise ValueError("launch_padding must be positive")

    if grid.dimensions == 1 and nrays_axis1 != 1:
        raise ValueError("nrays_axis1 must be one for a 1-D grid")
    if grid.dimensions < 3 and nrays_axis2 != 1:
        raise ValueError("nrays_axis2 must be one for a reduced-dimensional grid")
    if beams.dimensions != grid.dimensions or not np.allclose(
        np.asarray(beams.inactive_axis_lengths_m, dtype=np.float64),
        np.asarray(grid.inactive_axis_lengths_m, dtype=np.float64),
        rtol=1.0e-12,
        atol=0.0,
    ):
        raise ValueError(
            "beam power normalization does not match the grid dimensionality "
            "and inactive-axis lengths"
        )

    origins = np.asarray(beams.origin, dtype=np.float64)
    directions = np.asarray(beams.direction, dtype=np.float64)
    targets = np.asarray(beams.target, dtype=np.float64)
    if grid.dimensions == 1 and not np.allclose(
        origins[:, 1:], targets[:, 1:], rtol=0.0, atol=1.0e-12
    ):
        raise ValueError("1-D beam origins and targets must share y and z")
    if grid.dimensions == 2 and not np.allclose(
        origins[:, 2], targets[:, 2], rtol=0.0, atol=1.0e-12
    ):
        raise ValueError("2-D beam origins and targets must share z")
    if grid.dimensions == 1 and not np.allclose(
        directions[:, 1:], 0.0, rtol=0.0, atol=1.0e-12
    ):
        raise ValueError("1-D beam directions must be parallel to the x axis")
    if grid.dimensions == 2 and not np.allclose(
        directions[:, 2], 0.0, rtol=0.0, atol=1.0e-12
    ):
        raise ValueError("2-D beam directions must lie in the x-y plane")
    central_intersections = ray_grid_entry_distance(
        grid, origins, directions, forward_only=False
    )
    if np.any(~np.isfinite(central_intersections)):
        missing = [
            beams.names[index]
            for index, distance in enumerate(central_intersections)
            if not np.isfinite(distance)
        ]
        raise ValueError(
            "central beam lines do not intersect the grid for: " + ", ".join(missing)
        )
    central_entries = origins + central_intersections[:, None] * directions
    padding_distance = launch_padding * grid_characteristic_length(grid)
    launch_centres = central_entries - padding_distance * directions

    state = np.zeros(
        (beams.nbeams, nrays_axis1, nrays_axis2, RAY_STATE_LAYOUT.n_attributes),
        dtype=np.float64,
    )
    will_hit = np.zeros((beams.nbeams, nrays_axis1, nrays_axis2), dtype=bool)
    beam_entry_distance = np.full(
        (beams.nbeams, nrays_axis1, nrays_axis2), np.inf, dtype=np.float64
    )
    if grid.dimensions == 1:
        axes_x = np.broadcast_to((0.0, 1.0, 0.0), directions.shape)
        axes_y = np.broadcast_to((0.0, 0.0, 1.0), directions.shape)
    elif grid.dimensions == 2:
        axes_x = np.stack(
            (-directions[:, 1], directions[:, 0], np.zeros(directions.shape[0])),
            axis=-1,
        )
        axes_x /= np.linalg.norm(axes_x, axis=-1, keepdims=True)
        axes_y = np.broadcast_to((0.0, 0.0, 1.0), directions.shape)
    else:
        axes_x = np.asarray(beams.axis_x, dtype=np.float64)
        axes_y = np.asarray(beams.axis_y, dtype=np.float64)
    widths_x = np.asarray(beams.width_x, dtype=np.float64)
    widths_y = np.asarray(beams.width_y, dtype=np.float64)
    indices = np.asarray(beams.supergaussian_index, dtype=np.float64)
    frequencies = np.asarray(beams.omega, dtype=np.float64)
    powers = np.asarray(beams.power_fraction, dtype=np.float64)
    peak_intensities = np.asarray(beams.peak_intensity, dtype=np.float64)

    for beam_index in range(beams.nbeams):
        if grid.dimensions == 1:
            impact_x = np.zeros((1, 1), dtype=np.float64)
            impact_y = np.zeros_like(impact_x)
        elif grid.dimensions == 2:
            impact_x, impact_y = _sample_impact_parameter_line(
                widths_x[beam_index],
                indices[beam_index],
                nrays_axis1,
                intensity_cutoff,
            )
        else:
            impact_x, impact_y = _sample_impact_parameters(
                widths_x[beam_index],
                widths_y[beam_index],
                indices[beam_index],
                nrays_axis1,
                nrays_axis2,
                intensity_cutoff,
            )
        positions = (
            launch_centres[beam_index]
            + impact_x[..., None] * axes_x[beam_index]
            + impact_y[..., None] * axes_y[beam_index]
        )
        direction = directions[beam_index]
        ray_directions = np.broadcast_to(direction, positions.shape)
        primary_entry_distances = ray_grid_entry_distance(
            grid, positions, ray_directions
        )

        neighbour_offsets = _triangle_offsets(
            axes_x[beam_index], axes_y[beam_index], neighbour_spacing_m
        )
        neighbour_positions = positions[..., None, :] + neighbour_offsets
        neighbour_directions = np.broadcast_to(direction, neighbour_positions.shape)
        neighbour_entry_distances = ray_grid_entry_distance(
            grid, neighbour_positions, neighbour_directions
        )
        member_entry_distances = np.concatenate(
            (primary_entry_distances[..., None], neighbour_entry_distances), axis=-1
        )
        hit_mask = np.all(np.isfinite(member_entry_distances), axis=-1)
        will_hit[beam_index] = hit_mask
        if not np.all(hit_mask):
            missed_tubes = int(np.size(hit_mask) - np.count_nonzero(hit_mask))
            raise ValueError(
                f"{missed_tubes} sampled ray tube(s) from beam "
                f"{beams.names[beam_index]!r} do not have all four members "
                "intersecting the grid"
            )
        advance = np.max(member_entry_distances, axis=-1)
        beam_entry_distance[beam_index] = advance
        positions = positions + advance[..., None] * direction
        neighbour_positions = neighbour_positions + advance[..., None, None] * direction
        if grid.dimensions == 1:
            profile = np.ones_like(impact_x)
        elif grid.dimensions == 2:
            profile = np.exp(
                -(np.abs(impact_x / widths_x[beam_index]) ** indices[beam_index])
            )
        else:
            profile = np.exp(
                -(
                    np.abs(impact_x / widths_x[beam_index]) ** indices[beam_index]
                    + np.abs(impact_y / widths_y[beam_index]) ** indices[beam_index]
                )
            )
        normalized_profile = profile / np.sum(profile)
        initial_intensity = peak_intensities[beam_index] * profile
        initial_electric_field = np.sqrt(
            2.0 * initial_intensity / (VACUUM_PERMITTIVITY * SPEED_OF_LIGHT)
        )

        state[beam_index, ..., RAY_STATE_LAYOUT.position] = positions
        state[beam_index, ..., RAY_STATE_LAYOUT.frequency] = frequencies[beam_index]
        state[beam_index, ..., RAY_STATE_LAYOUT.arc_length] = advance
        state[beam_index, ..., RAY_STATE_LAYOUT.phase_length] = advance
        state[beam_index, ..., RAY_STATE_LAYOUT.path_length] = advance
        state[beam_index, ..., RAY_STATE_LAYOUT.impact_parameter_x] = impact_x
        state[beam_index, ..., RAY_STATE_LAYOUT.impact_parameter_y] = impact_y
        state[beam_index, ..., RAY_STATE_LAYOUT.neighbour_positions] = (
            neighbour_positions.reshape((*positions.shape[:-1], 9))
        )
        state[beam_index, ..., RAY_STATE_LAYOUT.ray_power] = (
            powers[beam_index] * normalized_profile
        )
        state[beam_index, ..., RAY_STATE_LAYOUT.initial_intensity] = initial_intensity
        state[beam_index, ..., RAY_STATE_LAYOUT.initial_electric_field] = (
            initial_electric_field
        )

    all_positions = jnp.asarray(state[..., RAY_STATE_LAYOUT.position])
    all_neighbour_positions = jnp.asarray(
        state[..., RAY_STATE_LAYOUT.neighbour_positions]
    ).reshape((*state.shape[:-1], 3, 3))
    primary_hydro = grid.interpolate(all_positions)
    neighbour_hydro = grid.interpolate(all_neighbour_positions)
    frequencies_jax = jnp.asarray(frequencies)[:, None, None]
    ncritical = critical_density(frequencies_jax)
    primary_epsilon = safe_permittivity(primary_hydro.ne / ncritical)
    neighbour_epsilon = safe_permittivity(neighbour_hydro.ne / ncritical[..., None])
    direction_jax = jnp.asarray(directions)[:, None, None, :]
    primary_momenta = direction_jax * jnp.sqrt(primary_epsilon)[..., None]
    neighbour_momenta = (
        direction_jax[..., None, :] * jnp.sqrt(neighbour_epsilon)[..., None]
    )
    state_jax = jnp.asarray(state)
    state_jax = state_jax.at[..., RAY_STATE_LAYOUT.momentum].set(primary_momenta)
    state_jax = state_jax.at[..., RAY_STATE_LAYOUT.neighbour_momenta].set(
        neighbour_momenta.reshape((*state.shape[:-1], 9))
    )
    state_jax = state_jax.at[..., RAY_STATE_LAYOUT.permittivity].set(primary_epsilon)

    position_tangents = all_neighbour_positions - all_positions[..., None, :]
    flat_positions = all_positions.reshape((-1, 3))
    flat_frequencies = jnp.broadcast_to(
        frequencies_jax, all_positions.shape[:-1]
    ).reshape((-1,))
    flat_directions = jnp.broadcast_to(direction_jax, all_positions.shape).reshape(
        (-1, 3)
    )
    flat_position_tangents = position_tangents.reshape((-1, 3, 3))

    def one_tube(position, frequency, direction, tangents):
        ncritical = critical_density(frequency)

        def initial_momentum(query_position):
            density = grid.interpolate(query_position).ne
            epsilon = safe_permittivity(density / ncritical)
            return direction * jnp.sqrt(epsilon)

        return jax.vmap(
            lambda tangent: jax.jvp(
                initial_momentum,
                (position,),
                (tangent,),
            )[1]
        )(tangents)

    neighbour_momentum_tangents = jax.vmap(one_tube)(
        flat_positions,
        flat_frequencies,
        flat_directions,
        flat_position_tangents,
    ).reshape((*state.shape[:-1], 9))
    return InitializedRays(
        state=state_jax,
        will_hit_grid=jnp.asarray(will_hit),
        beam_entry_distance=jnp.asarray(beam_entry_distance),
        beam_power=jnp.asarray(beams.beam_power),
        total_incident_power=jnp.sum(jnp.asarray(beams.beam_power)),
        neighbour_momentum_tangents=neighbour_momentum_tangents,
    )
