"""Batched JAX/Diffrax ray propagation and two-sheet resampling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import diffrax
import jax
import jax.numpy as jnp
import optimistix as optx

from pyGATH.grid import Geometry, Grid, convert_positions

from .plasma import (
    SPEED_OF_LIGHT,
    VACUUM_PERMITTIVITY,
    InverseBremsstrahlungOptions,
    critical_density,
    inverse_bremsstrahlung_depth_derivative,
    safe_density_ratio,
    safe_permittivity,
)
from .raystatelayout import RAY_STATE_LAYOUT


@dataclass(frozen=True)
class RayTracingOptions:
    """Numerical controls for propagation and sheet construction.

    ``maximum_path_length_grid_lengths`` is a failsafe upper bound. Normal
    solves terminate earlier, on the first accepted step for which every
    primary ray is outside the grid. ``rtol`` and ``atol`` act on an internal
    dimensionless state with a maximum norm, so one tolerance pair controls
    positions, momenta, path lengths, and tangent neighbour-ray variables at
    their own characteristic scales.
    """

    nsamples_per_sheet: int = 20
    diagnostic_samples: int = 512
    maximum_path_length_grid_lengths: float = 100.0
    dt0: float | None = None
    rtol: float = 1.0e-5
    atol: float = 1.0e-7
    max_steps: int = 4096
    area_floor: float = 1.0e-12
    minimum_amplitude_cap: float = 1.1

    def validate(self) -> None:
        """Raise ``ValueError`` when an option cannot define a valid solve."""
        integer_options = {
            "nsamples_per_sheet": self.nsamples_per_sheet,
            "diagnostic_samples": self.diagnostic_samples,
            "max_steps": self.max_steps,
        }
        for name, value in integer_options.items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.nsamples_per_sheet < 2:
            raise ValueError("nsamples_per_sheet must be at least two")
        if self.diagnostic_samples < 3:
            raise ValueError("diagnostic_samples must be at least three")
        positive_options = {
            "maximum_path_length_grid_lengths": (self.maximum_path_length_grid_lengths),
            "rtol": self.rtol,
            "atol": self.atol,
            "area_floor": self.area_floor,
            "minimum_amplitude_cap": self.minimum_amplitude_cap,
        }
        for name, value in positive_options.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.dt0 is not None and self.dt0 <= 0:
            raise ValueError("dt0 must be positive when supplied")


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class RayTraceResult:
    """Two ray sheets and the caustic metadata used to construct them.

    ``sheet_fields`` has shape
    ``(nbeams, 2, nrays_axis1, nrays_axis2, nsamples, 43)``. Its first 38
    entries use :data:`RAY_STATE_LAYOUT`; the final entries are described by
    :data:`RAY_SHEET_LAYOUT`.
    """

    sheet_fields: Any
    has_caustic: Any
    caustic_path: Any
    caustic_score: Any
    terminal_path: Any
    terminated: Any

    def tree_flatten(self):
        return (
            self.sheet_fields,
            self.has_caustic,
            self.caustic_path,
            self.caustic_score,
            self.terminal_path,
            self.terminated,
        ), None

    @classmethod
    def tree_unflatten(cls, _auxiliary, children):
        return cls(*children)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class _StateScaling:
    scales: Any
    position_reference: Any

    def tree_flatten(self):
        return (self.scales, self.position_reference), None

    @classmethod
    def tree_unflatten(cls, _auxiliary, children):
        return cls(*children)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class _PrimaryTraceArguments:
    grid: Grid
    position_scale: Any
    position_reference: Any
    momentum_scale: Any
    frequency: Any

    def tree_flatten(self):
        return (
            self.grid,
            self.position_scale,
            self.position_reference,
            self.momentum_scale,
            self.frequency,
        ), None

    @classmethod
    def tree_unflatten(cls, _auxiliary, children):
        return cls(*children)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class _TraceArguments:
    grid: Grid
    initial_area: Any
    area_floor: Any
    scaling: _StateScaling
    inverse_bremsstrahlung: InverseBremsstrahlungOptions

    def tree_flatten(self):
        return (
            self.grid,
            self.initial_area,
            self.area_floor,
            self.scaling,
        ), self.inverse_bremsstrahlung

    @classmethod
    def tree_unflatten(cls, inverse_bremsstrahlung, children):
        return cls(*children, inverse_bremsstrahlung)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class _InstantaneousDiagnostics:
    uncapped_amplitude: Any
    area_ratio: Any
    density_ratio: Any
    gradient_norm: Any
    inside: Any

    def tree_flatten(self):
        return (
            self.uncapped_amplitude,
            self.area_ratio,
            self.density_ratio,
            self.gradient_norm,
            self.inside,
        ), None

    @classmethod
    def tree_unflatten(cls, _auxiliary, children):
        return cls(*children)


def _neighbour_vectors(state, position_slice):
    leading_shape = state.shape[:-1]
    return state[..., position_slice].reshape((*leading_shape, 3, 3))


def _absolute_to_relative_state(state):
    """Replace absolute neighbour states by primary-relative differences."""
    primary_position = state[..., RAY_STATE_LAYOUT.position]
    primary_momentum = state[..., RAY_STATE_LAYOUT.momentum]
    neighbour_positions = _neighbour_vectors(
        state, RAY_STATE_LAYOUT.neighbour_positions
    )
    neighbour_momenta = _neighbour_vectors(state, RAY_STATE_LAYOUT.neighbour_momenta)
    relative = state.at[..., RAY_STATE_LAYOUT.neighbour_positions].set(
        (neighbour_positions - primary_position[..., None, :]).reshape(
            (*state.shape[:-1], 9)
        )
    )
    return relative.at[..., RAY_STATE_LAYOUT.neighbour_momenta].set(
        (neighbour_momenta - primary_momentum[..., None, :]).reshape(
            (*state.shape[:-1], 9)
        )
    )


def _relative_to_absolute_state(state):
    """Reconstruct public absolute neighbour states from differences."""
    primary_position = state[..., RAY_STATE_LAYOUT.position]
    primary_momentum = state[..., RAY_STATE_LAYOUT.momentum]
    position_offsets = _neighbour_vectors(state, RAY_STATE_LAYOUT.neighbour_positions)
    momentum_differences = _neighbour_vectors(state, RAY_STATE_LAYOUT.neighbour_momenta)
    absolute = state.at[..., RAY_STATE_LAYOUT.neighbour_positions].set(
        (primary_position[..., None, :] + position_offsets).reshape(
            (*state.shape[:-1], 9)
        )
    )
    return absolute.at[..., RAY_STATE_LAYOUT.neighbour_momenta].set(
        (primary_momentum[..., None, :] + momentum_differences).reshape(
            (*state.shape[:-1], 9)
        )
    )


def _build_state_scaling(relative_state, characteristic_length):
    """Build automatic physical scales for the dimensionless solver state.

    Primary positions and length-like quantities use the resolved physical
    grid length. Neighbour offsets use their initial triangle size, while
    neighbour momentum differences use their initial size or the angular
    scale implied by that triangle, whichever is larger.
    """
    dtype = relative_state.dtype
    tiny_scale = jnp.finfo(dtype).eps * characteristic_length
    position_offsets = _neighbour_vectors(
        relative_state, RAY_STATE_LAYOUT.neighbour_positions
    )
    momentum_differences = _neighbour_vectors(
        relative_state, RAY_STATE_LAYOUT.neighbour_momenta
    )
    offset_scale = jnp.max(jnp.linalg.norm(position_offsets, axis=-1), axis=-1)
    offset_scale = jnp.maximum(offset_scale, tiny_scale)
    momentum_difference_scale = jnp.max(
        jnp.linalg.norm(momentum_differences, axis=-1), axis=-1
    )
    momentum_difference_scale = jnp.maximum(
        momentum_difference_scale, offset_scale / characteristic_length
    )

    scales = jnp.ones_like(relative_state)
    scales = scales.at[..., RAY_STATE_LAYOUT.position].set(characteristic_length)
    frequency_scale = jnp.maximum(
        jnp.abs(relative_state[..., RAY_STATE_LAYOUT.frequency]),
        jnp.finfo(dtype).tiny,
    )
    scales = scales.at[..., RAY_STATE_LAYOUT.frequency].set(frequency_scale)
    for index in (
        RAY_STATE_LAYOUT.initial_intensity,
        RAY_STATE_LAYOUT.initial_electric_field,
    ):
        value_scale = jnp.maximum(
            jnp.abs(relative_state[..., index]), jnp.finfo(dtype).tiny
        )
        scales = scales.at[..., index].set(value_scale)
    for index in (
        RAY_STATE_LAYOUT.arc_length,
        RAY_STATE_LAYOUT.phase_length,
        RAY_STATE_LAYOUT.path_length,
        RAY_STATE_LAYOUT.impact_parameter_x,
        RAY_STATE_LAYOUT.impact_parameter_y,
    ):
        scales = scales.at[..., index].set(characteristic_length)
    scales = scales.at[..., RAY_STATE_LAYOUT.neighbour_positions].set(
        jnp.broadcast_to(offset_scale[..., None], (*offset_scale.shape, 9))
    )
    scales = scales.at[..., RAY_STATE_LAYOUT.neighbour_momenta].set(
        jnp.broadcast_to(
            momentum_difference_scale[..., None],
            (*momentum_difference_scale.shape, 9),
        )
    )
    return _StateScaling(
        scales=scales,
        position_reference=relative_state[..., RAY_STATE_LAYOUT.position],
    )


def _to_solver_state(relative_state, scaling):
    shifted = relative_state.at[..., RAY_STATE_LAYOUT.position].add(
        -scaling.position_reference
    )
    return shifted / scaling.scales


def _from_solver_state(solver_state, scaling):
    relative = solver_state * scaling.scales
    return relative.at[..., RAY_STATE_LAYOUT.position].add(scaling.position_reference)


def _to_primary_solver_state(solver_state):
    return jnp.concatenate(
        (
            solver_state[..., RAY_STATE_LAYOUT.position],
            solver_state[..., RAY_STATE_LAYOUT.momentum],
        ),
        axis=-1,
    )


def _from_primary_solver_state(primary_state, args):
    position = primary_state[..., :3] * args.position_scale + args.position_reference
    momentum = primary_state[..., 3:] * args.momentum_scale
    return position, momentum


def _maximum_norm(value):
    return jnp.max(jnp.abs(value))


def _tube_positions(state):
    primary = state[..., RAY_STATE_LAYOUT.position]
    neighbours = _neighbour_vectors(state, RAY_STATE_LAYOUT.neighbour_positions)
    return jnp.concatenate((primary[..., None, :], neighbours), axis=-2)


def _tube_momenta(state):
    primary = state[..., RAY_STATE_LAYOUT.momentum]
    neighbours = _neighbour_vectors(state, RAY_STATE_LAYOUT.neighbour_momenta)
    return jnp.concatenate((primary[..., None, :], neighbours), axis=-2)


def ray_rhs(_path, state, args):
    """Return derivatives for every primary and neighbour trajectory.

    ``args`` may be either a :class:`Grid` or the internal trace arguments.
    Each of the four positions in a ray tube is interpolated independently, so
    every momentum receives its own raw local density gradient.
    """
    if isinstance(args, _TraceArguments):
        grid = args.grid
        inverse_bremsstrahlung = args.inverse_bremsstrahlung
    else:
        grid = args
        inverse_bremsstrahlung = InverseBremsstrahlungOptions()
    if state.shape[-1] != RAY_STATE_LAYOUT.n_attributes:
        raise ValueError(
            f"state must end in {RAY_STATE_LAYOUT.n_attributes} attributes"
        )

    positions = _tube_positions(state)
    momenta = _tube_momenta(state)
    hydro = grid.interpolate(positions)
    omega = state[..., RAY_STATE_LAYOUT.frequency]
    ncritical = critical_density(omega)
    epsilon = safe_permittivity(hydro.ne / ncritical[..., None])
    momentum_derivative = -hydro.grad_ne / (2.0 * ncritical[..., None, None])

    derivative = jnp.zeros_like(state)
    derivative = derivative.at[..., RAY_STATE_LAYOUT.position].set(momenta[..., 0, :])
    derivative = derivative.at[..., RAY_STATE_LAYOUT.momentum].set(
        momentum_derivative[..., 0, :]
    )
    derivative = derivative.at[..., RAY_STATE_LAYOUT.neighbour_positions].set(
        momenta[..., 1:, :].reshape(
            (
                *state.shape[:-1],
                RAY_STATE_LAYOUT.neighbour_positions.stop
                - RAY_STATE_LAYOUT.neighbour_positions.start,
            )
        )
    )
    derivative = derivative.at[..., RAY_STATE_LAYOUT.neighbour_momenta].set(
        momentum_derivative[..., 1:, :].reshape(
            (
                *state.shape[:-1],
                RAY_STATE_LAYOUT.neighbour_momenta.stop
                - RAY_STATE_LAYOUT.neighbour_momenta.start,
            )
        )
    )
    derivative = derivative.at[..., RAY_STATE_LAYOUT.arc_length].set(
        jnp.linalg.norm(momenta[..., 0, :], axis=-1)
    )
    derivative = derivative.at[..., RAY_STATE_LAYOUT.phase_length].set(epsilon[..., 0])
    derivative = derivative.at[..., RAY_STATE_LAYOUT.path_length].set(1.0)
    if inverse_bremsstrahlung.enabled:
        inverse_bremsstrahlung_rate = inverse_bremsstrahlung_depth_derivative(
            hydro.ne[..., 0],
            hydro.Te[..., 0],
            omega,
            grid.composition.effective_charge,
            minimum_coulomb_log=(inverse_bremsstrahlung.minimum_coulomb_log),
            coulomb_log_override=(inverse_bremsstrahlung.coulomb_log_override),
            critical_collision_frequency_hz=(
                inverse_bremsstrahlung.critical_collision_frequency_hz
            ),
            inside=hydro.inside[..., 0],
        )
        derivative = derivative.at[..., RAY_STATE_LAYOUT.inverse_brems_depth].set(
            inverse_bremsstrahlung_rate
        )
    return derivative


def _acceleration_tangents(grid, positions, frequencies, position_tangents):
    """Apply the local acceleration Jacobian to three position tangents.

    Directly subtracting accelerations at nanometre-separated locations loses
    the small differential signal to cancellation. JAX's forward-mode
    directional derivative evaluates the infinitesimal ray-tube dynamics
    without constructing a dense spatial Jacobian.
    """
    flat_positions = positions.reshape((-1, 3))
    flat_frequencies = frequencies.reshape((-1,))
    flat_tangents = position_tangents.reshape((-1, 3, 3))

    def one_tube(position, frequency, tangents):
        ncritical = critical_density(frequency)

        def acceleration(query_position):
            gradient = grid.interpolate(query_position).grad_ne
            return -gradient / (2.0 * ncritical)

        def apply_tangent(tangent):
            return jax.jvp(
                acceleration,
                (position,),
                (tangent,),
            )[1]

        return jax.vmap(apply_tangent)(tangents)

    return jax.vmap(one_tube)(
        flat_positions,
        flat_frequencies,
        flat_tangents,
    ).reshape(position_tangents.shape)


def _scaled_tangent_ray_rhs(_path, solver_state, args):
    """Evolve the primary ray and its dimensionless tangent ray tube."""
    state = _from_solver_state(solver_state, args.scaling)
    positions = state[..., RAY_STATE_LAYOUT.position]
    momentum = state[..., RAY_STATE_LAYOUT.momentum]
    position_tangents = _neighbour_vectors(state, RAY_STATE_LAYOUT.neighbour_positions)
    momentum_tangents = _neighbour_vectors(state, RAY_STATE_LAYOUT.neighbour_momenta)
    hydro = args.grid.interpolate(positions)
    omega = state[..., RAY_STATE_LAYOUT.frequency]
    ncritical = critical_density(omega)
    epsilon = safe_permittivity(hydro.ne / ncritical)
    primary_acceleration = -hydro.grad_ne / (2.0 * ncritical[..., None])
    acceleration_tangents = _acceleration_tangents(
        args.grid,
        positions,
        omega,
        position_tangents,
    )

    derivative = jnp.zeros_like(state)
    derivative = derivative.at[..., RAY_STATE_LAYOUT.position].set(momentum)
    derivative = derivative.at[..., RAY_STATE_LAYOUT.momentum].set(primary_acceleration)
    derivative = derivative.at[..., RAY_STATE_LAYOUT.neighbour_positions].set(
        momentum_tangents.reshape((*state.shape[:-1], 9))
    )
    derivative = derivative.at[..., RAY_STATE_LAYOUT.neighbour_momenta].set(
        acceleration_tangents.reshape((*state.shape[:-1], 9))
    )
    derivative = derivative.at[..., RAY_STATE_LAYOUT.arc_length].set(
        jnp.linalg.norm(momentum, axis=-1)
    )
    derivative = derivative.at[..., RAY_STATE_LAYOUT.phase_length].set(epsilon)
    derivative = derivative.at[..., RAY_STATE_LAYOUT.path_length].set(1.0)
    inverse_bremsstrahlung = args.inverse_bremsstrahlung
    if inverse_bremsstrahlung.enabled:
        inverse_bremsstrahlung_rate = inverse_bremsstrahlung_depth_derivative(
            hydro.ne,
            hydro.Te,
            omega,
            args.grid.composition.effective_charge,
            minimum_coulomb_log=(inverse_bremsstrahlung.minimum_coulomb_log),
            coulomb_log_override=(inverse_bremsstrahlung.coulomb_log_override),
            critical_collision_frequency_hz=(
                inverse_bremsstrahlung.critical_collision_frequency_hz
            ),
            inside=hydro.inside,
        )
        derivative = derivative.at[..., RAY_STATE_LAYOUT.inverse_brems_depth].set(
            inverse_bremsstrahlung_rate
        )
    return derivative / args.scaling.scales


def _scaled_primary_ray_rhs(_path, primary_state, args):
    """Evolve only the scaled primary position and momentum used for exit."""
    position, momentum = _from_primary_solver_state(primary_state, args)
    hydro = args.grid.interpolate(position)
    ncritical = critical_density(args.frequency)
    acceleration = -hydro.grad_ne / (2.0 * ncritical[..., None])
    return jnp.concatenate(
        (
            momentum / args.position_scale,
            acceleration / args.momentum_scale,
        ),
        axis=-1,
    )


def _projected_area_from_neighbours(state, neighbours):
    edge_one = neighbours[..., 1, :] - neighbours[..., 0, :]
    edge_two = neighbours[..., 2, :] - neighbours[..., 0, :]
    area_vector = 0.5 * jnp.cross(edge_one, edge_two)
    momentum = state[..., RAY_STATE_LAYOUT.momentum]
    momentum_norm = jnp.linalg.norm(momentum, axis=-1, keepdims=True)
    safe_norm = jnp.maximum(momentum_norm, jnp.finfo(state.dtype).tiny)
    direction = momentum / safe_norm
    return jnp.abs(jnp.sum(area_vector * direction, axis=-1))


def _projected_triangle_area(state):
    neighbours = _neighbour_vectors(state, RAY_STATE_LAYOUT.neighbour_positions)
    return _projected_area_from_neighbours(state, neighbours)


def _relative_projected_triangle_area(state):
    offsets = _neighbour_vectors(state, RAY_STATE_LAYOUT.neighbour_positions)
    return _projected_area_from_neighbours(state, offsets)


def initial_ray_area(state):
    """Return each ray tube's initial projected neighbour-triangle area."""
    state = jnp.asarray(state, dtype=jnp.float64)
    if state.shape[-1] != RAY_STATE_LAYOUT.n_attributes:
        raise ValueError(
            f"state must end in {RAY_STATE_LAYOUT.n_attributes} attributes"
        )
    area = _projected_triangle_area(state)
    return jnp.maximum(area, jnp.finfo(state.dtype).tiny)


def _instantaneous_diagnostics(solver_state, args):
    state = _from_solver_state(solver_state, args.scaling)
    grid = args.grid
    area = _relative_projected_triangle_area(state)
    raw_ratio = area / args.initial_area
    area_floor = jnp.asarray(args.area_floor, dtype=state.dtype)
    area_ratio = jnp.sqrt(raw_ratio**2 + area_floor**2) / jnp.sqrt(1.0 + area_floor**2)

    positions = state[..., RAY_STATE_LAYOUT.position]
    hydro = grid.interpolate(positions)
    omega = state[..., RAY_STATE_LAYOUT.frequency]
    ncritical = critical_density(omega)
    density_ratio = safe_density_ratio(hydro.ne / ncritical)
    epsilon = safe_permittivity(hydro.ne / ncritical)
    uncapped = 1.0 / jnp.sqrt(jnp.sqrt(epsilon) * area_ratio)
    return _InstantaneousDiagnostics(
        uncapped_amplitude=uncapped,
        area_ratio=area_ratio,
        density_ratio=density_ratio,
        gradient_norm=jnp.linalg.norm(hydro.grad_ne, axis=-1),
        inside=hydro.inside,
    )


def _save_diagnostics(_path, state, args):
    return _instantaneous_diagnostics(state, args)


def _all_primary_rays_outside(t, y, args, **_kwargs):
    del t
    position, _momentum = _from_primary_solver_state(y, args)
    grid = args.grid
    grid_position = convert_positions(position, Geometry.CARTESIAN, grid.geom)
    active_position = grid_position[..., : grid.dimensions]
    active_extents = grid.extents[: grid.dimensions]
    coordinate_scale = jnp.maximum(1.0, jnp.max(jnp.abs(active_extents), axis=1))
    coordinate_tolerance = 512.0 * jnp.finfo(position.dtype).eps * coordinate_scale
    lower_margin = (
        active_position - active_extents[:, 0] + coordinate_tolerance
    ) / coordinate_scale
    upper_margin = (
        active_extents[:, 1] + coordinate_tolerance - active_position
    ) / coordinate_scale
    ray_margin = jnp.min(
        jnp.concatenate((lower_margin, upper_margin), axis=-1), axis=-1
    )
    return jnp.max(ray_margin)


def _amplitude_limit_history(
    diagnostics,
    omega,
    minimum_amplitude_cap,
):
    ncritical = critical_density(omega)

    def carry_lengthscale(previous, inputs):
        gradient_norm, inside = inputs
        usable = inside & jnp.isfinite(gradient_norm) & (gradient_norm > 0.0)
        current = ncritical / jnp.maximum(
            gradient_norm, jnp.finfo(gradient_norm.dtype).tiny
        )
        updated = jnp.where(usable, current, previous)
        return updated, updated

    initial = jnp.full_like(ncritical, jnp.inf)
    _, lengthscale = jax.lax.scan(
        carry_lengthscale,
        initial,
        (diagnostics.gradient_norm, diagnostics.inside),
    )
    zeta = 0.9 * jnp.cbrt(omega[None, ...] * lengthscale / SPEED_OF_LIGHT)
    nonnegative_density = jnp.maximum(diagnostics.density_ratio, 0.0)
    critical_cap_squared = zeta * jnp.sqrt(nonnegative_density)
    critical_cap = jnp.where(
        nonnegative_density > 0.0,
        jnp.sqrt(critical_cap_squared),
        jnp.inf,
    )
    geometric_cap = jnp.sqrt(zeta / diagnostics.area_ratio)
    limit = jnp.minimum(critical_cap, geometric_cap)
    finite_upper = jnp.sqrt(jnp.finfo(limit.dtype).max)
    limit = jnp.nan_to_num(
        limit,
        nan=finite_upper,
        posinf=finite_upper,
        neginf=minimum_amplitude_cap,
    )
    return jnp.maximum(limit, minimum_amplitude_cap)


def _caustic_locations(paths, uncapped, amplitude_limit):
    capped = jnp.minimum(uncapped, amplitude_limit)
    score = uncapped - capped
    maximum_index = jnp.argmax(score, axis=0)
    gather_index = maximum_index[None, ...]
    maximum_score = jnp.take_along_axis(score, gather_index, axis=0)[0]
    caustic_path = jnp.take_along_axis(
        paths.reshape((paths.shape[0],) + (1,) * maximum_index.ndim),
        gather_index,
        axis=0,
    )[0]

    previous_index = jnp.maximum(maximum_index - 1, 0)[None, ...]
    next_index = jnp.minimum(maximum_index + 1, paths.shape[0] - 1)[None, ...]
    previous_score = jnp.take_along_axis(score, previous_index, axis=0)[0]
    next_score = jnp.take_along_axis(score, next_index, axis=0)[0]
    curvature = previous_score - 2.0 * maximum_score + next_score
    safe_curvature = jnp.where(curvature == 0.0, 1.0, curvature)
    offset = 0.5 * (previous_score - next_score) / safe_curvature
    offset = jnp.clip(offset, -1.0, 1.0)
    interior = (maximum_index > 0) & (maximum_index < paths.shape[0] - 1)
    refine = interior & (curvature < 0.0) & jnp.isfinite(offset)
    path_spacing = paths[1] - paths[0]
    caustic_path = jnp.where(refine, caustic_path + offset * path_spacing, caustic_path)
    refined_score = maximum_score + 0.5 * (next_score - previous_score) * offset
    refined_score += 0.5 * curvature * offset**2
    maximum_score = jnp.where(refine, refined_score, maximum_score)
    has_caustic = maximum_score > 0.0
    return has_caustic, caustic_path, maximum_score


def _sheet_sample_paths(has_caustic, caustic_path, terminal_path, nsamples):
    fraction = jnp.linspace(0.0, 1.0, nsamples, dtype=terminal_path.dtype)
    first_end = jnp.where(has_caustic, caustic_path, terminal_path)
    first = first_end[..., None] * fraction
    second = (
        caustic_path[..., None] + (terminal_path - caustic_path[..., None]) * fraction
    )
    second = jnp.where(has_caustic[..., None], second, terminal_path)
    return jnp.stack((first, second), axis=-2)


def _resampling_paths(has_caustic, caustic_path, terminal_path, nsamples):
    epsilon = jnp.finfo(terminal_path.dtype).eps
    separation = epsilon * jnp.maximum(1.0, terminal_path)
    internal_caustic = jnp.clip(
        caustic_path,
        separation,
        jnp.maximum(terminal_path - separation, separation),
    )
    fraction = jnp.linspace(0.0, 1.0, nsamples, dtype=terminal_path.dtype)
    first = internal_caustic[..., None] * fraction
    second = (
        internal_caustic[..., None]
        + (terminal_path - internal_caustic[..., None]) * fraction[1:]
    )
    caustic_paths = jnp.concatenate((first, second), axis=-1)
    regular_paths = terminal_path * jnp.linspace(
        0.0, 1.0, 2 * nsamples - 1, dtype=terminal_path.dtype
    )
    selected_paths = jnp.where(has_caustic[..., None], caustic_paths, regular_paths)
    # The affine construction can round its final value one ULP above t1 for
    # some caustic locations. Diffrax validates SaveAt times strictly.
    return jnp.clip(selected_paths, 0.0, terminal_path)


def _resample_states(
    initial_state,
    grid,
    scaling,
    has_caustic,
    caustic_path,
    terminal_path,
    options,
    dt0,
    inverse_bremsstrahlung,
):
    leading_shape = initial_state.shape[:-1]
    flat_state = initial_state.reshape((-1, RAY_STATE_LAYOUT.n_attributes))
    flat_scales = scaling.scales.reshape((-1, RAY_STATE_LAYOUT.n_attributes))
    flat_references = scaling.position_reference.reshape((-1, 3))
    paths = _resampling_paths(
        has_caustic, caustic_path, terminal_path, options.nsamples_per_sheet
    )
    flat_paths = paths.reshape((-1, paths.shape[-1]))
    term = diffrax.ODETerm(_scaled_tangent_ray_rhs)
    solver = diffrax.Tsit5()
    controller = diffrax.PIDController(
        rtol=options.rtol,
        atol=options.atol,
        norm=_maximum_norm,
    )

    def solve_one(state, state_scales, position_reference, save_paths):
        args = _TraceArguments(
            grid=grid,
            initial_area=jnp.asarray(1.0, dtype=state.dtype),
            area_floor=jnp.asarray(options.area_floor, dtype=state.dtype),
            scaling=_StateScaling(state_scales, position_reference),
            inverse_bremsstrahlung=inverse_bremsstrahlung,
        )
        solution = diffrax.diffeqsolve(
            term,
            solver,
            t0=0.0,
            t1=terminal_path,
            dt0=dt0,
            y0=state,
            args=args,
            saveat=diffrax.SaveAt(ts=save_paths),
            stepsize_controller=controller,
            max_steps=options.max_steps,
            throw=False,
        )
        return solution.ys

    flat_values = jax.vmap(solve_one)(
        flat_state, flat_scales, flat_references, flat_paths
    )
    nsamples = options.nsamples_per_sheet
    caustic_first = flat_values[:, :nsamples, :]
    caustic_second = jnp.concatenate(
        (flat_values[:, nsamples - 1 : nsamples, :], flat_values[:, nsamples:, :]),
        axis=1,
    )
    regular_first = flat_values[:, ::2, :]
    regular_second = jnp.broadcast_to(flat_values[:, -1:, :], regular_first.shape)
    flat_has_caustic = has_caustic.reshape((-1, 1, 1))
    first = jnp.where(flat_has_caustic, caustic_first, regular_first)
    second = jnp.where(flat_has_caustic, caustic_second, regular_second)
    values = jnp.stack((first, second), axis=1)
    return values.reshape(
        (*leading_shape, 2, options.nsamples_per_sheet, RAY_STATE_LAYOUT.n_attributes)
    )


def _interpolate_history(paths, values, query_paths):
    ray_shape = values.shape[1:]
    flat_values = jnp.moveaxis(values, 0, -1).reshape((-1, values.shape[0]))
    flat_queries = query_paths.reshape(
        (-1, query_paths.shape[-2] * query_paths.shape[-1])
    )

    def interpolate_one(history, queries):
        return jnp.interp(queries, paths, history)

    result = jax.vmap(interpolate_one)(flat_values, flat_queries)
    return result.reshape((*ray_shape, *query_paths.shape[-2:]))


def _refresh_permittivity(state, grid):
    hydro = grid.interpolate(state[..., RAY_STATE_LAYOUT.position])
    ncritical = critical_density(state[..., RAY_STATE_LAYOUT.frequency])
    epsilon = safe_permittivity(hydro.ne / ncritical)
    return state.at[..., RAY_STATE_LAYOUT.permittivity].set(epsilon)


def _grid_characteristic_length(grid):
    """Return the norm of resolved, non-angular physical grid lengths.

    Explicitly inactive directions are omitted. Angular axes are never lengths,
    so cylindrical ``phi`` and spherical ``phi``/``theta`` do not contribute.
    """
    extents = grid.extents
    physical_axes = {
        "cartesian": (0, 1, 2),
        "cylindrical": (0, 2),
        "spherical": (0,),
    }[grid.geom.value]
    resolved_lengths = [
        extents[axis, 1] - extents[axis, 0]
        for axis in physical_axes
        if axis in grid.active_axes
    ]
    if not resolved_lengths:
        raise ValueError(
            "grid needs more than one cell in at least one non-angular physical "
            "direction to define the ray-tracing length scale"
        )
    return jnp.linalg.norm(jnp.stack(resolved_lengths))


def trace_rays(
    initial_rays,
    grid: Grid,
    *,
    options: RayTracingOptions | None = None,
    inverse_bremsstrahlung: InverseBremsstrahlungOptions | None = None,
):
    """Propagate initialized rays and return two caustic-resolved sheets.

    A lightweight first solve finds the common primary-ray exit event. A
    second global solve records uniformly spaced amplitude diagnostics over
    that actual interval. Finally, a vmapped solve resamples each independent
    ray tube directly at the requested sheet locations. This avoids retaining
    a dense full-state history.
    """
    options = RayTracingOptions() if options is None else options
    options.validate()
    inverse_bremsstrahlung = (
        InverseBremsstrahlungOptions()
        if inverse_bremsstrahlung is None
        else inverse_bremsstrahlung
    )
    inverse_bremsstrahlung.validate()
    state = getattr(initial_rays, "state", initial_rays)
    initial_momentum_tangents = getattr(
        initial_rays, "neighbour_momentum_tangents", None
    )
    state = jnp.asarray(state, dtype=jnp.float64)
    if state.ndim != 4 or state.shape[-1] != RAY_STATE_LAYOUT.n_attributes:
        raise ValueError(
            "initial ray state must have shape "
            f"(nbeams, nrays_axis1, nrays_axis2, {RAY_STATE_LAYOUT.n_attributes})"
        )

    characteristic_length = _grid_characteristic_length(grid)
    maximum_path = options.maximum_path_length_grid_lengths * characteristic_length
    dt0 = options.dt0
    initial_area = initial_ray_area(state)
    relative_state = _absolute_to_relative_state(state)
    if initial_momentum_tangents is not None:
        initial_momentum_tangents = jnp.asarray(
            initial_momentum_tangents, dtype=state.dtype
        )
        expected_shape = (*state.shape[:-1], 9)
        if initial_momentum_tangents.shape != expected_shape:
            raise ValueError(
                "initial neighbour momentum tangents must have shape "
                f"{expected_shape}, got {initial_momentum_tangents.shape}"
            )
        relative_state = relative_state.at[..., RAY_STATE_LAYOUT.neighbour_momenta].set(
            initial_momentum_tangents
        )
    scaling = _build_state_scaling(relative_state, characteristic_length)
    solver_state = _to_solver_state(relative_state, scaling)
    primary_solver_state = _to_primary_solver_state(solver_state)
    primary_args = _PrimaryTraceArguments(
        grid=grid,
        position_scale=scaling.scales[..., RAY_STATE_LAYOUT.position],
        position_reference=scaling.position_reference,
        momentum_scale=scaling.scales[..., RAY_STATE_LAYOUT.momentum],
        frequency=state[..., RAY_STATE_LAYOUT.frequency],
    )
    args = _TraceArguments(
        grid=grid,
        initial_area=initial_area,
        area_floor=jnp.asarray(options.area_floor, dtype=state.dtype),
        scaling=scaling,
        inverse_bremsstrahlung=inverse_bremsstrahlung,
    )
    controller = diffrax.PIDController(
        rtol=options.rtol,
        atol=options.atol,
        norm=_maximum_norm,
    )
    root_relative_tolerance = 512.0 * jnp.finfo(state.dtype).eps
    exit_root_finder = optx.Bisection(
        rtol=root_relative_tolerance,
        atol=root_relative_tolerance * characteristic_length,
    )
    exit_solution = diffrax.diffeqsolve(
        diffrax.ODETerm(_scaled_primary_ray_rhs),
        diffrax.Tsit5(),
        t0=0.0,
        t1=maximum_path,
        dt0=dt0,
        y0=primary_solver_state,
        args=primary_args,
        saveat=diffrax.SaveAt(t1=True),
        stepsize_controller=controller,
        event=diffrax.Event(
            _all_primary_rays_outside,
            root_finder=exit_root_finder,
            direction=False,
        ),
        max_steps=options.max_steps,
        throw=False,
    )
    terminal_path = exit_solution.ts[0]
    paths = jnp.linspace(
        0.0, terminal_path, options.diagnostic_samples, dtype=state.dtype
    )
    diagnostic_solution = diffrax.diffeqsolve(
        diffrax.ODETerm(_scaled_tangent_ray_rhs),
        diffrax.Tsit5(),
        t0=0.0,
        t1=terminal_path,
        dt0=dt0,
        y0=solver_state,
        args=args,
        saveat=diffrax.SaveAt(ts=paths, fn=_save_diagnostics),
        stepsize_controller=controller,
        max_steps=options.max_steps,
        throw=False,
    )
    diagnostics = diagnostic_solution.ys
    omega = state[..., RAY_STATE_LAYOUT.frequency]
    amplitude_limit = _amplitude_limit_history(
        diagnostics,
        omega,
        options.minimum_amplitude_cap,
    )
    has_caustic, caustic_path, caustic_score = _caustic_locations(
        paths,
        diagnostics.uncapped_amplitude,
        amplitude_limit,
    )
    sheet_paths = _sheet_sample_paths(
        has_caustic,
        caustic_path,
        terminal_path,
        options.nsamples_per_sheet,
    )
    sampled_state = _resample_states(
        solver_state,
        grid,
        scaling,
        has_caustic,
        caustic_path,
        terminal_path,
        options,
        dt0,
        inverse_bremsstrahlung,
    )
    sampled_scaling = _StateScaling(
        scales=scaling.scales[..., None, None, :],
        position_reference=scaling.position_reference[..., None, None, :],
    )
    sampled_diagnostics = _instantaneous_diagnostics(
        sampled_state,
        _TraceArguments(
            grid=grid,
            initial_area=initial_area[..., None, None],
            area_floor=jnp.asarray(options.area_floor, dtype=state.dtype),
            scaling=sampled_scaling,
            inverse_bremsstrahlung=inverse_bremsstrahlung,
        ),
    )
    sampled_relative_state = _from_solver_state(sampled_state, sampled_scaling)
    sampled_absolute_state = _relative_to_absolute_state(sampled_relative_state)
    sampled_absolute_state = _refresh_permittivity(sampled_absolute_state, grid)
    sampled_limit = _interpolate_history(paths, amplitude_limit, sheet_paths)
    capped_amplitude = jnp.minimum(
        sampled_diagnostics.uncapped_amplitude, sampled_limit
    )
    total_optical_depth = sum(
        sampled_absolute_state[..., index]
        for index in (
            RAY_STATE_LAYOUT.inverse_brems_depth,
            RAY_STATE_LAYOUT.cbet_depth,
            RAY_STATE_LAYOUT.srs_depth,
            RAY_STATE_LAYOUT.sbs_depth,
        )
    )
    electric_field = (
        sampled_absolute_state[..., RAY_STATE_LAYOUT.initial_electric_field]
        * capped_amplitude
        * jnp.exp(-0.5 * total_optical_depth)
    )
    refractive_index = jnp.sqrt(
        sampled_absolute_state[..., RAY_STATE_LAYOUT.permittivity]
    )
    intensity = (
        0.5
        * VACUUM_PERMITTIVITY
        * SPEED_OF_LIGHT
        * refractive_index
        * electric_field**2
    )
    sampled_hydro = grid.interpolate(
        sampled_absolute_state[..., RAY_STATE_LAYOUT.position]
    )
    if inverse_bremsstrahlung.enabled:
        inverse_bremsstrahlung_rate = inverse_bremsstrahlung_depth_derivative(
            sampled_hydro.ne,
            sampled_hydro.Te,
            sampled_absolute_state[..., RAY_STATE_LAYOUT.frequency],
            grid.composition.effective_charge,
            minimum_coulomb_log=(inverse_bremsstrahlung.minimum_coulomb_log),
            coulomb_log_override=(inverse_bremsstrahlung.coulomb_log_override),
            critical_collision_frequency_hz=(
                inverse_bremsstrahlung.critical_collision_frequency_hz
            ),
            inside=sampled_hydro.inside,
        )
    else:
        inverse_bremsstrahlung_rate = jnp.zeros_like(electric_field)
    inverse_brems_deposition = (
        0.5
        * VACUUM_PERMITTIVITY
        * SPEED_OF_LIGHT
        * electric_field**2
        * inverse_bremsstrahlung_rate
    )
    sheet_fields = jnp.concatenate(
        (
            sampled_absolute_state,
            sampled_diagnostics.uncapped_amplitude[..., None],
            capped_amplitude[..., None],
            electric_field[..., None],
            intensity[..., None],
            inverse_brems_deposition[..., None],
        ),
        axis=-1,
    )
    sheet_fields = jnp.moveaxis(sheet_fields, 3, 1)
    initial_path = state[..., RAY_STATE_LAYOUT.path_length]
    absolute_caustic_path = jnp.where(has_caustic, initial_path + caustic_path, jnp.inf)
    return RayTraceResult(
        sheet_fields=sheet_fields,
        has_caustic=has_caustic,
        caustic_path=absolute_caustic_path,
        caustic_score=caustic_score,
        terminal_path=terminal_path,
        terminated=jnp.asarray(exit_solution.event_mask),
    )


__all__ = [
    "RayTraceResult",
    "RayTracingOptions",
    "initial_ray_area",
    "ray_rhs",
    "trace_rays",
]
