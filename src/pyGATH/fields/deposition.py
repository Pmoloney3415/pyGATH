"""Conservative adaptive deposition of tetrahedral power onto hydro cells."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from numbers import Integral
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from pyGATH.grid import Geometry, Grid, convert_positions

from .tetrahedral import TetrahedralField


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class GridPowerDeposition:
    """Cell-centred conservative deposition results.

    ``power_density`` has units W/m^3 and ``cell_power`` has units W. The
    final three scalar diagnostics make the finite-grid conservation identity
    ``source_power = deposited_power + outside_power`` explicit.
    """

    power_density: Any
    cell_power: Any
    deposited_power: Any
    outside_power: Any
    source_power: Any

    @property
    def conservation_error(self):
        """Return the signed finite-grid power-balance residual in watts."""
        return self.source_power - self.deposited_power - self.outside_power

    def tree_flatten(self):
        return (
            self.power_density,
            self.cell_power,
            self.deposited_power,
            self.outside_power,
            self.source_power,
        ), None

    @classmethod
    def tree_unflatten(cls, _auxiliary, children):
        return cls(*children)


def grid_cell_volumes(grid: Grid):
    """Return exact physical volumes for every native rectilinear cell."""
    first_width = jnp.diff(grid.xb)
    second_width = jnp.diff(grid.yb)
    third_width = jnp.diff(grid.zb)
    if grid.geom is Geometry.CARTESIAN:
        first_factor = first_width
        third_factor = third_width
    elif grid.geom is Geometry.CYLINDRICAL:
        first_factor = 0.5 * (grid.xb[1:] ** 2 - grid.xb[:-1] ** 2)
        third_factor = third_width
    else:
        first_factor = (grid.xb[1:] ** 3 - grid.xb[:-1] ** 3) / 3.0
        third_factor = jnp.cos(grid.zb[:-1]) - jnp.cos(grid.zb[1:])
    return (
        first_factor[:, None, None]
        * second_width[None, :, None]
        * third_factor[None, None, :]
    )


def _axis_cell_index(vertices, query, is_uniform):
    finite_query = jnp.where(jnp.isfinite(query), query, vertices[0])
    if is_uniform:
        scaled = (finite_query - vertices[0]) / (vertices[1] - vertices[0])
        index = jnp.floor(scaled).astype(jnp.int32)
    else:
        index = jnp.searchsorted(vertices, finite_query, side="right") - 1
    index = jnp.clip(index, 0, vertices.size - 2)
    inside = jnp.isfinite(query) & (query >= vertices[0]) & (query <= vertices[-1])
    return index, inside


def _locate_cells(grid, cartesian_points):
    native = convert_positions(cartesian_points, Geometry.CARTESIAN, grid.geom)
    if grid.dimensions < 3:
        reference_centres = jnp.stack((grid.xc[0], grid.yc[0], grid.zc[0]))
        active = jnp.arange(3) < grid.dimensions
        native = jnp.where(active, native, reference_centres)
    axes = (grid.xb, grid.yb, grid.zb)
    locations = tuple(
        _axis_cell_index(axis, native[..., component], grid.is_uniform[component])
        for component, axis in enumerate(axes)
    )
    ix, iy, iz = (location[0] for location in locations)
    inside = locations[0][1] & locations[1][1] & locations[2][1]
    flat_index = (ix * grid.ncells[1] + iy) * grid.ncells[2] + iz
    return flat_index, inside, native, jnp.stack((ix, iy, iz), axis=-1)


def _tetrahedron_power(positions, values):
    edges = jnp.stack(
        tuple(
            positions[..., vertex, :] - positions[..., 0, :] for vertex in range(1, 4)
        ),
        axis=-1,
    )
    volume = jnp.abs(jnp.linalg.det(edges)) / 6.0
    return volume * jnp.mean(values, axis=-1)


_MIDPOINTS = {
    (0, 1): (0.5, 0.5, 0.0, 0.0),
    (0, 2): (0.5, 0.0, 0.5, 0.0),
    (0, 3): (0.5, 0.0, 0.0, 0.5),
    (1, 2): (0.0, 0.5, 0.5, 0.0),
    (1, 3): (0.0, 0.5, 0.0, 0.5),
    (2, 3): (0.0, 0.0, 0.5, 0.5),
}
_PARENT_POINTS = np.asarray(
    (
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
        _MIDPOINTS[(0, 1)],
        _MIDPOINTS[(0, 2)],
        _MIDPOINTS[(0, 3)],
        _MIDPOINTS[(1, 2)],
        _MIDPOINTS[(1, 3)],
        _MIDPOINTS[(2, 3)],
    ),
    dtype=np.float64,
)
_CHILD_CONNECTIVITY = np.asarray(
    (
        (0, 4, 5, 6),
        (1, 4, 7, 8),
        (2, 5, 7, 9),
        (3, 6, 8, 9),
        (4, 5, 6, 9),
        (4, 5, 7, 9),
        (4, 6, 8, 9),
        (4, 7, 8, 9),
    ),
    dtype=np.int32,
)
_CHILD_WEIGHTS = _PARENT_POINTS[_CHILD_CONNECTIVITY]


def _split_tetrahedron(positions, values):
    weights = jnp.asarray(_CHILD_WEIGHTS, dtype=positions.dtype)
    child_positions = jnp.einsum("cvp,...pd->...cvd", weights, positions)
    child_values = jnp.einsum("cvp,...p->...cv", weights, values)
    return child_positions, child_values


def _ball_angular_interval(angle, radial_distance, ball_radius):
    separated_from_axis = radial_distance > ball_radius
    safe_distance = jnp.maximum(radial_distance, jnp.finfo(ball_radius.dtype).tiny)
    half_width = jnp.arcsin(jnp.clip(ball_radius / safe_distance, 0.0, 1.0))
    return angle - half_width, angle + half_width, separated_from_axis


def _certified_contained(grid, positions, centroid, native, indices, inside):
    ix, iy, iz = jnp.moveaxis(indices, -1, 0)
    lower = jnp.stack((grid.xb[ix], grid.yb[iy], grid.zb[iz]), axis=-1)
    upper = jnp.stack((grid.xb[ix + 1], grid.yb[iy + 1], grid.zb[iz + 1]), axis=-1)
    if grid.geom is Geometry.CARTESIAN:
        active_positions = positions[..., : grid.dimensions]
        active_lower = lower[..., : grid.dimensions]
        active_upper = upper[..., : grid.dimensions]
        return (
            inside
            & jnp.all(jnp.min(active_positions, axis=-2) >= active_lower, axis=-1)
            & jnp.all(jnp.max(active_positions, axis=-2) <= active_upper, axis=-1)
        )

    physical_positions = positions[..., : grid.dimensions]
    physical_centroid = centroid[..., : grid.dimensions]
    ball_radius = jnp.max(
        jnp.linalg.norm(physical_positions - physical_centroid[..., None, :], axis=-1),
        axis=-1,
    )
    radial_contained = (native[..., 0] - ball_radius >= lower[..., 0]) & (
        native[..., 0] + ball_radius <= upper[..., 0]
    )
    cylindrical_radius = jnp.hypot(centroid[..., 0], centroid[..., 1])
    phi_lower, phi_upper, away_from_axis = _ball_angular_interval(
        native[..., 1], cylindrical_radius, ball_radius
    )
    full_phi = upper[..., 1] - lower[..., 1] >= 2.0 * jnp.pi
    phi_contained = full_phi | (
        away_from_axis & (phi_lower >= lower[..., 1]) & (phi_upper <= upper[..., 1])
    )
    if grid.geom is Geometry.CYLINDRICAL and grid.dimensions == 3:
        third_contained = (centroid[..., 2] - ball_radius >= lower[..., 2]) & (
            centroid[..., 2] + ball_radius <= upper[..., 2]
        )
    elif grid.geom is Geometry.CYLINDRICAL:
        third_contained = jnp.ones_like(inside)
    else:
        theta_lower, theta_upper, away_from_origin = _ball_angular_interval(
            native[..., 2], native[..., 0], ball_radius
        )
        full_theta = upper[..., 2] - lower[..., 2] >= jnp.pi
        third_contained = full_theta | (
            away_from_origin
            & (theta_lower >= lower[..., 2])
            & (theta_upper <= upper[..., 2])
        )
    return inside & radial_contained & phi_contained & third_contained


def _certified_outside(grid, positions, centroid, native):
    extents = grid.extents
    if grid.geom is Geometry.CARTESIAN:
        lower = jnp.min(positions[..., : grid.dimensions], axis=-2)
        upper = jnp.max(positions[..., : grid.dimensions], axis=-2)
        return jnp.any(
            (upper < extents[: grid.dimensions, 0])
            | (lower > extents[: grid.dimensions, 1]),
            axis=-1,
        )

    physical_positions = positions[..., : grid.dimensions]
    physical_centroid = centroid[..., : grid.dimensions]
    ball_radius = jnp.max(
        jnp.linalg.norm(physical_positions - physical_centroid[..., None, :], axis=-1),
        axis=-1,
    )
    radial_outside = (native[..., 0] + ball_radius < extents[0, 0]) | (
        native[..., 0] - ball_radius > extents[0, 1]
    )
    cylindrical_radius = jnp.hypot(centroid[..., 0], centroid[..., 1])
    phi_lower, phi_upper, away_from_axis = _ball_angular_interval(
        native[..., 1], cylindrical_radius, ball_radius
    )
    phi_outside = away_from_axis & (
        (phi_upper < extents[1, 0]) | (phi_lower > extents[1, 1])
    )
    if grid.geom is Geometry.CYLINDRICAL and grid.dimensions == 3:
        third_outside = (centroid[..., 2] + ball_radius < extents[2, 0]) | (
            centroid[..., 2] - ball_radius > extents[2, 1]
        )
    elif grid.geom is Geometry.CYLINDRICAL:
        third_outside = jnp.zeros_like(radial_outside)
    else:
        theta_lower, theta_upper, away_from_origin = _ball_angular_interval(
            native[..., 2], native[..., 0], ball_radius
        )
        third_outside = away_from_origin & (
            (theta_upper < extents[2, 0]) | (theta_lower > extents[2, 1])
        )
    return radial_outside | phi_outside | third_outside


def _deposit_adaptive_batch(
    positions,
    values,
    valid,
    grid,
    cell_power,
    outside_power,
    relative_tolerance,
    max_subdivision_levels,
):
    """Process one fixed-width batch using independent bounded DFS stacks."""
    number_of_cells = int(np.prod(grid.ncells))
    root_powers = jnp.where(valid, _tetrahedron_power(positions, values), 0.0)
    if max_subdivision_levels == 0:
        centroids = jnp.mean(positions, axis=-2)
        cells, inside, _, _ = _locate_cells(grid, centroids)
        safe_cells = jnp.minimum(cells, number_of_cells - 1)
        cell_power = cell_power.at[safe_cells].add(
            jnp.where(valid & inside, root_powers, 0.0)
        )
        outside_power += jnp.sum(jnp.where(valid & ~inside, root_powers, 0.0))
        return cell_power, outside_power, jnp.sum(root_powers)

    batch_size = positions.shape[0]
    stack_capacity = 1 + 7 * max_subdivision_levels
    stack_positions = (
        jnp.zeros((batch_size, stack_capacity, 4, 3), dtype=positions.dtype)
        .at[:, 0]
        .set(positions)
    )
    stack_values = (
        jnp.zeros((batch_size, stack_capacity, 4), dtype=values.dtype)
        .at[:, 0]
        .set(values)
    )
    stack_depths = jnp.zeros((batch_size, stack_capacity), dtype=jnp.int32)
    stack_sizes = valid.astype(jnp.int32)
    lanes = jnp.arange(batch_size, dtype=jnp.int32)
    children = jnp.arange(8, dtype=jnp.int32)

    def continue_work(carry):
        return jnp.any(carry[3] > 0)

    def process_nodes(carry):
        (
            current_positions,
            current_values,
            current_depths,
            current_sizes,
            current_cell_power,
            current_outside_power,
        ) = carry
        has_node = current_sizes > 0
        popped_sizes = jnp.maximum(current_sizes - 1, 0)
        node_positions = current_positions[lanes, popped_sizes]
        node_values = current_values[lanes, popped_sizes]
        node_depths = current_depths[lanes, popped_sizes]
        node_powers = _tetrahedron_power(node_positions, node_values)
        centroids = jnp.mean(node_positions, axis=-2)
        cells, inside, native, indices = _locate_cells(grid, centroids)
        target_cells = jnp.where(inside, cells, number_of_cells)
        contained = _certified_contained(
            grid,
            node_positions,
            centroids,
            native,
            indices,
            inside,
        )
        outside = _certified_outside(grid, node_positions, centroids, native)
        zero_field = jnp.all(node_values == 0.0, axis=-1)
        at_maximum = node_depths >= max_subdivision_levels

        child_positions, child_values = _split_tetrahedron(node_positions, node_values)
        child_centroids = jnp.mean(child_positions, axis=-2)
        child_cells, child_inside, _, _ = _locate_cells(grid, child_centroids)
        child_cells = jnp.where(child_inside, child_cells, number_of_cells)
        child_powers = _tetrahedron_power(child_positions, child_values)
        moved_power = jnp.sum(
            jnp.where(
                child_cells != target_cells[..., None],
                jnp.abs(child_powers),
                0.0,
            ),
            axis=-1,
        )
        absolute_power = jnp.sum(jnp.abs(child_powers), axis=-1)
        moved_fraction = moved_power / jnp.maximum(
            absolute_power, jnp.finfo(values.dtype).tiny
        )
        accept_estimate = (moved_power > 0.0) & (moved_fraction <= relative_tolerance)
        finish = has_node & (
            contained | outside | zero_field | at_maximum | accept_estimate
        )
        split = has_node & ~finish

        safe_cells = jnp.minimum(target_cells, number_of_cells - 1)
        current_cell_power = current_cell_power.at[safe_cells].add(
            jnp.where(finish & inside, node_powers, 0.0)
        )
        current_outside_power += jnp.sum(jnp.where(finish & ~inside, node_powers, 0.0))

        raw_stack_indices = popped_sizes[:, None] + children
        stack_indices = jnp.where(split[:, None], raw_stack_indices, 0)
        existing_positions = current_positions[lanes[:, None], stack_indices]
        existing_values = current_values[lanes[:, None], stack_indices]
        existing_depths = current_depths[lanes[:, None], stack_indices]
        position_updates = jnp.where(
            split[:, None, None, None], child_positions, existing_positions
        )
        value_updates = jnp.where(split[:, None, None], child_values, existing_values)
        depth_updates = jnp.where(
            split[:, None], node_depths[:, None] + 1, existing_depths
        )
        current_positions = current_positions.at[lanes[:, None], stack_indices].set(
            position_updates
        )
        current_values = current_values.at[lanes[:, None], stack_indices].set(
            value_updates
        )
        current_depths = current_depths.at[lanes[:, None], stack_indices].set(
            depth_updates
        )
        current_sizes = popped_sizes + 8 * split.astype(jnp.int32)
        return (
            current_positions,
            current_values,
            current_depths,
            current_sizes,
            current_cell_power,
            current_outside_power,
        )

    final = jax.lax.while_loop(
        continue_work,
        process_nodes,
        (
            stack_positions,
            stack_values,
            stack_depths,
            stack_sizes,
            cell_power,
            outside_power,
        ),
    )
    return final[4], final[5], jnp.sum(root_powers)


@partial(
    jax.jit,
    static_argnames=(
        "field_index",
        "selected_beam",
        "max_subdivision_levels",
        "tetrahedron_batch_size",
    ),
)
def _deposit_tetrahedral_power_core(
    field,
    grid,
    relative_tolerance,
    *,
    field_index,
    selected_beam,
    max_subdivision_levels,
    tetrahedron_batch_size,
):
    mesh = field.mesh
    number_of_cells = int(np.prod(grid.ncells))
    cell_power = jnp.zeros((number_of_cells,), dtype=jnp.float64)
    if mesh.ntetrahedra == 0:
        zero = jnp.asarray(0.0, dtype=jnp.float64)
        return cell_power, zero, zero

    selected_beams = mesh.nbeams if selected_beam < 0 else 1
    source_count = selected_beams * mesh.nsheets * mesh.ntetrahedra
    number_of_batches = (
        source_count + tetrahedron_batch_size - 1
    ) // tetrahedron_batch_size
    batch_offsets = jnp.arange(tetrahedron_batch_size, dtype=jnp.int32)

    def process_batch(batch_number, carry):
        accumulated_cells, accumulated_outside, accumulated_source = carry
        flat_indices = batch_number * tetrahedron_batch_size + batch_offsets
        real_source = flat_indices < source_count
        safe_indices = jnp.minimum(flat_indices, source_count - 1)
        per_beam = mesh.nsheets * mesh.ntetrahedra
        if selected_beam < 0:
            beam_indices = safe_indices // per_beam
            within_beam = safe_indices % per_beam
        else:
            beam_indices = jnp.full_like(safe_indices, selected_beam)
            within_beam = safe_indices
        sheet_indices = within_beam // mesh.ntetrahedra
        tetrahedron_indices = within_beam % mesh.ntetrahedra
        vertex_indices = mesh.connectivity[tetrahedron_indices]
        positions = mesh.vertex_positions[
            beam_indices[:, None], sheet_indices[:, None], vertex_indices
        ]
        values = field.vertex_values[
            beam_indices[:, None], sheet_indices[:, None], vertex_indices, field_index
        ]
        valid = (
            real_source & mesh.valid[beam_indices, sheet_indices, tetrahedron_indices]
        )
        valid &= jnp.all(jnp.isfinite(values), axis=-1)
        (
            accumulated_cells,
            accumulated_outside,
            batch_source,
        ) = _deposit_adaptive_batch(
            positions,
            values,
            valid,
            grid,
            accumulated_cells,
            accumulated_outside,
            relative_tolerance,
            max_subdivision_levels,
        )
        accumulated_source += batch_source
        return accumulated_cells, accumulated_outside, accumulated_source

    return jax.lax.fori_loop(
        0,
        number_of_batches,
        process_batch,
        (
            cell_power,
            jnp.asarray(0.0, dtype=jnp.float64),
            jnp.asarray(0.0, dtype=jnp.float64),
        ),
    )


def deposit_tetrahedral_power(
    field: TetrahedralField,
    grid: Grid,
    *,
    field_name: str = "inverse_brems_deposition",
    beam_index: int | None = None,
    max_subdivision_levels: int = 2,
    relative_tolerance: float = 1.0e-3,
    tetrahedron_batch_size: int = 4096,
) -> GridPowerDeposition:
    """Adaptively integrate a tetrahedral volumetric-power field onto cells.

    Source fields are barycentrically linear. Each accepted source tetrahedron
    is integrated exactly, accumulated in watts, and assigned to the native
    hydro cell containing its centroid. Tetrahedra which cannot be certified
    as belonging to one cell are recursively divided into eight children.

    ``beam_index=None`` sums every beam and sheet. Supplying one beam index
    still sums that beam's sheets. The maximum subdivision level and batch size
    are static JIT controls; the tolerance permits early acceptance when only a
    small fraction of one-level child power changes target cell.
    """
    if not isinstance(field_name, str):
        raise TypeError("field_name must be a string")
    try:
        field_index = getattr(field.selection, field_name)
    except AttributeError as error:
        raise ValueError(
            f"tetrahedral field does not include {field_name!r}"
        ) from error
    if not isinstance(field_index, Integral):
        raise TypeError("deposited tetrahedral field must be scalar")
    if beam_index is None:
        selected_beam = -1
    elif isinstance(beam_index, bool) or not isinstance(beam_index, Integral):
        raise TypeError("beam_index must be an integer or None")
    else:
        selected_beam = int(beam_index)
        if not 0 <= selected_beam < field.mesh.nbeams:
            raise IndexError(
                f"beam_index {selected_beam} is outside {field.mesh.nbeams} beams"
            )
    if isinstance(max_subdivision_levels, bool) or not isinstance(
        max_subdivision_levels, Integral
    ):
        raise TypeError("max_subdivision_levels must be an integer")
    max_subdivision_levels = int(max_subdivision_levels)
    if max_subdivision_levels < 0:
        raise ValueError("max_subdivision_levels cannot be negative")
    if max_subdivision_levels > 4:
        raise ValueError("max_subdivision_levels above four are not supported")
    if not np.isfinite(relative_tolerance) or relative_tolerance < 0.0:
        raise ValueError("relative_tolerance must be finite and nonnegative")
    if isinstance(tetrahedron_batch_size, bool) or not isinstance(
        tetrahedron_batch_size, Integral
    ):
        raise TypeError("tetrahedron_batch_size must be an integer")
    tetrahedron_batch_size = int(tetrahedron_batch_size)
    if tetrahedron_batch_size < 1:
        raise ValueError("tetrahedron_batch_size must be positive")
    flat_power, outside_power, source_power = _deposit_tetrahedral_power_core(
        field,
        grid,
        jnp.asarray(relative_tolerance, dtype=jnp.float64),
        field_index=int(field_index),
        selected_beam=selected_beam,
        max_subdivision_levels=max_subdivision_levels,
        tetrahedron_batch_size=tetrahedron_batch_size,
    )
    cell_power = flat_power.reshape(grid.ncells)
    power_density = cell_power / grid_cell_volumes(grid)
    return GridPowerDeposition(
        power_density=power_density,
        cell_power=cell_power,
        deposited_power=jnp.sum(cell_power),
        outside_power=outside_power,
        source_power=source_power,
    )


__all__ = [
    "GridPowerDeposition",
    "deposit_tetrahedral_power",
    "grid_cell_volumes",
]
