"""Conservative adaptive deposition for segments, triangles, and tetrahedra."""

from __future__ import annotations

import math
from functools import partial
from numbers import Integral

import jax
import jax.numpy as jnp
import numpy as np

from .deposition import (
    _CHILD_WEIGHTS as _TETRAHEDRON_CHILD_WEIGHTS,
)
from .deposition import (
    GridPowerDeposition,
    _certified_contained,
    _certified_outside,
    _locate_cells,
    grid_cell_volumes,
)
from .simplicial import SimplicialField

_SEGMENT_CHILD_WEIGHTS = np.asarray(
    (
        ((1.0, 0.0), (0.5, 0.5)),
        ((0.5, 0.5), (0.0, 1.0)),
    ),
    dtype=np.float64,
)

_TRIANGLE_CHILD_WEIGHTS = np.asarray(
    (
        ((1.0, 0.0, 0.0), (0.5, 0.5, 0.0), (0.5, 0.0, 0.5)),
        ((0.5, 0.5, 0.0), (0.0, 1.0, 0.0), (0.0, 0.5, 0.5)),
        ((0.5, 0.0, 0.5), (0.0, 0.5, 0.5), (0.0, 0.0, 1.0)),
        ((0.5, 0.5, 0.0), (0.0, 0.5, 0.5), (0.5, 0.0, 0.5)),
    ),
    dtype=np.float64,
)

_CHILD_WEIGHTS = {
    1: _SEGMENT_CHILD_WEIGHTS,
    2: _TRIANGLE_CHILD_WEIGHTS,
    3: _TETRAHEDRON_CHILD_WEIGHTS,
}


def _simplex_power(positions, values, dimension, inactive_measure):
    active = positions[..., :dimension]
    edges = jnp.stack(
        tuple(
            active[..., vertex, :] - active[..., 0, :]
            for vertex in range(1, dimension + 1)
        ),
        axis=-1,
    )
    measure = jnp.abs(jnp.linalg.det(edges)) / math.factorial(dimension)
    return inactive_measure * measure * jnp.mean(values, axis=-1)


def _split_simplex(positions, values, dimension):
    weights = jnp.asarray(_CHILD_WEIGHTS[dimension], dtype=positions.dtype)
    child_positions = jnp.einsum("cvp,...pd->...cvd", weights, positions)
    child_values = jnp.einsum("cvp,...p->...cv", weights, values)
    return child_positions, child_values


def _deposit_adaptive_batch(
    positions,
    values,
    valid,
    grid,
    cell_power,
    outside_power,
    relative_tolerance,
    max_subdivision_levels,
    dimension,
):
    number_of_cells = int(np.prod(grid.ncells))
    inactive_measure = grid.inactive_measure
    root_powers = jnp.where(
        valid,
        _simplex_power(positions, values, dimension, inactive_measure),
        0.0,
    )
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
    nvertices = dimension + 1
    nchildren = 2**dimension
    stack_capacity = 1 + (nchildren - 1) * max_subdivision_levels
    stack_positions = (
        jnp.zeros((batch_size, stack_capacity, nvertices, 3), dtype=positions.dtype)
        .at[:, 0]
        .set(positions)
    )
    stack_values = (
        jnp.zeros((batch_size, stack_capacity, nvertices), dtype=values.dtype)
        .at[:, 0]
        .set(values)
    )
    stack_depths = jnp.zeros((batch_size, stack_capacity), dtype=jnp.int32)
    stack_sizes = valid.astype(jnp.int32)
    lanes = jnp.arange(batch_size, dtype=jnp.int32)
    children = jnp.arange(nchildren, dtype=jnp.int32)

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
        node_powers = _simplex_power(
            node_positions, node_values, dimension, inactive_measure
        )
        centroids = jnp.mean(node_positions, axis=-2)
        cells, inside, native, indices = _locate_cells(grid, centroids)
        target_cells = jnp.where(inside, cells, number_of_cells)
        contained = _certified_contained(
            grid, node_positions, centroids, native, indices, inside
        )
        outside = _certified_outside(grid, node_positions, centroids, native)
        zero_field = jnp.all(node_values == 0.0, axis=-1)
        at_maximum = node_depths >= max_subdivision_levels

        child_positions, child_values = _split_simplex(
            node_positions, node_values, dimension
        )
        child_centroids = jnp.mean(child_positions, axis=-2)
        child_cells, child_inside, _, _ = _locate_cells(grid, child_centroids)
        child_cells = jnp.where(child_inside, child_cells, number_of_cells)
        child_powers = _simplex_power(
            child_positions, child_values, dimension, inactive_measure
        )
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
        current_positions = current_positions.at[lanes[:, None], stack_indices].set(
            jnp.where(split[:, None, None, None], child_positions, existing_positions)
        )
        current_values = current_values.at[lanes[:, None], stack_indices].set(
            jnp.where(split[:, None, None], child_values, existing_values)
        )
        current_depths = current_depths.at[lanes[:, None], stack_indices].set(
            jnp.where(split[:, None], node_depths[:, None] + 1, existing_depths)
        )
        current_sizes = popped_sizes + nchildren * split.astype(jnp.int32)
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
        "simplex_batch_size",
    ),
)
def _deposit_simplicial_power_core(
    field,
    grid,
    relative_tolerance,
    *,
    field_index,
    selected_beam,
    max_subdivision_levels,
    simplex_batch_size,
):
    mesh = field.mesh
    dimension = mesh.dimension
    number_of_cells = int(np.prod(grid.ncells))
    cell_power = jnp.zeros((number_of_cells,), dtype=jnp.float64)
    if mesh.nsimplices == 0:
        zero = jnp.asarray(0.0, dtype=jnp.float64)
        return cell_power, zero, zero

    selected_beams = mesh.nbeams if selected_beam < 0 else 1
    source_count = selected_beams * mesh.nsheets * mesh.nsimplices
    number_of_batches = (source_count + simplex_batch_size - 1) // simplex_batch_size
    batch_offsets = jnp.arange(simplex_batch_size, dtype=jnp.int32)

    def process_batch(batch_number, carry):
        accumulated_cells, accumulated_outside, accumulated_source = carry
        flat_indices = batch_number * simplex_batch_size + batch_offsets
        real_source = flat_indices < source_count
        safe_indices = jnp.minimum(flat_indices, source_count - 1)
        per_beam = mesh.nsheets * mesh.nsimplices
        if selected_beam < 0:
            beam_indices = safe_indices // per_beam
            within_beam = safe_indices % per_beam
        else:
            beam_indices = jnp.full_like(safe_indices, selected_beam)
            within_beam = safe_indices
        sheet_indices = within_beam // mesh.nsimplices
        simplex_indices = within_beam % mesh.nsimplices
        vertex_indices = mesh.connectivity[simplex_indices]
        positions = mesh.vertex_positions[
            beam_indices[:, None], sheet_indices[:, None], vertex_indices
        ]
        values = field.vertex_values[
            beam_indices[:, None], sheet_indices[:, None], vertex_indices, field_index
        ]
        valid = real_source & mesh.valid[beam_indices, sheet_indices, simplex_indices]
        valid &= jnp.all(jnp.isfinite(values), axis=-1)
        accumulated_cells, accumulated_outside, batch_source = _deposit_adaptive_batch(
            positions,
            values,
            valid,
            grid,
            accumulated_cells,
            accumulated_outside,
            relative_tolerance,
            max_subdivision_levels,
            dimension,
        )
        return (
            accumulated_cells,
            accumulated_outside,
            accumulated_source + batch_source,
        )

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


def deposit_simplicial_power(
    field: SimplicialField,
    grid,
    *,
    field_name: str = "inverse_brems_deposition",
    beam_index: int | None = None,
    max_subdivision_levels: int = 2,
    relative_tolerance: float = 1.0e-3,
    simplex_batch_size: int = 4096,
) -> GridPowerDeposition:
    """Conservatively integrate a reduced or 3-D volumetric source onto cells."""
    if field.mesh.dimension != grid.dimensions:
        raise ValueError("field and grid dimensions must match")
    if not isinstance(field_name, str):
        raise TypeError("field_name must be a string")
    try:
        field_index = getattr(field.selection, field_name)
    except AttributeError as error:
        raise ValueError(f"simplicial field does not include {field_name!r}") from error
    if not isinstance(field_index, Integral):
        raise TypeError("deposited simplicial field must be scalar")
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
    if not 0 <= max_subdivision_levels <= 4:
        raise ValueError("max_subdivision_levels must lie between zero and four")
    if not np.isfinite(relative_tolerance) or relative_tolerance < 0.0:
        raise ValueError("relative_tolerance must be finite and nonnegative")
    if isinstance(simplex_batch_size, bool) or not isinstance(
        simplex_batch_size, Integral
    ):
        raise TypeError("simplex_batch_size must be an integer")
    simplex_batch_size = int(simplex_batch_size)
    if simplex_batch_size < 1:
        raise ValueError("simplex_batch_size must be positive")

    flat_power, outside_power, source_power = _deposit_simplicial_power_core(
        field,
        grid,
        jnp.asarray(relative_tolerance, dtype=jnp.float64),
        field_index=int(field_index),
        selected_beam=selected_beam,
        max_subdivision_levels=max_subdivision_levels,
        simplex_batch_size=simplex_batch_size,
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


__all__ = ["deposit_simplicial_power"]
