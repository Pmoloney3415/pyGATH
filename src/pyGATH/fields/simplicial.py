"""Dimension-generic indexed laser fields and JAX interpolation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from pyGATH.grid import Geometry, convert_positions
from pyGATH.raytracing import RAY_STATE_LAYOUT

from .fieldlayout import FieldSelection, resolve_field_selection


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class SimplicialMesh:
    """Reusable segment, triangle, or tetrahedron geometry and its BVH."""

    vertex_positions: Any
    connectivity: Any
    origins: Any
    inverse_matrices: Any
    valid: Any
    leaf_order: Any
    internal_bounds_min: Any
    internal_bounds_max: Any
    dimension: int
    logical_shape: tuple[int, int, int]
    leaf_capacity: int
    tree_depth: int

    @property
    def nbeams(self) -> int:
        return self.vertex_positions.shape[0]

    @property
    def nsheets(self) -> int:
        return self.vertex_positions.shape[1]

    @property
    def nvertices(self) -> int:
        return self.vertex_positions.shape[2]

    @property
    def nsimplices(self) -> int:
        return self.connectivity.shape[0]

    @property
    def ntetrahedra(self) -> int:
        """Compatibility alias used by the legacy 3-D API."""
        return self.nsimplices

    @property
    def active_components(self) -> tuple[int, ...]:
        return tuple(range(self.dimension))

    def tree_flatten(self):
        children = (
            self.vertex_positions,
            self.connectivity,
            self.origins,
            self.inverse_matrices,
            self.valid,
            self.leaf_order,
            self.internal_bounds_min,
            self.internal_bounds_max,
        )
        auxiliary = (
            self.dimension,
            self.logical_shape,
            self.leaf_capacity,
            self.tree_depth,
        )
        return children, auxiliary

    @classmethod
    def tree_unflatten(cls, auxiliary, children):
        dimension, logical_shape, leaf_capacity, tree_depth = auxiliary
        return cls(
            *children,
            dimension,
            logical_shape,
            leaf_capacity,
            tree_depth,
        )


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class SimplicialField:
    """A reusable simplicial mesh with compact selected vertex values."""

    mesh: SimplicialMesh
    vertex_values: Any
    selection: FieldSelection

    def tree_flatten(self):
        return (self.mesh, self.vertex_values), self.selection

    @classmethod
    def tree_unflatten(cls, selection, children):
        mesh, vertex_values = children
        return cls(mesh, vertex_values, selection)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class InterpolatedSimplicialFields:
    """Beam- and sheet-resolved interpolation results."""

    values: Any
    inside: Any
    simplex_index: Any
    selection: FieldSelection

    @property
    def tet_index(self):
        """Compatibility alias for legacy tetrahedral callers."""
        return self.simplex_index

    def tree_flatten(self):
        return (self.values, self.inside, self.simplex_index), self.selection

    @classmethod
    def tree_unflatten(cls, selection, children):
        return cls(*children, selection)


def _segment_connectivity(logical_shape: tuple[int, int, int]):
    nsamples = logical_shape[2]
    starts = np.arange(max(nsamples - 1, 0), dtype=np.int32)
    return np.stack((starts, starts + 1), axis=-1)


def _triangular_connectivity(logical_shape: tuple[int, int, int]):
    nrays, _, nsamples = logical_shape
    ray, sample = np.meshgrid(
        np.arange(max(nrays - 1, 0), dtype=np.int32),
        np.arange(max(nsamples - 1, 0), dtype=np.int32),
        indexing="ij",
    )
    vertex_00 = ray * nsamples + sample
    vertex_10 = vertex_00 + nsamples
    vertex_01 = vertex_00 + 1
    vertex_11 = vertex_10 + 1
    triangles = np.stack(
        (
            np.stack((vertex_00, vertex_10, vertex_11), axis=-1),
            np.stack((vertex_00, vertex_01, vertex_11), axis=-1),
        ),
        axis=-2,
    )
    return triangles.reshape((-1, 3)).astype(np.int32, copy=False)


def _tetrahedral_connectivity(logical_shape: tuple[int, int, int]):
    nrays_one, nrays_two, nsamples = logical_shape
    first, second, sample = np.meshgrid(
        np.arange(max(nrays_one - 1, 0), dtype=np.int32),
        np.arange(max(nrays_two - 1, 0), dtype=np.int32),
        np.arange(max(nsamples - 1, 0), dtype=np.int32),
        indexing="ij",
    )
    vertex_000 = (first * nrays_two + second) * nsamples + sample
    vertex_100 = vertex_000 + nrays_two * nsamples
    vertex_010 = vertex_000 + nsamples
    vertex_001 = vertex_000 + 1
    vertex_110 = vertex_100 + nsamples
    vertex_101 = vertex_100 + 1
    vertex_011 = vertex_010 + 1
    vertex_111 = vertex_110 + 1
    tetrahedra = np.stack(
        (
            np.stack((vertex_000, vertex_100, vertex_110, vertex_111), axis=-1),
            np.stack((vertex_000, vertex_100, vertex_101, vertex_111), axis=-1),
            np.stack((vertex_000, vertex_010, vertex_110, vertex_111), axis=-1),
            np.stack((vertex_000, vertex_010, vertex_011, vertex_111), axis=-1),
            np.stack((vertex_000, vertex_001, vertex_101, vertex_111), axis=-1),
            np.stack((vertex_000, vertex_001, vertex_011, vertex_111), axis=-1),
        ),
        axis=-2,
    )
    return tetrahedra.reshape((-1, 4)).astype(np.int32, copy=False)


def _structured_connectivity(dimension: int, logical_shape: tuple[int, int, int]):
    if dimension == 1:
        return _segment_connectivity(logical_shape)
    if dimension == 2:
        return _triangular_connectivity(logical_shape)
    return _tetrahedral_connectivity(logical_shape)


def _spread_morton_bits(values):
    values = np.asarray(values, dtype=np.uint32) & np.uint32(0x000003FF)
    values = (values | (values << np.uint32(16))) & np.uint32(0x030000FF)
    values = (values | (values << np.uint32(8))) & np.uint32(0x0300F00F)
    values = (values | (values << np.uint32(4))) & np.uint32(0x030C30C3)
    values = (values | (values << np.uint32(2))) & np.uint32(0x09249249)
    return values


def _morton_codes(centroids):
    lower = np.min(centroids, axis=0)
    span = np.max(centroids, axis=0) - lower
    safe_span = np.where(span > 0.0, span, 1.0)
    normalized = np.clip((centroids - lower) / safe_span, 0.0, 1.0)
    quantized = np.floor(normalized * 1023.0).astype(np.uint32)
    codes = np.zeros((centroids.shape[0],), dtype=np.uint32)
    for component in range(centroids.shape[1]):
        codes |= _spread_morton_bits(quantized[:, component]) << np.uint32(component)
    return codes


def _build_global_bvh(valid, centroids, bounds_min, bounds_max):
    flat_valid = np.asarray(valid, dtype=bool).reshape(-1)
    valid_indices = np.flatnonzero(flat_valid)
    dimension = centroids.shape[-1]
    if valid_indices.size:
        flat_centroids = np.asarray(centroids).reshape((-1, dimension))[valid_indices]
        order = np.argsort(_morton_codes(flat_centroids), kind="stable")
        ordered_indices = valid_indices[order]
    else:
        ordered_indices = np.empty((0,), dtype=np.int64)

    nleaves = int(ordered_indices.size)
    leaf_capacity = 1 << max(0, (max(nleaves, 1) - 1).bit_length())
    tree_depth = leaf_capacity.bit_length() - 1
    if ordered_indices.size and ordered_indices[-1] > np.iinfo(np.int32).max:
        raise ValueError("simplicial field is too large for int32 BVH indices")
    leaf_order = np.full((leaf_capacity,), -1, dtype=np.int32)
    leaf_order[:nleaves] = ordered_indices.astype(np.int32, copy=False)

    leaf_minimum = np.full((leaf_capacity, dimension), np.inf, dtype=np.float64)
    leaf_maximum = np.full((leaf_capacity, dimension), -np.inf, dtype=np.float64)
    if nleaves:
        flat_minimum = np.asarray(bounds_min).reshape((-1, dimension))
        flat_maximum = np.asarray(bounds_max).reshape((-1, dimension))
        leaf_minimum[:nleaves] = flat_minimum[ordered_indices]
        leaf_maximum[:nleaves] = flat_maximum[ordered_indices]

    internal_minimum = np.empty((leaf_capacity - 1, dimension), dtype=np.float64)
    internal_maximum = np.empty((leaf_capacity - 1, dimension), dtype=np.float64)
    level_minimum = leaf_minimum
    level_maximum = leaf_maximum
    for depth in range(tree_depth - 1, -1, -1):
        level_minimum = np.minimum(level_minimum[0::2], level_minimum[1::2])
        level_maximum = np.maximum(level_maximum[0::2], level_maximum[1::2])
        start = (1 << depth) - 1
        stop = start + (1 << depth)
        internal_minimum[start:stop] = level_minimum
        internal_maximum[start:stop] = level_maximum
    return leaf_order, internal_minimum, internal_maximum, leaf_capacity, tree_depth


def _validate_dimension_and_shape(dimension: int, logical_shape):
    if isinstance(dimension, bool) or not isinstance(dimension, int):
        raise TypeError("dimension must be an integer")
    if dimension not in (1, 2, 3):
        raise ValueError("dimension must be one, two, or three")
    if dimension == 1 and logical_shape[:2] != (1, 1):
        raise ValueError("1-D sheet fields require singleton transverse ray axes")
    if dimension == 2 and logical_shape[1] != 1:
        raise ValueError("2-D sheet fields require nrays_axis2 to be one")


def simplicialise_sheet_fields(
    sheet_fields,
    *,
    dimension: int,
    fields=None,
    measure_tolerance: float = 1.0e-12,
) -> SimplicialField:
    """Build a segment, triangle, or tetrahedron field from traced sheets."""
    if measure_tolerance <= 0.0:
        raise ValueError("measure_tolerance must be positive")
    source = jnp.asarray(sheet_fields, dtype=jnp.float64)
    if source.ndim != 6:
        raise ValueError(
            "sheet_fields must have shape "
            "(nbeams, nsheets, nrays_axis1, nrays_axis2, nsamples, nfields)"
        )
    if source.shape[0] < 1 or source.shape[1] < 1:
        raise ValueError("sheet_fields must contain at least one beam and sheet")
    logical_shape = tuple(int(size) for size in source.shape[2:5])
    if any(size < 1 for size in logical_shape):
        raise ValueError("every sheet grid direction must contain at least one vertex")
    _validate_dimension_and_shape(dimension, logical_shape)

    selection = resolve_field_selection(fields, nsource=source.shape[-1])
    connectivity_host = _structured_connectivity(dimension, logical_shape)
    connectivity = jnp.asarray(connectivity_host, dtype=jnp.int32)
    nvertices = int(np.prod(logical_shape))
    positions = source[..., RAY_STATE_LAYOUT.position].reshape(
        (source.shape[0], source.shape[1], nvertices, 3)
    )
    selected_values = jnp.take(
        source, jnp.asarray(selection.source_indices, dtype=jnp.int32), axis=-1
    ).reshape(
        (
            source.shape[0],
            source.shape[1],
            nvertices,
            selection.n_attributes,
        )
    )

    simplex_positions = positions[..., connectivity, :]
    origins = simplex_positions[..., 0, :]
    active_positions = simplex_positions[..., :dimension]
    active_origins = active_positions[..., 0, :]
    edge_matrices = jnp.stack(
        tuple(
            active_positions[..., vertex, :] - active_origins
            for vertex in range(1, dimension + 1)
        ),
        axis=-1,
    )
    determinants = jnp.linalg.det(edge_matrices)
    pairwise_edges = (
        active_positions[..., :, None, :] - active_positions[..., None, :, :]
    )
    maximum_edge = jnp.max(jnp.linalg.norm(pairwise_edges, axis=-1), axis=(-2, -1))
    measure_scale = jnp.maximum(maximum_edge**dimension, jnp.finfo(source.dtype).tiny)
    finite = jnp.all(jnp.isfinite(simplex_positions), axis=(-2, -1))
    valid = finite & jnp.isfinite(determinants)
    valid &= jnp.abs(determinants) > measure_tolerance * measure_scale
    safe_matrices = jnp.where(
        valid[..., None, None],
        edge_matrices,
        jnp.eye(dimension, dtype=source.dtype),
    )
    inverse_matrices = jnp.linalg.inv(safe_matrices)
    bounds_min = jnp.min(active_positions, axis=-2)
    bounds_max = jnp.max(active_positions, axis=-2)
    centroids = jnp.mean(active_positions, axis=-2)
    bvh_data = _build_global_bvh(
        jax.device_get(valid),
        jax.device_get(centroids),
        jax.device_get(bounds_min),
        jax.device_get(bounds_max),
    )
    leaf_order, internal_min, internal_max, leaf_capacity, tree_depth = bvh_data
    mesh = SimplicialMesh(
        vertex_positions=positions,
        connectivity=connectivity,
        origins=origins,
        inverse_matrices=inverse_matrices,
        valid=valid,
        leaf_order=jnp.asarray(leaf_order, dtype=jnp.int32),
        internal_bounds_min=jnp.asarray(internal_min, dtype=jnp.float64),
        internal_bounds_max=jnp.asarray(internal_max, dtype=jnp.float64),
        dimension=dimension,
        logical_shape=logical_shape,
        leaf_capacity=leaf_capacity,
        tree_depth=tree_depth,
    )
    return SimplicialField(mesh, selected_values, selection)


def replace_simplicial_field_values(
    field: SimplicialField, sheet_fields
) -> SimplicialField:
    source = jnp.asarray(sheet_fields, dtype=jnp.float64)
    expected_prefix = (
        field.mesh.nbeams,
        field.mesh.nsheets,
        *field.mesh.logical_shape,
    )
    if source.ndim != 6 or source.shape[:5] != expected_prefix:
        raise ValueError(
            "updated sheet_fields must preserve the original beam, sheet, "
            "ray-grid, and sample dimensions"
        )
    if source.shape[-1] <= max(field.selection.source_indices):
        raise ValueError("updated sheet_fields do not contain every selected field")
    values = jnp.take(
        source,
        jnp.asarray(field.selection.source_indices, dtype=jnp.int32),
        axis=-1,
    ).reshape(
        (
            field.mesh.nbeams,
            field.mesh.nsheets,
            field.mesh.nvertices,
            field.selection.n_attributes,
        )
    )
    return SimplicialField(field.mesh, values, field.selection)


def _test_leaf(mesh, point, leaf_slot):
    flat_index = mesh.leaf_order[leaf_slot]
    safe_flat_index = jnp.maximum(flat_index, 0)
    simplex_index = safe_flat_index % mesh.nsimplices
    sheet_index = safe_flat_index // mesh.nsimplices
    active_point = point[: mesh.dimension]
    flat_origins = mesh.origins[..., : mesh.dimension].reshape((-1, mesh.dimension))
    flat_inverse = mesh.inverse_matrices.reshape((-1, mesh.dimension, mesh.dimension))
    flat_valid = mesh.valid.reshape((-1,))
    local = flat_inverse[safe_flat_index] @ (
        active_point - flat_origins[safe_flat_index]
    )
    weights = jnp.concatenate((jnp.asarray([1.0 - jnp.sum(local)]), local))
    return (
        sheet_index,
        simplex_index,
        weights,
        jnp.min(weights),
        (flat_index >= 0) & flat_valid[safe_flat_index],
    )


def _locate_point(mesh, point, barycentric_tolerance):
    nsheet_fields = mesh.nbeams * mesh.nsheets
    best_margin = jnp.full((nsheet_fields,), -jnp.inf, dtype=point.dtype)
    best_simplex = jnp.full((nsheet_fields,), -1, dtype=jnp.int32)
    best_weights = jnp.zeros((nsheet_fields, mesh.dimension + 1), dtype=point.dtype)
    if mesh.nsimplices == 0:
        return best_simplex, best_weights

    def update_leaf(leaf_slot, margins, simplices, weights_by_sheet):
        sheet, simplex, weights, margin, real_leaf = _test_leaf(mesh, point, leaf_slot)
        contained = real_leaf & jnp.all(weights >= -barycentric_tolerance)
        replace = contained & (margin > margins[sheet])
        margins = margins.at[sheet].set(jnp.where(replace, margin, margins[sheet]))
        simplices = simplices.at[sheet].set(
            jnp.where(replace, simplex, simplices[sheet])
        )
        weights_by_sheet = weights_by_sheet.at[sheet].set(
            jnp.where(replace, weights, weights_by_sheet[sheet])
        )
        return margins, simplices, weights_by_sheet

    if mesh.leaf_capacity == 1:
        _, best_simplex, best_weights = update_leaf(
            jnp.asarray(0, dtype=jnp.int32),
            best_margin,
            best_simplex,
            best_weights,
        )
        return best_simplex, best_weights

    internal_count = mesh.leaf_capacity - 1
    stack = jnp.full((mesh.tree_depth + 2,), -1, dtype=jnp.int32).at[0].set(0)

    def continue_traversal(carry):
        return carry[1] > 0

    def traverse(carry):
        node_stack, size, margins, simplices, weights_by_sheet = carry
        size -= 1
        node = node_stack[size]
        is_internal = node < internal_count

        def visit_internal(values):
            (
                current_stack,
                current_size,
                current_margins,
                current_simplices,
                current_weights,
            ) = values
            lower = mesh.internal_bounds_min[node]
            upper = mesh.internal_bounds_max[node]
            active_point = point[: mesh.dimension]
            coordinate_scale = jnp.maximum(
                1.0,
                jnp.max(jnp.abs(jnp.concatenate((active_point, lower, upper)))),
            )
            tolerance = 64.0 * jnp.finfo(point.dtype).eps * coordinate_scale
            overlaps = jnp.all(active_point >= lower - tolerance) & jnp.all(
                active_point <= upper + tolerance
            )

            def push_children(push_values):
                pushed_stack, pushed_size = push_values
                left = 2 * node + 1
                pushed_stack = pushed_stack.at[pushed_size].set(left + 1)
                pushed_stack = pushed_stack.at[pushed_size + 1].set(left)
                return pushed_stack, pushed_size + 2

            current_stack, current_size = jax.lax.cond(
                overlaps,
                push_children,
                lambda push_values: push_values,
                (current_stack, current_size),
            )
            return (
                current_stack,
                current_size,
                current_margins,
                current_simplices,
                current_weights,
            )

        def visit_leaf(values):
            (
                current_stack,
                current_size,
                current_margins,
                current_simplices,
                current_weights,
            ) = values
            leaf_slot = node - internal_count
            current_margins, current_simplices, current_weights = update_leaf(
                leaf_slot, current_margins, current_simplices, current_weights
            )
            return (
                current_stack,
                current_size,
                current_margins,
                current_simplices,
                current_weights,
            )

        return jax.lax.cond(
            is_internal,
            visit_internal,
            visit_leaf,
            (node_stack, size, margins, simplices, weights_by_sheet),
        )

    _, _, _, best_simplex, best_weights = jax.lax.while_loop(
        continue_traversal,
        traverse,
        (
            stack,
            jnp.asarray(1, dtype=jnp.int32),
            best_margin,
            best_simplex,
            best_weights,
        ),
    )
    return best_simplex, best_weights


def interpolate_simplicial_fields(
    field: SimplicialField,
    points,
    *,
    barycentric_tolerance: float = 1.0e-10,
) -> InterpolatedSimplicialFields:
    """Interpolate fields at Cartesian points, extruded through inactive axes."""
    query_points = jnp.asarray(points, dtype=jnp.float64)
    if query_points.ndim != 2 or query_points.shape[-1] != 3:
        raise ValueError("points must have shape (npoints, 3)")
    mesh = field.mesh
    npoints = query_points.shape[0]
    if mesh.nsimplices == 0:
        return InterpolatedSimplicialFields(
            values=jnp.zeros(
                (mesh.nbeams, mesh.nsheets, npoints, field.selection.n_attributes),
                dtype=query_points.dtype,
            ),
            inside=jnp.zeros((mesh.nbeams, mesh.nsheets, npoints), dtype=jnp.bool_),
            simplex_index=jnp.full(
                (mesh.nbeams, mesh.nsheets, npoints), -1, dtype=jnp.int32
            ),
            selection=field.selection,
        )

    flat_values = field.vertex_values.reshape(
        (
            mesh.nbeams * mesh.nsheets,
            mesh.nvertices,
            field.selection.n_attributes,
        )
    )
    sheet_indices = jnp.arange(mesh.nbeams * mesh.nsheets, dtype=jnp.int32)

    def interpolate_one(point):
        simplices, weights = _locate_point(mesh, point, barycentric_tolerance)
        inside = simplices >= 0
        safe_simplices = jnp.maximum(simplices, 0)
        vertex_indices = mesh.connectivity[safe_simplices]
        vertex_values = flat_values[sheet_indices[:, None], vertex_indices]
        values = jnp.sum(vertex_values * weights[..., None], axis=-2)
        return jnp.where(inside[..., None], values, 0.0), inside, simplices

    values, inside, simplices = jax.vmap(interpolate_one)(query_points)
    values = values.transpose((1, 0, 2)).reshape(
        (mesh.nbeams, mesh.nsheets, npoints, field.selection.n_attributes)
    )
    inside = inside.transpose((1, 0)).reshape((mesh.nbeams, mesh.nsheets, npoints))
    simplices = simplices.transpose((1, 0)).reshape(
        (mesh.nbeams, mesh.nsheets, npoints)
    )
    return InterpolatedSimplicialFields(values, inside, simplices, field.selection)


_JITTED_INTERPOLATE = jax.jit(interpolate_simplicial_fields)


def interpolate_simplicial_fields_batched(
    field: SimplicialField,
    points,
    *,
    point_batch_size: int = 8192,
    barycentric_tolerance: float = 1.0e-10,
) -> InterpolatedSimplicialFields:
    if isinstance(point_batch_size, bool) or not isinstance(point_batch_size, int):
        raise TypeError("point_batch_size must be an integer")
    if point_batch_size < 1:
        raise ValueError("point_batch_size must be positive")
    if barycentric_tolerance < 0.0:
        raise ValueError("barycentric_tolerance cannot be negative")
    query_points = jnp.asarray(points, dtype=jnp.float64)
    if query_points.ndim != 2 or query_points.shape[-1] != 3:
        raise ValueError("points must have shape (npoints, 3)")
    npoints = query_points.shape[0]
    if npoints == 0:
        return interpolate_simplicial_fields(
            field,
            query_points,
            barycentric_tolerance=barycentric_tolerance,
        )
    results = []
    for start in range(0, npoints, point_batch_size):
        stop = min(start + point_batch_size, npoints)
        batch = query_points[start:stop]
        padding = point_batch_size - batch.shape[0]
        if padding:
            batch = jnp.pad(batch, ((0, padding), (0, 0)))
        result = _JITTED_INTERPOLATE(
            field,
            batch,
            barycentric_tolerance=barycentric_tolerance,
        )
        results.append(
            InterpolatedSimplicialFields(
                result.values[..., : stop - start, :],
                result.inside[..., : stop - start],
                result.simplex_index[..., : stop - start],
                field.selection,
            )
        )
    return InterpolatedSimplicialFields(
        values=jnp.concatenate([result.values for result in results], axis=2),
        inside=jnp.concatenate([result.inside for result in results], axis=2),
        simplex_index=jnp.concatenate(
            [result.simplex_index for result in results], axis=2
        ),
        selection=field.selection,
    )


def interpolate_simplicial_fields_to_cells(
    field: SimplicialField,
    grid,
    *,
    point_batch_size: int = 8192,
    barycentric_tolerance: float = 1.0e-10,
) -> InterpolatedSimplicialFields:
    """Interpolate a field at every grid cell centre without reducing beams/sheets."""
    if field.mesh.dimension != grid.dimensions:
        raise ValueError("field and grid dimensions must match")
    native = jnp.stack(jnp.meshgrid(grid.xc, grid.yc, grid.zc, indexing="ij"), axis=-1)
    cartesian = convert_positions(native, grid.geom, Geometry.CARTESIAN)
    result = interpolate_simplicial_fields_batched(
        field,
        cartesian.reshape((-1, 3)),
        point_batch_size=point_batch_size,
        barycentric_tolerance=barycentric_tolerance,
    )
    prefix = (field.mesh.nbeams, field.mesh.nsheets, *grid.ncells)
    return InterpolatedSimplicialFields(
        values=result.values.reshape((*prefix, field.selection.n_attributes)),
        inside=result.inside.reshape(prefix),
        simplex_index=result.simplex_index.reshape(prefix),
        selection=result.selection,
    )


__all__ = [
    "InterpolatedSimplicialFields",
    "SimplicialField",
    "SimplicialMesh",
    "interpolate_simplicial_fields",
    "interpolate_simplicial_fields_batched",
    "interpolate_simplicial_fields_to_cells",
    "replace_simplicial_field_values",
    "simplicialise_sheet_fields",
]
