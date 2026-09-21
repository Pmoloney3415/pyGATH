"""Batched JAX deposition for one-, two-, and three-dimensional simplex meshes."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from functools import cache, partial
from numbers import Integral
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from .deposition import PowerDeposition
from .deposition_mesh import SimplicialDepositionMesh
from .simplicial import SimplicialField

ProgressCallback = Callable[[dict[str, Any]], None]
_TETRAHEDRON_FACE_INDICES = np.asarray(
    ((1, 2, 3), (0, 2, 3), (0, 1, 3), (0, 1, 2)), dtype=np.int32
)
_TETRAHEDRON_EDGE_INDICES = np.asarray(
    ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)), dtype=np.int32
)


@dataclass(frozen=True)
class _FlatBvh:
    bounds_min: np.ndarray
    bounds_max: np.ndarray
    left_child: np.ndarray
    right_child: np.ndarray
    leaf_targets: np.ndarray
    stack_capacity: int


def _cross_2d(first, second):
    return first[..., 0] * second[..., 1] - first[..., 1] * second[..., 0]


def _points_inside_triangle(points, triangle, relative_tolerance):
    following = jnp.roll(triangle, -1, axis=0)
    edges = following - triangle
    orientation = _cross_2d(edges[0], triangle[2] - triangle[0])
    orientation_sign = jnp.where(orientation >= 0.0, 1.0, -1.0)
    edge_scale = jnp.max(jnp.sum(edges * edges, axis=-1))
    signed = orientation_sign * _cross_2d(
        edges[None, :, :], points[:, None, :] - triangle[None, :, :]
    )
    return jnp.all(signed >= -relative_tolerance * edge_scale, axis=-1)


def _triangle_intersection_measure_centroid(source, target, relative_tolerance):
    source_inside = _points_inside_triangle(source, target, relative_tolerance)
    target_inside = _points_inside_triangle(target, source, relative_tolerance)

    source_vectors = jnp.roll(source, -1, axis=0) - source
    target_vectors = jnp.roll(target, -1, axis=0) - target
    first = source[:, None, :]
    second = target[None, :, :]
    first_vector = source_vectors[:, None, :]
    second_vector = target_vectors[None, :, :]
    separation = second - first
    denominator = _cross_2d(first_vector, second_vector)
    length_scale = jnp.maximum(
        jnp.max(jnp.linalg.norm(source_vectors, axis=-1)),
        jnp.max(jnp.linalg.norm(target_vectors, axis=-1)),
    )
    denominator_tolerance = relative_tolerance * length_scale**2
    nonparallel = jnp.abs(denominator) > denominator_tolerance
    safe_denominator = jnp.where(nonparallel, denominator, 1.0)
    source_fraction = _cross_2d(separation, second_vector) / safe_denominator
    target_fraction = _cross_2d(separation, first_vector) / safe_denominator
    edge_intersections = first + source_fraction[..., None] * first_vector
    edge_valid = nonparallel
    edge_valid &= source_fraction >= -relative_tolerance
    edge_valid &= source_fraction <= 1.0 + relative_tolerance
    edge_valid &= target_fraction >= -relative_tolerance
    edge_valid &= target_fraction <= 1.0 + relative_tolerance

    candidates = jnp.concatenate(
        (source, target, edge_intersections.reshape((9, 2))), axis=0
    )
    candidate_valid = jnp.concatenate(
        (source_inside, target_inside, edge_valid.reshape((9,))), axis=0
    )
    count = jnp.sum(candidate_valid, dtype=jnp.int32)
    safe_count = jnp.maximum(count, 1)
    centre = (
        jnp.sum(jnp.where(candidate_valid[:, None], candidates, 0.0), axis=0)
        / safe_count
    )
    angles = jnp.arctan2(candidates[:, 1] - centre[1], candidates[:, 0] - centre[0])
    sort_keys = jnp.where(candidate_valid, angles, jnp.inf)
    order = jnp.argsort(sort_keys)
    ordered = candidates[order]

    indices = jnp.arange(candidates.shape[0], dtype=jnp.int32)
    active = indices < count
    next_indices = jnp.where(indices + 1 < count, indices + 1, 0)
    following = ordered[next_indices]
    cross_terms = _cross_2d(ordered, following)
    cross_terms = jnp.where(active, cross_terms, 0.0)
    twice_signed_area = jnp.sum(cross_terms)
    absolute_area = 0.5 * jnp.abs(twice_signed_area)
    area_tolerance = relative_tolerance * length_scale**2
    valid_area = (count >= 3) & (absolute_area > area_tolerance)
    safe_twice_area = jnp.where(valid_area, twice_signed_area, 1.0)
    centroid = jnp.sum((ordered + following) * cross_terms[:, None], axis=0) / (
        3.0 * safe_twice_area
    )
    return jnp.where(valid_area, absolute_area, 0.0), centroid


def _pair_contribution(source, source_values, target, relative_tolerance):
    area, centroid = _triangle_intersection_measure_centroid(
        source, target, relative_tolerance
    )
    gradient = jnp.linalg.solve(
        source[1:] - source[0], source_values[1:] - source_values[0]
    )
    centroid_value = source_values[0] + jnp.dot(gradient, centroid - source[0])
    return area, area * centroid_value


def _tetrahedron_intersection_prepared(
    source,
    source_normals,
    source_offsets,
    source_scale,
    target,
    target_normals,
    target_offsets,
    target_scale,
    relative_tolerance,
):
    """Return intersection moments without solving plane triplet systems."""
    normals = jnp.concatenate((source_normals, target_normals), axis=0)
    offsets = jnp.concatenate((source_offsets, target_offsets), axis=0)
    coordinate_scale = jnp.maximum(
        jnp.max(jnp.abs(jnp.concatenate((source, target), axis=0))), 1.0
    )
    edge_scale = jnp.maximum(source_scale, target_scale)
    absolute_tolerance = jnp.maximum(
        relative_tolerance * edge_scale,
        128.0 * jnp.finfo(source.dtype).eps * coordinate_scale,
    )

    source_inside = jnp.all(
        source @ target_normals.T <= target_offsets[None, :] + absolute_tolerance,
        axis=1,
    )
    target_inside = jnp.all(
        target @ source_normals.T <= source_offsets[None, :] + absolute_tolerance,
        axis=1,
    )
    edges = jnp.asarray(_TETRAHEDRON_EDGE_INDICES)

    def edge_face_candidates(points, clipping_normals, clipping_offsets):
        starts = points[edges[:, 0]]
        stops = points[edges[:, 1]]
        directions = stops - starts
        start_distance = starts @ clipping_normals.T - clipping_offsets[None, :]
        stop_distance = stops @ clipping_normals.T - clipping_offsets[None, :]
        denominator = start_distance - stop_distance
        nonparallel = jnp.abs(denominator) > absolute_tolerance
        safe_denominator = jnp.where(nonparallel, denominator, 1.0)
        fraction = start_distance / safe_denominator
        candidates = starts[:, None, :] + fraction[..., None] * directions[:, None, :]
        strictly_on_edge = (fraction > relative_tolerance) & (
            fraction < 1.0 - relative_tolerance
        )
        feasible = jnp.all(
            candidates.reshape((-1, 3)) @ normals.T
            <= offsets[None, :] + absolute_tolerance,
            axis=1,
        ).reshape((6, 4))
        valid = nonparallel & strictly_on_edge & feasible
        return candidates.reshape((24, 3)), valid.reshape((24,))

    source_crossings, source_crossing_valid = edge_face_candidates(
        source, target_normals, target_offsets
    )
    target_crossings, target_crossing_valid = edge_face_candidates(
        target, source_normals, source_offsets
    )
    vertices = jnp.concatenate(
        (source, target, source_crossings, target_crossings), axis=0
    )
    vertex_valid = jnp.concatenate(
        (
            source_inside,
            target_inside,
            source_crossing_valid,
            target_crossing_valid,
        ),
        axis=0,
    )
    vertex_count = jnp.sum(vertex_valid, dtype=jnp.int32)
    safe_vertex_count = jnp.maximum(vertex_count, 1)
    interior = (
        jnp.sum(jnp.where(vertex_valid[:, None], vertices, 0.0), axis=0)
        / safe_vertex_count
    )

    normal_difference = jnp.linalg.norm(
        normals[:, None, :] - normals[None, :, :], axis=-1
    )
    offset_difference = jnp.abs(offsets[:, None] - offsets[None, :])
    earlier = jnp.tril(jnp.ones((8, 8), dtype=bool), k=-1)
    duplicate_plane = jnp.any(
        earlier
        & (normal_difference <= 16.0 * relative_tolerance)
        & (offset_difference <= absolute_tolerance),
        axis=1,
    )

    def face_contribution(normal, offset, plane_is_duplicate):
        distances = jnp.abs(vertices @ normal - offset)
        on_face = vertex_valid & (distances <= 4.0 * absolute_tolerance)
        count = jnp.sum(on_face, dtype=jnp.int32)
        safe_count = jnp.maximum(count, 1)
        centre = (
            jnp.sum(jnp.where(on_face[:, None], vertices, 0.0), axis=0) / safe_count
        )

        reference_axis = jnp.argmin(jnp.abs(normal))
        axis = jnp.eye(3, dtype=source.dtype)[reference_axis]
        first_basis = jnp.cross(normal, axis)
        first_basis /= jnp.maximum(jnp.linalg.norm(first_basis), 1.0e-30)
        second_basis = jnp.cross(normal, first_basis)
        relative = vertices - centre
        angles = jnp.arctan2(relative @ second_basis, relative @ first_basis)
        order = jnp.argsort(jnp.where(on_face, angles, jnp.inf))
        ordered = vertices[order]
        indices = jnp.arange(vertices.shape[0], dtype=jnp.int32)
        active = indices < count
        next_indices = jnp.where(indices + 1 < count, indices + 1, 0)
        following = ordered[next_indices]
        triple_products = jnp.einsum(
            "ij,ij->i",
            ordered - interior,
            jnp.cross(following - interior, centre - interior),
        )
        volumes = jnp.abs(triple_products) / 6.0
        active &= (count >= 3) & ~plane_is_duplicate
        volumes = jnp.where(active, volumes, 0.0)
        centroids = (interior[None, :] + centre[None, :] + ordered + following) / 4.0
        return jnp.sum(volumes), jnp.sum(volumes[:, None] * centroids, axis=0)

    face_volumes, face_moments = jax.vmap(face_contribution)(
        normals, offsets, duplicate_plane
    )
    volume = jnp.sum(face_volumes)
    moment = jnp.sum(face_moments, axis=0)
    volume_tolerance = relative_tolerance * edge_scale**3
    valid_volume = (vertex_count >= 4) & (volume > volume_tolerance)
    safe_volume = jnp.where(valid_volume, volume, 1.0)
    centroid = moment / safe_volume
    return jnp.where(valid_volume, volume, 0.0), centroid


def _flatten_bvh(root, leaf_size: int) -> _FlatBvh:
    bounds_min: list[np.ndarray] = []
    bounds_max: list[np.ndarray] = []
    left_child: list[int] = []
    right_child: list[int] = []
    leaf_targets: list[np.ndarray] = []

    def visit(node, depth: int) -> tuple[int, int]:
        index = len(bounds_min)
        bounds_min.append(np.asarray(node.lower, dtype=np.float64))
        bounds_max.append(np.asarray(node.upper, dtype=np.float64))
        left_child.append(-1)
        right_child.append(-1)
        targets = np.full((leaf_size,), -1, dtype=np.int32)
        leaf_targets.append(targets)
        maximum_depth = depth
        if node.indices is not None:
            targets[: node.indices.size] = node.indices
        else:
            left_index, left_depth = visit(node.left, depth + 1)
            right_index, right_depth = visit(node.right, depth + 1)
            left_child[index] = left_index
            right_child[index] = right_index
            maximum_depth = max(left_depth, right_depth)
        return index, maximum_depth

    _, maximum_depth = visit(root, 0)
    return _FlatBvh(
        bounds_min=np.asarray(bounds_min),
        bounds_max=np.asarray(bounds_max),
        left_child=np.asarray(left_child, dtype=np.int32),
        right_child=np.asarray(right_child, dtype=np.int32),
        leaf_targets=np.asarray(leaf_targets, dtype=np.int32),
        stack_capacity=maximum_depth + 2,
    )


def _query_bvh_count_one(
    query_lower,
    query_upper,
    node_lower,
    node_upper,
    left_child,
    right_child,
    leaf_targets,
    target_lower,
    target_upper,
    tolerance,
    stack_capacity: int,
):
    stack = jnp.full((stack_capacity,), -1, dtype=jnp.int32).at[0].set(0)

    def condition(state):
        return state[1] > 0

    def body(state):
        current_stack, stack_size, count = state
        stack_size -= 1
        node_index = current_stack[stack_size]
        overlaps_node = jnp.all(node_upper[node_index] >= query_lower - tolerance)
        overlaps_node &= jnp.all(node_lower[node_index] <= query_upper + tolerance)

        def visit_node(inner_state):
            left = left_child[node_index]

            def visit_leaf(leaf_state):
                leaf_stack, leaf_size_now, leaf_count = leaf_state
                targets = leaf_targets[node_index]
                safe_targets = jnp.maximum(targets, 0)
                matches = targets >= 0
                matches &= jnp.all(
                    target_upper[safe_targets] >= query_lower - tolerance, axis=1
                )
                matches &= jnp.all(
                    target_lower[safe_targets] <= query_upper + tolerance, axis=1
                )
                return (
                    leaf_stack,
                    leaf_size_now,
                    leaf_count + jnp.sum(matches, dtype=jnp.int32),
                )

            def visit_branch(branch_state):
                branch_stack, branch_size, branch_count = branch_state
                branch_stack = branch_stack.at[branch_size].set(left)
                branch_stack = branch_stack.at[branch_size + 1].set(
                    right_child[node_index]
                )
                return branch_stack, branch_size + 2, branch_count

            return jax.lax.cond(left < 0, visit_leaf, visit_branch, inner_state)

        return jax.lax.cond(
            overlaps_node,
            visit_node,
            lambda inner_state: inner_state,
            (current_stack, stack_size, count),
        )

    initial_size = jnp.asarray(1, dtype=jnp.int32)
    initial_count = jnp.asarray(0, dtype=jnp.int32)
    return jax.lax.while_loop(condition, body, (stack, initial_size, initial_count))[2]


@partial(jax.jit, static_argnums=(11,))
def _query_bvh_counts(
    query_lower,
    query_upper,
    query_valid,
    node_lower,
    node_upper,
    left_child,
    right_child,
    leaf_targets,
    target_lower,
    target_upper,
    tolerance,
    stack_capacity,
):
    counts = jax.vmap(
        _query_bvh_count_one,
        in_axes=(0, 0, None, None, None, None, None, None, None, None, None),
    )(
        query_lower,
        query_upper,
        node_lower,
        node_upper,
        left_child,
        right_child,
        leaf_targets,
        target_lower,
        target_upper,
        tolerance,
        stack_capacity,
    )
    return jnp.where(query_valid, counts, 0)


def _query_bvh_fill_one(
    query_lower,
    query_upper,
    node_lower,
    node_upper,
    left_child,
    right_child,
    leaf_targets,
    target_lower,
    target_upper,
    tolerance,
    stack_capacity: int,
    candidate_capacity: int,
):
    stack = jnp.full((stack_capacity,), -1, dtype=jnp.int32).at[0].set(0)
    candidates = jnp.full((candidate_capacity,), -1, dtype=jnp.int32)

    def condition(state):
        return state[1] > 0

    def body(state):
        current_stack, stack_size, count, current_candidates = state
        stack_size -= 1
        node_index = current_stack[stack_size]
        overlaps_node = jnp.all(node_upper[node_index] >= query_lower - tolerance)
        overlaps_node &= jnp.all(node_lower[node_index] <= query_upper + tolerance)

        def visit_node(inner_state):
            left = left_child[node_index]

            def visit_leaf(leaf_state):
                leaf_stack, leaf_size_now, leaf_count, leaf_candidates_now = leaf_state
                targets = leaf_targets[node_index]
                safe_targets = jnp.maximum(targets, 0)
                matches = targets >= 0
                matches &= jnp.all(
                    target_upper[safe_targets] >= query_lower - tolerance, axis=1
                )
                matches &= jnp.all(
                    target_lower[safe_targets] <= query_upper + tolerance, axis=1
                )
                ranks = jnp.cumsum(matches, dtype=jnp.int32) - 1
                slots = leaf_count + ranks
                scatter_slots = jnp.where(matches, slots, candidate_capacity)
                leaf_candidates_now = leaf_candidates_now.at[scatter_slots].set(
                    safe_targets, mode="drop"
                )
                return (
                    leaf_stack,
                    leaf_size_now,
                    leaf_count + jnp.sum(matches, dtype=jnp.int32),
                    leaf_candidates_now,
                )

            def visit_branch(branch_state):
                branch_stack, branch_size, branch_count, branch_candidates = (
                    branch_state
                )
                branch_stack = branch_stack.at[branch_size].set(left)
                branch_stack = branch_stack.at[branch_size + 1].set(
                    right_child[node_index]
                )
                return (
                    branch_stack,
                    branch_size + 2,
                    branch_count,
                    branch_candidates,
                )

            return jax.lax.cond(left < 0, visit_leaf, visit_branch, inner_state)

        return jax.lax.cond(
            overlaps_node,
            visit_node,
            lambda inner_state: inner_state,
            (current_stack, stack_size, count, current_candidates),
        )

    initial_size = jnp.asarray(1, dtype=jnp.int32)
    initial_count = jnp.asarray(0, dtype=jnp.int32)
    final = jax.lax.while_loop(
        condition, body, (stack, initial_size, initial_count, candidates)
    )
    return final[3], final[2]


def _make_bvh_fill_batch(
    stack_capacity: int,
    candidate_capacity: int,
    exact_batch_size: int,
):
    @jax.jit
    def fill_batch(
        source_points,
        source_values,
        source_output_indices,
        source_valid,
        node_lower,
        node_upper,
        left_child,
        right_child,
        leaf_targets,
        target_lower,
        target_upper,
        tolerance,
    ):
        query_lower = jnp.min(source_points, axis=1)
        query_upper = jnp.max(source_points, axis=1)
        candidates, counts = jax.vmap(
            _query_bvh_fill_one,
            in_axes=(
                0,
                0,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            ),
        )(
            query_lower,
            query_upper,
            node_lower,
            node_upper,
            left_child,
            right_child,
            leaf_targets,
            target_lower,
            target_upper,
            tolerance,
            stack_capacity,
            candidate_capacity,
        )
        counts = jnp.where(source_valid, counts, 0)
        candidate_slots = jnp.arange(candidate_capacity)[None, :]
        pair_valid = source_valid[:, None] & (candidate_slots < counts[:, None])
        flat_valid = pair_valid.reshape(-1)
        packed_indices = jnp.nonzero(flat_valid, size=exact_batch_size, fill_value=0)[0]
        packed_count = jnp.sum(counts, dtype=jnp.int32)
        packed_valid = jnp.arange(exact_batch_size) < packed_count
        repeated_points = jnp.repeat(source_points, candidate_capacity, axis=0)
        repeated_values = jnp.repeat(source_values, candidate_capacity, axis=0)
        repeated_outputs = jnp.repeat(source_output_indices, candidate_capacity, axis=0)
        flat_candidates = candidates.reshape(-1)
        overflow = jnp.any(counts > candidate_capacity)
        overflow |= packed_count > exact_batch_size
        return (
            repeated_points[packed_indices],
            repeated_values[packed_indices],
            repeated_outputs[packed_indices],
            flat_candidates[packed_indices],
            packed_valid,
            packed_count,
            overflow,
        )

    return fill_batch


@cache
def _make_indexed_bvh_fill_batch(
    stack_capacity: int,
    candidate_capacity: int,
    pair_batch_size: int,
):
    """Build a fixed-shape BVH fill kernel returning global source indices."""

    @jax.jit
    def fill_batch(
        source_indices,
        source_valid,
        all_source_points,
        node_lower,
        node_upper,
        left_child,
        right_child,
        leaf_targets,
        target_lower,
        target_upper,
        tolerance,
    ):
        safe_sources = jnp.maximum(source_indices, 0)
        source_points = all_source_points[safe_sources]
        query_lower = jnp.min(source_points, axis=1)
        query_upper = jnp.max(source_points, axis=1)
        candidates, counts = jax.vmap(
            _query_bvh_fill_one,
            in_axes=(
                0,
                0,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            ),
        )(
            query_lower,
            query_upper,
            node_lower,
            node_upper,
            left_child,
            right_child,
            leaf_targets,
            target_lower,
            target_upper,
            tolerance,
            stack_capacity,
            candidate_capacity,
        )
        counts = jnp.where(source_valid, counts, 0)
        slots = jnp.arange(candidate_capacity)[None, :]
        flat_valid = (source_valid[:, None] & (slots < counts[:, None])).reshape(-1)
        packed = jnp.nonzero(flat_valid, size=pair_batch_size, fill_value=0)[0]
        packed_count = jnp.sum(counts, dtype=jnp.int32)
        packed_valid = jnp.arange(pair_batch_size) < packed_count
        repeated_sources = jnp.repeat(source_indices, candidate_capacity)
        flat_candidates = candidates.reshape(-1)
        overflow = jnp.any(counts > candidate_capacity)
        overflow |= packed_count > pair_batch_size
        return (
            repeated_sources[packed],
            flat_candidates[packed],
            packed_valid,
            packed_count,
            overflow,
        )

    return fill_batch


@cache
def _make_tetrahedron_classification_batch(pair_batch_size: int):
    """Reject separated pairs and integrate tetrahedral containment cases."""

    @partial(jax.jit, donate_argnums=(0, 1))
    def classify_batch(
        accumulated_power,
        statistics,
        pair_source_indices,
        pair_target_indices,
        pair_valid,
        source_points,
        source_normals,
        source_offsets,
        source_volumes,
        source_centres,
        source_gradients,
        source_intercepts,
        source_output_indices,
        source_scales,
        target_points,
        target_normals,
        target_offsets,
        target_volumes,
        target_centres,
        target_scales,
        relative_tolerance,
        inactive_measure,
        overflow,
    ):
        safe_sources = jnp.maximum(pair_source_indices, 0)
        safe_targets = jnp.maximum(pair_target_indices, 0)
        selected_source = source_points[safe_sources]
        selected_target = target_points[safe_targets]
        selected_source_normals = source_normals[safe_sources]
        selected_target_normals = target_normals[safe_targets]
        selected_source_offsets = source_offsets[safe_sources]
        selected_target_offsets = target_offsets[safe_targets]
        edge_scale = jnp.maximum(
            source_scales[safe_sources], target_scales[safe_targets]
        )
        coordinate_scale = jnp.maximum(
            jnp.maximum(
                jnp.max(jnp.abs(selected_source), axis=(1, 2)),
                jnp.max(jnp.abs(selected_target), axis=(1, 2)),
            ),
            1.0,
        )
        absolute_tolerance = jnp.maximum(
            relative_tolerance * edge_scale,
            128.0 * jnp.finfo(selected_source.dtype).eps * coordinate_scale,
        )

        source_distances = (
            jnp.einsum("bvc,bfc->bvf", selected_source, selected_target_normals)
            - selected_target_offsets[:, None, :]
        )
        target_distances = (
            jnp.einsum("bvc,bfc->bvf", selected_target, selected_source_normals)
            - selected_source_offsets[:, None, :]
        )
        source_outside_face = jnp.any(
            jnp.all(source_distances > absolute_tolerance[:, None, None], axis=1),
            axis=1,
        )
        target_outside_face = jnp.any(
            jnp.all(target_distances > absolute_tolerance[:, None, None], axis=1),
            axis=1,
        )
        separated = pair_valid & (source_outside_face | target_outside_face)
        active = pair_valid & ~separated
        source_contained = active & jnp.all(
            source_distances <= absolute_tolerance[:, None, None], axis=(1, 2)
        )
        target_contained = (
            active
            & ~source_contained
            & jnp.all(
                target_distances <= absolute_tolerance[:, None, None], axis=(1, 2)
            )
        )
        general = active & ~source_contained & ~target_contained

        containment_measure = jnp.where(
            source_contained,
            source_volumes[safe_sources],
            target_volumes[safe_targets],
        )
        containment_centroid = jnp.where(
            source_contained[:, None],
            source_centres[safe_sources],
            target_centres[safe_targets],
        )
        containment_value = source_intercepts[safe_sources] + jnp.einsum(
            "bi,bi->b",
            source_gradients[safe_sources],
            containment_centroid,
        )
        containment_valid = source_contained | target_contained
        containment_power = jnp.where(
            containment_valid,
            inactive_measure * containment_measure * containment_value,
            0.0,
        )
        flat_output = (
            source_output_indices[safe_sources] * target_points.shape[0] + safe_targets
        )
        accumulated_power = accumulated_power.at[flat_output].add(containment_power)

        packed = jnp.nonzero(general, size=pair_batch_size, fill_value=0)[0]
        general_count = jnp.sum(general, dtype=jnp.int64)
        packed_valid = jnp.arange(pair_batch_size) < general_count
        increments = jnp.asarray(
            (
                jnp.sum(pair_valid, dtype=jnp.int64),
                jnp.sum(separated, dtype=jnp.int64),
                jnp.sum(source_contained, dtype=jnp.int64),
                jnp.sum(target_contained, dtype=jnp.int64),
                0,
                0,
                0,
                overflow.astype(jnp.int64),
            )
        )
        statistics += increments
        return (
            accumulated_power,
            statistics,
            pair_source_indices[packed],
            pair_target_indices[packed],
            packed_valid,
            general_count,
        )

    return classify_batch


@cache
def _make_tetrahedron_sat_batch(batch_size: int):
    """Compact pairs surviving all tetrahedron separating-axis tests."""

    @partial(jax.jit, donate_argnums=(0,))
    def separate_batch(
        statistics,
        pair_source_indices,
        pair_target_indices,
        pair_valid,
        source_points,
        source_scales,
        target_points,
        target_scales,
        relative_tolerance,
    ):
        safe_sources = jnp.maximum(pair_source_indices, 0)
        safe_targets = jnp.maximum(pair_target_indices, 0)
        selected_source = source_points[safe_sources]
        selected_target = target_points[safe_targets]
        edge_indices = jnp.asarray(_TETRAHEDRON_EDGE_INDICES)
        source_edges = (
            selected_source[:, edge_indices[:, 1]]
            - selected_source[:, edge_indices[:, 0]]
        )
        target_edges = (
            selected_target[:, edge_indices[:, 1]]
            - selected_target[:, edge_indices[:, 0]]
        )
        axes = jnp.cross(source_edges[:, :, None, :], target_edges[:, None, :, :])
        axis_lengths = jnp.linalg.norm(axes, axis=-1)
        edge_scale = jnp.maximum(
            source_scales[safe_sources], target_scales[safe_targets]
        )
        axis_valid = axis_lengths > relative_tolerance * edge_scale[:, None, None] ** 2
        safe_lengths = jnp.where(axis_valid, axis_lengths, 1.0)
        axes /= safe_lengths[..., None]
        source_projection = jnp.einsum("bvj,befj->bvef", selected_source, axes)
        target_projection = jnp.einsum("bvj,befj->bvef", selected_target, axes)
        coordinate_scale = jnp.maximum(
            jnp.maximum(
                jnp.max(jnp.abs(selected_source), axis=(1, 2)),
                jnp.max(jnp.abs(selected_target), axis=(1, 2)),
            ),
            1.0,
        )
        absolute_tolerance = jnp.maximum(
            relative_tolerance * edge_scale,
            128.0 * jnp.finfo(selected_source.dtype).eps * coordinate_scale,
        )
        source_below = jnp.max(source_projection, axis=1) < (
            jnp.min(target_projection, axis=1) - absolute_tolerance[:, None, None]
        )
        target_below = jnp.max(target_projection, axis=1) < (
            jnp.min(source_projection, axis=1) - absolute_tolerance[:, None, None]
        )
        separated = pair_valid & jnp.any(
            axis_valid & (source_below | target_below), axis=(1, 2)
        )
        survivors = pair_valid & ~separated
        survivor_count = jnp.sum(survivors, dtype=jnp.int64)
        packed = jnp.nonzero(survivors, size=batch_size, fill_value=0)[0]
        packed_valid = jnp.arange(batch_size) < survivor_count
        statistics = statistics.at[4].add(jnp.sum(separated, dtype=jnp.int64))
        statistics = statistics.at[5].add(survivor_count)
        return (
            statistics,
            pair_source_indices[packed],
            pair_target_indices[packed],
            packed_valid,
            survivor_count,
        )

    return separate_batch


@cache
def _make_prepared_tetrahedron_deposit_batch():
    """Integrate compacted partial intersections into a device accumulator."""

    @partial(jax.jit, donate_argnums=(0, 1))
    def deposit_batch(
        accumulated_power,
        statistics,
        pair_source_indices,
        pair_target_indices,
        pair_valid,
        source_points,
        source_normals,
        source_offsets,
        source_gradients,
        source_intercepts,
        source_output_indices,
        source_scales,
        target_points,
        target_normals,
        target_offsets,
        target_scales,
        relative_tolerance,
        inactive_measure,
    ):
        safe_sources = jnp.maximum(pair_source_indices, 0)
        safe_targets = jnp.maximum(pair_target_indices, 0)

        def contribution(source_index, target_index):
            volume, centroid = _tetrahedron_intersection_prepared(
                source_points[source_index],
                source_normals[source_index],
                source_offsets[source_index],
                source_scales[source_index],
                target_points[target_index],
                target_normals[target_index],
                target_offsets[target_index],
                target_scales[target_index],
                relative_tolerance,
            )
            value = source_intercepts[source_index] + jnp.dot(
                source_gradients[source_index], centroid
            )
            return volume, volume * value

        volumes, contributions = jax.vmap(contribution)(safe_sources, safe_targets)
        contributions = jnp.where(pair_valid, inactive_measure * contributions, 0.0)
        flat_output = (
            source_output_indices[safe_sources] * target_points.shape[0] + safe_targets
        )
        accumulated_power = accumulated_power.at[flat_output].add(contributions)
        statistics = statistics.at[6].add(
            jnp.sum(pair_valid & (volumes > 0.0), dtype=jnp.int64)
        )
        return accumulated_power, statistics

    return deposit_batch


def _make_exact_deposit_batch(output_size: int):
    @jax.jit
    def deposit_batch(
        pair_source_points,
        pair_source_values,
        pair_output_indices,
        pair_target_indices,
        pair_valid,
        target_points,
        tolerance,
        inactive_measure,
    ):
        safe_targets = jnp.maximum(pair_target_indices, 0)
        selected_targets = target_points[safe_targets]
        measures, contributions = jax.vmap(_pair_contribution, in_axes=(0, 0, 0, None))(
            pair_source_points,
            pair_source_values,
            selected_targets,
            tolerance,
        )
        contributions = jnp.where(pair_valid, inactive_measure * contributions, 0.0)
        flat_output_indices = (
            pair_output_indices * target_points.shape[0] + safe_targets
        )
        batch_power = (
            jnp.zeros((output_size,), dtype=contributions.dtype)
            .at[flat_output_indices]
            .add(contributions)
        )
        nonempty = jnp.sum(pair_valid & (measures > 0.0), dtype=jnp.int64)
        return batch_power, nonempty

    return deposit_batch


def _validate_selection(index, size: int, name: str) -> tuple[int, ...]:
    if index is None:
        return tuple(range(size))
    if isinstance(index, bool) or not isinstance(index, Integral):
        raise TypeError(f"{name} must be an integer or None")
    selected = int(index)
    if not 0 <= selected < size:
        raise IndexError(f"{name} {selected} is outside {size}")
    return (selected,)


def _emit(
    callback: ProgressCallback | None, stage: str, status: str, message: str, **data
):
    if callback is not None:
        callback({"stage": stage, "status": status, "message": message, **data})


def _collect_sources(
    field,
    dimension: int,
    field_index: int,
    beam_indices: tuple[int, ...],
    sheet_indices: tuple[int, ...],
):
    positions = np.asarray(field.mesh.vertex_positions, dtype=np.float64)
    connectivity = np.asarray(field.mesh.connectivity, dtype=np.int32)
    valid = np.asarray(field.mesh.valid, dtype=bool)
    values = np.asarray(field.vertex_values, dtype=np.float64)
    source_points_parts = []
    source_values_parts = []
    source_output_parts = []
    source_beam_parts = []
    source_sheet_parts = []
    for beam_output, current_beam in enumerate(beam_indices):
        for sheet_output, current_sheet in enumerate(sheet_indices):
            simplex_indices = np.flatnonzero(valid[current_beam, current_sheet])
            vertex_indices = connectivity[simplex_indices]
            selected_points = positions[
                current_beam, current_sheet, vertex_indices, :dimension
            ]
            selected_values = values[
                current_beam, current_sheet, vertex_indices, field_index
            ]
            finite = np.all(np.isfinite(selected_points), axis=(1, 2))
            finite &= np.all(np.isfinite(selected_values), axis=1)
            selected_points = selected_points[finite]
            selected_values = selected_values[finite]
            count = selected_points.shape[0]
            source_points_parts.append(selected_points)
            source_values_parts.append(selected_values)
            source_output_parts.append(
                np.full(
                    count,
                    beam_output * len(sheet_indices) + sheet_output,
                    dtype=np.int32,
                )
            )
            source_beam_parts.append(np.full(count, beam_output, dtype=np.int32))
            source_sheet_parts.append(np.full(count, sheet_output, dtype=np.int32))

    source_points = np.concatenate(source_points_parts, axis=0)
    source_values = np.concatenate(source_values_parts, axis=0)
    source_output_indices = np.concatenate(source_output_parts)
    source_beam_indices = np.concatenate(source_beam_parts)
    source_sheet_indices = np.concatenate(source_sheet_parts)
    return (
        source_points,
        source_values,
        source_output_indices,
        source_beam_indices,
        source_sheet_indices,
    )


def _source_simplex_measures(source_points: np.ndarray) -> np.ndarray:
    dimension = source_points.shape[-1]
    edge_matrices = source_points[:, 1:] - source_points[:, :1]
    if dimension == 1:
        return np.abs(edge_matrices[:, 0, 0])
    if dimension == 2:
        return 0.5 * np.abs(
            edge_matrices[:, 0, 0] * edge_matrices[:, 1, 1]
            - edge_matrices[:, 0, 1] * edge_matrices[:, 1, 0]
        )
    return np.abs(np.linalg.det(edge_matrices)) / 6.0


def _deposit_two_dimensional_jax_bvh(
    field: SimplicialField,
    mesh: SimplicialDepositionMesh,
    *,
    field_name: str,
    beam_index: int | None,
    sheet_index: int | None,
    resolve_beam_sheets: bool,
    intersection_tolerance: float,
    progress_interval_s: float | None,
    progress_callback: ProgressCallback | None,
    pair_batch_size: int,
    source_batch_size: int,
    bvh_leaf_size: int,
) -> PowerDeposition | Any:
    from .mesh_deposition import ResolvedPowerDeposition, _build_aabb_tree

    if field.mesh.dimension != 2 or mesh.dimension != 2:
        raise ValueError("triangle deposition requires matching 2-D meshes")
    for value, name in (
        (pair_batch_size, "pair_batch_size"),
        (source_batch_size, "source_batch_size"),
        (bvh_leaf_size, "bvh_leaf_size"),
    ):
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError(f"{name} must be an integer")
        if int(value) < 1:
            raise ValueError(f"{name} must be positive")
    pair_batch_size = int(pair_batch_size)
    source_batch_size = int(source_batch_size)
    bvh_leaf_size = int(bvh_leaf_size)

    try:
        field_index = getattr(field.selection, field_name)
    except AttributeError as error:
        raise ValueError(f"simplicial field does not include {field_name!r}") from error
    if not isinstance(field_index, Integral):
        raise TypeError("deposited simplicial field must be scalar")
    beam_indices = _validate_selection(beam_index, field.mesh.nbeams, "beam_index")
    sheet_indices = _validate_selection(sheet_index, field.mesh.nsheets, "sheet_index")
    dimension = 2
    (
        source_points,
        source_values,
        source_output_indices,
        source_beam_indices,
        source_sheet_indices,
    ) = _collect_sources(
        field,
        dimension,
        int(field_index),
        beam_indices,
        sheet_indices,
    )
    source_powers = (
        mesh.inactive_measure
        * _source_simplex_measures(source_points)
        * np.mean(source_values, axis=1)
    )
    source_power = np.zeros((len(beam_indices), len(sheet_indices)), dtype=np.float64)
    np.add.at(
        source_power,
        (source_beam_indices, source_sheet_indices),
        source_powers,
    )

    build_started = time.perf_counter()
    tree = _build_aabb_tree(
        mesh.cell_bounds_min, mesh.cell_bounds_max, leaf_size=bvh_leaf_size
    )
    flat_tree = _flatten_bvh(tree, bvh_leaf_size)
    build_elapsed = time.perf_counter() - build_started
    _emit(
        progress_callback,
        "jax bvh construction",
        "complete",
        f"flattened {flat_tree.bounds_min.shape[0]} BVH nodes in {build_elapsed:.2f} s",
        bvh_nodes=flat_tree.bounds_min.shape[0],
        elapsed_s=build_elapsed,
    )

    target_points = np.asarray(
        mesh.vertex_positions[mesh.simplex_connectivity, :dimension],
        dtype=np.float64,
    )
    mesh_scale = max(
        float(np.max(mesh.cell_bounds_max) - np.min(mesh.cell_bounds_min)),
        np.finfo(np.float64).tiny,
    )
    search_tolerance = intersection_tolerance * mesh_scale
    tree_device = tuple(
        jax.device_put(jnp.asarray(item))
        for item in (
            flat_tree.bounds_min,
            flat_tree.bounds_max,
            flat_tree.left_child,
            flat_tree.right_child,
            flat_tree.leaf_targets,
            mesh.cell_bounds_min[:, :dimension],
            mesh.cell_bounds_max[:, :dimension],
        )
    )

    total_sources = source_points.shape[0]
    candidate_counts = np.zeros((total_sources,), dtype=np.int32)
    count_started = time.perf_counter()
    last_progress = count_started
    _emit(
        progress_callback,
        "jax bvh count",
        "started",
        f"counting candidates for {total_sources} source simplices",
        total_source_simplices=total_sources,
    )
    for start in range(0, total_sources, source_batch_size):
        stop = min(start + source_batch_size, total_sources)
        count = stop - start
        batch_points = np.zeros(
            (source_batch_size, dimension + 1, dimension), dtype=np.float64
        )
        batch_valid = np.zeros((source_batch_size,), dtype=bool)
        batch_points[:count] = source_points[start:stop]
        batch_valid[:count] = True
        counts = _query_bvh_counts(
            jnp.asarray(np.min(batch_points, axis=1)),
            jnp.asarray(np.max(batch_points, axis=1)),
            jnp.asarray(batch_valid),
            *tree_device,
            jnp.asarray(search_tolerance),
            flat_tree.stack_capacity,
        )
        counts.block_until_ready()
        candidate_counts[start:stop] = np.asarray(counts[:count])
        now = time.perf_counter()
        if (
            progress_interval_s is not None
            and now - last_progress >= progress_interval_s
        ) or stop == total_sources:
            elapsed = now - count_started
            rate = stop / elapsed if elapsed > 0.0 else 0.0
            eta = (total_sources - stop) / rate if rate > 0.0 else None
            _emit(
                progress_callback,
                "jax bvh count",
                "running" if stop < total_sources else "complete",
                f"candidate count {stop}/{total_sources}; "
                f"{int(candidate_counts[:stop].sum())} pairs",
                processed_source_simplices=stop,
                total_source_simplices=total_sources,
                candidate_pairs=int(candidate_counts[:stop].sum()),
                estimated_remaining_s=eta,
            )
            last_progress = now

    capacities = np.zeros_like(candidate_counts)
    positive = candidate_counts > 0
    capacities[positive] = 1 << np.ceil(np.log2(candidate_counts[positive])).astype(
        np.int32
    )
    unique_capacities, capacity_populations = np.unique(
        capacities[positive], return_counts=True
    )
    capacity_histogram = {
        int(capacity): int(population)
        for capacity, population in zip(unique_capacities, capacity_populations)
    }
    total_pairs = int(candidate_counts.sum())
    _emit(
        progress_callback,
        "jax bvh count",
        "complete",
        f"counted {total_pairs} candidate pairs in "
        f"{time.perf_counter() - count_started:.2f} s; "
        f"capacities {capacity_histogram}",
        candidate_pairs=total_pairs,
        capacity_histogram=capacity_histogram,
        elapsed_s=time.perf_counter() - count_started,
    )

    output_size = len(beam_indices) * len(sheet_indices) * mesh.ncells
    accumulated_power = np.zeros((output_size,), dtype=np.float64)
    target_points_device = jax.device_put(jnp.asarray(target_points))
    maximum_capacity = int(np.max(unique_capacities)) if positive.any() else 1
    exact_batch_size = max(maximum_capacity, min(pair_batch_size, max(total_pairs, 1)))
    exact_batch_kernel = _make_exact_deposit_batch(output_size)
    dense_fill_limit = max(4 * exact_batch_size, maximum_capacity)
    fill_source_batch_size = min(
        source_batch_size,
        max(1, dense_fill_limit // maximum_capacity),
    )
    fill_batch_kernel = _make_bvh_fill_batch(
        flat_tree.stack_capacity,
        maximum_capacity,
        exact_batch_size,
    )
    fill_started = time.perf_counter()
    processed_sources = int(np.sum(~positive))
    processed_pairs = 0
    nonempty_overlaps = 0
    last_progress = fill_started
    positive_indices = np.flatnonzero(positive)
    positive_prefix = np.concatenate(
        ([0], np.cumsum(candidate_counts[positive_indices], dtype=np.int64))
    )
    group_start = 0
    while group_start < positive_indices.size:
        pair_limit = positive_prefix[group_start] + exact_batch_size
        pair_stop = int(np.searchsorted(positive_prefix, pair_limit, side="right") - 1)
        group_stop = min(
            positive_indices.size,
            group_start + fill_source_batch_size,
            max(group_start + 1, pair_stop),
        )
        selected = positive_indices[group_start:group_stop]
        count = selected.size
        batch_points = np.zeros(
            (fill_source_batch_size, dimension + 1, dimension), dtype=np.float64
        )
        batch_values = np.zeros(
            (fill_source_batch_size, dimension + 1), dtype=np.float64
        )
        batch_outputs = np.zeros((fill_source_batch_size,), dtype=np.int32)
        batch_valid = np.zeros((fill_source_batch_size,), dtype=bool)
        batch_points[:count] = source_points[selected]
        batch_values[:count] = source_values[selected]
        batch_outputs[:count] = source_output_indices[selected]
        batch_valid[:count] = True
        (
            pair_points,
            pair_values,
            pair_outputs,
            pair_targets,
            pair_valid,
            batch_pairs,
            overflow,
        ) = fill_batch_kernel(
            jnp.asarray(batch_points),
            jnp.asarray(batch_values),
            jnp.asarray(batch_outputs),
            jnp.asarray(batch_valid),
            *tree_device,
            jnp.asarray(search_tolerance),
        )
        pair_points.block_until_ready()
        if bool(np.asarray(overflow)):
            raise RuntimeError("JAX BVH candidate buffer overflowed")
        batch_power, batch_nonempty = exact_batch_kernel(
            pair_points,
            pair_values,
            pair_outputs,
            pair_targets,
            pair_valid,
            target_points_device,
            jnp.asarray(intersection_tolerance),
            jnp.asarray(mesh.inactive_measure),
        )
        batch_power.block_until_ready()
        accumulated_power += np.asarray(batch_power)
        current_pairs = int(np.asarray(batch_pairs))
        processed_pairs += current_pairs
        nonempty_overlaps += int(np.asarray(batch_nonempty))
        processed_sources += count
        group_start = group_stop
        now = time.perf_counter()
        if (
            progress_interval_s is not None
            and now - last_progress >= progress_interval_s
        ) or processed_sources == total_sources:
            elapsed = now - fill_started
            rate = processed_sources / elapsed if elapsed > 0.0 else 0.0
            eta = (total_sources - processed_sources) / rate if rate > 0.0 else None
            _emit(
                progress_callback,
                "jax bvh overlap",
                "running" if processed_sources < total_sources else "complete",
                f"device BVH overlap {processed_sources}/{total_sources}; "
                f"{processed_pairs}/{total_pairs} candidate pairs",
                processed_source_simplices=processed_sources,
                total_source_simplices=total_sources,
                processed_pairs=processed_pairs,
                total_pairs=total_pairs,
                nonempty_overlaps=nonempty_overlaps,
                estimated_remaining_s=eta,
            )
            last_progress = now

    cell_power = accumulated_power.reshape(
        (len(beam_indices), len(sheet_indices), mesh.ncells)
    )
    resolved = ResolvedPowerDeposition(
        cell_power=cell_power,
        source_power=source_power,
        cell_volumes=mesh.cell_volumes,
        beam_indices=np.asarray(beam_indices, dtype=np.int32),
        sheet_indices=np.asarray(sheet_indices, dtype=np.int32),
    )
    return resolved if resolve_beam_sheets else resolved.total


def _tetrahedron_geometry_host(points: np.ndarray):
    faces = points[:, _TETRAHEDRON_FACE_INDICES]
    first = faces[:, :, 0]
    normals = np.cross(faces[:, :, 1] - first, faces[:, :, 2] - first)
    excluded = points[:, np.arange(4)]
    inward = np.einsum("nfc,nfc->nf", normals, excluded - first) > 0.0
    normals = np.where(inward[..., None], -normals, normals)
    lengths = np.linalg.norm(normals, axis=-1)
    normals /= np.where(lengths > 0.0, lengths, 1.0)[..., None]
    offsets = np.einsum("nfc,nfc->nf", normals, first)
    edges = points[:, _TETRAHEDRON_EDGE_INDICES]
    scales = np.max(np.linalg.norm(edges[:, :, 1] - edges[:, :, 0], axis=-1), axis=1)
    edge_matrices = points[:, 1:] - points[:, :1]
    volumes = np.abs(np.linalg.det(edge_matrices)) / 6.0
    centres = np.mean(points, axis=1)
    return normals, offsets, volumes, centres, scales


def _deposit_three_dimensional_jax_bvh(
    field: SimplicialField,
    mesh: SimplicialDepositionMesh,
    *,
    field_name: str,
    beam_index: int | None,
    sheet_index: int | None,
    resolve_beam_sheets: bool,
    intersection_tolerance: float,
    progress_interval_s: float | None,
    progress_callback: ProgressCallback | None,
    pair_batch_size: int,
    source_batch_size: int,
    bvh_leaf_size: int,
) -> PowerDeposition | Any:
    """Optimized exact tetrahedral deposition with streamed device accumulation."""
    from .mesh_deposition import ResolvedPowerDeposition, _build_aabb_tree

    try:
        field_index = getattr(field.selection, field_name)
    except AttributeError as error:
        raise ValueError(f"simplicial field does not include {field_name!r}") from error
    if not isinstance(field_index, Integral):
        raise TypeError("deposited simplicial field must be scalar")
    beam_indices = _validate_selection(beam_index, field.mesh.nbeams, "beam_index")
    sheet_indices = _validate_selection(sheet_index, field.mesh.nsheets, "sheet_index")
    (
        source_points,
        source_values,
        source_output_indices,
        source_beam_indices,
        source_sheet_indices,
    ) = _collect_sources(
        field,
        3,
        int(field_index),
        beam_indices,
        sheet_indices,
    )
    source_measures = _source_simplex_measures(source_points)
    source_powers = (
        mesh.inactive_measure * source_measures * np.mean(source_values, axis=1)
    )
    source_power = np.zeros((len(beam_indices), len(sheet_indices)), dtype=np.float64)
    np.add.at(
        source_power,
        (source_beam_indices, source_sheet_indices),
        source_powers,
    )

    active_source = (source_measures > 0.0) & np.any(source_values != 0.0, axis=1)
    skipped_zero_sources = int(np.count_nonzero(~active_source))
    source_points = source_points[active_source]
    source_values = source_values[active_source]
    source_output_indices = source_output_indices[active_source]
    if not resolve_beam_sheets:
        source_output_indices = np.zeros_like(source_output_indices)

    if source_points.shape[0] == 0:
        empty_power = np.zeros((mesh.ncells,), dtype=np.float64)
        if not resolve_beam_sheets:
            total_source_power = float(np.sum(source_power))
            return PowerDeposition(
                power_density=empty_power,
                cell_power=empty_power,
                deposited_power=0.0,
                outside_power=total_source_power,
                source_power=total_source_power,
            )
        return ResolvedPowerDeposition(
            cell_power=np.zeros(
                (len(beam_indices), len(sheet_indices), mesh.ncells),
                dtype=np.float64,
            ),
            source_power=source_power,
            cell_volumes=mesh.cell_volumes,
            beam_indices=np.asarray(beam_indices, dtype=np.int32),
            sheet_indices=np.asarray(sheet_indices, dtype=np.int32),
        )

    preparation_started = time.perf_counter()
    (
        source_normals,
        source_offsets,
        source_volumes,
        source_centres,
        source_scales,
    ) = _tetrahedron_geometry_host(source_points)
    source_gradients = np.linalg.solve(
        source_points[:, 1:] - source_points[:, :1],
        (source_values[:, 1:] - source_values[:, :1])[..., None],
    ).squeeze(-1)
    source_intercepts = source_values[:, 0] - np.einsum(
        "ni,ni->n", source_gradients, source_points[:, 0]
    )
    target_points = np.asarray(
        mesh.vertex_positions[mesh.simplex_connectivity, :3], dtype=np.float64
    )
    target_normals = np.asarray(mesh.cell_plane_normals, dtype=np.float64)
    target_offsets = np.asarray(mesh.cell_plane_offsets, dtype=np.float64)
    target_volumes = np.asarray(mesh.cell_volumes, dtype=np.float64)
    target_centres = np.asarray(mesh.cell_centres[:, :3], dtype=np.float64)
    target_edges = target_points[:, _TETRAHEDRON_EDGE_INDICES]
    target_scales = np.max(
        np.linalg.norm(target_edges[:, :, 1] - target_edges[:, :, 0], axis=-1),
        axis=1,
    )
    source_lower = np.min(source_points, axis=1)
    source_upper = np.max(source_points, axis=1)
    preparation_elapsed = time.perf_counter() - preparation_started
    _emit(
        progress_callback,
        "jax geometry preparation",
        "complete",
        f"prepared {source_points.shape[0]} active source and {mesh.ncells} target "
        f"tetrahedra in {preparation_elapsed:.2f} s; skipped "
        f"{skipped_zero_sources} zero sources",
        active_source_simplices=source_points.shape[0],
        skipped_zero_source_simplices=skipped_zero_sources,
        elapsed_s=preparation_elapsed,
    )

    build_started = time.perf_counter()
    tree = _build_aabb_tree(
        mesh.cell_bounds_min, mesh.cell_bounds_max, leaf_size=bvh_leaf_size
    )
    flat_tree = _flatten_bvh(tree, bvh_leaf_size)
    build_elapsed = time.perf_counter() - build_started
    _emit(
        progress_callback,
        "jax bvh construction",
        "complete",
        f"flattened {flat_tree.bounds_min.shape[0]} BVH nodes in {build_elapsed:.2f} s",
        bvh_nodes=flat_tree.bounds_min.shape[0],
        elapsed_s=build_elapsed,
    )

    mesh_scale = max(
        float(np.max(mesh.cell_bounds_max) - np.min(mesh.cell_bounds_min)),
        np.finfo(np.float64).tiny,
    )
    search_tolerance = intersection_tolerance * mesh_scale
    tree_device = tuple(
        jax.device_put(jnp.asarray(item))
        for item in (
            flat_tree.bounds_min,
            flat_tree.bounds_max,
            flat_tree.left_child,
            flat_tree.right_child,
            flat_tree.leaf_targets,
            mesh.cell_bounds_min[:, :3],
            mesh.cell_bounds_max[:, :3],
        )
    )

    total_sources = source_points.shape[0]
    candidate_counts = np.zeros((total_sources,), dtype=np.int32)
    count_started = time.perf_counter()
    last_progress = count_started
    _emit(
        progress_callback,
        "jax bvh count",
        "started",
        f"counting candidates for {total_sources} active source simplices",
        total_source_simplices=total_sources,
    )
    for start in range(0, total_sources, source_batch_size):
        stop = min(start + source_batch_size, total_sources)
        count = stop - start
        batch_lower = np.zeros((source_batch_size, 3), dtype=np.float64)
        batch_upper = np.zeros((source_batch_size, 3), dtype=np.float64)
        batch_valid = np.zeros((source_batch_size,), dtype=bool)
        batch_lower[:count] = source_lower[start:stop]
        batch_upper[:count] = source_upper[start:stop]
        batch_valid[:count] = True
        counts = _query_bvh_counts(
            jnp.asarray(batch_lower),
            jnp.asarray(batch_upper),
            jnp.asarray(batch_valid),
            *tree_device,
            jnp.asarray(search_tolerance),
            flat_tree.stack_capacity,
        )
        counts.block_until_ready()
        candidate_counts[start:stop] = np.asarray(counts[:count])
        now = time.perf_counter()
        if (
            progress_interval_s is not None
            and now - last_progress >= progress_interval_s
        ) or stop == total_sources:
            elapsed = now - count_started
            rate = stop / elapsed if elapsed > 0.0 else 0.0
            eta = (total_sources - stop) / rate if rate > 0.0 else None
            _emit(
                progress_callback,
                "jax bvh count",
                "running" if stop < total_sources else "complete",
                f"candidate count {stop}/{total_sources}; "
                f"{int(candidate_counts[:stop].sum())} pairs",
                processed_source_simplices=stop,
                total_source_simplices=total_sources,
                candidate_pairs=int(candidate_counts[:stop].sum()),
                estimated_remaining_s=eta,
            )
            last_progress = now

    capacities = np.zeros_like(candidate_counts)
    positive = candidate_counts > 0
    capacities[positive] = 1 << np.ceil(np.log2(candidate_counts[positive])).astype(
        np.int32
    )
    bucket_capacities = capacities.copy()
    bucket_capacities[(bucket_capacities > 0) & (bucket_capacities <= 128)] = 128
    unique_capacities, capacity_populations = np.unique(
        bucket_capacities[positive], return_counts=True
    )
    capacity_histogram = {
        int(capacity): int(population)
        for capacity, population in zip(unique_capacities, capacity_populations)
    }
    total_pairs = int(candidate_counts.sum())
    dense_candidate_slots = int(
        np.sum(unique_capacities.astype(np.int64) * capacity_populations)
    )
    candidate_buffer_utilization = (
        total_pairs / dense_candidate_slots if dense_candidate_slots else 1.0
    )
    _emit(
        progress_callback,
        "jax bvh count",
        "complete",
        f"counted {total_pairs} candidate pairs in "
        f"{time.perf_counter() - count_started:.2f} s; "
        f"execution buckets {capacity_histogram}; buffers "
        f"{candidate_buffer_utilization:.1%} full",
        candidate_pairs=total_pairs,
        capacity_histogram=capacity_histogram,
        dense_candidate_slots=dense_candidate_slots,
        candidate_buffer_utilization=candidate_buffer_utilization,
        elapsed_s=time.perf_counter() - count_started,
    )

    output_channels = (
        len(beam_indices) * len(sheet_indices) if resolve_beam_sheets else 1
    )
    output_size = output_channels * mesh.ncells
    maximum_capacity = int(np.max(unique_capacities)) if positive.any() else 1
    exact_batch_size = max(maximum_capacity, pair_batch_size)
    source_device = tuple(
        jax.device_put(jnp.asarray(item))
        for item in (
            source_points,
            source_normals,
            source_offsets,
            source_volumes,
            source_centres,
            source_gradients,
            source_intercepts,
            source_output_indices,
            source_scales,
        )
    )
    target_device = tuple(
        jax.device_put(jnp.asarray(item))
        for item in (
            target_points,
            target_normals,
            target_offsets,
            target_volumes,
            target_centres,
            target_scales,
        )
    )
    accumulated_power = jnp.zeros((output_size,), dtype=jnp.float64)
    statistics = jnp.zeros((8,), dtype=jnp.int64)
    classify_kernel = _make_tetrahedron_classification_batch(exact_batch_size)
    sat_kernels = {}
    exact_kernels = {}

    overlap_started = time.perf_counter()
    last_progress = overlap_started
    processed_sources = int(np.sum(~positive))
    processed_pairs = 0
    batches_since_sync = 0
    for capacity in unique_capacities:
        capacity = int(capacity)
        bucket_indices = np.flatnonzero(bucket_capacities == capacity)
        fill_source_batch_size = min(
            source_batch_size, max(1, exact_batch_size // capacity)
        )
        fill_kernel = _make_indexed_bvh_fill_batch(
            flat_tree.stack_capacity,
            capacity,
            exact_batch_size,
        )
        for start in range(0, bucket_indices.size, fill_source_batch_size):
            selected = bucket_indices[start : start + fill_source_batch_size]
            count = selected.size
            batch_sources = np.zeros((fill_source_batch_size,), dtype=np.int32)
            batch_valid = np.zeros((fill_source_batch_size,), dtype=bool)
            batch_sources[:count] = selected
            batch_valid[:count] = True
            (
                pair_sources,
                pair_targets,
                pair_valid,
                _batch_pairs,
                overflow,
            ) = fill_kernel(
                jnp.asarray(batch_sources),
                jnp.asarray(batch_valid),
                source_device[0],
                *tree_device,
                jnp.asarray(search_tolerance),
            )
            (
                accumulated_power,
                statistics,
                partial_sources,
                partial_targets,
                partial_valid,
                general_count,
            ) = classify_kernel(
                accumulated_power,
                statistics,
                pair_sources,
                pair_targets,
                pair_valid,
                *source_device,
                *target_device,
                jnp.asarray(intersection_tolerance),
                jnp.asarray(mesh.inactive_measure),
                overflow,
            )
            general_count = int(np.asarray(general_count))
            if general_count:
                sat_capacity = max(256, 1 << (general_count - 1).bit_length())
                sat_kernel = sat_kernels.setdefault(
                    sat_capacity, _make_tetrahedron_sat_batch(sat_capacity)
                )
                (
                    statistics,
                    exact_sources,
                    exact_targets,
                    exact_valid,
                    exact_count,
                ) = sat_kernel(
                    statistics,
                    partial_sources[:sat_capacity],
                    partial_targets[:sat_capacity],
                    partial_valid[:sat_capacity],
                    source_device[0],
                    source_device[8],
                    target_device[0],
                    target_device[5],
                    jnp.asarray(intersection_tolerance),
                )
                exact_count = int(np.asarray(exact_count))
            else:
                exact_count = 0
            if exact_count:
                clip_capacity = max(256, 1 << (exact_count - 1).bit_length())
                exact_kernel = exact_kernels.setdefault(
                    clip_capacity,
                    _make_prepared_tetrahedron_deposit_batch(),
                )
                accumulated_power, statistics = exact_kernel(
                    accumulated_power,
                    statistics,
                    exact_sources[:clip_capacity],
                    exact_targets[:clip_capacity],
                    exact_valid[:clip_capacity],
                    source_device[0],
                    source_device[1],
                    source_device[2],
                    source_device[5],
                    source_device[6],
                    source_device[7],
                    source_device[8],
                    target_device[0],
                    target_device[1],
                    target_device[2],
                    target_device[5],
                    jnp.asarray(intersection_tolerance),
                    jnp.asarray(mesh.inactive_measure),
                )
            processed_sources += count
            processed_pairs += int(candidate_counts[selected].sum())
            batches_since_sync += 1
            final_batch = (
                capacity == int(unique_capacities[-1])
                and start + fill_source_batch_size >= bucket_indices.size
            )
            if batches_since_sync >= 8 or final_batch:
                statistics.block_until_ready()
                batches_since_sync = 0
                now = time.perf_counter()
                if (
                    progress_interval_s is not None
                    and now - last_progress >= progress_interval_s
                ) or final_batch:
                    current_statistics = np.asarray(statistics)
                    elapsed = now - overlap_started
                    rate = processed_pairs / elapsed if elapsed > 0.0 else 0.0
                    eta = (total_pairs - processed_pairs) / rate if rate > 0.0 else None
                    nonempty = int(
                        current_statistics[2]
                        + current_statistics[3]
                        + current_statistics[6]
                    )
                    _emit(
                        progress_callback,
                        "jax bvh overlap",
                        "complete" if final_batch else "running",
                        f"optimized overlap {processed_sources}/{total_sources}; "
                        f"{processed_pairs}/{total_pairs} AABB pairs; "
                        f"{int(current_statistics[1])} face and "
                        f"{int(current_statistics[4])} edge-axis rejects; "
                        f"{int(current_statistics[5])} exact clips",
                        processed_source_simplices=processed_sources,
                        total_source_simplices=total_sources,
                        processed_pairs=processed_pairs,
                        total_pairs=total_pairs,
                        face_rejected_pairs=int(current_statistics[1]),
                        edge_axis_rejected_pairs=int(current_statistics[4]),
                        source_containment_pairs=int(current_statistics[2]),
                        target_containment_pairs=int(current_statistics[3]),
                        exact_intersections=int(current_statistics[5]),
                        nonempty_overlaps=nonempty,
                        estimated_remaining_s=eta,
                    )
                    last_progress = now

    accumulated_power.block_until_ready()
    final_statistics = np.asarray(statistics)
    if int(final_statistics[7]) != 0:
        raise RuntimeError("JAX BVH candidate buffer overflowed")
    cell_power_flat = np.asarray(accumulated_power)
    if not resolve_beam_sheets:
        deposited_power = float(np.sum(cell_power_flat))
        total_source_power = float(np.sum(source_power))
        return PowerDeposition(
            power_density=cell_power_flat / mesh.cell_volumes,
            cell_power=cell_power_flat,
            deposited_power=deposited_power,
            outside_power=total_source_power - deposited_power,
            source_power=total_source_power,
        )

    cell_power = cell_power_flat.reshape(
        (len(beam_indices), len(sheet_indices), mesh.ncells)
    )
    return ResolvedPowerDeposition(
        cell_power=cell_power,
        source_power=source_power,
        cell_volumes=mesh.cell_volumes,
        beam_indices=np.asarray(beam_indices, dtype=np.int32),
        sheet_indices=np.asarray(sheet_indices, dtype=np.int32),
    )


def _make_segment_deposit_batch(output_size: int):
    @jax.jit
    def deposit_batch(
        source_points,
        source_values,
        source_output_indices,
        target_bounds_min,
        target_bounds_max,
        pair_source_indices,
        pair_target_indices,
        pair_valid,
        inactive_measure,
    ):
        safe_sources = jnp.maximum(pair_source_indices, 0)
        safe_targets = jnp.maximum(pair_target_indices, 0)
        points = source_points[safe_sources, :, 0]
        values = source_values[safe_sources]
        lower = jnp.maximum(jnp.min(points, axis=1), target_bounds_min[safe_targets])
        upper = jnp.minimum(jnp.max(points, axis=1), target_bounds_max[safe_targets])
        length = jnp.maximum(upper - lower, 0.0)
        slope = (values[:, 1] - values[:, 0]) / (points[:, 1] - points[:, 0])
        integral = values[:, 0] * length + 0.5 * slope * (
            (upper - points[:, 0]) ** 2 - (lower - points[:, 0]) ** 2
        )
        contribution = jnp.where(
            pair_valid & (length > 0.0),
            inactive_measure * integral,
            0.0,
        )
        flat_output = (
            source_output_indices[safe_sources] * target_bounds_min.size + safe_targets
        )
        power = jnp.zeros((output_size,), dtype=contribution.dtype)
        return power.at[flat_output].add(contribution)

    return deposit_batch


def _deposit_one_dimensional(
    field: SimplicialField,
    mesh: SimplicialDepositionMesh,
    *,
    field_name: str,
    beam_index: int | None,
    sheet_index: int | None,
    resolve_beam_sheets: bool,
    intersection_tolerance: float,
    progress_interval_s: float | None,
    progress_callback: ProgressCallback | None,
    pair_batch_size: int,
    source_batch_size: int,
    bvh_leaf_size: int,
) -> PowerDeposition | Any:
    del source_batch_size, bvh_leaf_size
    from .mesh_deposition import ResolvedPowerDeposition

    if field.mesh.dimension != 1 or mesh.dimension != 1:
        raise ValueError("segment deposition requires matching 1-D meshes")
    if mesh.topology != "linear":
        raise ValueError("one-dimensional deposition requires a linear target mesh")
    try:
        field_index = getattr(field.selection, field_name)
    except AttributeError as error:
        raise ValueError(f"simplicial field does not include {field_name!r}") from error
    if not isinstance(field_index, Integral):
        raise TypeError("deposited simplicial field must be scalar")

    beam_indices = _validate_selection(beam_index, field.mesh.nbeams, "beam_index")
    sheet_indices = _validate_selection(sheet_index, field.mesh.nsheets, "sheet_index")
    (
        source_points,
        source_values,
        source_output_indices,
        source_beam_indices,
        source_sheet_indices,
    ) = _collect_sources(
        field,
        1,
        int(field_index),
        beam_indices,
        sheet_indices,
    )
    measures = _source_simplex_measures(source_points)
    source_powers = mesh.inactive_measure * measures * np.mean(source_values, axis=1)
    source_power = np.zeros((len(beam_indices), len(sheet_indices)), dtype=np.float64)
    np.add.at(
        source_power,
        (source_beam_indices, source_sheet_indices),
        source_powers,
    )

    target_lower = np.asarray(mesh.cell_bounds_min[:, 0], dtype=np.float64)
    target_upper = np.asarray(mesh.cell_bounds_max[:, 0], dtype=np.float64)
    scale = max(
        float(target_upper[-1] - target_lower[0]),
        np.finfo(np.float64).tiny,
    )
    tolerance = intersection_tolerance * scale
    source_lower = np.min(source_points[:, :, 0], axis=1)
    source_upper = np.max(source_points[:, :, 0], axis=1)
    starts = np.searchsorted(target_upper, source_lower - tolerance, side="left")
    stops = np.searchsorted(target_lower, source_upper + tolerance, side="right")
    counts = np.maximum(stops - starts, 0)
    pair_sources = np.repeat(np.arange(source_points.shape[0], dtype=np.int32), counts)
    pair_targets = (
        np.concatenate(
            [
                np.arange(start, stop, dtype=np.int32)
                for start, stop in zip(starts, stops, strict=True)
                if stop > start
            ]
        )
        if np.any(counts)
        else np.empty((0,), dtype=np.int32)
    )

    output_size = len(beam_indices) * len(sheet_indices) * mesh.ncells
    accumulated_power = np.zeros((output_size,), dtype=np.float64)
    kernel = _make_segment_deposit_batch(output_size)
    source_points_device = jax.device_put(jnp.asarray(source_points))
    source_values_device = jax.device_put(jnp.asarray(source_values))
    source_output_device = jax.device_put(jnp.asarray(source_output_indices))
    target_lower_device = jax.device_put(jnp.asarray(target_lower))
    target_upper_device = jax.device_put(jnp.asarray(target_upper))

    overlap_started = time.perf_counter()
    last_progress = overlap_started
    _emit(
        progress_callback,
        "jax segment overlap",
        "started",
        f"integrating {pair_sources.size} candidate pairs",
        total_source_simplices=source_points.shape[0],
        total_pairs=pair_sources.size,
    )
    for start in range(0, pair_sources.size, pair_batch_size):
        stop = min(start + pair_batch_size, pair_sources.size)
        count = stop - start
        batch_sources = np.zeros((pair_batch_size,), dtype=np.int32)
        batch_targets = np.zeros((pair_batch_size,), dtype=np.int32)
        batch_valid = np.zeros((pair_batch_size,), dtype=bool)
        batch_sources[:count] = pair_sources[start:stop]
        batch_targets[:count] = pair_targets[start:stop]
        batch_valid[:count] = True
        batch_power = kernel(
            source_points_device,
            source_values_device,
            source_output_device,
            target_lower_device,
            target_upper_device,
            jnp.asarray(batch_sources),
            jnp.asarray(batch_targets),
            jnp.asarray(batch_valid),
            jnp.asarray(mesh.inactive_measure),
        )
        batch_power.block_until_ready()
        accumulated_power += np.asarray(batch_power)
        now = time.perf_counter()
        if (
            progress_interval_s is not None
            and now - last_progress >= progress_interval_s
        ) or stop == pair_sources.size:
            elapsed = now - overlap_started
            rate = stop / elapsed if elapsed > 0.0 else 0.0
            eta = (pair_sources.size - stop) / rate if rate > 0.0 else None
            _emit(
                progress_callback,
                "jax segment overlap",
                "running" if stop < pair_sources.size else "complete",
                f"candidate pairs {stop}/{pair_sources.size}",
                processed_pairs=stop,
                total_pairs=pair_sources.size,
                estimated_remaining_s=eta,
            )
            last_progress = now

    _emit(
        progress_callback,
        "jax segment overlap",
        "complete",
        f"deposited {source_points.shape[0]} source segments through "
        f"{pair_sources.size} candidate pairs",
        total_source_simplices=source_points.shape[0],
        candidate_pairs=pair_sources.size,
    )
    resolved = ResolvedPowerDeposition(
        cell_power=accumulated_power.reshape(
            (len(beam_indices), len(sheet_indices), mesh.ncells)
        ),
        source_power=source_power,
        cell_volumes=mesh.cell_volumes,
        beam_indices=np.asarray(beam_indices, dtype=np.int32),
        sheet_indices=np.asarray(sheet_indices, dtype=np.int32),
    )
    return resolved if resolve_beam_sheets else resolved.total


def deposit_simplicial_power_to_mesh_jax(
    field: SimplicialField,
    mesh: SimplicialDepositionMesh,
    *,
    field_name: str,
    beam_index: int | None,
    sheet_index: int | None,
    resolve_beam_sheets: bool,
    intersection_tolerance: float,
    progress_interval_s: float | None,
    progress_callback: ProgressCallback | None,
    pair_batch_size: int,
    source_batch_size: int,
    bvh_leaf_size: int,
) -> PowerDeposition | Any:
    """Deposit an affine simplicial field using the sole retained JAX path."""
    options = {
        "field_name": field_name,
        "beam_index": beam_index,
        "sheet_index": sheet_index,
        "resolve_beam_sheets": resolve_beam_sheets,
        "intersection_tolerance": intersection_tolerance,
        "progress_interval_s": progress_interval_s,
        "progress_callback": progress_callback,
        "pair_batch_size": pair_batch_size,
        "source_batch_size": source_batch_size,
        "bvh_leaf_size": bvh_leaf_size,
    }
    if mesh.dimension == 1:
        return _deposit_one_dimensional(field, mesh, **options)
    if mesh.dimension == 2:
        return _deposit_two_dimensional_jax_bvh(field, mesh, **options)
    if mesh.dimension == 3:
        return _deposit_three_dimensional_jax_bvh(field, mesh, **options)
    raise ValueError(
        "deposition supports only one-, two-, and three-dimensional meshes"
    )


__all__ = ["deposit_simplicial_power_to_mesh_jax"]
