"""Straight-sided simplicial deposition meshes in one, two, and three dimensions."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral
from typing import Any

import numpy as np

from pyGATH.grid import Geometry, Grid
from pyGATH.reporting import ProgressTracker, reported_stage


@dataclass(frozen=True)
class SimplicialDepositionMesh:
    """An unstructured segment, triangle, or tetrahedron deposition mesh.

    ``simplex_radial_layer`` groups cells between successive entries of
    ``radial_boundaries``. Surface resolution may therefore vary with radius
    without imposing a rectangular radial-by-angular result shape.
    """

    vertex_positions: np.ndarray
    simplex_connectivity: np.ndarray
    simplex_radial_layer: np.ndarray
    radial_boundaries: np.ndarray
    surface_element_counts: np.ndarray
    surface_angular_counts: np.ndarray
    cell_volumes: np.ndarray
    cell_centres: np.ndarray
    cell_bounds_min: np.ndarray
    cell_bounds_max: np.ndarray
    cell_plane_normals: np.ndarray
    cell_plane_offsets: np.ndarray
    dimension: int
    inactive_measure: float
    topology: str

    @property
    def ncells(self) -> int:
        return int(self.simplex_connectivity.shape[0])

    @property
    def nsimplices(self) -> int:
        return self.ncells

    @property
    def nradial(self) -> int:
        return int(self.radial_boundaries.size - 1)


def _describe_mesh(mesh: SimplicialDepositionMesh) -> str:
    simplex_name = ("segments", "triangles", "tetrahedra")[mesh.dimension - 1]
    return f"{mesh.topology}, {mesh.ncells:,} {simplex_name}"


def _validate_radial_boundaries(radial_boundaries: Any) -> np.ndarray:
    radii = np.asarray(radial_boundaries, dtype=np.float64)
    if radii.ndim != 1 or radii.size < 2:
        raise ValueError("radial_boundaries must be one-dimensional with two entries")
    if not np.all(np.isfinite(radii)):
        raise ValueError("radial_boundaries must be finite")
    if radii[0] < 0.0 or np.any(np.diff(radii) <= 0.0):
        raise ValueError(
            "radial_boundaries must be nonnegative and strictly increasing"
        )
    return radii


def _validate_integer(value: Any, name: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _validate_inactive_measure(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise TypeError("inactive_length_m must be a number") from error
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError("inactive_length_m must be finite and positive")
    return result


def _orient_triangles(vertices: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    result = np.asarray(triangles, dtype=np.int32).copy()
    points = vertices[result, :2]
    first = points[:, 1] - points[:, 0]
    second = points[:, 2] - points[:, 0]
    signed_twice_area = first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0]
    reverse = signed_twice_area < 0.0
    result[reverse, 1], result[reverse, 2] = (
        result[reverse, 2].copy(),
        result[reverse, 1].copy(),
    )
    return result


def _orient_tetrahedra(vertices: np.ndarray, tetrahedra: np.ndarray) -> np.ndarray:
    result = np.asarray(tetrahedra, dtype=np.int32).copy()
    points = vertices[result]
    determinants = np.linalg.det(
        np.stack(
            (
                points[:, 1] - points[:, 0],
                points[:, 2] - points[:, 0],
                points[:, 3] - points[:, 0],
            ),
            axis=-1,
        )
    )
    reverse = determinants < 0.0
    result[reverse, 1], result[reverse, 2] = (
        result[reverse, 2].copy(),
        result[reverse, 1].copy(),
    )
    return result


def _simplex_geometry(
    vertices: np.ndarray,
    connectivity: np.ndarray,
    dimension: int,
    inactive_measure: float,
):
    points = vertices[connectivity, :dimension]
    edges = np.stack(
        tuple(points[:, index] - points[:, 0] for index in range(1, dimension + 1)),
        axis=-1,
    )
    measure = np.abs(np.linalg.det(edges)) / math.factorial(dimension)
    volumes = measure * inactive_measure
    centres = np.zeros((connectivity.shape[0], 3), dtype=np.float64)
    centres[:, :dimension] = np.mean(points, axis=1)
    bounds_min = np.min(points, axis=1)
    bounds_max = np.max(points, axis=1)

    nplanes = dimension + 1
    normals = np.empty((connectivity.shape[0], nplanes, dimension), dtype=np.float64)
    offsets = np.empty((connectivity.shape[0], nplanes), dtype=np.float64)
    for cell_index, simplex in enumerate(points):
        for excluded in range(nplanes):
            face = np.delete(simplex, excluded, axis=0)
            if dimension == 1:
                normal = np.asarray((1.0 if excluded == 0 else -1.0,))
            elif dimension == 2:
                edge = face[1] - face[0]
                normal = np.asarray((edge[1], -edge[0]))
            else:
                normal = np.cross(face[1] - face[0], face[2] - face[0])
            normal /= np.linalg.norm(normal)
            offset = float(np.dot(normal, face[0]))
            if np.dot(normal, simplex[excluded]) > offset:
                normal = -normal
                offset = -offset
            normals[cell_index, excluded] = normal
            offsets[cell_index, excluded] = offset
    return volumes, centres, bounds_min, bounds_max, normals, offsets


def _make_mesh(
    vertices,
    connectivity,
    radial_layers,
    radii,
    surface_element_counts,
    surface_angular_counts,
    *,
    dimension,
    inactive_measure,
    topology,
) -> SimplicialDepositionMesh:
    vertex_array = np.asarray(vertices, dtype=np.float64)
    simplex_array = np.asarray(connectivity, dtype=np.int32)
    if dimension == 2:
        simplex_array = _orient_triangles(vertex_array, simplex_array)
    elif dimension == 3:
        simplex_array = _orient_tetrahedra(vertex_array, simplex_array)
    geometry = _simplex_geometry(
        vertex_array, simplex_array, dimension, inactive_measure
    )
    if np.any(geometry[0] <= 0.0):
        raise ValueError("deposition mesh contains a degenerate simplex")
    return SimplicialDepositionMesh(
        vertex_positions=vertex_array,
        simplex_connectivity=simplex_array,
        simplex_radial_layer=np.asarray(radial_layers, dtype=np.int32),
        radial_boundaries=np.asarray(radii, dtype=np.float64),
        surface_element_counts=np.asarray(surface_element_counts, dtype=np.int32),
        surface_angular_counts=np.asarray(surface_angular_counts, dtype=np.int32),
        cell_volumes=geometry[0],
        cell_centres=geometry[1],
        cell_bounds_min=geometry[2],
        cell_bounds_max=geometry[3],
        cell_plane_normals=geometry[4],
        cell_plane_offsets=geometry[5],
        dimension=dimension,
        inactive_measure=inactive_measure,
        topology=topology,
    )


@reported_stage("build deposition mesh", describe=_describe_mesh)
def build_linear_deposition_mesh(
    boundaries,
    *,
    inactive_area_m2: float = 1.0,
) -> SimplicialDepositionMesh:
    """Build a line-segment target mesh with an extruded physical area."""
    points = np.asarray(boundaries, dtype=np.float64)
    if points.ndim != 1 or points.size < 2:
        raise ValueError("boundaries must be one-dimensional with at least two entries")
    if not np.all(np.isfinite(points)) or np.any(np.diff(points) <= 0.0):
        raise ValueError("boundaries must be finite and strictly increasing")
    inactive_measure = _validate_inactive_measure(inactive_area_m2)
    vertices = np.zeros((points.size, 3), dtype=np.float64)
    vertices[:, 0] = points
    connectivity = np.column_stack(
        (
            np.arange(points.size - 1, dtype=np.int32),
            np.arange(1, points.size, dtype=np.int32),
        )
    )
    counts = np.ones((points.size,), dtype=np.int32)
    return _make_mesh(
        vertices,
        connectivity,
        np.arange(points.size - 1, dtype=np.int32),
        points,
        counts,
        counts,
        dimension=1,
        inactive_measure=inactive_measure,
        topology="linear",
    )


@reported_stage("build deposition mesh", describe=_describe_mesh)
def build_cartesian_deposition_mesh_from_grid(grid: Grid) -> SimplicialDepositionMesh:
    """Triangulate or tetrahedralise a 2-D or 3-D Cartesian grid."""
    if grid.geom is not Geometry.CARTESIAN or grid.dimensions not in (2, 3):
        raise ValueError(
            "Cartesian deposition meshes require a 2-D or 3-D Cartesian grid"
        )
    axes = [np.asarray(grid.xb), np.asarray(grid.yb)]
    if grid.dimensions == 3:
        axes.append(np.asarray(grid.zb))
    native = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
    vertices = np.zeros((*native.shape[:-1], 3), dtype=np.float64)
    vertices[..., : grid.dimensions] = native
    vertices = vertices.reshape((-1, 3))

    connectivity = []
    layers = []
    progress = ProgressTracker(
        "build deposition mesh", grid.ncells[0], "first-axis cell layers"
    )
    if grid.dimensions == 2:
        ny = grid.ncells[1] + 1
        for ix in range(grid.ncells[0]):
            for iy in range(grid.ncells[1]):
                v00 = ix * ny + iy
                v10 = v00 + ny
                v01 = v00 + 1
                v11 = v10 + 1
                connectivity.extend(((v00, v10, v11), (v00, v11, v01)))
                layers.extend((ix, ix))
            progress.update(ix + 1)
    else:
        ny = grid.ncells[1] + 1
        nz = grid.ncells[2] + 1
        for ix in range(grid.ncells[0]):
            for iy in range(grid.ncells[1]):
                for iz in range(grid.ncells[2]):
                    v000 = (ix * ny + iy) * nz + iz
                    v100 = v000 + ny * nz
                    v010 = v000 + nz
                    v001 = v000 + 1
                    v110 = v100 + nz
                    v101 = v100 + 1
                    v011 = v010 + 1
                    v111 = v110 + 1
                    connectivity.extend(
                        (
                            (v000, v100, v110, v111),
                            (v000, v100, v101, v111),
                            (v000, v010, v110, v111),
                            (v000, v010, v011, v111),
                            (v000, v001, v101, v111),
                            (v000, v001, v011, v111),
                        )
                    )
                    layers.extend((ix,) * 6)
            progress.update(ix + 1)
    counts = np.zeros((grid.ncells[0] + 1,), dtype=np.int32)
    return _make_mesh(
        vertices,
        connectivity,
        layers,
        np.asarray(grid.xb),
        counts,
        counts,
        dimension=grid.dimensions,
        inactive_measure=float(grid.inactive_measure),
        topology="cartesian",
    )


def _stitch_rings(inner: np.ndarray, outer: np.ndarray) -> list[tuple[int, int, int]]:
    """Triangulate an annulus by merging the two rings' angular events."""
    ninner = inner.size
    nouter = outer.size
    inner_step = 0
    outer_step = 0
    triangles = []
    tolerance = 16.0 * np.finfo(np.float64).eps
    while inner_step < ninner or outer_step < nouter:
        next_inner = (inner_step + 1) / ninner if inner_step < ninner else np.inf
        next_outer = (outer_step + 1) / nouter if outer_step < nouter else np.inf
        current_inner = int(inner[inner_step % ninner])
        current_outer = int(outer[outer_step % nouter])
        if next_inner <= next_outer + tolerance:
            following_inner = int(inner[(inner_step + 1) % ninner])
            triangles.append((current_inner, current_outer, following_inner))
            inner_step += 1
        else:
            following_outer = int(outer[(outer_step + 1) % nouter])
            triangles.append((current_inner, current_outer, following_outer))
            outer_step += 1
    return triangles


@reported_stage("build deposition mesh", describe=_describe_mesh)
def build_circular_deposition_mesh(
    radial_boundaries,
    *,
    maximum_angular_cells: int,
    inactive_length_m: float = 1.0,
    angular_offset: float = 0.0,
) -> SimplicialDepositionMesh:
    """Build an adaptive physical-resolution triangle mesh of a circle."""
    radii = _validate_radial_boundaries(radial_boundaries)
    maximum = _validate_integer(maximum_angular_cells, "maximum_angular_cells", 3)
    inactive_measure = _validate_inactive_measure(inactive_length_m)
    if not np.isfinite(angular_offset):
        raise ValueError("angular_offset must be finite")

    counts = np.zeros(radii.size, dtype=np.int32)
    positive = radii > 0.0
    counts[positive] = np.maximum(
        3, np.rint(maximum * radii[positive] / radii[-1]).astype(np.int32)
    )
    counts[-1] = maximum
    counts = np.maximum.accumulate(counts)

    vertices: list[np.ndarray] = []
    rings: list[np.ndarray | None] = []
    for radius, count in zip(radii, counts, strict=True):
        if radius == 0.0:
            rings.append(None)
            vertices.append(np.zeros(3, dtype=np.float64))
            continue
        angles = angular_offset + 2.0 * np.pi * np.arange(count) / count
        start = len(vertices)
        vertices.extend(
            np.column_stack(
                (
                    radius * np.cos(angles),
                    radius * np.sin(angles),
                    np.zeros(count),
                )
            )
        )
        rings.append(np.arange(start, start + count, dtype=np.int32))

    triangles: list[tuple[int, int, int]] = []
    radial_layers: list[int] = []
    origin_index = 0 if radii[0] == 0.0 else None
    progress = ProgressTracker("build deposition mesh", radii.size - 1, "radial layers")
    for layer in range(radii.size - 1):
        inner = rings[layer]
        outer = rings[layer + 1]
        assert outer is not None
        if inner is None:
            assert origin_index is not None
            layer_triangles = [
                (origin_index, int(outer[index]), int(outer[(index + 1) % outer.size]))
                for index in range(outer.size)
            ]
        else:
            layer_triangles = _stitch_rings(inner, outer)
        triangles.extend(layer_triangles)
        radial_layers.extend((layer,) * len(layer_triangles))
        progress.update(layer + 1)

    return _make_mesh(
        vertices,
        triangles,
        radial_layers,
        radii,
        counts,
        counts,
        dimension=2,
        inactive_measure=inactive_measure,
        topology="adaptive_circular",
    )


def _orient_surface_faces(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    result = np.asarray(faces, dtype=np.int32).copy()
    points = vertices[result]
    normals = np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0])
    inward = np.einsum("ij,ij->i", normals, np.mean(points, axis=1)) < 0.0
    result[inward, 1], result[inward, 2] = (
        result[inward, 2].copy(),
        result[inward, 1].copy(),
    )
    return result


def _icosahedron() -> tuple[np.ndarray, np.ndarray]:
    golden_ratio = 0.5 * (1.0 + math.sqrt(5.0))
    vertices = np.asarray(
        (
            (-1, golden_ratio, 0),
            (1, golden_ratio, 0),
            (-1, -golden_ratio, 0),
            (1, -golden_ratio, 0),
            (0, -1, golden_ratio),
            (0, 1, golden_ratio),
            (0, -1, -golden_ratio),
            (0, 1, -golden_ratio),
            (golden_ratio, 0, -1),
            (golden_ratio, 0, 1),
            (-golden_ratio, 0, -1),
            (-golden_ratio, 0, 1),
        ),
        dtype=np.float64,
    )
    vertices /= np.linalg.norm(vertices, axis=1)[:, None]
    faces = np.asarray(
        (
            (0, 11, 5),
            (0, 5, 1),
            (0, 1, 7),
            (0, 7, 10),
            (0, 10, 11),
            (1, 5, 9),
            (5, 11, 4),
            (11, 10, 2),
            (10, 7, 6),
            (7, 1, 8),
            (3, 9, 4),
            (3, 4, 2),
            (3, 2, 6),
            (3, 6, 8),
            (3, 8, 9),
            (4, 9, 5),
            (2, 4, 11),
            (6, 2, 10),
            (8, 6, 7),
            (9, 8, 1),
        ),
        dtype=np.int32,
    )
    return vertices, _orient_surface_faces(vertices, faces)


def _subdivide_surface(vertices: np.ndarray, faces: np.ndarray):
    vertex_list = [vertex.copy() for vertex in vertices]
    midpoints: dict[tuple[int, int], int] = {}

    def midpoint(first, second):
        edge = tuple(sorted((int(first), int(second))))
        if edge not in midpoints:
            point = vertex_list[edge[0]] + vertex_list[edge[1]]
            point /= np.linalg.norm(point)
            midpoints[edge] = len(vertex_list)
            vertex_list.append(point)
        return midpoints[edge]

    children = []
    parents = []
    for parent, (first, second, third) in enumerate(faces):
        first_second = midpoint(first, second)
        second_third = midpoint(second, third)
        third_first = midpoint(third, first)
        children.extend(
            (
                (first, first_second, third_first),
                (second, second_third, first_second),
                (third, third_first, second_third),
                (first_second, second_third, third_first),
            )
        )
        parents.extend((parent,) * 4)
    child_vertices = np.asarray(vertex_list, dtype=np.float64)
    child_faces = _orient_surface_faces(
        child_vertices, np.asarray(children, dtype=np.int32)
    )
    return child_vertices, child_faces, np.asarray(parents, dtype=np.int32)


def _surface_levels(maximum_level: int):
    vertices, faces = _icosahedron()
    levels = [(vertices, faces)]
    parents = [None]
    for _ in range(maximum_level):
        vertices, faces, parent = _subdivide_surface(vertices, faces)
        levels.append((vertices, faces))
        parents.append(parent)
    return levels, parents


def _boundary_edge_chains(
    coarse_face: np.ndarray, outer_faces: np.ndarray
) -> list[list[int]]:
    edge_counts: dict[tuple[int, int], int] = {}
    for face in outer_faces:
        for first, second in (
            (face[0], face[1]),
            (face[1], face[2]),
            (face[2], face[0]),
        ):
            edge = tuple(sorted((int(first), int(second))))
            edge_counts[edge] = edge_counts.get(edge, 0) + 1
    boundary = [edge for edge, count in edge_counts.items() if count == 1]
    adjacency: dict[int, list[int]] = {}
    for first, second in boundary:
        adjacency.setdefault(first, []).append(second)
        adjacency.setdefault(second, []).append(first)

    chains = []
    for first, second in (
        (coarse_face[0], coarse_face[1]),
        (coarse_face[1], coarse_face[2]),
        (coarse_face[2], coarse_face[0]),
    ):
        start, stop = int(first), int(second)
        chain = [start]
        previous = -1
        current = start
        while current != stop:
            options = [item for item in adjacency[current] if item != previous]
            if not options:
                raise RuntimeError("geodesic patch boundary is disconnected")
            following = stop if stop in options else options[0]
            chain.append(following)
            previous, current = current, following
        chains.append(chain)
    return chains


def _fan_polygon(indices: list[int]) -> list[tuple[int, int, int]]:
    return [
        (indices[0], indices[index], indices[index + 1])
        for index in range(1, len(indices) - 1)
    ]


@reported_stage("build deposition mesh", describe=_describe_mesh)
def build_geodesic_deposition_mesh(
    radial_boundaries,
    *,
    maximum_angular_cells: int,
) -> SimplicialDepositionMesh:
    """Build adaptive geodesic surfaces joined by tetrahedra.

    The angular control is an approximate upper bound on cells around a great
    circle. Hierarchical icosahedral levels give roughly ``10 * 2**level``.
    """
    radii = _validate_radial_boundaries(radial_boundaries)
    maximum = _validate_integer(maximum_angular_cells, "maximum_angular_cells", 10)
    maximum_level = max(0, math.floor(math.log2(maximum / 10.0)))
    levels, parent_maps = _surface_levels(maximum_level)

    desired = maximum * radii / radii[-1]
    radial_levels = np.zeros(radii.size, dtype=np.int32)
    positive = desired >= 10.0
    radial_levels[positive] = np.floor(np.log2(desired[positive] / 10.0)).astype(
        np.int32
    )
    radial_levels = np.clip(radial_levels, 0, maximum_level)
    radial_levels = np.maximum.accumulate(radial_levels)
    surface_faces = np.asarray(
        [
            levels[int(level)][1].shape[0] if radius > 0.0 else 0
            for radius, level in zip(radii, radial_levels, strict=True)
        ],
        dtype=np.int32,
    )
    angular_counts = np.asarray(
        [
            10 * (1 << int(level)) if radius > 0.0 else 0
            for radius, level in zip(radii, radial_levels, strict=True)
        ],
        dtype=np.int32,
    )

    vertices: list[np.ndarray] = []
    surface_indices: list[np.ndarray | None] = []
    for radius, level in zip(radii, radial_levels, strict=True):
        if radius == 0.0:
            surface_indices.append(None)
            vertices.append(np.zeros(3, dtype=np.float64))
            continue
        directions = levels[int(level)][0]
        start = len(vertices)
        vertices.extend(radius * directions)
        surface_indices.append(
            np.arange(start, start + directions.shape[0], dtype=np.int32)
        )

    tetrahedra: list[tuple[int, int, int, int]] = []
    radial_layers: list[int] = []
    origin_index = 0 if radii[0] == 0.0 else None
    progress = ProgressTracker("build deposition mesh", radii.size - 1, "radial layers")
    for layer in range(radii.size - 1):
        inner_indices = surface_indices[layer]
        outer_indices = surface_indices[layer + 1]
        assert outer_indices is not None
        outer_level = int(radial_levels[layer + 1])
        _, outer_surface_faces = levels[outer_level]
        if inner_indices is None:
            assert origin_index is not None
            for face in outer_surface_faces:
                tetrahedra.append(
                    (origin_index, *(int(outer_indices[index]) for index in face))
                )
                radial_layers.append(layer)
            progress.update(layer + 1)
            continue

        inner_level = int(radial_levels[layer])
        _, inner_surface_faces = levels[inner_level]
        ancestors = np.arange(outer_surface_faces.shape[0], dtype=np.int32)
        for level in range(outer_level, inner_level, -1):
            parent = parent_maps[level]
            assert parent is not None
            ancestors = parent[ancestors]

        for coarse_index, coarse_face in enumerate(inner_surface_faces):
            patch_faces = outer_surface_faces[ancestors == coarse_index]
            boundary_triangles: list[tuple[int, int, int]] = [
                tuple(int(inner_indices[index]) for index in coarse_face)
            ]
            boundary_triangles.extend(
                tuple(int(outer_indices[index]) for index in face)
                for face in patch_faces
            )
            for chain in _boundary_edge_chains(coarse_face, patch_faces):
                if chain[0] > chain[-1]:
                    chain = chain[::-1]
                inner_start = int(inner_indices[chain[0]])
                inner_stop = int(inner_indices[chain[-1]])
                outer_chain = [int(outer_indices[index]) for index in chain]
                polygon = [inner_start, *outer_chain, inner_stop]
                boundary_triangles.extend(_fan_polygon(polygon))

            boundary_vertices = np.unique(np.asarray(boundary_triangles))
            interior = np.mean(
                np.asarray([vertices[index] for index in boundary_vertices]), axis=0
            )
            centre_index = len(vertices)
            vertices.append(interior)
            for triangle in boundary_triangles:
                tetrahedra.append((centre_index, *triangle))
                radial_layers.append(layer)
        progress.update(layer + 1)

    return _make_mesh(
        vertices,
        tetrahedra,
        radial_layers,
        radii,
        surface_faces,
        angular_counts,
        dimension=3,
        inactive_measure=1.0,
        topology="adaptive_geodesic",
    )


def build_circular_deposition_mesh_from_grid(
    grid: Grid,
    *,
    maximum_angular_cells: int | None = None,
    angular_offset: float | None = None,
) -> SimplicialDepositionMesh:
    """Build an adaptive circular mesh using a cylindrical grid's radii."""
    if grid.dimensions != 2 or grid.geom is not Geometry.CYLINDRICAL:
        raise ValueError("circular deposition meshes require a 2-D cylindrical grid")
    phi = np.asarray(grid.yb)
    if not np.isclose(phi[-1] - phi[0], 2.0 * np.pi):
        raise ValueError("circular deposition meshes require a full-azimuth grid")
    maximum = grid.ncells[1] if maximum_angular_cells is None else maximum_angular_cells
    offset = float(phi[0]) if angular_offset is None else angular_offset
    return build_circular_deposition_mesh(
        np.asarray(grid.xb),
        maximum_angular_cells=maximum,
        inactive_length_m=float(grid.inactive_measure),
        angular_offset=offset,
    )


def build_linear_deposition_mesh_from_grid(grid: Grid) -> SimplicialDepositionMesh:
    """Build a line mesh from a one-dimensional Cartesian hydro grid."""
    if grid.dimensions != 1 or grid.geom is not Geometry.CARTESIAN:
        raise ValueError("linear deposition meshes require a 1-D Cartesian grid")
    return build_linear_deposition_mesh(
        np.asarray(grid.xb),
        inactive_area_m2=float(grid.inactive_measure),
    )


def build_geodesic_deposition_mesh_from_grid(
    grid: Grid,
    *,
    maximum_angular_cells: int,
) -> SimplicialDepositionMesh:
    """Build adaptive geodesic surfaces using a spherical grid's radii."""
    if grid.dimensions != 3 or grid.geom is not Geometry.SPHERICAL:
        raise ValueError("geodesic deposition meshes require a 3-D spherical grid")
    phi = np.asarray(grid.yb)
    theta = np.asarray(grid.zb)
    if not (
        np.isclose(phi[-1] - phi[0], 2.0 * np.pi)
        and np.isclose(theta[0], 0.0)
        and np.isclose(theta[-1], np.pi)
    ):
        raise ValueError("geodesic deposition meshes require a full-sphere grid")
    return build_geodesic_deposition_mesh(
        np.asarray(grid.xb), maximum_angular_cells=maximum_angular_cells
    )


__all__ = [
    "SimplicialDepositionMesh",
    "build_cartesian_deposition_mesh_from_grid",
    "build_circular_deposition_mesh",
    "build_circular_deposition_mesh_from_grid",
    "build_geodesic_deposition_mesh",
    "build_geodesic_deposition_mesh_from_grid",
    "build_linear_deposition_mesh",
    "build_linear_deposition_mesh_from_grid",
]
