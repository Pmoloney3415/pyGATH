"""Small Matplotlib helpers for grids and traced ray sheets."""

from __future__ import annotations

from collections.abc import Sequence
from itertools import combinations

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from pyGATH.fields import SimplicialField, SimplicialMesh
from pyGATH.grid import Geometry, Grid, convert_positions, convert_vectors
from pyGATH.raytracing import RAY_STATE_LAYOUT, RayTraceResult

_GRID_AXIS_NAMES = {
    Geometry.CARTESIAN: ("x", "y", "z"),
    Geometry.CYLINDRICAL: ("r", "phi", "z"),
    Geometry.SPHERICAL: ("r", "phi", "theta"),
}
_CARTESIAN_COMPONENTS = {"x": 0, "y": 1, "z": 2}
_SCALAR_FIELDS = {"ne": "ne", "ni": "ni", "te": "Te", "ti": "Ti"}
_VECTOR_FIELDS = {"grad_ne": "grad_ne", "velocity": "velocity"}


def _projection_indices(projection: str) -> tuple[int, int]:
    projection = projection.lower()
    if len(projection) != 2 or len(set(projection)) != 2:
        raise ValueError("projection must be one of 'xy', 'xz', or 'yz'")
    try:
        return tuple(_CARTESIAN_COMPONENTS[name] for name in projection)
    except KeyError as error:
        raise ValueError("projection must be one of 'xy', 'xz', or 'yz'") from error


def _slice_coordinates(
    grid: Grid,
    axes: tuple[str, str],
    index: int,
):
    axis_names = _GRID_AXIS_NAMES[grid.geom]
    if len(axes) != 2 or axes[0] == axes[1]:
        raise ValueError("axes must contain two different grid-axis names")
    try:
        plotted_axes = tuple(axis_names.index(name) for name in axes)
    except ValueError as error:
        choices = ", ".join(axis_names)
        raise ValueError(
            f"axes for this grid must be selected from {choices}"
        ) from error

    fixed_axis = next(axis for axis in range(3) if axis not in plotted_axes)
    vertices = (grid.xb, grid.yb, grid.zb)
    fixed_vertices = vertices[fixed_axis]
    if isinstance(index, bool) or not isinstance(index, int):
        raise TypeError("index must be an integer vertex index")
    if not -fixed_vertices.size <= index < fixed_vertices.size:
        raise IndexError(
            f"index {index} is outside the fixed {axis_names[fixed_axis]} vertex axis"
        )
    index %= fixed_vertices.size

    first, second = jnp.meshgrid(
        vertices[plotted_axes[0]], vertices[plotted_axes[1]], indexing="ij"
    )
    native_positions = jnp.empty((*first.shape, 3), dtype=grid.xb.dtype)
    native_positions = native_positions.at[..., plotted_axes[0]].set(first)
    native_positions = native_positions.at[..., plotted_axes[1]].set(second)
    native_positions = native_positions.at[..., fixed_axis].set(fixed_vertices[index])
    cartesian_positions = convert_positions(
        native_positions, grid.geom, Geometry.CARTESIAN
    )
    return native_positions, cartesian_positions, plotted_axes, fixed_axis, index


def _take_vertex_plane(values, plotted_axes, fixed_axis, index):
    plane = np.take(np.asarray(values), index, axis=fixed_axis)
    remaining_axes = [axis for axis in range(3) if axis != fixed_axis]
    permutation = [remaining_axes.index(axis) for axis in plotted_axes]
    if plane.ndim == 3:
        permutation.append(2)
    return np.transpose(plane, permutation)


def _hydro_vertex_values(
    grid,
    variable,
    component,
    native_positions,
    plotted_axes,
    fixed_axis,
    index,
):
    key = variable.lower()
    if key in _SCALAR_FIELDS:
        if component is not None:
            raise ValueError(
                f"component cannot be supplied for scalar field {variable!r}"
            )
        field = getattr(grid.hydro, _SCALAR_FIELDS[key])
        return _take_vertex_plane(field, plotted_axes, fixed_axis, index)

    if key not in _VECTOR_FIELDS:
        choices = ", ".join((*_SCALAR_FIELDS, *_VECTOR_FIELDS))
        raise ValueError(
            f"unknown hydro variable {variable!r}; expected one of {choices}"
        )
    if component is None:
        raise ValueError(f"component is required for vector field {variable!r}")
    try:
        component_index = _CARTESIAN_COMPONENTS[component.lower()]
    except (AttributeError, KeyError) as error:
        raise ValueError("component must be 'x', 'y', or 'z'") from error

    field = grid.grad_ne if key == "grad_ne" else grid.hydro.velocity
    local_vectors = _take_vertex_plane(field, plotted_axes, fixed_axis, index)
    cartesian_vectors = convert_vectors(
        jnp.asarray(local_vectors), native_positions, grid.geom, Geometry.CARTESIAN
    )
    return np.asarray(cartesian_vectors)[..., component_index]


def plot_hydro_slice(
    grid: Grid,
    variable: str,
    *,
    axes: tuple[str, str],
    index: int,
    component: str | None = None,
    projection: str = "xy",
    normalization: float = 1.0,
    ax=None,
    **pcolormesh_kwargs,
):
    """Plot one grid-vertex plane as cell averages in Cartesian coordinates.

    ``axes`` names the two varying native grid coordinates. ``index`` selects
    a vertex on the remaining grid axis. Scalar variables are ``ne``, ``ni``,
    ``Te``, and ``Ti``. Vector variables ``grad_ne`` and ``velocity`` require
    a Cartesian ``component``. Values are divided by ``normalization``.

    Returns ``(ax, mesh)``, where ``mesh`` is Matplotlib's ``QuadMesh``.
    """
    normalization = float(normalization)
    if not np.isfinite(normalization) or normalization <= 0.0:
        raise ValueError("normalization must be a finite positive number")
    projection_indices = _projection_indices(projection)
    native, cartesian, plotted_axes, fixed_axis, index = _slice_coordinates(
        grid, axes, index
    )
    vertex_values = _hydro_vertex_values(
        grid,
        variable,
        component,
        native,
        plotted_axes,
        fixed_axis,
        index,
    )
    cell_values = 0.25 * (
        vertex_values[:-1, :-1]
        + vertex_values[1:, :-1]
        + vertex_values[:-1, 1:]
        + vertex_values[1:, 1:]
    )
    cartesian = np.asarray(cartesian)

    if ax is None:
        _, ax = plt.subplots()
    pcolormesh_kwargs.setdefault("shading", "flat")
    mesh = ax.pcolormesh(
        cartesian[..., projection_indices[0]],
        cartesian[..., projection_indices[1]],
        cell_values / normalization,
        **pcolormesh_kwargs,
    )
    first_name, second_name = projection.lower()
    ax.set_xlabel(f"{first_name} [m]")
    ax.set_ylabel(f"{second_name} [m]")
    return ax, mesh


def plot_ray_trajectories(
    result: RayTraceResult,
    *,
    beam_index: int = 0,
    projection: str = "xy",
    ray_stride: int = 1,
    sheet_colors: Sequence[str] = ("tab:blue", "tab:orange"),
    ax=None,
    line_kwargs: dict | None = None,
    caustic_kwargs: dict | None = None,
):
    """Plot both ray sheets for one beam and scatter its detected caustics.

    Every primary ray is plotted by default. ``ray_stride`` applies to both
    transverse ray axes. Returns ``(ax, artists)``; ``artists`` contains line
    lists under ``sheet_1`` and ``sheet_2`` and the caustic scatter artist.
    """
    projection_indices = _projection_indices(projection)
    if isinstance(beam_index, bool) or not isinstance(beam_index, int):
        raise TypeError("beam_index must be an integer")
    if isinstance(ray_stride, bool) or not isinstance(ray_stride, int):
        raise TypeError("ray_stride must be an integer")
    if ray_stride < 1:
        raise ValueError("ray_stride must be positive")
    if len(sheet_colors) != 2:
        raise ValueError("sheet_colors must contain exactly two colors")

    fields = np.asarray(result.sheet_fields)
    if fields.ndim != 6 or fields.shape[1] != 2:
        raise ValueError(
            "result.sheet_fields must have shape (nbeams, 2, ..., nfields)"
        )
    if not -fields.shape[0] <= beam_index < fields.shape[0]:
        raise IndexError(f"beam_index {beam_index} is outside the trace result")
    beam_index %= fields.shape[0]
    positions = fields[
        beam_index,
        :,
        ::ray_stride,
        ::ray_stride,
        :,
        RAY_STATE_LAYOUT.position,
    ]

    if ax is None:
        _, ax = plt.subplots()
    base_line_kwargs = {} if line_kwargs is None else dict(line_kwargs)
    base_line_kwargs.pop("label", None)
    base_line_kwargs.setdefault("linewidth", 1.0)
    base_line_kwargs.setdefault("alpha", 0.8)
    artists = {"sheet_1": [], "sheet_2": [], "caustics": None}
    for sheet_index in range(2):
        key = f"sheet_{sheet_index + 1}"
        for first_index in range(positions.shape[1]):
            for second_index in range(positions.shape[2]):
                trajectory = positions[sheet_index, first_index, second_index]
                kwargs = dict(base_line_kwargs)
                kwargs["color"] = sheet_colors[sheet_index]
                if not artists[key]:
                    kwargs["label"] = f"Sheet {sheet_index + 1}"
                (line,) = ax.plot(
                    trajectory[:, projection_indices[0]],
                    trajectory[:, projection_indices[1]],
                    **kwargs,
                )
                artists[key].append(line)

    caustic_mask = np.asarray(result.has_caustic[beam_index])[
        ::ray_stride, ::ray_stride
    ]
    if np.any(caustic_mask):
        caustics = positions[0, ..., -1, :][caustic_mask]
        scatter_kwargs = {} if caustic_kwargs is None else dict(caustic_kwargs)
        scatter_kwargs.setdefault("color", "black")
        scatter_kwargs.setdefault("marker", "x")
        scatter_kwargs.setdefault("s", 35.0)
        scatter_kwargs.setdefault("zorder", 3)
        scatter_kwargs.setdefault("label", "Caustic")
        artists["caustics"] = ax.scatter(
            caustics[:, projection_indices[0]],
            caustics[:, projection_indices[1]],
            **scatter_kwargs,
        )

    first_name, second_name = projection.lower()
    ax.set_xlabel(f"{first_name} [m]")
    ax.set_ylabel(f"{second_name} [m]")
    return ax, artists


def plot_simplicial_mesh(
    simplicial_field: SimplicialField | SimplicialMesh,
    *,
    beam_index: int = 0,
    sheet_index: int = 0,
    simplex_stride: int = 1,
    ax=None,
):
    """Plot valid segment, triangle, or tetrahedron edges for one ray sheet."""
    mesh = (
        simplicial_field.mesh
        if isinstance(simplicial_field, SimplicialField)
        else simplicial_field
    )
    if not isinstance(mesh, SimplicialMesh):
        raise TypeError("simplicial_field must be a SimplicialField or mesh")
    for name, index, size in (
        ("beam_index", beam_index, mesh.nbeams),
        ("sheet_index", sheet_index, mesh.nsheets),
    ):
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError(f"{name} must be an integer")
        if not -size <= index < size:
            raise IndexError(f"{name} {index} is outside a size of {size}")
    if isinstance(simplex_stride, bool) or not isinstance(simplex_stride, int):
        raise TypeError("simplex_stride must be an integer")
    if simplex_stride < 1:
        raise ValueError("simplex_stride must be positive")
    beam_index %= mesh.nbeams
    sheet_index %= mesh.nsheets

    positions = np.asarray(mesh.vertex_positions[beam_index, sheet_index])
    connectivity = np.asarray(mesh.connectivity)
    valid = np.asarray(mesh.valid[beam_index, sheet_index], dtype=bool)
    simplices = connectivity[valid][::simplex_stride]
    if simplices.size == 0:
        raise ValueError("the selected beam sheet has no valid simplices")
    local_edges = np.asarray(
        tuple(combinations(range(mesh.dimension + 1), 2)), dtype=np.int32
    )
    mesh_edges = np.unique(
        np.sort(simplices[:, local_edges].reshape((-1, 2)), axis=-1), axis=0
    )
    mesh_vertices = np.unique(simplices)
    if ax is None:
        figure = plt.figure()
        ax = figure.add_subplot(111, projection="3d")
    if not hasattr(ax, "get_zlim"):
        raise TypeError("ax must be a three-dimensional Matplotlib axis")
    edges = Line3DCollection(
        positions[mesh_edges], colors="grey", linewidths=0.65, alpha=0.4
    )
    ax.add_collection3d(edges)
    vertices = ax.scatter(
        *positions[mesh_vertices].T,
        color="black",
        s=7.0,
        alpha=0.7,
        depthshade=False,
    )
    plotted = positions[mesh_vertices]
    ax.auto_scale_xyz(plotted[:, 0], plotted[:, 1], plotted[:, 2])
    coordinate_range = np.ptp(plotted, axis=0)
    positive = coordinate_range[coordinate_range > 0.0]
    fallback = np.min(positive) if positive.size else 1.0
    ax.set_box_aspect(np.where(coordinate_range > 0.0, coordinate_range, fallback))
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    return ax, {"background_edges": edges, "background_vertices": vertices}


__all__ = [
    "plot_hydro_slice",
    "plot_ray_trajectories",
    "plot_simplicial_mesh",
]
