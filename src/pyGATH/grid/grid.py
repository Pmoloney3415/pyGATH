"""Rectilinear grid construction and JAX-compatible hydro interpolation."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from .composition import PlasmaComposition
from .geometry import Geometry, convert_positions, convert_vectors
from .hydro import HydroFields, HydroState, SafeHydroState, VertexHydroFields
from .initial_conditions import resolve as resolve_initial_condition


@dataclass(frozen=True)
class GradedAxis:
    """Description of smoothly varying cell sizes along one spatial axis.

    ``boundaries`` contains the internal physical positions separating regions.
    There must be one more target ``cell_sizes`` value than boundaries, and one
    ``transition_widths`` value per boundary. A transition width is the full
    distance over which the tanh blend moves from 10% to 90%.
    """

    boundaries: Sequence[float]
    cell_sizes: Sequence[float]
    transition_widths: Sequence[float]


@dataclass(frozen=True)
class GridCoordinates:
    """Coordinate-only grid view passed to initial-condition functions."""

    geom: Geometry
    dimensions: int
    xb: Any
    yb: Any
    zb: Any
    xc: Any
    yc: Any
    zc: Any

    @property
    def ncells(self) -> tuple[int, int, int]:
        return (self.xb.size - 1, self.yb.size - 1, self.zb.size - 1)

    @property
    def vertex_shape(self) -> tuple[int, int, int]:
        return (self.xb.size, self.yb.size, self.zb.size)

    def vertex_mesh(self):
        """Return broadcast vertex coordinates with ``ij`` indexing."""
        return jnp.meshgrid(self.xb, self.yb, self.zb, indexing="ij")

    @property
    def active_axes(self) -> tuple[int, ...]:
        """Indices of coordinates resolved by this simulation."""
        return tuple(range(self.dimensions))

    @property
    def inactive_axes(self) -> tuple[int, ...]:
        """Indices of invariant reference-extrusion coordinates."""
        return tuple(range(self.dimensions, 3))


def _validate_dimensions(geom: Geometry, dimensions: int) -> int:
    if isinstance(dimensions, bool) or not isinstance(dimensions, (int, np.integer)):
        raise TypeError("dimensions must be an integer")
    dimensions = int(dimensions)
    if dimensions not in (1, 2, 3):
        raise ValueError("dimensions must be one, two, or three")
    supported = {
        1: (Geometry.CARTESIAN,),
        2: (Geometry.CARTESIAN, Geometry.CYLINDRICAL),
        3: tuple(Geometry),
    }[dimensions]
    if geom not in supported:
        choices = ", ".join(item.value for item in supported)
        raise ValueError(
            f"{dimensions}-D grids support only the following geometries: {choices}"
        )
    return dimensions


def _expand_reduced_grid_inputs(
    geom: Geometry,
    dimensions: int,
    extents,
    ncells,
    graded_axes,
    inactive_axis_lengths_m,
):
    """Append one centred reference cell for every inactive direction."""
    dimensions = _validate_dimensions(geom, dimensions)
    extents = tuple(extents)
    ncells = tuple(ncells)
    if len(extents) != dimensions:
        raise ValueError(
            f"{dimensions}-D extents must contain exactly {dimensions} entries"
        )
    if len(ncells) != dimensions:
        raise ValueError(
            f"{dimensions}-D ncells must contain exactly {dimensions} entries"
        )
    if graded_axes is None:
        graded_axes = (None,) * dimensions
    else:
        graded_axes = tuple(graded_axes)
        if len(graded_axes) != dimensions:
            raise ValueError(
                f"{dimensions}-D graded_axes must contain exactly {dimensions} entries"
            )

    ninactive = 3 - dimensions
    if inactive_axis_lengths_m is None:
        inactive_lengths = (1.0,) * ninactive
    else:
        inactive_lengths = tuple(inactive_axis_lengths_m)
        if len(inactive_lengths) != ninactive:
            raise ValueError(
                "inactive_axis_lengths_m must contain one value per inactive axis"
            )
        if any(
            isinstance(length, bool)
            or not isinstance(length, (int, float, np.integer, np.floating))
            for length in inactive_lengths
        ):
            raise TypeError("inactive-axis lengths must be numbers")
        if any(not np.isfinite(length) or length <= 0.0 for length in inactive_lengths):
            raise ValueError("inactive-axis lengths must be finite and positive")
        inactive_lengths = tuple(float(length) for length in inactive_lengths)

    inactive_extents = tuple(
        (-0.5 * length, 0.5 * length) for length in inactive_lengths
    )
    return (
        dimensions,
        (*extents, *inactive_extents),
        (*ncells, *((1,) * ninactive)),
        (*graded_axes, *((None,) * ninactive)),
    )


def _validate_extents(geom: Geometry, extents) -> np.ndarray:
    extents = np.asarray(extents, dtype=float)
    if extents.shape != (3, 2):
        raise ValueError(f"extents must have shape (3, 2), got {extents.shape}")
    if not np.all(np.isfinite(extents)):
        raise ValueError("extents must contain only finite values")
    if np.any(extents[:, 1] <= extents[:, 0]):
        raise ValueError("each upper extent must be greater than its lower extent")

    if geom in (Geometry.CYLINDRICAL, Geometry.SPHERICAL) and extents[0, 0] < 0:
        raise ValueError("radial extents cannot include negative radii")

    if geom in (Geometry.CYLINDRICAL, Geometry.SPHERICAL):
        phi_min, phi_max = extents[1]
        if phi_min < -np.pi or phi_max > np.pi:
            raise ValueError("azimuthal extents must lie within [-pi, pi]")

    if geom is Geometry.SPHERICAL:
        theta_min, theta_max = extents[2]
        if theta_min < 0 or theta_max > np.pi:
            raise ValueError("polar extents must lie within [0, pi]")
    return extents


def _validate_graded_axis(spec: GradedAxis, extent: np.ndarray) -> None:
    boundaries = np.asarray(spec.boundaries, dtype=float)
    cell_sizes = np.asarray(spec.cell_sizes, dtype=float)
    transition_widths = np.asarray(spec.transition_widths, dtype=float)
    if boundaries.ndim != 1 or cell_sizes.ndim != 1 or transition_widths.ndim != 1:
        raise ValueError("graded-axis inputs must be one-dimensional")
    if cell_sizes.size != boundaries.size + 1:
        raise ValueError("cell_sizes must contain one more value than boundaries")
    if transition_widths.size != boundaries.size:
        raise ValueError("transition_widths must contain one value per boundary")
    if not (
        np.all(np.isfinite(boundaries))
        and np.all(np.isfinite(cell_sizes))
        and np.all(np.isfinite(transition_widths))
    ):
        raise ValueError("graded-axis inputs must contain only finite values")
    if np.any(cell_sizes <= 0):
        raise ValueError("graded-axis cell sizes must be positive")
    if np.any(transition_widths <= 0):
        raise ValueError("graded-axis transition widths must be positive")
    if boundaries.size and (
        boundaries[0] <= extent[0]
        or boundaries[-1] >= extent[1]
        or np.any(np.diff(boundaries) <= 0)
    ):
        raise ValueError("graded-axis boundaries must increase inside the extent")


def _graded_cell_size(position: float, spec: GradedAxis) -> float:
    boundaries = np.asarray(spec.boundaries, dtype=float)
    sizes = np.asarray(spec.cell_sizes, dtype=float)
    widths = np.asarray(spec.transition_widths, dtype=float)
    cell_size = sizes[0]
    ten_to_ninety_factor = 2.0 * np.arctanh(0.8)
    for index, boundary in enumerate(boundaries):
        tanh_scale = widths[index] / ten_to_ninety_factor
        blend = 0.5 * (1.0 + np.tanh((position - boundary) / tanh_scale))
        cell_size += (sizes[index + 1] - sizes[index]) * blend
    return float(cell_size)


def _build_graded_axis(extent: np.ndarray, spec: GradedAxis, dtype) -> jax.Array:
    _validate_graded_axis(spec, extent)
    lower, upper = extent
    cursor = float(lower)
    widths: list[float] = []
    while cursor < upper:
        width = _graded_cell_size(cursor, spec)
        if not np.isfinite(width) or width <= 0:
            raise ValueError(
                "overlapping graded-axis transitions produced a non-positive cell size"
            )
        widths.append(width)
        cursor += width
        if len(widths) > 1_000_000:
            raise ValueError("graded axis would contain more than 1,000,000 cells")

    width_array = jnp.asarray(widths, dtype=dtype)
    return jnp.asarray(lower, dtype=dtype) + jnp.concatenate(
        (jnp.zeros(1, dtype=dtype), jnp.cumsum(width_array))
    )


def _build_uniform_axis(extent: np.ndarray, ncells: int, dtype) -> jax.Array:
    if isinstance(ncells, bool) or not isinstance(ncells, (int, np.integer)):
        raise TypeError("uniform-axis cell counts must be integers")
    if ncells < 1:
        raise ValueError("every uniform axis must contain at least one cell")
    return jnp.linspace(extent[0], extent[1], ncells + 1, dtype=dtype)


def _is_uniform(vertices) -> bool:
    widths = np.diff(np.asarray(vertices, dtype=float))
    if widths.size <= 1:
        return True
    scale = max(1.0, float(np.max(np.abs(vertices))))
    tolerance = np.finfo(widths.dtype).eps * scale * 64.0
    return bool(np.allclose(widths, widths[0], rtol=1.0e-6, atol=tolerance))


def _coerce_hydro(
    fields: HydroFields,
    shape,
    dtype,
    composition: PlasmaComposition,
) -> VertexHydroFields:
    if not isinstance(fields, HydroFields):
        raise TypeError("initial-condition functions must return HydroFields")

    def scalar_field(value, name):
        try:
            return jnp.broadcast_to(jnp.asarray(value, dtype=dtype), shape)
        except ValueError as error:
            raise ValueError(
                f"{name} cannot be broadcast to vertex shape {shape}"
            ) from error

    try:
        velocity = jnp.broadcast_to(
            jnp.asarray(fields.velocity, dtype=dtype), (*shape, 3)
        )
    except ValueError as error:
        raise ValueError(
            f"velocity cannot be broadcast to vertex shape {(*shape, 3)}"
        ) from error

    ne = scalar_field(fields.ne, "ne")
    return VertexHydroFields(
        ne=ne,
        ni=ne / composition.mean_charge,
        Te=scalar_field(fields.Te, "Te"),
        Ti=scalar_field(fields.Ti, "Ti"),
        velocity=velocity,
    )


def _validate_regular_density(geom: Geometry, coordinates: GridCoordinates, ne) -> None:
    """Check scalar density regularity at duplicated coordinate singularities."""
    values = np.asarray(ne)
    xb = np.asarray(coordinates.xb)
    zb = np.asarray(coordinates.zb)

    if (
        geom is Geometry.CYLINDRICAL
        and np.isclose(xb[0], 0.0)
        and not np.allclose(values[0], values[0, 0:1, :])
    ):
        raise ValueError("ne must be independent of phi on the cylindrical axis")

    if geom is Geometry.SPHERICAL:
        if np.isclose(xb[0], 0.0) and not np.allclose(values[0], values[0, 0, 0]):
            raise ValueError("ne must have one value across all angles at r=0")
        if np.isclose(zb[0], 0.0) and not np.allclose(
            values[:, :, 0], values[:, 0:1, 0]
        ):
            raise ValueError("ne must be independent of phi at theta=0")
        if np.isclose(zb[-1], np.pi) and not np.allclose(
            values[:, :, -1], values[:, 0:1, -1]
        ):
            raise ValueError("ne must be independent of phi at theta=pi")


def _validate_reduced_hydro(
    coordinates: GridCoordinates,
    hydro: VertexHydroFields,
    safe: SafeHydroState,
) -> None:
    """Require reduced hydro and safe states to be invariant out of simulation."""
    if coordinates.dimensions == 3:
        return

    def invariant(values, axis: int) -> bool:
        array = np.asarray(values)
        reference = np.take(array, 0, axis=axis)
        reference = np.expand_dims(reference, axis=axis)
        return bool(np.allclose(array, reference, rtol=1.0e-12, atol=0.0))

    for name in ("ne", "ni", "Te", "Ti", "velocity"):
        values = getattr(hydro, name)
        for axis in coordinates.inactive_axes:
            if not invariant(values, axis):
                raise ValueError(
                    f"{name} must be invariant along inactive grid axis {axis}"
                )

    inactive_components = tuple(range(coordinates.dimensions, 3))
    velocity = np.asarray(hydro.velocity)
    if inactive_components and not np.allclose(
        velocity[..., inactive_components], 0.0, rtol=0.0, atol=0.0
    ):
        raise ValueError("velocity must have zero inactive physical components")
    for name in ("grad_ne", "velocity"):
        values = np.asarray(getattr(safe, name))
        if inactive_components and not np.allclose(
            values[list(inactive_components)], 0.0, rtol=0.0, atol=0.0
        ):
            raise ValueError(
                f"safe-state {name} must have zero inactive physical components"
            )


def _density_gradient(geom: Geometry, coordinates: GridCoordinates, ne):
    d_first = jnp.gradient(ne, coordinates.xb, axis=0)
    d_second = jnp.gradient(ne, coordinates.yb, axis=1)
    d_third = jnp.gradient(ne, coordinates.zb, axis=2)

    if geom is Geometry.CARTESIAN:
        return jnp.stack((d_first, d_second, d_third), axis=-1)

    radius = coordinates.xb[:, None, None]
    safe_radius = jnp.where(radius == 0, 1.0, radius)
    if geom is Geometry.CYLINDRICAL:
        angular = jnp.where(radius == 0, 0.0, d_second / safe_radius)
        return jnp.stack((d_first, angular, d_third), axis=-1)

    theta = coordinates.zb[None, None, :]
    sin_theta = jnp.sin(theta)
    angular_tolerance = 32.0 * jnp.finfo(ne.dtype).eps
    regular_angle = jnp.abs(sin_theta) > angular_tolerance
    regular_radius = radius != 0
    safe_metric = jnp.where(regular_radius & regular_angle, radius * sin_theta, 1.0)
    phi_component = jnp.where(
        regular_radius & regular_angle, d_second / safe_metric, 0.0
    )
    theta_component = jnp.where(
        regular_radius & regular_angle, d_third / safe_radius, 0.0
    )
    return jnp.stack((d_first, phi_component, theta_component), axis=-1)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class Grid:
    """Hydrodynamic data on a rectilinear mesh with three-axis storage.

    Use :meth:`create` to construct a grid. Coordinate and vector fields are
    stored in ``geom`` coordinates; interpolation accepts Cartesian positions
    and returns vectors in Cartesian components.
    """

    geom: Geometry
    dimensions: int
    xb: Any
    yb: Any
    zb: Any
    xc: Any
    yc: Any
    zc: Any
    hydro: VertexHydroFields
    grad_ne: Any
    safe_state: SafeHydroState
    composition: PlasmaComposition
    ncells: tuple[int, int, int]
    is_uniform: tuple[bool, bool, bool]

    @classmethod
    def create(
        cls,
        *,
        geom: Geometry | str,
        extents: Sequence[Sequence[float]],
        ncells: Sequence[int | None],
        dimensions: int = 3,
        inactive_axis_lengths_m: Sequence[float] | None = None,
        initial_condition: str | Callable[[GridCoordinates], HydroFields] = "uniform",
        graded_axes: Sequence[GradedAxis | None] | None = None,
        safe_state: SafeHydroState | None = None,
        composition: PlasmaComposition | None = None,
        initial_condition_parameters: dict[str, Any] | None = None,
    ) -> Grid:
        """Construct coordinates, initialise hydro fields, and derive ``grad_ne``.

        Each active direction must have either an integer entry in ``ncells``
        or a :class:`GradedAxis` entry in ``graded_axes``. Inactive directions
        are generated as centred one-cell reference extrusions. Angular grid
        directions must always use uniform spacing.
        """
        geom = Geometry.parse(geom)
        dimensions, extents, ncells, graded_axes = _expand_reduced_grid_inputs(
            geom,
            dimensions,
            extents,
            ncells,
            graded_axes,
            inactive_axis_lengths_m,
        )
        extents_array = _validate_extents(geom, extents)

        angular_axes = {
            Geometry.CARTESIAN: (),
            Geometry.CYLINDRICAL: (1,),
            Geometry.SPHERICAL: (1, 2),
        }[geom]
        for axis in angular_axes:
            if graded_axes[axis] is not None:
                raise ValueError("angular grid directions must use uniform spacing")

        dtype = jnp.dtype(jnp.float64)
        vertices = []
        for axis, (count, graded) in enumerate(zip(ncells, graded_axes, strict=True)):
            if graded is None:
                if count is None:
                    raise ValueError(
                        f"axis {axis} requires ncells when no GradedAxis is supplied"
                    )
                vertices.append(_build_uniform_axis(extents_array[axis], count, dtype))
            else:
                if count is not None:
                    raise ValueError(
                        f"axis {axis} cannot specify both ncells and a GradedAxis"
                    )
                vertices.append(_build_graded_axis(extents_array[axis], graded, dtype))

        xb, yb, zb = vertices
        xc = 0.5 * (xb[:-1] + xb[1:])
        yc = 0.5 * (yb[:-1] + yb[1:])
        zc = 0.5 * (zb[:-1] + zb[1:])
        coordinates = GridCoordinates(geom, dimensions, xb, yb, zb, xc, yc, zc)
        composition = composition or PlasmaComposition.create()
        parameters = initial_condition_parameters or {}
        initialise = resolve_initial_condition(initial_condition, **parameters)
        hydro = _coerce_hydro(
            initialise(coordinates),
            coordinates.vertex_shape,
            dtype,
            composition,
        )
        _validate_regular_density(geom, coordinates, hydro.ne)
        safe = (safe_state or SafeHydroState()).as_arrays(dtype)
        if safe.grad_ne.shape != (3,) or safe.velocity.shape != (3,):
            raise ValueError("safe-state vectors must have shape (3,)")
        _validate_reduced_hydro(coordinates, hydro, safe)
        grad_ne = _density_gradient(geom, coordinates, hydro.ne)
        final_ncells = coordinates.ncells
        uniform = tuple(_is_uniform(axis) for axis in vertices)
        return cls(
            geom=geom,
            dimensions=dimensions,
            xb=xb,
            yb=yb,
            zb=zb,
            xc=xc,
            yc=yc,
            zc=zc,
            hydro=hydro,
            grad_ne=grad_ne,
            safe_state=safe,
            composition=composition,
            ncells=final_ncells,
            is_uniform=uniform,
        )

    @property
    def vertex_shape(self) -> tuple[int, int, int]:
        return (self.xb.size, self.yb.size, self.zb.size)

    @property
    def active_axes(self) -> tuple[int, ...]:
        return tuple(range(self.dimensions))

    @property
    def inactive_axes(self) -> tuple[int, ...]:
        return tuple(range(self.dimensions, 3))

    @property
    def inactive_axis_lengths_m(self):
        extents = self.extents
        return extents[self.dimensions :, 1] - extents[self.dimensions :, 0]

    @property
    def inactive_measure(self):
        lengths = self.inactive_axis_lengths_m
        return (
            jnp.prod(lengths) if lengths.size else jnp.asarray(1.0, dtype=self.xb.dtype)
        )

    @property
    def extents(self):
        """Actual grid extents, including any graded-axis upper overshoot."""
        return jnp.stack(
            (
                jnp.stack((self.xb[0], self.xb[-1])),
                jnp.stack((self.yb[0], self.yb[-1])),
                jnp.stack((self.zb[0], self.zb[-1])),
            )
        )

    def interpolate(self, cartesian_positions) -> HydroState:
        """Sample all hydro fields at Cartesian positions shaped ``(..., 3)``."""
        return interpolate_hydro(self, cartesian_positions)

    def contains(self, cartesian_positions):
        """Return whether Cartesian positions lie within the grid extents."""
        return contains(self, cartesian_positions)

    def tree_flatten(self):
        children = (
            self.xb,
            self.yb,
            self.zb,
            self.xc,
            self.yc,
            self.zc,
            self.hydro,
            self.grad_ne,
            self.safe_state,
            self.composition,
        )
        auxiliary = (self.geom, self.dimensions, self.ncells, self.is_uniform)
        return children, auxiliary

    @classmethod
    def tree_unflatten(cls, auxiliary, children):
        geom, dimensions, ncells, is_uniform = auxiliary
        xb, yb, zb, xc, yc, zc, hydro, grad_ne, safe_state, composition = children
        return cls(
            geom,
            dimensions,
            xb,
            yb,
            zb,
            xc,
            yc,
            zc,
            hydro,
            grad_ne,
            safe_state,
            composition,
            ncells,
            is_uniform,
        )


def _locate_axis(vertices, query, is_uniform: bool):
    finite_query = jnp.where(jnp.isfinite(query), query, vertices[0])
    if is_uniform:
        scaled = (finite_query - vertices[0]) / (vertices[1] - vertices[0])
        lower = jnp.floor(scaled).astype(jnp.int32)
    else:
        lower = jnp.searchsorted(vertices, finite_query, side="right") - 1
    lower = jnp.clip(lower, 0, vertices.size - 2)
    left = vertices[lower]
    right = vertices[lower + 1]
    weight = (finite_query - left) / (right - left)
    inside = jnp.isfinite(query) & (query >= vertices[0]) & (query <= vertices[-1])
    return lower, weight, inside


def _trilinear(field, indices, weights):
    ix, iy, iz = indices
    tx, ty, tz = weights
    trailing_dimensions = field.ndim - 3

    def expand(weight):
        return weight.reshape(weight.shape + (1,) * trailing_dimensions)

    tx = expand(tx)
    ty = expand(ty)
    tz = expand(tz)
    f000 = field[ix, iy, iz]
    f100 = field[ix + 1, iy, iz]
    f010 = field[ix, iy + 1, iz]
    f110 = field[ix + 1, iy + 1, iz]
    f001 = field[ix, iy, iz + 1]
    f101 = field[ix + 1, iy, iz + 1]
    f011 = field[ix, iy + 1, iz + 1]
    f111 = field[ix + 1, iy + 1, iz + 1]
    lower_z = (
        (1.0 - tx) * (1.0 - ty) * f000
        + tx * (1.0 - ty) * f100
        + (1.0 - tx) * ty * f010
        + tx * ty * f110
    )
    upper_z = (
        (1.0 - tx) * (1.0 - ty) * f001
        + tx * (1.0 - ty) * f101
        + (1.0 - tx) * ty * f011
        + tx * ty * f111
    )
    return (1.0 - tz) * lower_z + tz * upper_z


def interpolate_hydro(grid: Grid, cartesian_positions) -> HydroState:
    """Trilinearly interpolate grid hydro at batched Cartesian positions."""
    cartesian_positions = jnp.asarray(cartesian_positions, dtype=grid.xb.dtype)
    if cartesian_positions.ndim < 1 or cartesian_positions.shape[-1] != 3:
        raise ValueError(
            f"cartesian_positions must have shape (..., 3), got "
            f"{cartesian_positions.shape}"
        )
    grid_positions = convert_positions(
        cartesian_positions, Geometry.CARTESIAN, grid.geom
    )
    if grid.dimensions < 3:
        reference_centres = jnp.stack((grid.xc[0], grid.yc[0], grid.zc[0]))
        active = jnp.arange(3) < grid.dimensions
        grid_positions = jnp.where(active, grid_positions, reference_centres)
    axes = (grid.xb, grid.yb, grid.zb)
    locations = tuple(
        _locate_axis(axis, grid_positions[..., index], grid.is_uniform[index])
        for index, axis in enumerate(axes)
    )
    indices = tuple(location[0] for location in locations)
    weights = tuple(location[1] for location in locations)
    inside = locations[0][2] & locations[1][2] & locations[2][2]
    inside &= jnp.all(jnp.isfinite(cartesian_positions), axis=-1)

    ne = _trilinear(grid.hydro.ne, indices, weights)
    ni = _trilinear(grid.hydro.ni, indices, weights)
    Te = _trilinear(grid.hydro.Te, indices, weights)
    Ti = _trilinear(grid.hydro.Ti, indices, weights)
    local_gradient = _trilinear(grid.grad_ne, indices, weights)
    local_velocity = _trilinear(grid.hydro.velocity, indices, weights)
    gradient = convert_vectors(
        local_gradient, grid_positions, grid.geom, Geometry.CARTESIAN
    )
    velocity = convert_vectors(
        local_velocity, grid_positions, grid.geom, Geometry.CARTESIAN
    )

    safe = grid.safe_state
    return HydroState(
        ne=jnp.where(inside, ne, safe.ne),
        ni=jnp.where(inside, ni, safe.ne / grid.composition.mean_charge),
        Te=jnp.where(inside, Te, safe.Te),
        Ti=jnp.where(inside, Ti, safe.Ti),
        grad_ne=jnp.where(inside[..., None], gradient, safe.grad_ne),
        velocity=jnp.where(inside[..., None], velocity, safe.velocity),
        inside=inside,
    )


def contains(grid: Grid, cartesian_positions):
    """Return a JAX boolean mask for Cartesian positions inside ``grid``.

    A small dtype-scaled tolerance keeps rays initialized exactly on a grid
    face from being classified as outside because of coordinate-conversion
    roundoff. The function performs no hydro interpolation and is suitable for
    inexpensive use in a Diffrax event condition.
    """
    cartesian_positions = jnp.asarray(cartesian_positions, dtype=grid.xb.dtype)
    if cartesian_positions.ndim < 1 or cartesian_positions.shape[-1] != 3:
        raise ValueError(
            f"cartesian_positions must have shape (..., 3), got "
            f"{cartesian_positions.shape}"
        )
    grid_positions = convert_positions(
        cartesian_positions, Geometry.CARTESIAN, grid.geom
    )
    finite = jnp.all(jnp.isfinite(cartesian_positions), axis=-1)
    if grid.dimensions < 3:
        reference_centres = jnp.stack((grid.xc[0], grid.yc[0], grid.zc[0]))
        active = jnp.arange(3) < grid.dimensions
        grid_positions = jnp.where(active, grid_positions, reference_centres)
    extents = grid.extents
    scales = jnp.maximum(1.0, jnp.max(jnp.abs(extents), axis=1))
    tolerance = 512.0 * jnp.finfo(grid.xb.dtype).eps * scales
    above_lower = grid_positions >= extents[:, 0] - tolerance
    below_upper = grid_positions <= extents[:, 1] + tolerance
    return finite & jnp.all(above_lower & below_upper, axis=-1)
