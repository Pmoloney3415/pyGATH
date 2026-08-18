"""Physical intersections between Cartesian rays and supported grid domains."""

from __future__ import annotations

import numpy as np

from pyGATH.grid import Geometry, Grid


def _characteristic_length(
    geometry: Geometry, dimensions: int, extents: np.ndarray
) -> float:
    if geometry is Geometry.CARTESIAN:
        active = extents[:dimensions]
        return float(np.linalg.norm(active[:, 1] - active[:, 0]))
    if geometry is Geometry.CYLINDRICAL:
        diameter = 2.0 * extents[0, 1]
        if dimensions == 2:
            return float(diameter)
        axial_length = extents[2, 1] - extents[2, 0]
        return float(np.hypot(diameter, axial_length))
    return float(2.0 * extents[0, 1])


def grid_characteristic_length(grid: Grid) -> float:
    """Return a conservative physical length scale for launch-plane padding."""
    extents = np.asarray(grid.extents, dtype=np.float64)
    return _characteristic_length(grid.geom, grid.dimensions, extents)


def _cartesian_to_grid(points: np.ndarray, geometry: Geometry) -> np.ndarray:
    if geometry is Geometry.CARTESIAN:
        return points
    x, y, z = np.moveaxis(points, -1, 0)
    rho = np.hypot(x, y)
    phi = np.where(rho == 0, 0.0, np.arctan2(y, x))
    if geometry is Geometry.CYLINDRICAL:
        return np.stack((rho, phi, z), axis=-1)
    radius = np.sqrt(x * x + y * y + z * z)
    safe_radius = np.where(radius == 0, 1.0, radius)
    theta = np.where(
        radius == 0,
        0.0,
        np.arccos(np.clip(z / safe_radius, -1.0, 1.0)),
    )
    return np.stack((radius, phi, theta), axis=-1)


def _inside(
    points: np.ndarray,
    geometry: Geometry,
    dimensions: int,
    extents: np.ndarray,
    tolerances: np.ndarray,
) -> np.ndarray:
    coordinates = _cartesian_to_grid(points, geometry)[..., :dimensions]
    active_extents = extents[:dimensions]
    return np.all(
        (coordinates >= active_extents[:, 0] - tolerances)
        & (coordinates <= active_extents[:, 1] + tolerances),
        axis=-1,
    )


def _plane_candidates(
    origins: np.ndarray,
    directions: np.ndarray,
    normal: np.ndarray,
    offset: float,
) -> np.ndarray:
    denominator = directions @ normal
    numerator = offset - origins @ normal
    candidates = np.full(origins.shape[0], np.inf, dtype=np.float64)
    valid = np.abs(denominator) > 64.0 * np.finfo(np.float64).eps
    np.divide(numerator, denominator, out=candidates, where=valid)
    return candidates[:, None]


def _quadratic_candidates(a, b, c) -> np.ndarray:
    a, b, c = np.broadcast_arrays(a, b, c)
    scale = np.maximum.reduce((np.abs(a), np.abs(b), np.abs(c), np.ones_like(a)))
    tolerance = 128.0 * np.finfo(np.float64).eps * scale
    linear = (np.abs(a) <= tolerance) & (np.abs(b) > tolerance)
    discriminant = b * b - 4.0 * a * c
    quadratic = (np.abs(a) > tolerance) & (discriminant >= -tolerance)

    candidates = np.full((*a.shape, 2), np.inf, dtype=np.float64)
    np.divide(-c, b, out=candidates[..., 0], where=linear)
    root = np.sqrt(np.maximum(discriminant, 0.0))
    denominator = 2.0 * a
    first_root = np.full(a.shape, np.inf, dtype=np.float64)
    second_root = np.full(a.shape, np.inf, dtype=np.float64)
    np.divide(-b - root, denominator, out=first_root, where=quadratic)
    np.divide(-b + root, denominator, out=second_root, where=quadratic)
    candidates[..., 0] = np.where(quadratic, first_root, candidates[..., 0])
    candidates[..., 1] = np.where(quadratic, second_root, candidates[..., 1])
    return candidates


def _radial_candidates(
    origins: np.ndarray,
    directions: np.ndarray,
    radii: np.ndarray,
    geometry: Geometry,
) -> list[np.ndarray]:
    candidates = []
    for radius in radii:
        if radius == 0:
            continue
        if geometry is Geometry.CYLINDRICAL:
            a = directions[:, 0] ** 2 + directions[:, 1] ** 2
            b = 2.0 * (
                origins[:, 0] * directions[:, 0] + origins[:, 1] * directions[:, 1]
            )
            c = origins[:, 0] ** 2 + origins[:, 1] ** 2 - radius**2
        else:
            a = np.sum(directions * directions, axis=1)
            b = 2.0 * np.sum(origins * directions, axis=1)
            c = np.sum(origins * origins, axis=1) - radius**2
        candidates.append(_quadratic_candidates(a, b, c))
    return candidates


def _phi_candidates(
    origins: np.ndarray,
    directions: np.ndarray,
    phi_min: float,
    phi_max: float,
) -> list[np.ndarray]:
    if np.isclose(phi_max - phi_min, 2.0 * np.pi):
        return []
    return [
        _plane_candidates(
            origins,
            directions,
            np.asarray((-np.sin(phi), np.cos(phi), 0.0)),
            0.0,
        )
        for phi in (phi_min, phi_max)
    ]


def _theta_candidates(
    origins: np.ndarray, directions: np.ndarray, theta: float
) -> np.ndarray | None:
    if np.isclose(theta, 0.0) or np.isclose(theta, np.pi):
        return None
    if np.isclose(theta, np.pi / 2.0):
        return _plane_candidates(
            origins,
            directions,
            np.asarray((0.0, 0.0, 1.0)),
            0.0,
        )
    cosine_squared = np.cos(theta) ** 2
    sine_squared = np.sin(theta) ** 2
    ox, oy, oz = np.moveaxis(origins, -1, 0)
    dx, dy, dz = np.moveaxis(directions, -1, 0)
    return _quadratic_candidates(
        cosine_squared * (dx * dx + dy * dy) - sine_squared * dz * dz,
        2.0 * (cosine_squared * (ox * dx + oy * dy) - sine_squared * oz * dz),
        cosine_squared * (ox * ox + oy * oy) - sine_squared * oz * oz,
    )


def _boundary_candidates(
    origins: np.ndarray,
    directions: np.ndarray,
    geometry: Geometry,
    dimensions: int,
    extents: np.ndarray,
) -> np.ndarray:
    candidates: list[np.ndarray] = []
    if geometry is Geometry.CARTESIAN:
        for axis in range(dimensions):
            normal = np.zeros(3)
            normal[axis] = 1.0
            candidates.extend(
                _plane_candidates(origins, directions, normal, boundary)
                for boundary in extents[axis]
            )
        return np.concatenate(candidates, axis=1)

    candidates.extend(_radial_candidates(origins, directions, extents[0], geometry))
    candidates.extend(_phi_candidates(origins, directions, *extents[1]))
    if geometry is Geometry.CYLINDRICAL:
        if dimensions == 3:
            normal = np.asarray((0.0, 0.0, 1.0))
            candidates.extend(
                _plane_candidates(origins, directions, normal, boundary)
                for boundary in extents[2]
            )
    else:
        for theta in extents[2]:
            candidate = _theta_candidates(origins, directions, theta)
            if candidate is not None:
                candidates.append(candidate)
    return np.concatenate(candidates, axis=1)


def ray_grid_entry_distance(
    grid: Grid,
    origins,
    directions,
    *,
    forward_only: bool = True,
):
    """Return physical distances to the first grid intersection along each ray.

    ``origins`` and ``directions`` have shape ``(..., 3)``. Directions are
    normalized internally. A ray that has no qualifying intersection receives
    ``inf``. With ``forward_only=False``, the earliest point on each oriented
    infinite line is returned and its signed distance may be negative.
    """
    origins = np.asarray(origins, dtype=np.float64)
    directions = np.asarray(directions, dtype=np.float64)
    if origins.shape[-1:] != (3,) or directions.shape[-1:] != (3,):
        raise ValueError("origins and directions must have shape (..., 3)")
    origins, directions = np.broadcast_arrays(origins, directions)
    leading_shape = origins.shape[:-1]
    flat_origins = origins.reshape((-1, 3))
    flat_directions = directions.reshape((-1, 3))
    norms = np.linalg.norm(flat_directions, axis=1)
    if np.any(~np.isfinite(flat_origins)) or np.any(~np.isfinite(flat_directions)):
        raise ValueError("ray origins and directions must be finite")
    if np.any(norms == 0):
        raise ValueError("ray directions must be nonzero")
    flat_directions = flat_directions / norms[:, None]

    extents = np.asarray(grid.extents, dtype=np.float64)
    active_extents = extents[: grid.dimensions]
    scales = np.maximum(1.0, np.max(np.abs(active_extents), axis=1))
    coordinate_tolerances = 512.0 * np.finfo(np.float64).eps * scales
    distance_tolerance = (
        512.0
        * np.finfo(np.float64).eps
        * max(1.0, _characteristic_length(grid.geom, grid.dimensions, extents))
    )

    candidates = _boundary_candidates(
        flat_origins,
        flat_directions,
        grid.geom,
        grid.dimensions,
        extents,
    )
    finite = np.isfinite(candidates)
    safe_candidates = np.where(finite, candidates, 0.0)
    candidate_points = (
        flat_origins[:, None, :]
        + safe_candidates[..., None] * flat_directions[:, None, :]
    )
    valid = finite & _inside(
        candidate_points,
        grid.geom,
        grid.dimensions,
        extents,
        coordinate_tolerances,
    )
    if forward_only:
        valid &= candidates >= -distance_tolerance
        candidate_distances = np.maximum(candidates, 0.0)
    else:
        candidate_distances = candidates
    distances = np.min(
        np.where(valid, candidate_distances, np.inf),
        axis=1,
    )
    if forward_only:
        origins_inside = _inside(
            flat_origins,
            grid.geom,
            grid.dimensions,
            extents,
            coordinate_tolerances,
        )
        distances = np.where(origins_inside, 0.0, distances)
    return distances.reshape(leading_shape)
