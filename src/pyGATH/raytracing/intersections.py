"""Physical intersections between Cartesian rays and supported grid domains."""

from __future__ import annotations

import numpy as np

from pyGATH.grid import Geometry, Grid


def grid_characteristic_length(grid: Grid) -> float:
    """Return a conservative physical length scale for launch-plane padding."""
    extents = np.asarray(grid.extents, dtype=np.float64)
    if grid.geom is Geometry.CARTESIAN:
        active = extents[np.asarray(grid.active_axes)]
        return float(np.linalg.norm(active[:, 1] - active[:, 0]))
    if grid.geom is Geometry.CYLINDRICAL:
        diameter = 2.0 * extents[0, 1]
        if grid.dimensions == 2:
            return float(diameter)
        axial_length = extents[2, 1] - extents[2, 0]
        return float(np.hypot(diameter, axial_length))
    return float(2.0 * extents[0, 1])


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


def _inside(grid: Grid, point: np.ndarray) -> bool:
    coordinates = _cartesian_to_grid(point[None, :], grid.geom)[0]
    extents = np.asarray(grid.extents, dtype=np.float64)
    active = np.asarray(grid.active_axes)
    coordinates = coordinates[active]
    extents = extents[active]
    scales = np.maximum(1.0, np.max(np.abs(extents), axis=1))
    tolerance = 512.0 * np.finfo(np.float64).eps * scales
    return bool(
        np.all(coordinates >= extents[:, 0] - tolerance)
        and np.all(coordinates <= extents[:, 1] + tolerance)
    )


def _plane_candidate(
    candidates: list[float],
    origin: np.ndarray,
    direction: np.ndarray,
    normal: np.ndarray,
    offset: float,
) -> None:
    denominator = float(np.dot(normal, direction))
    tolerance = 64.0 * np.finfo(np.float64).eps
    if abs(denominator) > tolerance:
        candidates.append(float((offset - np.dot(normal, origin)) / denominator))


def _quadratic_candidates(
    candidates: list[float],
    a: float,
    b: float,
    c: float,
) -> None:
    scale = max(abs(a), abs(b), abs(c), 1.0)
    tolerance = 128.0 * np.finfo(np.float64).eps * scale
    if abs(a) <= tolerance:
        if abs(b) > tolerance:
            candidates.append(float(-c / b))
        return
    discriminant = b * b - 4.0 * a * c
    if discriminant < -tolerance:
        return
    root = np.sqrt(max(discriminant, 0.0))
    candidates.extend((float((-b - root) / (2.0 * a)), float((-b + root) / (2.0 * a))))


def _radial_cylinder_candidates(
    candidates: list[float],
    origin: np.ndarray,
    direction: np.ndarray,
    radius: float,
) -> None:
    if radius == 0:
        return
    ox, oy = origin[:2]
    dx, dy = direction[:2]
    _quadratic_candidates(
        candidates,
        dx * dx + dy * dy,
        2.0 * (ox * dx + oy * dy),
        ox * ox + oy * oy - radius * radius,
    )


def _sphere_candidates(
    candidates: list[float],
    origin: np.ndarray,
    direction: np.ndarray,
    radius: float,
) -> None:
    if radius == 0:
        return
    _quadratic_candidates(
        candidates,
        float(np.dot(direction, direction)),
        2.0 * float(np.dot(origin, direction)),
        float(np.dot(origin, origin) - radius * radius),
    )


def _phi_candidates(
    candidates: list[float],
    origin: np.ndarray,
    direction: np.ndarray,
    phi_min: float,
    phi_max: float,
) -> None:
    if np.isclose(phi_max - phi_min, 2.0 * np.pi):
        return
    for phi in (phi_min, phi_max):
        normal = np.asarray((-np.sin(phi), np.cos(phi), 0.0))
        _plane_candidate(candidates, origin, direction, normal, 0.0)


def _theta_candidates(
    candidates: list[float],
    origin: np.ndarray,
    direction: np.ndarray,
    theta: float,
) -> None:
    if np.isclose(theta, 0.0) or np.isclose(theta, np.pi):
        return
    if np.isclose(theta, np.pi / 2.0):
        _plane_candidate(
            candidates,
            origin,
            direction,
            np.asarray((0.0, 0.0, 1.0)),
            0.0,
        )
        return
    cosine_squared = np.cos(theta) ** 2
    sine_squared = np.sin(theta) ** 2
    ox, oy, oz = origin
    dx, dy, dz = direction
    _quadratic_candidates(
        candidates,
        cosine_squared * (dx * dx + dy * dy) - sine_squared * dz * dz,
        2.0 * (cosine_squared * (ox * dx + oy * dy) - sine_squared * oz * dz),
        cosine_squared * (ox * ox + oy * oy) - sine_squared * oz * oz,
    )


def _boundary_candidates(
    grid: Grid,
    origin: np.ndarray,
    direction: np.ndarray,
) -> list[float]:
    extents = np.asarray(grid.extents, dtype=np.float64)
    candidates: list[float] = []
    if grid.geom is Geometry.CARTESIAN:
        for axis in grid.active_axes:
            normal = np.zeros(3)
            normal[axis] = 1.0
            for boundary in extents[axis]:
                _plane_candidate(candidates, origin, direction, normal, boundary)
        return candidates

    for radius in extents[0]:
        if grid.geom is Geometry.CYLINDRICAL:
            _radial_cylinder_candidates(candidates, origin, direction, radius)
        else:
            _sphere_candidates(candidates, origin, direction, radius)
    _phi_candidates(candidates, origin, direction, *extents[1])

    if grid.geom is Geometry.CYLINDRICAL:
        if grid.dimensions == 3:
            z_normal = np.asarray((0.0, 0.0, 1.0))
            for boundary in extents[2]:
                _plane_candidate(candidates, origin, direction, z_normal, boundary)
    else:
        for theta in extents[2]:
            _theta_candidates(candidates, origin, direction, theta)
    return candidates


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

    distances = np.full(flat_origins.shape[0], np.inf, dtype=np.float64)
    for ray_index, (origin, direction) in enumerate(
        zip(flat_origins, flat_directions, strict=True)
    ):
        if forward_only and _inside(grid, origin):
            distances[ray_index] = 0.0
            continue
        candidates = _boundary_candidates(grid, origin, direction)
        tolerance = (
            512.0
            * np.finfo(np.float64).eps
            * max(1.0, grid_characteristic_length(grid))
        )
        valid = []
        for distance in candidates:
            if not np.isfinite(distance):
                continue
            if forward_only and distance < -tolerance:
                continue
            point = origin + distance * direction
            if _inside(grid, point):
                valid.append(max(distance, 0.0) if forward_only else distance)
        if valid:
            distances[ray_index] = min(valid)
    return distances.reshape(leading_shape)
