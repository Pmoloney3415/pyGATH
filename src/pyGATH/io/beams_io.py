"""Strict CSV input for large, tabular beam lists."""

from __future__ import annotations

import csv
import math
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from pyGATH.beam import BeamBatch, spot_basis
from pyGATH.grid import Geometry, convert_positions

DEFAULT_OMEGA_RAD_S = 5.361e15

_REQUIRED_COLUMNS = {
    "beam_id",
    "origin_geometry",
    "origin_1",
    "origin_2",
    "origin_3",
    "target_geometry",
    "target_1",
    "target_2",
    "target_3",
    "width_x_m",
    "width_y_m",
    "rotation_pi",
    "supergaussian_index",
}
_OPTIONAL_COLUMNS = {
    "omega_rad_s",
    "power_fraction",
    "peak_intensity_w_m2",
    "beam_power_w",
}
_POWER_MODES = {"peak_intensity", "total_power", "per_beam_power"}


class BeamFileError(ValueError):
    """Raised when a beam CSV file has an invalid schema or value."""


def _number(row: dict[str, str], key: str, row_number: int, default=None) -> float:
    raw_value = row.get(key, "").strip()
    if not raw_value:
        if default is not None:
            return float(default)
        raise BeamFileError(f"row {row_number}: {key!r} cannot be empty")
    try:
        value = float(raw_value)
    except ValueError as error:
        raise BeamFileError(f"row {row_number}: {key!r} must be a number") from error
    if not np.isfinite(value):
        raise BeamFileError(f"row {row_number}: {key!r} must be finite")
    return value


def _geometry(row: dict[str, str], key: str, row_number: int) -> Geometry:
    raw_value = row.get(key, "").strip()
    try:
        return Geometry.parse(raw_value)
    except ValueError as error:
        raise BeamFileError(f"row {row_number}: invalid {key}: {error}") from error


def _position(
    row: dict[str, str],
    prefix: str,
    geometry: Geometry,
    row_number: int,
) -> np.ndarray:
    position = np.asarray(
        [_number(row, f"{prefix}_{axis}", row_number) for axis in range(1, 4)],
        dtype=np.float64,
    )
    if geometry is Geometry.CYLINDRICAL:
        if position[0] < 0:
            raise BeamFileError(
                f"row {row_number}: {prefix}_1 radius cannot be negative"
            )
        if not -1.0 <= position[1] <= 1.0:
            raise BeamFileError(
                f"row {row_number}: {prefix}_2 phi fraction must lie within [-1, 1]"
            )
        position[1] *= np.pi
    elif geometry is Geometry.SPHERICAL:
        if position[0] < 0:
            raise BeamFileError(
                f"row {row_number}: {prefix}_1 radius cannot be negative"
            )
        if not -1.0 <= position[1] <= 1.0:
            raise BeamFileError(
                f"row {row_number}: {prefix}_2 phi fraction must lie within [-1, 1]"
            )
        if not 0.0 <= position[2] <= 1.0:
            raise BeamFileError(
                f"row {row_number}: {prefix}_3 theta fraction must lie within [0, 1]"
            )
        position[1:] *= np.pi
    return np.asarray(
        convert_positions(position, geometry, Geometry.CARTESIAN), dtype=np.float64
    )


def load_beams_csv(
    path: str | Path,
    *,
    power_mode: str = "total_power",
    total_power_w: float | None = 1.0,
    dimensions: int = 3,
    inactive_axis_lengths_m=None,
) -> BeamBatch:
    """Load beams and resolve their physical powers and peak intensities.

    For cylindrical positions, columns ``*_1, *_2, *_3`` mean
    ``radius, phi/pi, z``. For spherical positions they mean
    ``radius, phi/pi, theta/pi``. Cartesian columns are ordinary SI positions.
    Spot rotation is likewise supplied as a fraction of pi.
    """
    if not isinstance(power_mode, str) or power_mode.lower() not in _POWER_MODES:
        choices = ", ".join(sorted(_POWER_MODES))
        raise BeamFileError(f"power_mode must be one of: {choices}")
    power_mode = power_mode.lower()
    if isinstance(dimensions, bool) or not isinstance(dimensions, int):
        raise BeamFileError("dimensions must be an integer")
    if dimensions not in (1, 2, 3):
        raise BeamFileError("dimensions must be one, two, or three")
    ninactive = 3 - dimensions
    inactive_lengths = (
        (1.0,) * ninactive
        if inactive_axis_lengths_m is None
        else tuple(inactive_axis_lengths_m)
    )
    if len(inactive_lengths) != ninactive:
        raise BeamFileError(
            "inactive_axis_lengths_m must contain one value per inactive axis"
        )
    if any(
        isinstance(length, bool)
        or not isinstance(length, (int, float))
        or not np.isfinite(length)
        or length <= 0.0
        for length in inactive_lengths
    ):
        raise BeamFileError("inactive-axis lengths must be finite positive numbers")
    inactive_lengths = tuple(float(length) for length in inactive_lengths)
    if power_mode == "total_power":
        if total_power_w is None or not np.isfinite(total_power_w):
            raise BeamFileError("total_power mode requires finite total_power_w")
        if total_power_w <= 0.0:
            raise BeamFileError("total_power_w must be positive")
    source = Path(path)
    try:
        stream = source.open("r", encoding="utf-8-sig", newline="")
    except OSError as error:
        raise BeamFileError(f"could not read beam file {source}: {error}") from error

    with stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise BeamFileError(f"beam file {source} has no header")
        fieldnames = [name.strip() for name in reader.fieldnames]
        if len(fieldnames) != len(set(fieldnames)):
            raise BeamFileError(f"beam file {source} contains duplicate columns")
        missing = sorted(_REQUIRED_COLUMNS - set(fieldnames))
        unknown = sorted(set(fieldnames) - _REQUIRED_COLUMNS - _OPTIONAL_COLUMNS)
        if missing:
            raise BeamFileError(
                f"missing required beam column(s): {', '.join(missing)}"
            )
        if unknown:
            raise BeamFileError(f"unknown beam column(s): {', '.join(unknown)}")

        names: list[str] = []
        origins: list[np.ndarray] = []
        targets: list[np.ndarray] = []
        width_x: list[float] = []
        width_y: list[float] = []
        rotations: list[float] = []
        indices: list[float] = []
        frequencies: list[float] = []
        input_powers: list[float] = []
        for row_number, row in enumerate(reader, start=2):
            if None in row:
                raise BeamFileError(
                    f"row {row_number}: contains more values than the CSV header"
                )
            row = {key.strip(): (value or "") for key, value in row.items()}
            name = row["beam_id"].strip()
            if not name:
                raise BeamFileError(f"row {row_number}: 'beam_id' cannot be empty")
            if name in names:
                raise BeamFileError(f"row {row_number}: duplicate beam_id {name!r}")
            origin_geometry = _geometry(row, "origin_geometry", row_number)
            target_geometry = _geometry(row, "target_geometry", row_number)
            origin = _position(row, "origin", origin_geometry, row_number)
            target = _position(row, "target", target_geometry, row_number)
            x_width = _number(row, "width_x_m", row_number)
            y_width = _number(row, "width_y_m", row_number)
            index = _number(row, "supergaussian_index", row_number)
            frequency = _number(
                row, "omega_rad_s", row_number, default=DEFAULT_OMEGA_RAD_S
            )
            if power_mode == "total_power":
                input_power = _number(row, "power_fraction", row_number, default=1.0)
                input_name = "power_fraction"
            elif power_mode == "peak_intensity":
                input_power = _number(row, "peak_intensity_w_m2", row_number)
                input_name = "peak_intensity_w_m2"
            else:
                input_power = _number(row, "beam_power_w", row_number)
                input_name = "beam_power_w"
            if x_width <= 0 or y_width <= 0:
                raise BeamFileError(f"row {row_number}: beam widths must be positive")
            if index <= 0:
                raise BeamFileError(
                    f"row {row_number}: supergaussian_index must be positive"
                )
            if frequency <= 0:
                raise BeamFileError(f"row {row_number}: omega_rad_s must be positive")
            if input_power < 0:
                raise BeamFileError(
                    f"row {row_number}: {input_name} cannot be negative"
                )
            if np.array_equal(origin, target):
                raise BeamFileError(
                    f"row {row_number}: beam target must differ from origin"
                )
            names.append(name)
            origins.append(origin)
            targets.append(target)
            width_x.append(x_width)
            width_y.append(y_width)
            rotations.append(_number(row, "rotation_pi", row_number))
            indices.append(index)
            frequencies.append(frequency)
            input_powers.append(input_power)

    if not names:
        raise BeamFileError(f"beam file {source} contains no beam rows")
    input_power_array = np.asarray(input_powers, dtype=np.float64)
    if not np.any(input_power_array > 0.0):
        raise BeamFileError("at least one beam must have positive incident power")
    widths_x_array = np.asarray(width_x, dtype=np.float64)
    widths_y_array = np.asarray(width_y, dtype=np.float64)
    indices_array = np.asarray(indices, dtype=np.float64)
    gamma_factors = np.asarray(
        [math.gamma(1.0 + 1.0 / index) for index in indices_array]
    )
    inactive_measure = float(np.prod(inactive_lengths)) if inactive_lengths else 1.0
    if dimensions == 1:
        profile_integrals = np.full_like(widths_x_array, inactive_measure)
    elif dimensions == 2:
        profile_integrals = 2.0 * widths_x_array * gamma_factors * inactive_measure
    else:
        profile_integrals = 4.0 * widths_x_array * widths_y_array * gamma_factors**2
    if power_mode == "total_power":
        power_fraction = input_power_array / np.sum(input_power_array)
        beam_power = float(total_power_w) * power_fraction
        peak_intensity = beam_power / profile_integrals
    elif power_mode == "peak_intensity":
        peak_intensity = input_power_array
        beam_power = peak_intensity * profile_integrals
        power_fraction = beam_power / np.sum(beam_power)
    else:
        beam_power = input_power_array
        peak_intensity = beam_power / profile_integrals
        power_fraction = beam_power / np.sum(beam_power)
    origin_array = jnp.asarray(np.stack(origins), dtype=jnp.float64)
    target_array = jnp.asarray(np.stack(targets), dtype=jnp.float64)
    directions = target_array - origin_array
    directions /= jnp.linalg.norm(directions, axis=1, keepdims=True)
    rotation_array = jnp.asarray(rotations, dtype=jnp.float64)
    axis_x, axis_y = spot_basis(directions, rotation_array)
    return BeamBatch(
        names=tuple(names),
        origin=origin_array,
        target=target_array,
        direction=directions,
        axis_x=axis_x,
        axis_y=axis_y,
        width_x=jnp.asarray(widths_x_array, dtype=jnp.float64),
        width_y=jnp.asarray(widths_y_array, dtype=jnp.float64),
        rotation_pi=rotation_array,
        supergaussian_index=jnp.asarray(indices, dtype=jnp.float64),
        omega=jnp.asarray(frequencies, dtype=jnp.float64),
        power_fraction=jnp.asarray(power_fraction, dtype=jnp.float64),
        peak_intensity=jnp.asarray(peak_intensity, dtype=jnp.float64),
        beam_power=jnp.asarray(beam_power, dtype=jnp.float64),
        dimensions=dimensions,
        inactive_axis_lengths_m=inactive_lengths,
    )
