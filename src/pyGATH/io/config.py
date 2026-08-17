"""Read and validate TOML simulation input decks."""

from __future__ import annotations

import tomllib
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np

from pyGATH.beam import BeamBatch, InitializedRays, initialize_rays
from pyGATH.grid import (
    Geometry,
    GradedAxis,
    Grid,
    HydroFields,
    PlasmaComposition,
    SafeHydroState,
)
from pyGATH.raytracing import (
    InverseBremsstrahlungOptions,
    RayTraceResult,
    RayTracingOptions,
    trace_rays,
)

from .beams_io import load_beams_csv


class ConfigError(ValueError):
    """Raised when a simulation input deck has an invalid schema or value."""


_AXIS_NAMES = {
    Geometry.CARTESIAN: ("x", "y", "z"),
    Geometry.CYLINDRICAL: ("r", "phi", "z"),
    Geometry.SPHERICAL: ("r", "phi", "theta"),
}

_ANGULAR_AXES = {
    Geometry.CARTESIAN: frozenset(),
    Geometry.CYLINDRICAL: frozenset({"phi"}),
    Geometry.SPHERICAL: frozenset({"phi", "theta"}),
}


def _expect_table(value: Any, location: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigError(f"{location} must be a TOML table")
    return value


def _reject_unknown(table: Mapping[str, Any], allowed: set[str], location: str) -> None:
    unknown = sorted(set(table) - allowed)
    if unknown:
        names = ", ".join(repr(name) for name in unknown)
        raise ConfigError(f"unknown key(s) in {location}: {names}")


def _require(table: Mapping[str, Any], keys: set[str], location: str) -> None:
    missing = sorted(keys - set(table))
    if missing:
        names = ", ".join(repr(name) for name in missing)
        raise ConfigError(f"missing required key(s) in {location}: {names}")


def _number(value: Any, location: str) -> float:
    if isinstance(value, bool):
        raise ConfigError(f"{location} must be a number")
    if isinstance(value, (int, float)):
        result = float(value)
    else:
        raise ConfigError(f"{location} must be a number")
    if not np.isfinite(result):
        raise ConfigError(f"{location} must be finite")
    return result


def _number_list(value: Any, location: str) -> tuple[float, ...]:
    if not isinstance(value, list):
        raise ConfigError(f"{location} must be an array")
    return tuple(
        _number(item, f"{location}[{index}]") for index, item in enumerate(value)
    )


def _vector(value: Any, location: str) -> tuple[float, float, float]:
    values = _number_list(value, location)
    if len(values) != 3:
        raise ConfigError(f"{location} must contain exactly three values")
    return values


def _positive_integer(value: Any, location: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ConfigError(f"{location} must be a positive integer")
    return value


def _positive_number(value: Any, location: str) -> float:
    result = _number(value, location)
    if result <= 0:
        raise ConfigError(f"{location} must be positive")
    return result


def _boolean(value: Any, location: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(f"{location} must be true or false")
    return value


def _parse_axis(
    raw_axis: Any,
    *,
    name: str,
    angular: bool,
) -> tuple[tuple[float, float], int | None, GradedAxis | None]:
    location = f"[grid.axes.{name}]"
    axis = _expect_table(raw_axis, location)
    common = {"min", "max", "spacing"}
    uniform_keys = common | {"ncells"}
    graded_keys = common | {"boundaries", "cell_sizes", "transition_widths"}
    _require(axis, common, location)
    spacing = axis["spacing"]
    if not isinstance(spacing, str):
        raise ConfigError(f"{location}.spacing must be 'uniform' or 'graded'")
    spacing = spacing.lower()
    lower = _number(axis["min"], f"{location}.min")
    upper = _number(axis["max"], f"{location}.max")
    if upper <= lower:
        raise ConfigError(f"{location}.max must be greater than {location}.min")
    if name == "r" and lower < 0:
        raise ConfigError(f"{location} cannot include negative radii")
    if name == "phi" and (lower < -1 or upper > 1):
        raise ConfigError(f"{location} fractions of pi must lie within [-1, 1]")
    if name == "theta" and (lower < 0 or upper > 1):
        raise ConfigError(f"{location} fractions of pi must lie within [0, 1]")
    extent = (lower * np.pi, upper * np.pi) if angular else (lower, upper)

    if spacing == "uniform":
        _reject_unknown(axis, uniform_keys, location)
        _require(axis, {"ncells"}, location)
        return extent, _positive_integer(axis["ncells"], f"{location}.ncells"), None

    if spacing == "graded":
        if angular:
            raise ConfigError(f"{location} is angular and must use uniform spacing")
        _reject_unknown(axis, graded_keys, location)
        _require(
            axis,
            {"boundaries", "cell_sizes", "transition_widths"},
            location,
        )
        boundaries = _number_list(axis["boundaries"], f"{location}.boundaries")
        cell_sizes = _number_list(axis["cell_sizes"], f"{location}.cell_sizes")
        transition_widths = _number_list(
            axis["transition_widths"], f"{location}.transition_widths"
        )
        if len(cell_sizes) != len(boundaries) + 1:
            raise ConfigError(
                f"{location}.cell_sizes must contain one more value than boundaries"
            )
        if len(transition_widths) != len(boundaries):
            raise ConfigError(
                f"{location}.transition_widths must contain one value per boundary"
            )
        if any(size <= 0 for size in cell_sizes):
            raise ConfigError(f"{location}.cell_sizes values must be positive")
        if any(width <= 0 for width in transition_widths):
            raise ConfigError(f"{location}.transition_widths values must be positive")
        if boundaries and (
            boundaries[0] <= lower
            or boundaries[-1] >= upper
            or any(right <= left for left, right in pairwise(boundaries))
        ):
            raise ConfigError(
                f"{location}.boundaries must increase strictly inside the axis extent"
            )
        specification = GradedAxis(
            boundaries=boundaries,
            cell_sizes=cell_sizes,
            transition_widths=transition_widths,
        )
        return extent, None, specification

    raise ConfigError(f"{location}.spacing must be 'uniform' or 'graded'")


def _parse_safe_state(raw_safe_state: Any) -> SafeHydroState:
    if raw_safe_state is None:
        return SafeHydroState()
    location = "[grid.safe_state]"
    safe = _expect_table(raw_safe_state, location)
    allowed = {"ne", "Te", "Ti", "grad_ne", "velocity"}
    _reject_unknown(safe, allowed, location)
    defaults = SafeHydroState()
    return SafeHydroState(
        ne=_number(safe.get("ne", defaults.ne), f"{location}.ne"),
        Te=_number(safe.get("Te", defaults.Te), f"{location}.Te"),
        Ti=_number(safe.get("Ti", defaults.Ti), f"{location}.Ti"),
        grad_ne=_vector(
            safe.get("grad_ne", list(defaults.grad_ne)), f"{location}.grad_ne"
        ),
        velocity=_vector(
            safe.get("velocity", list(defaults.velocity)), f"{location}.velocity"
        ),
    )


def _parse_composition(raw_composition: Any) -> PlasmaComposition:
    if raw_composition is None:
        return PlasmaComposition.create()
    location = "[grid.composition]"
    composition = _expect_table(raw_composition, location)
    _reject_unknown(composition, {"elements"}, location)
    _require(composition, {"elements"}, location)
    raw_elements = composition["elements"]
    if not isinstance(raw_elements, list) or not raw_elements:
        raise ConfigError(f"{location}.elements must be a non-empty array of tables")

    elements: list[tuple[str, float, float, float]] = []
    for index, raw_element in enumerate(raw_elements):
        element_location = f"[[grid.composition.elements]] entry {index}"
        element = _expect_table(raw_element, element_location)
        _reject_unknown(element, {"name", "A", "Z", "fraction"}, element_location)
        _require(element, {"name", "A", "Z", "fraction"}, element_location)
        name = element["name"]
        if not isinstance(name, str) or not name.strip():
            raise ConfigError(f"{element_location}.name must be a non-empty string")
        elements.append(
            (
                name.strip(),
                _positive_number(element["A"], f"{element_location}.A"),
                _positive_number(element["Z"], f"{element_location}.Z"),
                _number(element["fraction"], f"{element_location}.fraction"),
            )
        )
        if elements[-1][3] < 0.0:
            raise ConfigError(f"{element_location}.fraction cannot be negative")
    try:
        return PlasmaComposition.create(elements)
    except ValueError as error:
        raise ConfigError(f"{location}: {error}") from error


def _parse_initial_condition(raw_initial_condition: Any) -> tuple[str, dict[str, Any]]:
    if raw_initial_condition is None:
        return "uniform", {}
    location = "[grid.initial_condition]"
    initial = _expect_table(raw_initial_condition, location)
    _reject_unknown(initial, {"name", "parameters"}, location)
    name = initial.get("name", "uniform")
    if not isinstance(name, str) or not name.strip():
        raise ConfigError(f"{location}.name must be a non-empty string")
    parameters = initial.get("parameters", {})
    parameters = _expect_table(parameters, f"{location}.parameters")
    return name.strip(), dict(parameters)


@dataclass(frozen=True)
class GridConfig:
    """Validated configuration needed to construct a :class:`Grid`."""

    geometry: Geometry
    dimensions: int
    extents: tuple[tuple[float, float], ...]
    ncells: tuple[int | None, ...]
    graded_axes: tuple[GradedAxis | None, ...]
    inactive_axis_lengths_m: tuple[float, ...]
    initial_condition: str
    initial_condition_parameters: Mapping[str, Any]
    safe_state: SafeHydroState
    composition: PlasmaComposition

    def build(
        self,
        initial_conditions: Mapping[str, Callable[[Any], HydroFields]] | None = None,
    ) -> Grid:
        """Construct a Grid, optionally resolving a custom initial condition."""
        registry = {} if initial_conditions is None else initial_conditions
        initialise = registry.get(self.initial_condition, self.initial_condition)
        return Grid.create(
            geom=self.geometry,
            dimensions=self.dimensions,
            extents=self.extents,
            ncells=self.ncells,
            graded_axes=self.graded_axes,
            inactive_axis_lengths_m=self.inactive_axis_lengths_m,
            initial_condition=initialise,
            initial_condition_parameters=dict(self.initial_condition_parameters),
            safe_state=self.safe_state,
            composition=self.composition,
        )


@dataclass(frozen=True)
class BeamPowerConfig:
    """Select how physical beam powers and peak intensities are supplied."""

    mode: str
    total_power_w: float | None = None


@dataclass(frozen=True)
class BeamsConfig:
    """Validated settings shared by every beam and ray sample."""

    file: Path
    dimensions: int = 3
    inactive_axis_lengths_m: tuple[float, ...] = ()
    nrays_axis1: int = 20
    nrays_axis2: int = 20
    intensity_cutoff: float = 2.0e-4
    neighbour_spacing_m: float = 1.0e-9
    launch_padding: float = 10.0
    power: BeamPowerConfig = BeamPowerConfig("total_power", 1.0)

    def load(self) -> BeamBatch:
        """Read and validate the referenced CSV beam list."""
        return load_beams_csv(
            self.file,
            power_mode=self.power.mode,
            total_power_w=self.power.total_power_w,
            dimensions=self.dimensions,
            inactive_axis_lengths_m=self.inactive_axis_lengths_m,
        )

    def initialize(self, grid: Grid, beams: BeamBatch | None = None) -> InitializedRays:
        """Load beams if needed, then sample and place their ray states."""
        beam_batch = self.load() if beams is None else beams
        return initialize_rays(
            beam_batch,
            grid,
            nrays_axis1=self.nrays_axis1,
            nrays_axis2=self.nrays_axis2,
            intensity_cutoff=self.intensity_cutoff,
            neighbour_spacing_m=self.neighbour_spacing_m,
            launch_padding=self.launch_padding,
        )


@dataclass(frozen=True)
class RayTracingConfig:
    """Validated controls for the Diffrax solve and ray-sheet sampling.

    The solver applies ``rtol`` and ``atol`` to automatically scaled,
    dimensionless state groups using a maximum norm.
    """

    nsamples_per_sheet: int = 20
    diagnostic_samples: int = 512
    maximum_path_length_grid_lengths: float = 100.0
    dt0: float | None = None
    rtol: float = 1.0e-5
    atol: float = 1.0e-7
    max_steps: int = 4096
    area_floor: float = 1.0e-12
    minimum_amplitude_cap: float = 1.1

    def options(self) -> RayTracingOptions:
        """Build the immutable numerical options used by ``trace_rays``."""
        return RayTracingOptions(
            nsamples_per_sheet=self.nsamples_per_sheet,
            diagnostic_samples=self.diagnostic_samples,
            maximum_path_length_grid_lengths=(self.maximum_path_length_grid_lengths),
            dt0=self.dt0,
            rtol=self.rtol,
            atol=self.atol,
            max_steps=self.max_steps,
            area_floor=self.area_floor,
            minimum_amplitude_cap=self.minimum_amplitude_cap,
        )

    def trace(
        self,
        initial_rays: InitializedRays,
        grid: Grid,
        inverse_bremsstrahlung: InverseBremsstrahlungOptions | None = None,
    ) -> RayTraceResult:
        """Propagate initialized rays and construct their two sheets."""
        return trace_rays(
            initial_rays,
            grid,
            options=self.options(),
            inverse_bremsstrahlung=inverse_bremsstrahlung,
        )


@dataclass(frozen=True)
class PhysicsConfig:
    """Laser-plasma physics controls independent of numerical tolerances."""

    inverse_bremsstrahlung: InverseBremsstrahlungOptions


@dataclass(frozen=True)
class SimulationConfig:
    """Validated simulation deck with room for future top-level sections."""

    grid: GridConfig
    beams: BeamsConfig | None
    raytracing: RayTracingConfig
    physics: PhysicsConfig
    extra_sections: Mapping[str, Any]
    source: Path

    def build_grid(
        self,
        initial_conditions: Mapping[str, Callable[[Any], HydroFields]] | None = None,
    ) -> Grid:
        """Build the configured grid using built-in or supplied initial conditions."""
        return self.grid.build(initial_conditions=initial_conditions)

    def load_beams(self) -> BeamBatch:
        """Load the beam list referenced by the simulation deck."""
        if self.beams is None:
            raise ConfigError("input deck does not contain a [beams] section")
        return self.beams.load()

    def initialize_rays(
        self, grid: Grid, beams: BeamBatch | None = None
    ) -> InitializedRays:
        """Build the packed global ray state from the configured beams."""
        if self.beams is None:
            raise ConfigError("input deck does not contain a [beams] section")
        return self.beams.initialize(grid, beams=beams)

    def trace_rays(self, initial_rays: InitializedRays, grid: Grid) -> RayTraceResult:
        """Run the configured ray trace and two-sheet resampling."""
        return self.raytracing.trace(
            initial_rays,
            grid,
            self.physics.inverse_bremsstrahlung,
        )


def _parse_beams(raw_beams: Any, source: Path, grid_config: GridConfig) -> BeamsConfig:
    location = "[beams]"
    beams = _expect_table(raw_beams, location)
    allowed = {
        "file",
        "nrays_axis1",
        "nrays_axis2",
        "intensity_cutoff",
        "neighbour_spacing_m",
        "launch_padding",
        "power",
    }
    _reject_unknown(beams, allowed, location)
    _require(beams, {"file", "power"}, location)
    raw_file = beams["file"]
    if not isinstance(raw_file, str) or not raw_file.strip():
        raise ConfigError(f"{location}.file must be a non-empty path string")
    beam_file = Path(raw_file)
    if not beam_file.is_absolute():
        beam_file = source.parent / beam_file
    beam_file = beam_file.resolve()
    cutoff = _number(
        beams.get("intensity_cutoff", 2.0e-4), f"{location}.intensity_cutoff"
    )
    if not 0.0 < cutoff < 1.0:
        raise ConfigError(f"{location}.intensity_cutoff must lie between zero and one")
    raw_power = _expect_table(beams["power"], "[beams.power]")
    _reject_unknown(raw_power, {"mode", "total_power_w"}, "[beams.power]")
    _require(raw_power, {"mode"}, "[beams.power]")
    mode = raw_power["mode"]
    allowed_modes = {"peak_intensity", "total_power", "per_beam_power"}
    if not isinstance(mode, str) or mode.lower() not in allowed_modes:
        choices = ", ".join(sorted(allowed_modes))
        raise ConfigError(f"[beams.power].mode must be one of: {choices}")
    mode = mode.lower()
    if mode == "total_power":
        _require(raw_power, {"total_power_w"}, "[beams.power]")
        total_power_w = _positive_number(
            raw_power["total_power_w"], "[beams.power].total_power_w"
        )
    else:
        total_power_w = None
    default_axis1 = 1 if grid_config.dimensions == 1 else 20
    default_axis2 = 20 if grid_config.dimensions == 3 else 1
    nrays_axis1 = _positive_integer(
        beams.get("nrays_axis1", default_axis1), f"{location}.nrays_axis1"
    )
    nrays_axis2 = _positive_integer(
        beams.get("nrays_axis2", default_axis2), f"{location}.nrays_axis2"
    )
    if grid_config.dimensions == 1 and nrays_axis1 != 1:
        raise ConfigError("[beams].nrays_axis1 must be one for a 1-D grid")
    if grid_config.dimensions < 3 and nrays_axis2 != 1:
        raise ConfigError(
            "[beams].nrays_axis2 must be one for a reduced-dimensional grid"
        )

    return BeamsConfig(
        file=beam_file,
        dimensions=grid_config.dimensions,
        inactive_axis_lengths_m=grid_config.inactive_axis_lengths_m,
        nrays_axis1=nrays_axis1,
        nrays_axis2=nrays_axis2,
        intensity_cutoff=cutoff,
        neighbour_spacing_m=_positive_number(
            beams.get("neighbour_spacing_m", 1.0e-9),
            f"{location}.neighbour_spacing_m",
        ),
        launch_padding=_positive_number(
            beams.get("launch_padding", 10.0), f"{location}.launch_padding"
        ),
        power=BeamPowerConfig(mode=mode, total_power_w=total_power_w),
    )


def _parse_raytracing(raw_raytracing: Any) -> RayTracingConfig:
    location = "[raytracing]"
    if raw_raytracing is None:
        return RayTracingConfig()
    raytracing = _expect_table(raw_raytracing, location)
    allowed = {
        "nsamples_per_sheet",
        "diagnostic_samples",
        "maximum_path_length_grid_lengths",
        "dt0",
        "rtol",
        "atol",
        "max_steps",
        "area_floor",
        "minimum_amplitude_cap",
    }
    _reject_unknown(raytracing, allowed, location)
    defaults = RayTracingConfig()
    nsamples = _positive_integer(
        raytracing.get("nsamples_per_sheet", defaults.nsamples_per_sheet),
        f"{location}.nsamples_per_sheet",
    )
    diagnostics = _positive_integer(
        raytracing.get("diagnostic_samples", defaults.diagnostic_samples),
        f"{location}.diagnostic_samples",
    )
    if nsamples < 2:
        raise ConfigError(f"{location}.nsamples_per_sheet must be at least two")
    if diagnostics < 3:
        raise ConfigError(f"{location}.diagnostic_samples must be at least three")
    return RayTracingConfig(
        nsamples_per_sheet=nsamples,
        diagnostic_samples=diagnostics,
        maximum_path_length_grid_lengths=_positive_number(
            raytracing.get(
                "maximum_path_length_grid_lengths",
                defaults.maximum_path_length_grid_lengths,
            ),
            f"{location}.maximum_path_length_grid_lengths",
        ),
        dt0=(
            _positive_number(raytracing["dt0"], f"{location}.dt0")
            if "dt0" in raytracing
            else defaults.dt0
        ),
        rtol=_positive_number(
            raytracing.get("rtol", defaults.rtol), f"{location}.rtol"
        ),
        atol=_positive_number(
            raytracing.get("atol", defaults.atol), f"{location}.atol"
        ),
        max_steps=_positive_integer(
            raytracing.get("max_steps", defaults.max_steps),
            f"{location}.max_steps",
        ),
        area_floor=_positive_number(
            raytracing.get("area_floor", defaults.area_floor),
            f"{location}.area_floor",
        ),
        minimum_amplitude_cap=_positive_number(
            raytracing.get("minimum_amplitude_cap", defaults.minimum_amplitude_cap),
            f"{location}.minimum_amplitude_cap",
        ),
    )


def _parse_physics(raw_physics: Any) -> PhysicsConfig:
    if raw_physics is None:
        return PhysicsConfig(InverseBremsstrahlungOptions())
    physics = _expect_table(raw_physics, "[physics]")
    _reject_unknown(physics, {"inverse_bremsstrahlung"}, "[physics]")
    raw_inverse = physics.get("inverse_bremsstrahlung", {})
    inverse = _expect_table(raw_inverse, "[physics.inverse_bremsstrahlung]")
    allowed = {
        "enabled",
        "minimum_coulomb_log",
        "coulomb_log_override",
        "critical_collision_frequency_hz",
    }
    _reject_unknown(inverse, allowed, "[physics.inverse_bremsstrahlung]")
    minimum = _positive_number(
        inverse.get("minimum_coulomb_log", 2.0),
        "[physics.inverse_bremsstrahlung].minimum_coulomb_log",
    )
    override = (
        _positive_number(
            inverse["coulomb_log_override"],
            "[physics.inverse_bremsstrahlung].coulomb_log_override",
        )
        if "coulomb_log_override" in inverse
        else None
    )
    critical_collision_frequency = (
        _positive_number(
            inverse["critical_collision_frequency_hz"],
            "[physics.inverse_bremsstrahlung].critical_collision_frequency_hz",
        )
        if "critical_collision_frequency_hz" in inverse
        else None
    )
    options = InverseBremsstrahlungOptions(
        enabled=_boolean(
            inverse.get("enabled", False),
            "[physics.inverse_bremsstrahlung].enabled",
        ),
        minimum_coulomb_log=minimum,
        coulomb_log_override=override,
        critical_collision_frequency_hz=critical_collision_frequency,
    )
    try:
        options.validate()
    except ValueError as error:
        raise ConfigError(f"[physics.inverse_bremsstrahlung]: {error}") from error
    return PhysicsConfig(options)


def _parse_grid(raw_grid: Any) -> GridConfig:
    location = "[grid]"
    grid = _expect_table(raw_grid, location)
    allowed = {
        "geometry",
        "dimensions",
        "inactive_axis_lengths_m",
        "axes",
        "initial_condition",
        "safe_state",
        "composition",
    }
    _reject_unknown(grid, allowed, location)
    _require(grid, {"geometry", "axes"}, location)

    try:
        geometry = Geometry.parse(grid["geometry"])
    except (TypeError, ValueError) as error:
        raise ConfigError(f"{location}.geometry: {error}") from error

    dimensions = _positive_integer(grid.get("dimensions", 3), "[grid].dimensions")
    if dimensions > 3:
        raise ConfigError("[grid].dimensions must be one, two, or three")
    supported = {
        1: (Geometry.CARTESIAN,),
        2: (Geometry.CARTESIAN, Geometry.CYLINDRICAL),
        3: tuple(Geometry),
    }[dimensions]
    if geometry not in supported:
        choices = ", ".join(item.value for item in supported)
        raise ConfigError(
            f"[grid].geometry is not supported in {dimensions}D; expected: {choices}"
        )

    raw_inactive_lengths = grid.get("inactive_axis_lengths_m", [1.0] * (3 - dimensions))
    inactive_axis_lengths_m = _number_list(
        raw_inactive_lengths, "[grid].inactive_axis_lengths_m"
    )
    if len(inactive_axis_lengths_m) != 3 - dimensions:
        raise ConfigError(
            "[grid].inactive_axis_lengths_m must contain one value per inactive axis"
        )
    if any(length <= 0.0 for length in inactive_axis_lengths_m):
        raise ConfigError("[grid].inactive_axis_lengths_m values must be positive")

    axes = _expect_table(grid["axes"], "[grid.axes]")
    expected_names = _AXIS_NAMES[geometry][:dimensions]
    _reject_unknown(axes, set(expected_names), "[grid.axes]")
    _require(axes, set(expected_names), "[grid.axes]")
    angular_names = _ANGULAR_AXES[geometry]
    parsed_axes = [
        _parse_axis(axes[name], name=name, angular=name in angular_names)
        for name in expected_names
    ]
    extents = tuple(item[0] for item in parsed_axes)
    ncells = tuple(item[1] for item in parsed_axes)
    graded_axes = tuple(item[2] for item in parsed_axes)
    initial_condition, parameters = _parse_initial_condition(
        grid.get("initial_condition")
    )
    return GridConfig(
        geometry=geometry,
        dimensions=dimensions,
        extents=extents,
        ncells=ncells,
        graded_axes=graded_axes,
        inactive_axis_lengths_m=inactive_axis_lengths_m,
        initial_condition=initial_condition,
        initial_condition_parameters=parameters,
        safe_state=_parse_safe_state(grid.get("safe_state")),
        composition=_parse_composition(grid.get("composition")),
    )


def load_simulation_config(path: str | Path) -> SimulationConfig:
    """Load a TOML simulation deck and strictly validate its grid section."""
    source = Path(path)
    try:
        with source.open("rb") as stream:
            raw_config = tomllib.load(stream)
    except tomllib.TOMLDecodeError as error:
        raise ConfigError(
            f"could not parse TOML input deck {source}: {error}"
        ) from error
    except OSError as error:
        raise ConfigError(f"could not read input deck {source}: {error}") from error

    if "grid" not in raw_config:
        raise ConfigError("input deck must contain a [grid] section")
    grid = _parse_grid(raw_config["grid"])
    beams = (
        _parse_beams(raw_config["beams"], source, grid)
        if "beams" in raw_config
        else None
    )
    raytracing = _parse_raytracing(raw_config.get("raytracing"))
    physics = _parse_physics(raw_config.get("physics"))
    extra_sections = {
        name: value
        for name, value in raw_config.items()
        if name not in {"grid", "beams", "raytracing", "physics"}
    }
    return SimulationConfig(
        grid=grid,
        beams=beams,
        raytracing=raytracing,
        physics=physics,
        extra_sections=extra_sections,
        source=source,
    )
