"""Shared workflow helpers used by the example notebooks and regression tests.

This private module keeps the notebooks focused on explanation and plots. It is
not intended to be run directly; the notebooks are the user-facing examples.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from pyGATH.fields import (
    GridPowerDeposition,
    deposit_simplicial_power,
    grid_cell_volumes,
    interpolate_simplicial_fields_batched,
    interpolate_simplicial_fields_to_cells,
    simplicialise_sheet_fields,
)
from pyGATH.grid import Geometry, convert_positions
from pyGATH.io import load_simulation_config
from pyGATH.raytracing import (
    RAY_SHEET_LAYOUT,
    RAY_STATE_LAYOUT,
    SPEED_OF_LIGHT,
    VACUUM_PERMITTIVITY,
    critical_density,
    inverse_bremsstrahlung_depth_derivative,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIRECTORY = PROJECT_ROOT / "configs" / "example_configs"
UNIFORM_1D_CONFIG = CONFIG_DIRECTORY / "uniform_1d_deposition.toml"
PAPER_S64_CONFIGS = {
    "cartesian": CONFIG_DIRECTORY / "paper_s64_ten_beam_cartesian_deposition.toml",
    "cylindrical": CONFIG_DIRECTORY / "paper_s64_ten_beam_cylindrical_deposition.toml",
}
PAPER_S64_NO_CBET_ABSORPTION = 0.919
ONE_DIMENSIONAL_FIELD_CONFIG = (
    CONFIG_DIRECTORY / "one_dimensional_field_reconstruction.toml"
)
TWO_DIMENSIONAL_FIELD_CONFIGS = {
    "cartesian": (
        CONFIG_DIRECTORY / "two_dimensional_cartesian_field_reconstruction.toml"
    ),
    "cylindrical": (
        CONFIG_DIRECTORY / "two_dimensional_cylindrical_field_reconstruction.toml"
    ),
}
LINEAR_GRADIENT_CONFIG = (
    PROJECT_ROOT / "configs" / "test_configs" / "linear_gradient_turning.toml"
)
ELECTRON_MASS = 9.109_383_713_9e-31
ELEMENTARY_CHARGE = 1.602_176_634e-19
_RECONSTRUCTION_FIELDS = (
    "phase_length",
    "initial_electric_field",
    "electric_field",
)


def coherent_cell_deposition(trace, grid, beams, inverse_bremsstrahlung):
    """Reconstruct each beam's two sheets coherently at hydro-cell centres."""
    field = simplicialise_sheet_fields(
        trace.sheet_fields,
        dimension=grid.dimensions,
        fields=("phase_length", "electric_field"),
    )
    sampled = interpolate_simplicial_fields_to_cells(field, grid, point_batch_size=4096)
    values = np.asarray(sampled.values)
    inside = np.asarray(sampled.inside)
    phase = (
        np.asarray(beams.omega)[:, None, None, None, None]
        / SPEED_OF_LIGHT
        * values[..., sampled.selection.phase_length]
    )
    phase -= 0.5 * np.pi * np.arange(2)[None, :, None, None, None]
    sheet_field = np.where(
        inside,
        values[..., sampled.selection.electric_field] * np.exp(1j * phase),
        0.0,
    )
    coherent_field_squared = np.abs(np.sum(sheet_field, axis=1)) ** 2

    native_centres = np.stack(
        np.meshgrid(
            np.asarray(grid.xc),
            np.asarray(grid.yc),
            np.asarray(grid.zc),
            indexing="ij",
        ),
        axis=-1,
    )
    cartesian_centres = convert_positions(native_centres, grid.geom, Geometry.CARTESIAN)
    hydro = grid.interpolate(cartesian_centres)
    omega = np.asarray(beams.omega)[:, None, None, None]
    rate = inverse_bremsstrahlung_depth_derivative(
        np.asarray(hydro.ne)[None, ...],
        np.asarray(hydro.Te)[None, ...],
        omega,
        grid.composition.effective_charge,
        minimum_coulomb_log=inverse_bremsstrahlung.minimum_coulomb_log,
        coulomb_log_override=inverse_bremsstrahlung.coulomb_log_override,
        critical_collision_frequency_hz=(
            inverse_bremsstrahlung.critical_collision_frequency_hz
        ),
        inside=np.asarray(hydro.inside)[None, ...],
    )
    power_density = np.sum(
        0.5
        * VACUUM_PERMITTIVITY
        * SPEED_OF_LIGHT
        * coherent_field_squared
        * np.asarray(rate),
        axis=0,
    )
    volumes = np.asarray(grid_cell_volumes(grid))
    cell_power = power_density * volumes
    deposited_power = np.sum(cell_power)
    return GridPowerDeposition(
        power_density=power_density,
        cell_power=cell_power,
        deposited_power=deposited_power,
        outside_power=np.asarray(0.0),
        source_power=deposited_power,
    )


def run_simulation(simulation, *, return_coherent: bool = False):
    """Trace a loaded simulation and deposit its inverse-bremsstrahlung source.

    Set ``return_coherent=True`` to include the coherent cell-centred
    deposition between the raw conservative deposition and the checks mapping.
    """
    grid = simulation.build_grid()
    beams = simulation.load_beams()
    initial_rays = simulation.initialize_rays(grid, beams=beams)
    trace = simulation.trace_rays(initial_rays, grid)
    field = simplicialise_sheet_fields(
        trace.sheet_fields,
        dimension=grid.dimensions,
        fields="inverse_brems_deposition",
    )
    options = simulation.extra_sections.get("deposition", {})
    deposition = deposit_simplicial_power(
        field,
        grid,
        max_subdivision_levels=int(options.get("max_subdivision_levels", 2)),
        relative_tolerance=float(options.get("relative_tolerance", 1.0e-3)),
        simplex_batch_size=int(options.get("simplex_batch_size", 4096)),
    )
    coherent_deposition = (
        coherent_cell_deposition(
            trace, grid, beams, simulation.physics.inverse_bremsstrahlung
        )
        if grid.dimensions == 2
        else None
    )

    sheet_fields = np.asarray(trace.sheet_fields)
    terminal_depth = np.max(
        sheet_fields[..., RAY_STATE_LAYOUT.inverse_brems_depth], axis=(1, 4)
    )
    ray_power_fraction = np.asarray(initial_rays.state[..., RAY_STATE_LAYOUT.ray_power])
    direct_absorption_fraction = float(
        np.sum(ray_power_fraction * (1.0 - np.exp(-terminal_depth)))
    )
    incident_power = float(initial_rays.total_incident_power)
    checks = {
        "incident_power_w": incident_power,
        "direct_absorption_fraction": direct_absorption_fraction,
        "source_absorption_fraction": float(deposition.source_power) / incident_power,
        "deposited_absorption_fraction": (
            float(deposition.deposited_power) / incident_power
        ),
        "outside_power_fraction": float(deposition.outside_power) / incident_power,
        "deposition_conservation_error_fraction": (
            float(deposition.conservation_error) / incident_power
        ),
        "maximum_cbet_depth": float(
            np.max(np.abs(sheet_fields[..., RAY_STATE_LAYOUT.cbet_depth]))
        ),
    }
    if coherent_deposition is not None:
        checks["coherent_deposited_absorption_fraction"] = (
            float(coherent_deposition.deposited_power) / incident_power
        )
    if return_coherent:
        return trace, grid, beams, deposition, coherent_deposition, checks
    return trace, grid, beams, deposition, checks


def run_case(config_path: Path):
    """Load and run one configured inverse-bremsstrahlung case."""
    return run_simulation(load_simulation_config(config_path))


def run_uniform_1d(config_path: Path = UNIFORM_1D_CONFIG):
    """Run the uniform 1-D case and append its analytical absorption."""
    trace, grid, beams, deposition, checks = run_case(config_path)
    omega = float(beams.omega[0])
    density = float(grid.hydro.ne[0, 0, 0])
    density_ratio = density / float(critical_density(omega))
    refractive_index = np.sqrt(1.0 - density_ratio)
    collision_frequency = load_simulation_config(
        config_path
    ).physics.inverse_bremsstrahlung.critical_collision_frequency_hz
    if collision_frequency is None:
        raise RuntimeError("the analytical case requires a critical collision rate")
    length = float(grid.xb[-1] - grid.xb[0])
    optical_depth = (
        2.0
        * collision_frequency
        / SPEED_OF_LIGHT
        * density_ratio**2
        * length
        / refractive_index
    )
    checks.update(
        {
            "analytical_optical_depth": optical_depth,
            "analytical_absorption_fraction": 1.0 - np.exp(-optical_depth),
        }
    )
    return trace, grid, beams, deposition, checks


def run_paper_s64(configs: dict[str, Path] | None = None):
    """Run both coordinate representations of the ten-beam paper case."""
    selected = PAPER_S64_CONFIGS if configs is None else configs
    return {geometry: run_case(path) for geometry, path in selected.items()}


def reconstruct_one_dimensional_electric_field(result, grid, omega_rad_s: float):
    """Interpolate and coherently sum both sheets of a one-dimensional trace."""
    source_fields = np.array(result.sheet_fields, copy=True)
    source_fields[..., RAY_STATE_LAYOUT.initial_electric_field] *= source_fields[
        ..., RAY_SHEET_LAYOUT.uncapped_amplitude
    ]
    field = simplicialise_sheet_fields(
        source_fields,
        dimension=1,
        fields=("phase_length", "initial_electric_field", "electric_field"),
    )
    sampled = interpolate_simplicial_fields_to_cells(field, grid)
    values = np.asarray(sampled.values[0, :, :, 0, 0])
    inside = np.asarray(sampled.inside[0, :, :, 0, 0])
    selection = sampled.selection

    phase_length = values[..., selection.phase_length]
    incident_indices = np.flatnonzero(inside[0])
    if incident_indices.size == 0:
        raise RuntimeError("the incident sheet does not cover any grid cell centres")
    phase_reference = phase_length[0, incident_indices[0]]
    phase = omega_rad_s / SPEED_OF_LIGHT * (phase_length - phase_reference)
    phase -= 0.5 * np.pi * np.arange(values.shape[0])[:, None]

    initial_field = float(
        np.asarray(result.sheet_fields)[
            0, 0, 0, 0, 0, RAY_STATE_LAYOUT.initial_electric_field
        ]
    )
    uncapped_magnitude = values[..., selection.initial_electric_field]
    capped_magnitude = values[..., selection.electric_field]
    phasor = np.exp(1j * phase)
    uncapped_sheets = np.where(inside, uncapped_magnitude * phasor, 0.0)
    capped_sheets = np.where(inside, capped_magnitude * phasor, 0.0)

    return {
        "x_m": np.asarray(grid.xc),
        "inside": inside,
        "phase_rad": phase,
        "initial_electric_field_v_m": initial_field,
        "quiver_amplitude_per_v_m": (
            ELEMENTARY_CHARGE / (ELECTRON_MASS * SPEED_OF_LIGHT * omega_rad_s)
        ),
        "uncapped_sheet_magnitude_v_m": uncapped_magnitude,
        "capped_sheet_magnitude_v_m": capped_magnitude,
        "uncapped_field_v_m": np.sum(uncapped_sheets, axis=0),
        "capped_field_v_m": np.sum(capped_sheets, axis=0),
    }


def run_one_dimensional_field_reconstruction(
    config: Path = ONE_DIMENSIONAL_FIELD_CONFIG,
):
    """Trace and reconstruct the one-dimensional reflected-beam case."""
    simulation = load_simulation_config(config)
    grid = simulation.build_grid()
    beams = simulation.load_beams()
    initial_rays = simulation.initialize_rays(grid, beams=beams)
    result = simulation.trace_rays(initial_rays, grid)
    omega = float(beams.omega[0])
    reconstruction = reconstruct_one_dimensional_electric_field(result, grid, omega)

    parameters = simulation.grid.initial_condition_parameters
    density_radius = 343.0e-6 * float(parameters["scale"])
    critical_radius = density_radius * float(
        parameters["density_at_scale_over_ncritical"]
    ) ** (1.0 / float(parameters["density_power"]))
    expected_critical_x = float(parameters["coordinate_offset_m"]) - critical_radius
    sheet_fields = np.asarray(result.sheet_fields)
    numerical_caustic_x = float(
        sheet_fields[0, 0, 0, 0, -1, RAY_STATE_LAYOUT.position][0]
    )
    caustic_density_ratio = float(
        grid.interpolate(np.asarray((numerical_caustic_x, 0.0, 0.0))).ne
        / critical_density(omega)
    )
    maximum_deposition = float(
        np.max(np.abs(sheet_fields[..., RAY_SHEET_LAYOUT.inverse_brems_deposition]))
    )

    if grid.dimensions != 1:
        raise RuntimeError("the field-reconstruction example must use a 1-D grid")
    if not bool(result.terminated) or not bool(result.has_caustic[0, 0, 0]):
        raise RuntimeError("the paper ray did not reflect and exit through a caustic")
    if not np.isclose(numerical_caustic_x, expected_critical_x, rtol=2.0e-3):
        raise RuntimeError("the numerical caustic does not match the critical surface")
    if maximum_deposition != 0.0:
        raise RuntimeError("inverse-bremsstrahlung deposition must be disabled")

    checks = {
        "expected_critical_x_m": expected_critical_x,
        "numerical_caustic_x_m": numerical_caustic_x,
        "caustic_density_over_ncritical": caustic_density_ratio,
        "maximum_inverse_brems_deposition_w_m3": maximum_deposition,
    }
    return result, grid, beams, reconstruction, checks


def _coherent_sum(values, inside, selection, omega_rad_s, phase_reference):
    phase = (
        omega_rad_s
        / SPEED_OF_LIGHT
        * (values[..., selection.phase_length] - phase_reference)
    )
    phase -= (
        0.5
        * np.pi
        * np.arange(values.shape[0]).reshape(
            (values.shape[0],) + (1,) * (values.ndim - 2)
        )
    )
    phasor = np.exp(1j * phase)
    uncapped_magnitude = values[..., selection.initial_electric_field]
    capped_magnitude = values[..., selection.electric_field]
    uncapped_sheets = np.where(inside, uncapped_magnitude * phasor, 0.0)
    capped_sheets = np.where(inside, capped_magnitude * phasor, 0.0)
    return {
        "inside": inside,
        "phase_rad": phase,
        "uncapped_sheet_magnitude_v_m": uncapped_magnitude,
        "capped_sheet_magnitude_v_m": capped_magnitude,
        "uncapped_field_v_m": np.sum(uncapped_sheets, axis=0),
        "capped_field_v_m": np.sum(capped_sheets, axis=0),
    }


def reconstruct_two_dimensional_electric_field(result, grid, omega_rad_s: float):
    """Triangulate and coherently reconstruct a two-dimensional trace."""
    source_fields = np.array(result.sheet_fields, copy=True)
    peak_initial_field = float(
        np.max(source_fields[0, 0, :, 0, 0, RAY_STATE_LAYOUT.initial_electric_field])
    )
    source_fields[..., RAY_STATE_LAYOUT.initial_electric_field] *= source_fields[
        ..., RAY_SHEET_LAYOUT.uncapped_amplitude
    ]
    field = simplicialise_sheet_fields(
        source_fields,
        dimension=2,
        fields=_RECONSTRUCTION_FIELDS,
    )

    line_x = np.linspace(-15.5e-6, -4.5e-6, 1400)
    line_points = np.stack(
        (line_x, np.zeros_like(line_x), np.zeros_like(line_x)), axis=-1
    )
    line_sample = interpolate_simplicial_fields_batched(
        field, line_points, point_batch_size=2048
    )
    line_values = np.asarray(line_sample.values[0])
    line_inside = np.asarray(line_sample.inside[0])
    incident_line = np.flatnonzero(line_inside[0])
    if incident_line.size == 0:
        raise RuntimeError("the incident sheet does not cover the comparison line")
    phase_reference = line_values[
        0, incident_line[0], line_sample.selection.phase_length
    ]
    line_reconstruction = _coherent_sum(
        line_values,
        line_inside,
        line_sample.selection,
        omega_rad_s,
        phase_reference,
    )
    line_reconstruction["x_m"] = line_x

    cell_sample = interpolate_simplicial_fields_to_cells(
        field, grid, point_batch_size=4096
    )
    cell_values = np.asarray(cell_sample.values[0, :, :, :, 0])
    cell_inside = np.asarray(cell_sample.inside[0, :, :, :, 0])
    cell_reconstruction = _coherent_sum(
        cell_values,
        cell_inside,
        cell_sample.selection,
        omega_rad_s,
        phase_reference,
    )

    first, second = np.meshgrid(np.asarray(grid.xc), np.asarray(grid.yc), indexing="ij")
    native_centres = np.stack((first, second, np.zeros_like(first)), axis=-1)
    cartesian_centres = np.asarray(
        convert_positions(native_centres, grid.geom, Geometry.CARTESIAN)
    )
    first_boundary, second_boundary = np.meshgrid(
        np.asarray(grid.xb), np.asarray(grid.yb), indexing="ij"
    )
    native_boundaries = np.stack(
        (first_boundary, second_boundary, np.zeros_like(first_boundary)), axis=-1
    )
    cartesian_boundaries = np.asarray(
        convert_positions(native_boundaries, grid.geom, Geometry.CARTESIAN)
    )
    cell_reconstruction.update(
        {
            "peak_initial_electric_field_v_m": peak_initial_field,
            "quiver_amplitude_per_v_m": (
                ELEMENTARY_CHARGE / (ELECTRON_MASS * SPEED_OF_LIGHT * omega_rad_s)
            ),
            "native_centres": native_centres,
            "cartesian_centres_m": cartesian_centres,
            "cartesian_boundaries_m": cartesian_boundaries,
            "lineout": line_reconstruction,
        }
    )
    return cell_reconstruction


def run_two_dimensional_field_reconstruction(config: Path):
    """Trace and reconstruct one coordinate representation of the 2-D case."""
    simulation = load_simulation_config(config)
    grid = simulation.build_grid()
    beams = simulation.load_beams()
    initial_rays = simulation.initialize_rays(grid, beams=beams)
    result = simulation.trace_rays(initial_rays, grid)
    omega = float(beams.omega[0])
    reconstruction = reconstruct_two_dimensional_electric_field(result, grid, omega)
    sheet_fields = np.asarray(result.sheet_fields)
    caustic_mask = np.asarray(result.has_caustic[0, :, 0], dtype=bool)
    caustic_positions = sheet_fields[0, 0, :, 0, -1, RAY_STATE_LAYOUT.position][
        caustic_mask
    ]
    caustic_density_ratio = np.asarray(grid.interpolate(caustic_positions).ne) / float(
        critical_density(omega)
    )
    maximum_deposition = float(
        np.max(np.abs(sheet_fields[..., RAY_SHEET_LAYOUT.inverse_brems_deposition]))
    )
    active = np.any(reconstruction["inside"], axis=0)

    if grid.dimensions != 2:
        raise RuntimeError("the field-reconstruction example must use a 2-D grid")
    if not bool(result.terminated) or not np.any(caustic_mask):
        raise RuntimeError("the 2-D beam did not produce reflected caustic sheets")
    if maximum_deposition != 0.0:
        raise RuntimeError("inverse-bremsstrahlung deposition must be disabled")
    if not np.all(np.isfinite(reconstruction["capped_field_v_m"][active])):
        raise RuntimeError("the capped coherent field contains non-finite values")

    checks = {
        "geometry": grid.geom.value,
        "rays_with_caustics": int(np.count_nonzero(caustic_mask)),
        "total_rays": int(caustic_mask.size),
        "maximum_caustic_density_over_ncritical": float(np.max(caustic_density_ratio)),
        "maximum_inverse_brems_deposition_w_m3": maximum_deposition,
    }
    return result, grid, beams, reconstruction, checks


def run_linear_gradient_turning(config: Path = LINEAR_GRADIENT_CONFIG):
    """Trace the single-ray linear-gradient regression workflow."""
    simulation = load_simulation_config(config)
    grid = simulation.build_grid()
    beams = simulation.load_beams()
    initial_rays = simulation.initialize_rays(grid, beams=beams)
    result = simulation.trace_rays(initial_rays, grid)

    direction = np.asarray(beams.direction[0])
    incidence_angle = np.arctan2(direction[1], direction[0])
    omega = float(beams.omega[0])
    density_scale = float(critical_density(omega)) / float(grid.grad_ne[0, 0, 0, 0])
    expected_density_ratio = np.cos(incidence_angle) ** 2
    expected_turn_x = density_scale * expected_density_ratio

    fields = np.asarray(result.sheet_fields[0, :, 0, 0])
    positions = fields[..., RAY_STATE_LAYOUT.position]
    numerical_turn_x = float(np.max(positions[..., 0]))
    caustic_x = float(positions[0, -1, 0])
    join_error = float(
        np.max(
            np.abs(
                fields[0, -1, : RAY_STATE_LAYOUT.n_attributes]
                - fields[1, 0, : RAY_STATE_LAYOUT.n_attributes]
            )
        )
    )

    if fields.shape[0] != 2:
        raise RuntimeError(f"expected two sheets, got {fields.shape[0]}")
    if not bool(result.has_caustic[0, 0, 0]):
        raise RuntimeError("the ray-tube caustic was not detected")
    if not np.isclose(numerical_turn_x, expected_turn_x, rtol=2.0e-3):
        raise RuntimeError(
            "numerical turning point does not match the linear-gradient solution"
        )
    if not np.isclose(caustic_x, expected_turn_x, rtol=3.0e-3):
        raise RuntimeError("detected caustic is not close to the ray turning point")
    if join_error > 1.0e-10:
        raise RuntimeError("the two sheets do not share their caustic sample")

    return result, {
        "incidence_angle_deg": float(np.degrees(incidence_angle)),
        "expected_density_ratio": float(expected_density_ratio),
        "expected_turn_x_m": float(expected_turn_x),
        "numerical_turn_x_m": numerical_turn_x,
        "caustic_x_m": caustic_x,
        "sheet_join_error": join_error,
    }
