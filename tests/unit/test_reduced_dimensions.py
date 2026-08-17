import math

import jax.numpy as jnp
import numpy as np
import pytest

from pyGATH.beam import initialize_rays
from pyGATH.fields import (
    deposit_simplicial_power,
    grid_cell_volumes,
    interpolate_simplicial_fields,
    interpolate_simplicial_fields_to_cells,
    simplicialise_sheet_fields,
)
from pyGATH.grid import Grid, HydroFields, SafeHydroState
from pyGATH.io import load_simulation_config
from pyGATH.io.beams_io import load_beams_csv
from pyGATH.raytracing import (
    RAY_SHEET_LAYOUT,
    RAY_STATE_LAYOUT,
    RayTracingOptions,
    grid_characteristic_length,
    trace_rays,
)


def _write_beam(path):
    path.write_text(
        "beam_id,origin_geometry,origin_1,origin_2,origin_3,"
        "target_geometry,target_1,target_2,target_3,width_x_m,width_y_m,"
        "rotation_pi,supergaussian_index,power_fraction\n"
        "beam,cartesian,-1,0,0,cartesian,0,0,0,0.2,0.3,0.25,2,1\n",
        encoding="utf-8",
    )


def test_reduced_grids_generate_reference_axes_and_effective_volumes():
    one_d = Grid.create(
        geom="cartesian",
        dimensions=1,
        extents=((0.0, 2.0),),
        ncells=(2,),
        inactive_axis_lengths_m=(3.0, 4.0),
    )
    assert one_d.ncells == (2, 1, 1)
    np.testing.assert_allclose(one_d.yb, (-1.5, 1.5))
    np.testing.assert_allclose(one_d.zb, (-2.0, 2.0))
    np.testing.assert_allclose(grid_cell_volumes(one_d), 12.0)

    polar = Grid.create(
        geom="cylindrical",
        dimensions=2,
        extents=((1.0, 2.0), (0.0, 0.5 * np.pi)),
        ncells=(1, 1),
        inactive_axis_lengths_m=(2.0,),
    )
    assert polar.ncells == (1, 1, 1)
    np.testing.assert_allclose(grid_cell_volumes(polar), 1.5 * np.pi)


def test_reduced_config_requires_only_active_axes(tmp_path):
    deck = tmp_path / "one_d.toml"
    deck.write_text(
        """
[grid]
geometry = "cartesian"
dimensions = 1
inactive_axis_lengths_m = [2.0, 3.0]

[grid.axes.x]
min = 0.0
max = 1.0
spacing = "uniform"
ncells = 4
""",
        encoding="utf-8",
    )
    config = load_simulation_config(deck)
    grid = config.build_grid()
    assert config.grid.dimensions == 1
    assert grid.ncells == (4, 1, 1)
    np.testing.assert_allclose(grid.inactive_axis_lengths_m, (2.0, 3.0))


def test_reduced_grid_rejects_out_of_plane_hydro_and_safe_vectors():
    def varying_y(coordinates):
        x, y, z = coordinates.vertex_mesh()
        del x, z
        shape = coordinates.vertex_shape
        return HydroFields(
            ne=1.0 + y,
            Te=jnp.ones(shape),
            Ti=jnp.ones(shape),
            velocity=jnp.zeros((*shape, 3)),
        )

    with pytest.raises(ValueError, match="ne must be invariant"):
        Grid.create(
            geom="cartesian",
            dimensions=1,
            extents=((0.0, 1.0),),
            ncells=(2,),
            initial_condition=varying_y,
        )

    with pytest.raises(ValueError, match="safe-state grad_ne"):
        Grid.create(
            geom="cartesian",
            dimensions=1,
            extents=((0.0, 1.0),),
            ncells=(2,),
            safe_state=SafeHydroState(grad_ne=(0.0, 1.0, 0.0)),
        )


def test_inactive_reference_lengths_do_not_set_ray_tracing_length_scale():
    grid = Grid.create(
        geom="cartesian",
        dimensions=1,
        extents=((0.0, 2.0e-6),),
        ncells=(2,),
    )
    np.testing.assert_allclose(grid_characteristic_length(grid), 2.0e-6)


def test_inactive_coordinates_are_projected_onto_the_reference_extrusion(tmp_path):
    one_d_file = tmp_path / "one_d_offset.csv"
    _write_beam(one_d_file)
    one_d_file.write_text(
        one_d_file.read_text(encoding="utf-8").replace(
            "cartesian,-1,0,0,cartesian,0,0,0",
            "cartesian,-1,10,-7,cartesian,0,10,-7",
        ),
        encoding="utf-8",
    )
    grid = Grid.create(
        geom="cartesian",
        dimensions=1,
        extents=((0.0, 1.0),),
        ncells=(2,),
    )
    hydro = grid.interpolate(np.asarray((0.5, 10.0, -7.0)))
    assert bool(hydro.inside)
    assert bool(grid.contains(np.asarray((0.5, 10.0, -7.0))))
    beams = load_beams_csv(one_d_file, dimensions=1)
    rays = initialize_rays(beams, grid, launch_padding=1.0)
    inactive_positions = np.asarray(rays.state[..., RAY_STATE_LAYOUT.position][..., 1:])
    np.testing.assert_allclose(
        inactive_positions,
        np.broadcast_to((10.0, -7.0), inactive_positions.shape),
    )

    two_d_file = tmp_path / "two_d_offset.csv"
    _write_beam(two_d_file)
    two_d_file.write_text(
        two_d_file.read_text(encoding="utf-8").replace(
            "cartesian,-1,0,0,cartesian,0,0,0",
            "cartesian,-1,0,8,cartesian,0,0,8",
        ),
        encoding="utf-8",
    )
    polar = Grid.create(
        geom="cylindrical",
        dimensions=2,
        extents=((0.0, 1.0), (-np.pi, np.pi)),
        ncells=(2, 4),
    )
    polar_beams = load_beams_csv(two_d_file, dimensions=2)
    polar_rays = initialize_rays(polar_beams, polar, nrays_axis1=3, launch_padding=1.0)
    np.testing.assert_allclose(
        polar_rays.state[..., RAY_STATE_LAYOUT.position][..., 2],
        np.full((1, 3, 1), 8.0),
    )


def test_reduced_beam_power_normalization_and_primary_sampling(tmp_path):
    beam_file = tmp_path / "beam.csv"
    _write_beam(beam_file)

    one_d_beams = load_beams_csv(
        beam_file,
        total_power_w=12.0,
        dimensions=1,
        inactive_axis_lengths_m=(3.0, 4.0),
    )
    np.testing.assert_allclose(one_d_beams.peak_intensity, 1.0)
    one_d_grid = Grid.create(
        geom="cartesian",
        dimensions=1,
        extents=((0.0, 1.0),),
        ncells=(8,),
        inactive_axis_lengths_m=(3.0, 4.0),
    )
    one_d_rays = initialize_rays(one_d_beams, one_d_grid, launch_padding=1.0)
    assert one_d_rays.state.shape == (1, 1, 1, RAY_STATE_LAYOUT.n_attributes)
    np.testing.assert_allclose(
        one_d_rays.state[..., RAY_STATE_LAYOUT.impact_parameter], 0.0
    )
    np.testing.assert_allclose(
        one_d_rays.state[..., RAY_STATE_LAYOUT.position][..., 1:], 0.0
    )

    two_d_beams = load_beams_csv(
        beam_file,
        total_power_w=2.0,
        dimensions=2,
        inactive_axis_lengths_m=(1.0,),
    )
    expected_integral = 2.0 * 0.2 * math.gamma(1.5)
    np.testing.assert_allclose(two_d_beams.peak_intensity, 2.0 / expected_integral)
    two_d_grid = Grid.create(
        geom="cartesian",
        dimensions=2,
        extents=((0.0, 1.0), (-1.0, 1.0)),
        ncells=(8, 8),
    )
    two_d_rays = initialize_rays(
        two_d_beams,
        two_d_grid,
        nrays_axis1=3,
        launch_padding=1.0,
    )
    assert two_d_rays.state.shape == (1, 3, 1, RAY_STATE_LAYOUT.n_attributes)
    np.testing.assert_allclose(
        two_d_rays.state[..., RAY_STATE_LAYOUT.position][..., 2], 0.0
    )
    np.testing.assert_allclose(
        two_d_rays.state[..., RAY_STATE_LAYOUT.momentum][..., 2], 0.0
    )


def test_reduced_beams_ignore_unused_spot_columns_and_check_reference_lengths(
    tmp_path,
):
    baseline_file = tmp_path / "baseline.csv"
    one_d_changed_file = tmp_path / "one_d_changed.csv"
    two_d_changed_file = tmp_path / "two_d_changed.csv"
    _write_beam(baseline_file)
    one_d_changed_file.write_text(
        baseline_file.read_text(encoding="utf-8").replace(
            "0.2,0.3,0.25,2,1", "5.0,7.0,0.9,2,1"
        ),
        encoding="utf-8",
    )
    two_d_changed_file.write_text(
        baseline_file.read_text(encoding="utf-8").replace(
            "0.2,0.3,0.25,2,1", "0.2,7.0,0.9,2,1"
        ),
        encoding="utf-8",
    )

    for dimensions, lengths, changed_file in (
        (1, (2.0, 3.0), one_d_changed_file),
        (2, (2.0,), two_d_changed_file),
    ):
        baseline = load_beams_csv(
            baseline_file,
            dimensions=dimensions,
            inactive_axis_lengths_m=lengths,
        )
        changed = load_beams_csv(
            changed_file,
            dimensions=dimensions,
            inactive_axis_lengths_m=lengths,
        )
        np.testing.assert_allclose(baseline.peak_intensity, changed.peak_intensity)
        extents = ((0.0, 1.0),) if dimensions == 1 else ((0.0, 1.0), (-1.0, 1.0))
        grid = Grid.create(
            geom="cartesian",
            dimensions=dimensions,
            extents=extents,
            ncells=(2,) * dimensions,
            inactive_axis_lengths_m=lengths,
        )
        count = 1 if dimensions == 1 else 3
        baseline_rays = initialize_rays(
            baseline, grid, nrays_axis1=count, launch_padding=1.0
        )
        changed_rays = initialize_rays(
            changed, grid, nrays_axis1=count, launch_padding=1.0
        )
        np.testing.assert_allclose(baseline_rays.state, changed_rays.state)

    grid = Grid.create(
        geom="cartesian",
        dimensions=1,
        extents=((0.0, 1.0),),
        ncells=(2,),
    )
    mismatched = load_beams_csv(
        baseline_file,
        dimensions=1,
        inactive_axis_lengths_m=(2.0, 1.0),
    )
    with pytest.raises(ValueError, match="power normalization"):
        initialize_rays(mismatched, grid)


def test_one_dimensional_ray_trace_stays_on_the_beam_axis(tmp_path):
    beam_file = tmp_path / "beam.csv"
    _write_beam(beam_file)
    beams = load_beams_csv(
        beam_file,
        dimensions=1,
        inactive_axis_lengths_m=(1.0, 1.0),
    )
    grid = Grid.create(
        geom="cartesian",
        dimensions=1,
        extents=((0.0, 1.0),),
        ncells=(4,),
    )
    initial_rays = initialize_rays(beams, grid, launch_padding=1.0)
    result = trace_rays(
        initial_rays,
        grid,
        options=RayTracingOptions(
            nsamples_per_sheet=3,
            diagnostic_samples=16,
            maximum_path_length_grid_lengths=3.0,
            rtol=1.0e-6,
            atol=1.0e-9,
            max_steps=256,
        ),
    )

    primary = np.asarray(result.sheet_fields[..., RAY_STATE_LAYOUT.position])
    momentum = np.asarray(result.sheet_fields[..., RAY_STATE_LAYOUT.momentum])
    np.testing.assert_allclose(primary[..., 1:], 0.0, atol=1.0e-14)
    np.testing.assert_allclose(momentum[..., 1:], 0.0, atol=1.0e-14)
    assert bool(result.terminated)


@pytest.mark.parametrize(
    ("dimensions", "nrays_axis1", "nrays_axis2", "message"),
    (
        (1, 2, 1, "nrays_axis1"),
        (2, 2, 2, "nrays_axis2"),
    ),
)
def test_reduced_initialization_rejects_conflicting_counts(
    tmp_path, dimensions, nrays_axis1, nrays_axis2, message
):
    beam_file = tmp_path / "beam.csv"
    _write_beam(beam_file)
    lengths = (1.0,) * (3 - dimensions)
    beams = load_beams_csv(
        beam_file,
        dimensions=dimensions,
        inactive_axis_lengths_m=lengths,
    )
    extents = ((0.0, 1.0),) if dimensions == 1 else ((0.0, 1.0), (-1.0, 1.0))
    grid = Grid.create(
        geom="cartesian",
        dimensions=dimensions,
        extents=extents,
        ncells=(2,) * dimensions,
    )
    with pytest.raises(ValueError, match=message):
        initialize_rays(
            beams,
            grid,
            nrays_axis1=nrays_axis1,
            nrays_axis2=nrays_axis2,
        )


def test_segment_field_interpolates_linearly_through_inactive_directions():
    fields = np.zeros((1, 1, 1, 1, 3, RAY_SHEET_LAYOUT.n_attributes))
    for sample in range(3):
        fields[0, 0, 0, 0, sample, RAY_STATE_LAYOUT.position] = (
            float(sample),
            0.2,
            -0.3,
        )
        fields[0, 0, 0, 0, sample, RAY_STATE_LAYOUT.ray_power] = 1.0 + 2.0 * sample
    field = simplicialise_sheet_fields(fields, dimension=1, fields="ray_power")
    result = interpolate_simplicial_fields(
        field, np.asarray(((0.5, 20.0, -10.0), (1.5, -3.0, 4.0)))
    )

    assert field.mesh.connectivity.shape == (2, 2)
    np.testing.assert_array_equal(result.inside, True)
    np.testing.assert_allclose(result.values[0, 0, :, 0], (2.0, 4.0))


def test_triangular_field_and_cell_centre_interpolation_are_affine_exact():
    fields = np.zeros((1, 1, 2, 1, 2, RAY_SHEET_LAYOUT.n_attributes))
    for ray in range(2):
        for sample in range(2):
            position = np.asarray((float(sample), float(ray), 0.25))
            fields[0, 0, ray, 0, sample, RAY_STATE_LAYOUT.position] = position
            fields[0, 0, ray, 0, sample, RAY_STATE_LAYOUT.ray_power] = (
                1.0 + 2.0 * position[0] + 3.0 * position[1]
            )
    field = simplicialise_sheet_fields(fields, dimension=2, fields="ray_power")
    point_result = interpolate_simplicial_fields(
        field, np.asarray(((0.25, 0.5, -100.0),))
    )
    np.testing.assert_allclose(point_result.values[0, 0, 0, 0], 3.0)

    grid = Grid.create(
        geom="cartesian",
        dimensions=2,
        extents=((0.0, 1.0), (0.0, 1.0)),
        ncells=(2, 2),
    )
    cells = interpolate_simplicial_fields_to_cells(field, grid, point_batch_size=3)
    assert cells.values.shape == (1, 1, 2, 2, 1, 1)
    x, y = np.meshgrid(np.asarray(grid.xc), np.asarray(grid.yc), indexing="ij")
    np.testing.assert_allclose(cells.values[0, 0, :, :, 0, 0], 1.0 + 2.0 * x + 3.0 * y)
    np.testing.assert_array_equal(cells.inside, True)


@pytest.mark.parametrize(
    ("dimension", "inactive_lengths", "source_value", "expected_power"),
    (
        (1, (2.0, 3.0), 4.0, 24.0),
        (2, (2.0,), 5.0, 10.0),
    ),
)
def test_reduced_simplicial_deposition_is_conservative_and_volumetric(
    dimension, inactive_lengths, source_value, expected_power
):
    if dimension == 1:
        fields = np.zeros((1, 1, 1, 1, 3, RAY_SHEET_LAYOUT.n_attributes))
        for sample, x in enumerate((0.0, 0.5, 1.0)):
            fields[0, 0, 0, 0, sample, RAY_STATE_LAYOUT.position] = (x, 2.0, -3.0)
        extents = ((0.0, 1.0),)
        ncells = (2,)
    else:
        fields = np.zeros((1, 1, 2, 1, 2, RAY_SHEET_LAYOUT.n_attributes))
        for ray in range(2):
            for sample in range(2):
                fields[0, 0, ray, 0, sample, RAY_STATE_LAYOUT.position] = (
                    float(sample),
                    float(ray),
                    4.0,
                )
        extents = ((0.0, 1.0), (0.0, 1.0))
        ncells = (1, 1)
    fields[..., RAY_SHEET_LAYOUT.inverse_brems_deposition] = source_value
    field = simplicialise_sheet_fields(
        fields, dimension=dimension, fields="inverse_brems_deposition"
    )
    grid = Grid.create(
        geom="cartesian",
        dimensions=dimension,
        extents=extents,
        ncells=ncells,
        inactive_axis_lengths_m=inactive_lengths,
    )
    deposition = deposit_simplicial_power(
        field,
        grid,
        relative_tolerance=0.0,
        max_subdivision_levels=2,
        simplex_batch_size=4,
    )

    np.testing.assert_allclose(deposition.source_power, expected_power, rtol=1.0e-13)
    np.testing.assert_allclose(deposition.deposited_power, expected_power, rtol=1.0e-13)
    np.testing.assert_allclose(deposition.outside_power, 0.0, atol=1.0e-14)
    np.testing.assert_allclose(deposition.power_density, source_value, rtol=1.0e-13)
    np.testing.assert_allclose(deposition.conservation_error, 0.0, atol=1.0e-13)
