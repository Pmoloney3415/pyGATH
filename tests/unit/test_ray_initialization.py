import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pyGATH.beam import initialize_rays
from pyGATH.grid import Grid, HydroFields
from pyGATH.io import load_beams_csv
from pyGATH.raytracing import (
    RAY_STATE_LAYOUT,
    SPEED_OF_LIGHT,
    VACUUM_PERMITTIVITY,
    critical_density,
    ray_grid_entry_distance,
    safe_density_ratio,
    safe_permittivity,
)


def _write_beams(path):
    path.write_text(
        "beam_id,origin_geometry,origin_1,origin_2,origin_3,"
        "target_geometry,target_1,target_2,target_3,"
        "width_x_m,width_y_m,rotation_pi,supergaussian_index,"
        "omega_rad_s,power_fraction\n"
        "one,cartesian,-3,0,0,cartesian,0,0,0,0.1,0.2,0,2,5.361e15,2\n"
        "two,cartesian,-3,0,0,cartesian,0,0,0,0.2,0.1,0.25,4,5.361e15,1\n",
        encoding="utf-8",
    )


def _cartesian_grid(initial_condition="uniform"):
    return Grid.create(
        geom="cartesian",
        extents=((-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)),
        ncells=(4, 4, 4),
        initial_condition=initial_condition,
    )


def test_ray_state_shape_layout_neighbours_and_global_power(tmp_path):
    beam_file = tmp_path / "beams.csv"
    _write_beams(beam_file)
    beams = load_beams_csv(beam_file)
    grid = _cartesian_grid()
    rays = initialize_rays(
        beams,
        grid,
        nrays_axis1=4,
        nrays_axis2=6,
        neighbour_spacing_m=1.0e-9,
    )
    state = rays.state
    layout = RAY_STATE_LAYOUT
    assert state.shape == (2, 4, 6, layout.n_attributes)
    assert state.dtype == jnp.float64
    assert rays.neighbour_momentum_tangents.shape == (2, 4, 6, 9)
    np.testing.assert_allclose(rays.neighbour_momentum_tangents, 0.0)
    assert rays.will_hit_grid.shape == (2, 4, 6)
    assert np.all(rays.will_hit_grid)
    np.testing.assert_allclose(state[..., layout.ray_power].sum(), 1.0)
    np.testing.assert_allclose(rays.total_incident_power, 1.0)
    np.testing.assert_allclose(rays.beam_power.sum(), 1.0)
    np.testing.assert_allclose(
        state[0, ..., layout.ray_power].sum(), 2.0 / 3.0, rtol=1.0e-14
    )
    np.testing.assert_allclose(
        state[1, ..., layout.ray_power].sum(), 1.0 / 3.0, rtol=1.0e-14
    )
    neighbours = state[..., layout.neighbour_positions].reshape((2, 4, 6, 3, 3))
    positions = state[..., layout.position]
    neighbour_distance = np.linalg.norm(neighbours - positions[..., None, :], axis=-1)
    np.testing.assert_allclose(neighbour_distance, 1.0e-9, rtol=1.0e-7)
    assert rays.beam_entry_distance.shape == (2, 4, 6)
    np.testing.assert_allclose(state[..., layout.arc_length], rays.beam_entry_distance)
    np.testing.assert_allclose(
        state[..., layout.phase_length], rays.beam_entry_distance
    )
    np.testing.assert_allclose(state[..., layout.path_length], rays.beam_entry_distance)
    np.testing.assert_allclose(state[..., layout.cbet_depth], 0.0)
    np.testing.assert_allclose(state[..., layout.srs_depth], 0.0)
    np.testing.assert_allclose(state[..., layout.sbs_depth], 0.0)


def test_each_angled_ray_tube_advances_by_its_latest_member_entry(tmp_path):
    beam_file = tmp_path / "beams.csv"
    beam_file.write_text(
        "beam_id,origin_geometry,origin_1,origin_2,origin_3,"
        "target_geometry,target_1,target_2,target_3,"
        "width_x_m,width_y_m,rotation_pi,supergaussian_index,"
        "omega_rad_s,power_fraction\n"
        "angled,cartesian,-3,-1.0919107028,0,cartesian,0,0,0,"
        "0.05,0.05,0,2,5.361e15,1\n",
        encoding="utf-8",
    )
    beams = load_beams_csv(beam_file)
    grid = _cartesian_grid()
    rays = initialize_rays(
        beams,
        grid,
        nrays_axis1=3,
        nrays_axis2=3,
        neighbour_spacing_m=1.0e-9,
    )
    positions = rays.state[..., RAY_STATE_LAYOUT.position]
    neighbours = rays.state[..., RAY_STATE_LAYOUT.neighbour_positions].reshape(
        (1, 3, 3, 3, 3)
    )
    members = np.concatenate((positions[..., None, :], neighbours), axis=-2)
    np.testing.assert_allclose(np.min(members[..., 0], axis=-1), -1.0, atol=1.0e-13)
    assert rays.beam_entry_distance.shape == (1, 3, 3)
    assert np.ptp(rays.beam_entry_distance) > 0.0
    assert np.all(rays.beam_entry_distance > 10.0)
    assert np.all(grid.interpolate(positions).inside)
    assert np.all(grid.interpolate(neighbours).inside)


def test_supergaussian_power_uses_two_impact_coordinates(tmp_path):
    beam_file = tmp_path / "beams.csv"
    _write_beams(beam_file)
    beams = load_beams_csv(beam_file)
    rays = initialize_rays(beams, _cartesian_grid(), nrays_axis1=5, nrays_axis2=5)
    state = rays.state[0]
    x = state[..., RAY_STATE_LAYOUT.impact_parameter_x]
    y = state[..., RAY_STATE_LAYOUT.impact_parameter_y]
    expected = np.exp(
        -((np.abs(x / beams.width_x[0]) ** 2) + (np.abs(y / beams.width_y[0]) ** 2))
    )
    expected *= float(beams.power_fraction[0]) / expected.sum()
    np.testing.assert_allclose(state[..., RAY_STATE_LAYOUT.ray_power], expected)
    profile = np.exp(
        -((np.abs(x / beams.width_x[0]) ** 2) + (np.abs(y / beams.width_y[0]) ** 2))
    )
    expected_intensity = float(beams.peak_intensity[0]) * profile
    np.testing.assert_allclose(
        state[..., RAY_STATE_LAYOUT.initial_intensity], expected_intensity
    )
    np.testing.assert_allclose(
        state[..., RAY_STATE_LAYOUT.initial_electric_field],
        np.sqrt(2.0 * expected_intensity / (VACUUM_PERMITTIVITY * SPEED_OF_LIGHT)),
    )


def test_safe_density_cap_changes_only_permittivity_not_grid_derivative(tmp_path):
    beam_file = tmp_path / "beam.csv"
    _write_beams(beam_file)
    beams = load_beams_csv(beam_file)
    ncritical = float(critical_density(beams.omega[0]))

    def high_density(grid):
        x, _y, _z = grid.vertex_mesh()
        shape = grid.vertex_shape
        return HydroFields(
            ne=ncritical * (1.1 + 0.01 * x),
            Te=jnp.ones(shape),
            Ti=jnp.ones(shape),
            velocity=jnp.zeros((*shape, 3)),
        )

    grid = _cartesian_grid(high_density)
    raw_gradient = np.asarray(grid.grad_ne).copy()
    rays = initialize_rays(beams, grid, nrays_axis1=3, nrays_axis2=3)
    epsilon = rays.state[..., RAY_STATE_LAYOUT.permittivity]
    assert np.all(epsilon > 0)
    np.testing.assert_allclose(grid.grad_ne, raw_gradient)
    assert np.max(np.abs(grid.grad_ne[..., 0])) > 0


def test_safe_density_ratio_is_smooth_and_jittable():
    threshold = 0.999
    epsilon = 1.0e-7
    values = jnp.asarray([threshold - epsilon, threshold, threshold + epsilon, 2.0])
    safe = jax.jit(safe_density_ratio)(values)
    assert safe[-1] < 1.0
    left_slope = (safe[1] - safe[0]) / epsilon
    right_slope = (safe[2] - safe[1]) / epsilon
    np.testing.assert_allclose(left_slope, 1.0, rtol=1.0e-9)
    np.testing.assert_allclose(right_slope, 1.0, rtol=2.0e-4)
    assert np.all(safe_permittivity(values) > 0)


@pytest.mark.parametrize(
    ("geom", "extents", "origin", "expected"),
    [
        ("cartesian", ((-1, 1), (-1, 1), (-1, 1)), (-2, 0, 0), 1.0),
        ("cylindrical", ((0, 1), (-np.pi, np.pi), (-1, 1)), (-2, 0, 0), 1.0),
        ("spherical", ((0, 1), (-np.pi, np.pi), (0, np.pi)), (-2, 0, 0), 1.0),
    ],
)
def test_entry_distance_for_all_grid_geometries(geom, extents, origin, expected):
    grid = Grid.create(geom=geom, extents=extents, ncells=(2, 4, 2))
    distance = ray_grid_entry_distance(grid, np.asarray(origin), np.asarray((1, 0, 0)))
    np.testing.assert_allclose(distance, expected)
