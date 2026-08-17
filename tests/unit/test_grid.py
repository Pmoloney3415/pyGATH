import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pyGATH.grid import (
    GradedAxis,
    Grid,
    HydroFields,
    PlasmaComposition,
    SafeHydroState,
    interpolate_hydro,
)


def linear_cartesian_hydro(grid):
    x, y, z = grid.vertex_mesh()
    shape = grid.vertex_shape
    return HydroFields(
        ne=x + 2.0 * y + 3.0 * z,
        Te=jnp.full(shape, 5.0),
        Ti=jnp.full(shape, 7.0),
        velocity=jnp.stack(
            (
                jnp.broadcast_to(x, shape),
                jnp.broadcast_to(y, shape),
                jnp.broadcast_to(z, shape),
            ),
            axis=-1,
        ),
    )


def test_uniform_grid_builds_vertices_centres_and_default_hydro():
    grid = Grid.create(
        geom="cartesian",
        extents=((0.0, 2.0), (-1.0, 1.0), (3.0, 5.0)),
        ncells=(2, 4, 1),
    )
    assert grid.ncells == (2, 4, 1)
    assert grid.vertex_shape == (3, 5, 2)
    assert grid.is_uniform == (True, True, True)
    np.testing.assert_allclose(grid.xb, [0.0, 1.0, 2.0])
    np.testing.assert_allclose(grid.xc, [0.5, 1.5])
    np.testing.assert_allclose(grid.hydro.ne, 1.0)
    np.testing.assert_allclose(grid.grad_ne, 0.0)


def test_fully_ionized_mixture_derives_ion_density_and_effective_charge():
    composition = PlasmaComposition.create(
        (("carbon", 12.0, 6.0, 1.0), ("hydrogen", 1.0, 1.0, 1.0))
    )
    grid = Grid.create(
        geom="cartesian",
        extents=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
        ncells=(2, 2, 2),
        composition=composition,
        initial_condition_parameters={"ne": 7.0},
    )
    np.testing.assert_allclose(composition.mean_charge, 3.5)
    np.testing.assert_allclose(composition.effective_charge, 18.5 / 3.5)
    np.testing.assert_allclose(grid.hydro.ni, 2.0)


def test_omega_lilac_fit_reproduces_density_temperature_and_radial_flow():
    omega = 5.36e15
    grid = Grid.create(
        geom="spherical",
        extents=((3.5e-4, 1.2e-3), (-np.pi, np.pi), (0.0, np.pi)),
        ncells=(3, 4, 2),
        initial_condition="omega_lilac_fit",
        initial_condition_parameters={"omega_rad_s": omega},
    )
    epsilon_0 = 8.854_187_812_8e-12
    electron_mass = 9.109_383_713_9e-31
    elementary_charge = 1.602_176_634e-19
    ncritical = epsilon_0 * electron_mass * omega**2 / elementary_charge**2
    expected_density = 1.165 * ncritical * (343.0e-6 / np.asarray(grid.xb)) ** 3.78

    np.testing.assert_allclose(grid.hydro.ne[:, 0, 0], expected_density)
    np.testing.assert_allclose(grid.hydro.Te, 2.0e3)
    np.testing.assert_allclose(grid.hydro.Ti, 1.0e3)
    assert np.all(np.asarray(grid.hydro.velocity[..., 0]) > 0.0)
    np.testing.assert_allclose(grid.hydro.velocity[..., 1:], 0.0)


def test_graded_axis_uses_cumulative_cell_sizes_and_covers_upper_extent():
    specification = GradedAxis(
        boundaries=(0.45,),
        cell_sizes=(0.12, 0.025),
        transition_widths=(0.2,),
    )
    grid = Grid.create(
        geom="cartesian",
        extents=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
        ncells=(None, 2, 2),
        graded_axes=(specification, None, None),
    )
    widths = np.diff(grid.xb)
    assert grid.xb[-1] >= 1.0
    assert grid.xb[-2] < 1.0
    assert widths[0] > widths[-1]
    assert grid.ncells[0] == widths.size
    assert grid.is_uniform == (False, True, True)


def test_nonuniform_interpolation_and_density_gradient_are_linear_exact():
    grid = Grid.create(
        geom="cartesian",
        extents=((0.0, 1.0), (-1.0, 1.0), (0.0, 2.0)),
        ncells=(None, 4, 4),
        graded_axes=(
            GradedAxis(
                boundaries=(0.4,),
                cell_sizes=(0.13, 0.035),
                transition_widths=(0.25,),
            ),
            None,
            None,
        ),
        initial_condition=linear_cartesian_hydro,
    )
    positions = jnp.array([[0.15, -0.5, 0.25], [0.83, 0.2, 1.25]])
    sampled = grid.interpolate(positions)
    expected_ne = positions[:, 0] + 2.0 * positions[:, 1] + 3.0 * positions[:, 2]
    np.testing.assert_allclose(sampled.ne, expected_ne, rtol=2.0e-5, atol=2.0e-5)
    np.testing.assert_allclose(sampled.ni, expected_ne, rtol=2.0e-5, atol=2.0e-5)
    np.testing.assert_allclose(
        sampled.grad_ne,
        jnp.broadcast_to(jnp.array([1.0, 2.0, 3.0]), positions.shape),
        rtol=2.0e-5,
        atol=2.0e-5,
    )
    np.testing.assert_allclose(sampled.velocity, positions, atol=2.0e-5)
    assert np.all(sampled.inside)


def test_exact_upper_vertex_is_inside():
    grid = Grid.create(
        geom="cartesian",
        extents=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
        ncells=(2, 2, 2),
        initial_condition=linear_cartesian_hydro,
    )
    position = jnp.array([grid.xb[-1], grid.yb[-1], grid.zb[-1]])
    sampled = grid.interpolate(position)
    assert sampled.inside
    np.testing.assert_allclose(sampled.ne, 6.0)


def test_outside_and_nonfinite_positions_use_safe_state():
    safe = SafeHydroState(
        ne=11.0,
        Te=13.0,
        Ti=14.0,
        grad_ne=(1.0, 2.0, 3.0),
        velocity=(4.0, 5.0, 6.0),
    )
    grid = Grid.create(
        geom="cartesian",
        extents=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
        ncells=(2, 2, 2),
        safe_state=safe,
    )
    sampled = grid.interpolate(jnp.array([[2.0, 0.5, 0.5], [jnp.nan, 0.5, 0.5]]))
    np.testing.assert_array_equal(sampled.inside, [False, False])
    np.testing.assert_allclose(sampled.ne, [11.0, 11.0])
    np.testing.assert_allclose(sampled.ni, [11.0, 11.0])
    np.testing.assert_allclose(sampled.grad_ne, [[1.0, 2.0, 3.0]] * 2)
    np.testing.assert_allclose(sampled.velocity, [[4.0, 5.0, 6.0]] * 2)


def cylindrical_hydro(grid):
    radius, _phi, _z = grid.vertex_mesh()
    shape = grid.vertex_shape
    local_radial = jnp.broadcast_to(jnp.array([1.0, 0.0, 0.0]), (*shape, 3))
    return HydroFields(
        ne=jnp.broadcast_to(radius, shape),
        Te=jnp.ones(shape),
        Ti=jnp.ones(shape),
        velocity=local_radial,
    )


def test_cylindrical_interpolation_converts_vectors_after_interpolation():
    grid = Grid.create(
        geom="cylindrical",
        extents=((0.0, 2.0), (-jnp.pi, jnp.pi), (-1.0, 1.0)),
        ncells=(4, 8, 4),
        initial_condition=cylindrical_hydro,
    )
    sampled = grid.interpolate(jnp.array([[0.0, 1.0, 0.0]]))
    np.testing.assert_allclose(sampled.ne, [1.0], atol=1.0e-6)
    np.testing.assert_allclose(sampled.grad_ne, [[0.0, 1.0, 0.0]], atol=1.0e-6)
    np.testing.assert_allclose(sampled.velocity, [[0.0, 1.0, 0.0]], atol=1.0e-6)


def test_coordinate_singularities_do_not_produce_nonfinite_output():
    cylindrical = Grid.create(
        geom="cylindrical",
        extents=((0.0, 1.0), (-jnp.pi, jnp.pi), (-1.0, 1.0)),
        ncells=(2, 4, 2),
    )
    spherical = Grid.create(
        geom="spherical",
        extents=((0.0, 1.0), (-jnp.pi, jnp.pi), (0.0, jnp.pi)),
        ncells=(2, 4, 4),
    )
    for grid in (cylindrical, spherical):
        sampled = grid.interpolate(jnp.array([[0.0, 0.0, 0.0]]))
        assert np.all(np.isfinite(sampled.grad_ne))
        assert np.all(np.isfinite(sampled.velocity))


@pytest.mark.parametrize(
    ("geom", "axis"),
    [("cylindrical", 1), ("spherical", 1), ("spherical", 2)],
)
def test_graded_angular_axes_are_rejected(geom, axis):
    graded = [None, None, None]
    counts = [2, 2, 2]
    graded[axis] = GradedAxis((), (0.1,), ())
    counts[axis] = None
    extents = ((0.0, 1.0), (-jnp.pi, jnp.pi), (0.0, jnp.pi))
    with pytest.raises(ValueError, match="angular"):
        Grid.create(
            geom=geom,
            extents=extents,
            ncells=counts,
            graded_axes=graded,
        )


@pytest.mark.parametrize(
    ("geom", "extents", "message"),
    [
        ("cylindrical", ((-1.0, 1.0), (-1.0, 1.0), (0.0, 1.0)), "radii"),
        ("cylindrical", ((0.0, 1.0), (-4.0, 1.0), (0.0, 1.0)), "azimuthal"),
        ("spherical", ((0.0, 1.0), (-1.0, 1.0), (-0.1, 1.0)), "polar"),
    ],
)
def test_invalid_curvilinear_extents_are_rejected(geom, extents, message):
    with pytest.raises(ValueError, match=message):
        Grid.create(geom=geom, extents=extents, ncells=(2, 2, 2))


def test_grid_and_interpolation_are_jittable_pytrees():
    grid = Grid.create(
        geom="cartesian",
        extents=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
        ncells=(None, 2, 2),
        graded_axes=(GradedAxis((0.5,), (0.15, 0.05), (0.2,)), None, None),
        initial_condition=linear_cartesian_hydro,
    )
    sample = jax.jit(interpolate_hydro)(grid, jnp.array([[0.25, 0.5, 0.75]]))
    np.testing.assert_allclose(sample.ne, [3.5], atol=2.0e-5)
    np.testing.assert_array_equal(sample.inside, [True])
