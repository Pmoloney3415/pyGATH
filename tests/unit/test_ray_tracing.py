import jax
import jax.numpy as jnp
import numpy as np

from pyGATH.grid import Grid, HydroFields
from pyGATH.raytracing import (
    RAY_SHEET_LAYOUT,
    RAY_STATE_LAYOUT,
    SPEED_OF_LIGHT,
    VACUUM_PERMITTIVITY,
    InverseBremsstrahlungOptions,
    RayTracingOptions,
    critical_density,
    inverse_bremsstrahlung_depth_derivative,
    ray_rhs,
    trace_rays,
    tracing,
)

OMEGA = 5.361e15


def _single_vacuum_ray_state():
    state = jnp.zeros((1, 1, 1, RAY_STATE_LAYOUT.n_attributes), dtype=jnp.float64)
    state = state.at[..., RAY_STATE_LAYOUT.position].set(jnp.asarray((-1.0, 0.0, 0.0)))
    state = state.at[..., RAY_STATE_LAYOUT.momentum].set(jnp.asarray((1.0, 0.0, 0.0)))
    neighbours = jnp.asarray(
        (
            (-1.0, 1.0e-6, 0.0),
            (-1.0, -0.5e-6, 0.5 * jnp.sqrt(3.0) * 1.0e-6),
            (-1.0, -0.5e-6, -0.5 * jnp.sqrt(3.0) * 1.0e-6),
        )
    )
    neighbour_momenta = jnp.broadcast_to(jnp.asarray((1.0, 0.0, 0.0)), (3, 3))
    state = state.at[..., RAY_STATE_LAYOUT.neighbour_positions].set(
        neighbours.reshape(9)
    )
    state = state.at[..., RAY_STATE_LAYOUT.neighbour_momenta].set(
        neighbour_momenta.reshape(9)
    )
    state = state.at[..., RAY_STATE_LAYOUT.frequency].set(OMEGA)
    state = state.at[..., RAY_STATE_LAYOUT.permittivity].set(1.0)
    state = state.at[..., RAY_STATE_LAYOUT.initial_intensity].set(2.0)
    state = state.at[..., RAY_STATE_LAYOUT.initial_electric_field].set(
        jnp.sqrt(4.0 / (VACUUM_PERMITTIVITY * SPEED_OF_LIGHT))
    )
    return state


def _vacuum_grid():
    return Grid.create(
        geom="cartesian",
        extents=((-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)),
        ncells=(2, 2, 2),
    )


def test_ray_rhs_evolves_each_neighbour_with_its_own_density_gradient():
    ncritical = float(critical_density(OMEGA))

    def varying_density(coordinates):
        x, _y, _z = coordinates.vertex_mesh()
        shape = coordinates.vertex_shape
        return HydroFields(
            ne=ncritical * (0.1 + 0.05 * x**2),
            Te=jnp.ones(shape),
            Ti=jnp.ones(shape),
            velocity=jnp.zeros((*shape, 3)),
        )

    grid = Grid.create(
        geom="cartesian",
        extents=((-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)),
        ncells=(4, 2, 2),
        initial_condition=varying_density,
    )
    state = _single_vacuum_ray_state()
    state = state.at[..., RAY_STATE_LAYOUT.position].set((0.0, 0.0, 0.0))
    neighbours = jnp.asarray(((-0.75, 0.0, 0.0), (-0.25, 0.0, 0.0), (0.75, 0.0, 0.0)))
    state = state.at[..., RAY_STATE_LAYOUT.neighbour_positions].set(
        neighbours.reshape(9)
    )

    derivative = jax.jit(ray_rhs)(0.0, state, grid)
    primary_acceleration = derivative[..., RAY_STATE_LAYOUT.momentum]
    neighbour_accelerations = derivative[
        ..., RAY_STATE_LAYOUT.neighbour_momenta
    ].reshape((1, 1, 1, 3, 3))

    np.testing.assert_allclose(primary_acceleration, 0.0, atol=1.0e-15)
    assert neighbour_accelerations[0, 0, 0, 0, 0] > 0.0
    assert neighbour_accelerations[0, 0, 0, 2, 0] < 0.0
    assert not np.isclose(
        neighbour_accelerations[0, 0, 0, 0, 0],
        neighbour_accelerations[0, 0, 0, 1, 0],
    )


def test_primary_exit_rhs_matches_primary_components_of_tangent_rhs():
    grid = _vacuum_grid()
    state = _single_vacuum_ray_state()
    characteristic_length = tracing._grid_characteristic_length(grid)
    relative_state = tracing._absolute_to_relative_state(state)
    scaling = tracing._build_state_scaling(relative_state, characteristic_length)
    solver_state = tracing._to_solver_state(relative_state, scaling)
    primary_state = tracing._to_primary_solver_state(solver_state)
    primary_args = tracing._PrimaryTraceArguments(
        grid=grid,
        position_scale=scaling.scales[..., RAY_STATE_LAYOUT.position],
        position_reference=scaling.position_reference,
        momentum_scale=scaling.scales[..., RAY_STATE_LAYOUT.momentum],
        frequency=state[..., RAY_STATE_LAYOUT.frequency],
    )
    tangent_args = tracing._TraceArguments(
        grid=grid,
        initial_area=tracing.initial_ray_area(state),
        area_floor=jnp.asarray(1.0e-12),
        scaling=scaling,
        inverse_bremsstrahlung=InverseBremsstrahlungOptions(),
    )

    primary_derivative = jax.jit(tracing._scaled_primary_ray_rhs)(
        0.0, primary_state, primary_args
    )
    tangent_derivative = jax.jit(tracing._scaled_tangent_ray_rhs)(
        0.0, solver_state, tangent_args
    )

    assert primary_state.shape[-1] == 6
    np.testing.assert_allclose(
        primary_derivative,
        tracing._to_primary_solver_state(tangent_derivative),
        rtol=1.0e-14,
        atol=1.0e-14,
    )


def test_vacuum_trace_terminates_outside_and_builds_degenerate_second_sheet():
    options = RayTracingOptions(
        nsamples_per_sheet=5,
        diagnostic_samples=32,
        maximum_path_length_grid_lengths=10.0,
        rtol=1.0e-6,
        atol=1.0e-9,
        max_steps=512,
    )
    initial_state = _single_vacuum_ray_state()
    initial_position = initial_state[0, 0, 0, RAY_STATE_LAYOUT.position]
    initial_momentum = initial_state[0, 0, 0, RAY_STATE_LAYOUT.momentum]
    initial_neighbour_positions = initial_state[
        0, 0, 0, RAY_STATE_LAYOUT.neighbour_positions
    ].reshape((3, 3))
    initial_neighbour_momenta = initial_state[
        0, 0, 0, RAY_STATE_LAYOUT.neighbour_momenta
    ].reshape((3, 3))
    initial_offsets = initial_neighbour_positions - initial_position
    initial_momentum_differences = initial_neighbour_momenta - initial_momentum

    result = trace_rays(initial_state, _vacuum_grid(), options=options)
    fields = result.sheet_fields

    assert fields.shape == (1, 2, 1, 1, 5, RAY_SHEET_LAYOUT.n_attributes)
    assert bool(result.terminated)
    assert not bool(result.has_caustic[0, 0, 0])
    assert np.isinf(result.caustic_path[0, 0, 0])
    assert result.terminal_path > 2.0
    np.testing.assert_allclose(result.terminal_path, 2.0, rtol=1.0e-10)

    first_sheet = fields[0, 0, 0, 0]
    second_sheet = fields[0, 1, 0, 0]
    neighbour_positions = first_sheet[:, RAY_STATE_LAYOUT.neighbour_positions].reshape(
        (-1, 3, 3)
    )
    neighbour_momenta = first_sheet[:, RAY_STATE_LAYOUT.neighbour_momenta].reshape(
        (-1, 3, 3)
    )
    position_offsets = (
        neighbour_positions
        - first_sheet[:, RAY_STATE_LAYOUT.position][:, np.newaxis, :]
    )
    momentum_differences = (
        neighbour_momenta - first_sheet[:, RAY_STATE_LAYOUT.momentum][:, np.newaxis, :]
    )
    np.testing.assert_allclose(
        position_offsets,
        np.broadcast_to(initial_offsets, position_offsets.shape),
        rtol=1.0e-9,
        atol=1.0e-15,
    )
    np.testing.assert_allclose(
        momentum_differences,
        np.broadcast_to(initial_momentum_differences, momentum_differences.shape),
        atol=1.0e-14,
    )
    np.testing.assert_allclose(
        first_sheet[:, RAY_STATE_LAYOUT.path_length],
        np.linspace(0.0, float(result.terminal_path), 5),
        rtol=1.0e-8,
    )
    np.testing.assert_allclose(
        first_sheet[:, RAY_STATE_LAYOUT.arc_length],
        first_sheet[:, RAY_STATE_LAYOUT.path_length],
        rtol=1.0e-8,
    )
    np.testing.assert_allclose(
        first_sheet[:, RAY_STATE_LAYOUT.phase_length],
        first_sheet[:, RAY_STATE_LAYOUT.path_length],
        rtol=1.0e-8,
    )
    assert first_sheet[-1, RAY_STATE_LAYOUT.position][0] > 1.0
    np.testing.assert_allclose(
        second_sheet,
        np.broadcast_to(second_sheet[-1], second_sheet.shape),
        rtol=1.0e-12,
    )
    np.testing.assert_allclose(
        fields[..., RAY_SHEET_LAYOUT.uncapped_amplitude], 1.0, rtol=1.0e-10
    )
    np.testing.assert_allclose(
        fields[..., RAY_SHEET_LAYOUT.capped_amplitude], 1.0, rtol=1.0e-10
    )
    np.testing.assert_allclose(
        fields[..., RAY_SHEET_LAYOUT.intensity], 2.0, rtol=1.0e-10
    )


def test_nrl_inverse_bremsstrahlung_depth_and_sheet_attenuation():
    ncritical = float(critical_density(OMEGA))
    density = 0.1 * ncritical
    temperature = 1.0e5
    rate = inverse_bremsstrahlung_depth_derivative(
        density,
        temperature,
        OMEGA,
        1.0,
        minimum_coulomb_log=2.0,
        coulomb_log_override=2.0,
    )
    expected_rate = 3.1e-17 * density**2 * 2.0 * temperature**-1.5 * OMEGA**-2
    np.testing.assert_allclose(rate, expected_rate, rtol=1.0e-14)

    grid = Grid.create(
        geom="cartesian",
        extents=((-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)),
        ncells=(2, 2, 2),
        initial_condition_parameters={"ne": density, "Te": temperature},
        safe_state=None,
    )
    state = _single_vacuum_ray_state()
    refractive_index = np.sqrt(0.9)
    state = state.at[..., RAY_STATE_LAYOUT.momentum].set(
        jnp.asarray((refractive_index, 0.0, 0.0))
    )
    neighbour_momenta = jnp.broadcast_to(
        jnp.asarray((refractive_index, 0.0, 0.0)), (3, 3)
    )
    state = state.at[..., RAY_STATE_LAYOUT.neighbour_momenta].set(
        neighbour_momenta.reshape(9)
    )
    state = state.at[..., RAY_STATE_LAYOUT.permittivity].set(0.9)
    result = trace_rays(
        state,
        grid,
        options=RayTracingOptions(
            nsamples_per_sheet=5,
            diagnostic_samples=32,
            maximum_path_length_grid_lengths=10.0,
            rtol=1.0e-6,
            atol=1.0e-9,
            max_steps=512,
        ),
        inverse_bremsstrahlung=InverseBremsstrahlungOptions(
            enabled=True,
            coulomb_log_override=2.0,
        ),
    )
    first_sheet = np.asarray(result.sheet_fields[0, 0, 0, 0])
    depth = first_sheet[:, RAY_STATE_LAYOUT.inverse_brems_depth]
    assert depth[-1] > depth[0] >= 0.0
    np.testing.assert_allclose(
        first_sheet[0, RAY_SHEET_LAYOUT.intensity], 2.0, rtol=2.0e-8
    )
    np.testing.assert_allclose(
        first_sheet[:, RAY_SHEET_LAYOUT.intensity],
        2.0 * np.exp(-depth),
        rtol=2.0e-8,
    )
    expected_deposition = (
        0.5
        * VACUUM_PERMITTIVITY
        * SPEED_OF_LIGHT
        * first_sheet[:, RAY_SHEET_LAYOUT.electric_field] ** 2
        * float(rate)
    )
    expected_deposition = np.where(
        np.asarray(grid.contains(first_sheet[:, RAY_STATE_LAYOUT.position])),
        expected_deposition,
        0.0,
    )
    np.testing.assert_allclose(
        first_sheet[:, RAY_SHEET_LAYOUT.inverse_brems_deposition],
        expected_deposition,
        rtol=2.0e-10,
    )


def test_critical_collision_frequency_inverse_bremsstrahlung_rate():
    ncritical = float(critical_density(OMEGA))
    density = 0.37 * ncritical
    collision_frequency = 206.45e12
    rate = inverse_bremsstrahlung_depth_derivative(
        density,
        70.0,
        OMEGA,
        3.1,
        critical_collision_frequency_hz=collision_frequency,
    )
    expected = 2.0 * collision_frequency / SPEED_OF_LIGHT * (density / ncritical) ** 2
    np.testing.assert_allclose(rate, expected, rtol=1.0e-14)
