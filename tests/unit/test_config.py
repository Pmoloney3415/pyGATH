from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pyGATH.grid import HydroFields
from pyGATH.io import ConfigError, load_simulation_config

PROJECT_ROOT = Path(__file__).parents[2]
EXAMPLE_DECK = PROJECT_ROOT / "configs" / "example_configs" / "example_simulation.toml"


def _cartesian_deck(extra_grid: str = "", initial_condition: str = "") -> str:
    return f"""
[simulation]
name = "unit-test"

[grid]
geometry = "cartesian"
{extra_grid}

[grid.axes.x]
min = 0.0
max = 1.0
spacing = "uniform"
ncells = 2

[grid.axes.y]
min = -1.0
max = 1.0
spacing = "uniform"
ncells = 4

[grid.axes.z]
min = 0.0
max = 2.0
spacing = "uniform"
ncells = 2

{initial_condition}
"""


def test_example_deck_loads_and_builds_float64_grid():
    simulation = load_simulation_config(EXAMPLE_DECK)
    grid = simulation.build_grid()
    assert jax.config.jax_enable_x64
    assert grid.xb.dtype == jnp.float64
    assert grid.hydro.ne.dtype == jnp.float64
    assert grid.geom.value == "cylindrical"
    assert grid.is_uniform == (False, True, True)
    assert grid.extents[0, 1] >= 0.01
    assert simulation.beams is not None
    assert simulation.beams.file == (
        PROJECT_ROOT / "beam_csvs" / "example_beam_csvs" / "example_beams.csv"
    )
    assert simulation.beams.intensity_cutoff == 2.0e-4
    assert simulation.beams.neighbour_spacing_m == 1.0e-9
    assert simulation.beams.power.mode == "total_power"
    assert simulation.beams.power.total_power_w == 1.0e12
    assert simulation.raytracing.nsamples_per_sheet == 20
    assert simulation.raytracing.dt0 is None
    assert simulation.physics.inverse_bremsstrahlung.enabled
    np.testing.assert_allclose(grid.composition.mean_charge, 3.5)
    np.testing.assert_allclose(grid.hydro.ni, grid.hydro.ne / 3.5)


def test_unknown_top_level_sections_are_preserved(tmp_path):
    path = tmp_path / "simulation.toml"
    path.write_text(_cartesian_deck(), encoding="utf-8")
    simulation = load_simulation_config(path)
    assert simulation.extra_sections == {"simulation": {"name": "unit-test"}}
    assert simulation.build_grid().ncells == (2, 4, 2)


def test_custom_initial_condition_is_resolved_from_explicit_registry(tmp_path):
    path = tmp_path / "custom.toml"
    initial_condition = """
[grid.initial_condition]
name = "linear-density"

[grid.initial_condition.parameters]
offset = 4.0
"""
    path.write_text(
        _cartesian_deck(initial_condition=initial_condition), encoding="utf-8"
    )

    def linear_density(grid, *, offset):
        x, y, z = grid.vertex_mesh()
        shape = grid.vertex_shape
        return HydroFields(
            ne=x + y + z + offset,
            Te=jnp.ones(shape),
            Ti=jnp.ones(shape),
            velocity=jnp.zeros((*shape, 3)),
        )

    simulation = load_simulation_config(path)
    grid = simulation.build_grid(initial_conditions={"linear-density": linear_density})
    np.testing.assert_allclose(grid.hydro.ne[0, 0, 0], 3.0)
    np.testing.assert_allclose(grid.grad_ne, 1.0)


def test_grid_section_rejects_unknown_keys(tmp_path):
    path = tmp_path / "unknown.toml"
    path.write_text(
        _cartesian_deck(extra_grid="geometery = 'cartesian'"), encoding="utf-8"
    )
    with pytest.raises(ConfigError, match="geometery"):
        load_simulation_config(path)


def test_fractional_pi_angular_extents_are_converted(tmp_path):
    path = tmp_path / "angles.toml"
    path.write_text(
        """
[grid]
geometry = "spherical"

[grid.axes.r]
min = 0.0
max = 1.0
spacing = "uniform"
ncells = 2

[grid.axes.phi]
min = -0.75
max = 0.25
spacing = "uniform"
ncells = 8

[grid.axes.theta]
min = 0.0
max = 0.37
spacing = "uniform"
ncells = 4
""",
        encoding="utf-8",
    )
    grid = load_simulation_config(path).build_grid()
    np.testing.assert_allclose(
        grid.yb[jnp.array([0, -1])], [-0.75 * np.pi, 0.25 * np.pi]
    )
    np.testing.assert_allclose(grid.zb[jnp.array([0, -1])], [0.0, 0.37 * np.pi])


def test_out_of_range_pi_fraction_is_rejected(tmp_path):
    path = tmp_path / "bad-angle-fraction.toml"
    path.write_text(
        """
[grid]
geometry = "cylindrical"

[grid.axes.r]
min = 0.0
max = 1.0
spacing = "uniform"
ncells = 2

[grid.axes.phi]
min = -1.01
max = 1.0
spacing = "uniform"
ncells = 8

[grid.axes.z]
min = 0.0
max = 1.0
spacing = "uniform"
ncells = 2
""",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match=r"fractions of pi must lie within \[-1, 1\]"):
        load_simulation_config(path)


def test_angular_axis_cannot_be_graded_in_deck(tmp_path):
    path = tmp_path / "graded-angle.toml"
    path.write_text(
        """
[grid]
geometry = "cylindrical"

[grid.axes.r]
min = 0.0
max = 1.0
spacing = "uniform"
ncells = 2

[grid.axes.phi]
min = -1.0
max = 1.0
spacing = "graded"
boundaries = [0.0]
cell_sizes = [0.1, 0.2]
transition_widths = [0.1]

[grid.axes.z]
min = 0.0
max = 1.0
spacing = "uniform"
ncells = 2
""",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="must use uniform spacing"):
        load_simulation_config(path)


def test_semantically_invalid_axis_is_rejected_during_load(tmp_path):
    path = tmp_path / "invalid-axis.toml"
    path.write_text(
        _cartesian_deck().replace("max = 1.0", "max = 0.0", 1), encoding="utf-8"
    )
    with pytest.raises(ConfigError, match="max must be greater"):
        load_simulation_config(path)


def test_beam_section_is_strict_and_path_is_relative_to_deck(tmp_path):
    path = tmp_path / "simulation.toml"
    path.write_text(
        _cartesian_deck()
        + """
[beams]
file = "inputs/beams.csv"
nrays_axis1 = 3
nrays_axis2 = 5
intensity_cutoff = 0.001
neighbour_spacing_m = 2e-9
launch_padding = 7.0

[beams.power]
mode = "total_power"
total_power_w = 2e12
""",
        encoding="utf-8",
    )
    simulation = load_simulation_config(path)
    assert simulation.beams is not None
    assert simulation.beams.file == tmp_path / "inputs" / "beams.csv"
    assert simulation.beams.nrays_axis1 == 3
    assert simulation.beams.nrays_axis2 == 5
    assert simulation.beams.power.total_power_w == 2.0e12

    path.write_text(
        _cartesian_deck() + "\n[beams]\nfile = 'beams.csv'\ntypo = 1\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="typo"):
        load_simulation_config(path)


def test_raytracing_section_is_validated(tmp_path):
    path = tmp_path / "simulation.toml"
    path.write_text(
        _cartesian_deck()
        + """
[raytracing]
nsamples_per_sheet = 7
diagnostic_samples = 31
dt0 = 1e-5
""",
        encoding="utf-8",
    )
    simulation = load_simulation_config(path)
    assert simulation.raytracing.nsamples_per_sheet == 7
    assert simulation.raytracing.diagnostic_samples == 31
    assert simulation.raytracing.dt0 == 1.0e-5
    assert simulation.raytracing.options().dt0 == 1.0e-5

    path.write_text(
        _cartesian_deck() + "\n[raytracing]\nnsamples_per_sheet = 1\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="at least two"):
        load_simulation_config(path)
