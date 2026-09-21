from pathlib import Path

import numpy as np
import pytest

from pyGATH.fields import (
    build_linear_deposition_mesh_from_grid,
    deposit_simplicial_power_to_mesh,
    simplicialise_sheet_fields,
)
from pyGATH.io import load_simulation_config

pytestmark = pytest.mark.regression

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG = PROJECT_ROOT / "configs" / "example_configs" / "uniform_1d_deposition.toml"


def test_complete_one_dimensional_conservative_deposition_pipeline():
    simulation = load_simulation_config(CONFIG)
    grid = simulation.build_grid()
    beams = simulation.load_beams()
    initial_rays = simulation.initialize_rays(grid, beams=beams)
    trace = simulation.trace_rays(initial_rays, grid)
    field = simplicialise_sheet_fields(
        trace.sheet_fields,
        dimension=grid.dimensions,
        fields="inverse_brems_deposition",
    )
    mesh = build_linear_deposition_mesh_from_grid(grid)
    deposition = deposit_simplicial_power_to_mesh(field, mesh)

    assert grid.dimensions == 1
    assert bool(trace.terminated)
    assert deposition.source_power > 0.0
    assert deposition.deposited_power > 0.0
    assert deposition.outside_power < 1.0e-8 * deposition.source_power
    np.testing.assert_allclose(
        deposition.conservation_error,
        0.0,
        atol=1.0e-12 * deposition.source_power,
    )
    assert np.all(np.asarray(deposition.power_density) >= 0.0)
