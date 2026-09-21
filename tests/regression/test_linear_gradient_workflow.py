from pathlib import Path

import numpy as np
import pytest

from pyGATH.io import load_simulation_config
from pyGATH.raytracing import (
    RAY_SHEET_LAYOUT,
    RAY_STATE_LAYOUT,
    critical_density,
)

pytestmark = pytest.mark.regression

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG = PROJECT_ROOT / "configs" / "test_configs" / "linear_gradient_turning.toml"


def test_linear_gradient_ray_turns_analytically_and_builds_two_sheets():
    simulation = load_simulation_config(CONFIG)
    grid = simulation.build_grid()
    beams = simulation.load_beams()
    initial = simulation.initialize_rays(grid, beams=beams)
    result = simulation.trace_rays(initial, grid)
    fields = np.asarray(result.sheet_fields)

    assert fields.shape == (1, 2, 1, 1, 40, RAY_SHEET_LAYOUT.n_attributes)
    assert bool(result.terminated)
    assert bool(result.has_caustic[0, 0, 0])

    direction = np.asarray(beams.direction[0])
    angle = np.arctan2(direction[1], direction[0])
    density_scale = float(critical_density(float(beams.omega[0]))) / float(
        grid.grad_ne[0, 0, 0, 0]
    )
    expected_turn_x = density_scale * np.cos(angle) ** 2
    positions = fields[0, :, 0, 0, ..., RAY_STATE_LAYOUT.position]
    np.testing.assert_allclose(np.max(positions[..., 0]), expected_turn_x, rtol=2.0e-3)
    np.testing.assert_allclose(positions[0, -1, 0], expected_turn_x, rtol=3.0e-3)
    np.testing.assert_allclose(
        fields[0, 0, 0, 0, -1, : RAY_STATE_LAYOUT.n_attributes],
        fields[0, 1, 0, 0, 0, : RAY_STATE_LAYOUT.n_attributes],
        atol=1.0e-12,
    )
