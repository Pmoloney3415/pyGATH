import numpy as np
import pytest

from examples._workflows import run_linear_gradient_turning
from pyGATH.raytracing import RAY_SHEET_LAYOUT, RAY_STATE_LAYOUT

pytestmark = pytest.mark.regression


def test_linear_gradient_ray_turns_analytically_and_builds_two_sheets():
    result, checks = run_linear_gradient_turning()
    fields = np.asarray(result.sheet_fields)

    assert fields.shape == (1, 2, 1, 1, 40, RAY_SHEET_LAYOUT.n_attributes)
    assert bool(result.terminated)
    assert bool(result.has_caustic[0, 0, 0])

    expected_turn_x = 1.0e-3 * np.cos(np.deg2rad(20.0)) ** 2
    np.testing.assert_allclose(
        checks["expected_density_ratio"], np.cos(np.deg2rad(20.0)) ** 2
    )
    np.testing.assert_allclose(
        checks["numerical_turn_x_m"], expected_turn_x, rtol=2.0e-3
    )
    np.testing.assert_allclose(checks["caustic_x_m"], expected_turn_x, rtol=3.0e-3)

    first_sheet = fields[0, 0, 0, 0]
    second_sheet = fields[0, 1, 0, 0]
    np.testing.assert_allclose(
        first_sheet[-1, : RAY_STATE_LAYOUT.n_attributes],
        second_sheet[0, : RAY_STATE_LAYOUT.n_attributes],
        rtol=0.0,
        atol=1.0e-12,
    )
    assert not np.allclose(
        second_sheet[0, RAY_STATE_LAYOUT.position],
        second_sheet[-1, RAY_STATE_LAYOUT.position],
    )
    np.testing.assert_allclose(
        fields[..., RAY_STATE_LAYOUT.position][..., 2], 0.0, atol=1.0e-15
    )
    np.testing.assert_allclose(
        fields[..., RAY_STATE_LAYOUT.momentum][..., 2], 0.0, atol=1.0e-15
    )
