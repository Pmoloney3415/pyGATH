import numpy as np
import pytest

from examples._workflows import run_one_dimensional_field_reconstruction
from pyGATH.raytracing import RAY_SHEET_LAYOUT, RAY_STATE_LAYOUT

pytestmark = pytest.mark.regression


def test_one_dimensional_paper_field_reconstruction_reflects_and_is_capped():
    result, _grid, _beams, reconstruction, checks = (
        run_one_dimensional_field_reconstruction()
    )
    fields = np.asarray(result.sheet_fields)

    assert fields.shape == (1, 2, 1, 1, 1200, RAY_SHEET_LAYOUT.n_attributes)
    assert bool(result.terminated)
    assert bool(result.has_caustic[0, 0, 0])
    np.testing.assert_allclose(
        checks["numerical_caustic_x_m"],
        checks["expected_critical_x_m"],
        rtol=2.0e-3,
    )
    np.testing.assert_allclose(
        checks["caustic_density_over_ncritical"], 1.0, rtol=2.0e-3
    )
    assert checks["maximum_inverse_brems_deposition_w_m3"] == 0.0

    active = np.any(reconstruction["inside"], axis=0)
    capped = reconstruction["capped_field_v_m"][active]
    assert np.all(np.isfinite(capped))
    assert reconstruction["quiver_amplitude_per_v_m"] > 0.0
    assert np.max(reconstruction["uncapped_sheet_magnitude_v_m"][:, active]) > np.max(
        reconstruction["capped_sheet_magnitude_v_m"][:, active]
    )
    np.testing.assert_allclose(
        fields[..., RAY_STATE_LAYOUT.position][..., 1:], 0.0, atol=1.0e-15
    )
    np.testing.assert_allclose(
        fields[..., RAY_STATE_LAYOUT.momentum][..., 1:], 0.0, atol=1.0e-15
    )
