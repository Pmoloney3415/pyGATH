import numpy as np
import pytest

from examples._workflows import (
    TWO_DIMENSIONAL_FIELD_CONFIGS,
    run_two_dimensional_field_reconstruction,
)
from pyGATH.raytracing import RAY_SHEET_LAYOUT, RAY_STATE_LAYOUT

pytestmark = pytest.mark.regression


@pytest.mark.parametrize(
    ("geometry", "minimum_caustic_rays"),
    (("cartesian", 64), ("cylindrical", 60)),
)
def test_two_dimensional_paper_field_reconstruction_in_both_coordinates(
    geometry, minimum_caustic_rays
):
    result, grid, _beams, reconstruction, checks = (
        run_two_dimensional_field_reconstruction(
            TWO_DIMENSIONAL_FIELD_CONFIGS[geometry]
        )
    )
    fields = np.asarray(result.sheet_fields)

    assert grid.geom.value == geometry
    assert grid.dimensions == 2
    assert fields.shape == (1, 2, 64, 1, 400, RAY_SHEET_LAYOUT.n_attributes)
    assert bool(result.terminated)
    assert checks["rays_with_caustics"] >= minimum_caustic_rays
    assert checks["maximum_caustic_density_over_ncritical"] > 0.95
    assert checks["maximum_inverse_brems_deposition_w_m3"] == 0.0

    active = np.any(reconstruction["inside"], axis=0)
    assert np.all(np.isfinite(reconstruction["capped_field_v_m"][active]))
    assert reconstruction["quiver_amplitude_per_v_m"] > 0.0
    valid_sheet = reconstruction["inside"]
    uncapped_sheet = reconstruction["uncapped_sheet_magnitude_v_m"]
    capped_sheet = reconstruction["capped_sheet_magnitude_v_m"]
    tolerance = 1.0e-12 * np.max(uncapped_sheet[valid_sheet])
    assert np.all(uncapped_sheet[valid_sheet] + tolerance >= capped_sheet[valid_sheet])
    assert np.any(capped_sheet[valid_sheet] < uncapped_sheet[valid_sheet] - tolerance)
    np.testing.assert_allclose(
        fields[..., RAY_STATE_LAYOUT.position][..., 2], 0.0, atol=1.0e-15
    )
    np.testing.assert_allclose(
        fields[..., RAY_STATE_LAYOUT.momentum][..., 2], 0.0, atol=1.0e-15
    )
