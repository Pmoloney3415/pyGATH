import math

import numpy as np
import pytest

from examples._workflows import (
    PAPER_S64_NO_CBET_ABSORPTION,
    run_paper_s64,
    run_uniform_1d,
)
from pyGATH.raytracing import RAY_STATE_LAYOUT

pytestmark = pytest.mark.regression


def test_uniform_1d_grid_deposition_matches_total_analytical_absorption():
    trace, grid, beams, deposition, checks = run_uniform_1d()

    assert grid.dimensions == 1
    assert beams.nbeams == 1
    assert bool(trace.terminated)
    np.testing.assert_allclose(
        checks["direct_absorption_fraction"],
        checks["analytical_absorption_fraction"],
        rtol=5.0e-6,
    )
    np.testing.assert_allclose(
        checks["deposited_absorption_fraction"],
        checks["analytical_absorption_fraction"],
        rtol=7.0e-4,
    )
    np.testing.assert_allclose(
        checks["source_absorption_fraction"],
        checks["analytical_absorption_fraction"],
        rtol=7.0e-4,
    )
    assert checks["outside_power_fraction"] < 1.0e-8
    assert abs(checks["deposition_conservation_error_fraction"]) < 1.0e-12
    assert checks["maximum_cbet_depth"] == 0.0
    assert np.all(np.asarray(deposition.cell_power) >= 0.0)


def test_paper_s64_no_cbet_deposition_in_cartesian_and_cylindrical_grids():
    cases = run_paper_s64()
    raw_deposited = {}
    coherent_deposited = {}

    for geometry, (trace, grid, beams, _deposition, checks) in cases.items():
        assert grid.geom.value == geometry
        assert grid.dimensions == 2
        assert beams.nbeams == 10
        assert bool(trace.terminated)
        assert checks["maximum_cbet_depth"] == 0.0
        assert abs(checks["deposition_conservation_error_fraction"]) < 1.0e-12

        angles = np.mod(
            np.arctan2(np.asarray(beams.origin)[:, 1], np.asarray(beams.origin)[:, 0]),
            2.0 * np.pi,
        )
        wrapped_gaps = np.diff(np.r_[np.sort(angles), np.sort(angles)[0] + 2 * np.pi])
        np.testing.assert_allclose(wrapped_gaps, 2.0 * np.pi / 10.0, atol=1.0e-14)
        np.testing.assert_allclose(np.asarray(beams.peak_intensity), 1.625e18)
        np.testing.assert_allclose(np.asarray(grid.hydro.Te), 70.0)
        np.testing.assert_allclose(np.asarray(grid.hydro.Ti), 35.0)
        expected_incident_power = (
            10 * 1.625e18 * 2.0 * 6.043942984636073e-6 * math.gamma(1.25)
        )
        np.testing.assert_allclose(checks["incident_power_w"], expected_incident_power)

        np.testing.assert_allclose(
            checks["direct_absorption_fraction"],
            PAPER_S64_NO_CBET_ABSORPTION,
            atol=1.5e-2,
        )
        assert checks["outside_power_fraction"] < 2.0e-3
        assert (
            checks["coherent_deposited_absorption_fraction"]
            > checks["deposited_absorption_fraction"]
        )
        np.testing.assert_allclose(
            checks["coherent_deposited_absorption_fraction"],
            PAPER_S64_NO_CBET_ABSORPTION,
            atol=2.5e-2,
        )
        raw_deposited[geometry] = checks["deposited_absorption_fraction"]
        coherent_deposited[geometry] = checks["coherent_deposited_absorption_fraction"]

        fields = np.asarray(trace.sheet_fields)
        assert np.max(fields[..., RAY_STATE_LAYOUT.inverse_brems_depth]) > 0.0

    np.testing.assert_allclose(
        raw_deposited["cartesian"], raw_deposited["cylindrical"], atol=3.0e-3
    )
    np.testing.assert_allclose(
        coherent_deposited["cartesian"],
        coherent_deposited["cylindrical"],
        atol=5.0e-3,
    )
