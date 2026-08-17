from pathlib import Path

import numpy as np
import pytest

from pyGATH.io import BeamFileError, load_beams_csv

HEADER = (
    "beam_id,origin_geometry,origin_1,origin_2,origin_3,"
    "target_geometry,target_1,target_2,target_3,"
    "width_x_m,width_y_m,rotation_pi,supergaussian_index,"
    "omega_rad_s,power_fraction,peak_intensity_w_m2,beam_power_w\n"
)


def test_csv_loads_coordinate_systems_and_normalizes_relative_power(tmp_path):
    path = tmp_path / "beams.csv"
    path.write_text(
        HEADER
        + "cart,cartesian,-2,0,0,cartesian,0,0,0,0.1,0.2,0,2,,2\n"
        + "cyl,cylindrical,2,1,0,cartesian,0,0,0,0.2,0.1,0.5,4,1e15,1\n"
        + "sph,spherical,2,0,0.5,cartesian,0,0,0,0.1,0.1,0,2,2e15,0\n",
        encoding="utf-8",
    )
    beams = load_beams_csv(path)
    assert beams.names == ("cart", "cyl", "sph")
    np.testing.assert_allclose(beams.power_fraction, [2 / 3, 1 / 3, 0])
    np.testing.assert_allclose(beams.power_fraction.sum(), 1.0)
    np.testing.assert_allclose(beams.beam_power.sum(), 1.0)
    np.testing.assert_allclose(beams.origin[1], [-2.0, 0.0, 0.0], atol=1.0e-14)
    np.testing.assert_allclose(beams.origin[2], [2.0, 0.0, 0.0], atol=1.0e-14)
    np.testing.assert_allclose(beams.omega[0], 5.361e15)
    np.testing.assert_allclose(np.linalg.norm(beams.direction, axis=1), 1.0)
    np.testing.assert_allclose(np.sum(beams.axis_x * beams.direction, axis=1), 0.0)
    np.testing.assert_allclose(np.sum(beams.axis_y * beams.direction, axis=1), 0.0)


def test_zero_rotation_spot_y_projects_global_z(tmp_path):
    path = tmp_path / "basis.csv"
    path.write_text(
        HEADER + "beam,cartesian,-2,0,0,cartesian,0,0,0,0.1,0.1,0,2,1e15,1\n",
        encoding="utf-8",
    )
    beam = load_beams_csv(path)[0]
    np.testing.assert_allclose(beam.axis_y, [0.0, 0.0, 1.0])
    np.testing.assert_allclose(beam.axis_x, [0.0, 1.0, 0.0])


def test_peak_intensity_and_per_beam_power_modes_resolve_beam_fractions(tmp_path):
    path = tmp_path / "physical-power.csv"
    path.write_text(
        HEADER + "one,cartesian,-2,0,0,cartesian,0,0,0,0.1,0.2,0,2,1e15,"
        "ignored,100,3\n" + "two,cartesian,-2,0,0,cartesian,0,0,0,0.2,0.1,0,2,1e15,"
        "ignored,50,1\n",
        encoding="utf-8",
    )
    intensity_beams = load_beams_csv(path, power_mode="peak_intensity")
    np.testing.assert_allclose(intensity_beams.peak_intensity, [100.0, 50.0])
    np.testing.assert_allclose(intensity_beams.power_fraction, [2.0 / 3.0, 1.0 / 3.0])

    power_beams = load_beams_csv(path, power_mode="per_beam_power")
    np.testing.assert_allclose(power_beams.beam_power, [3.0, 1.0])
    np.testing.assert_allclose(power_beams.power_fraction, [0.75, 0.25])


@pytest.mark.parametrize(
    ("row", "message"),
    [
        (
            "bad,cylindrical,1,1.1,0,cartesian,0,0,0,1,1,0,2,1e15,1\n",
            "phi fraction",
        ),
        (
            "bad,spherical,1,0,-0.1,cartesian,0,0,0,1,1,0,2,1e15,1\n",
            "theta fraction",
        ),
        (
            "bad,cartesian,0,0,0,cartesian,0,0,0,-1,1,0,2,1e15,1\n",
            "widths",
        ),
    ],
)
def test_invalid_beam_rows_are_rejected(tmp_path, row, message):
    path = tmp_path / "invalid.csv"
    path.write_text(HEADER + row, encoding="utf-8")
    with pytest.raises(BeamFileError, match=message):
        load_beams_csv(path)


def test_unknown_csv_columns_are_rejected(tmp_path):
    path = Path(tmp_path) / "unknown.csv"
    path.write_text(HEADER.strip() + ",typo\n", encoding="utf-8")
    with pytest.raises(BeamFileError, match="unknown beam column"):
        load_beams_csv(path)
