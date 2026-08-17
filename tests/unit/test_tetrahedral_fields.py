import jax
import numpy as np
import pytest

from pyGATH.fields import (
    deposit_tetrahedral_power,
    grid_cell_volumes,
    interpolate_tetrahedral_fields,
    interpolate_tetrahedral_fields_batched,
    replace_tetrahedral_field_values,
    tetrahedralise_sheet_fields,
)
from pyGATH.grid import Grid
from pyGATH.raytracing import RAY_SHEET_LAYOUT, RAY_STATE_LAYOUT


def _sheet_fields(*, inverted=False, include_collapsed_sheet=True):
    nsheets = 2 if include_collapsed_sheet else 1
    fields = np.zeros((1, nsheets, 2, 2, 2, RAY_SHEET_LAYOUT.n_attributes))
    for first in range(2):
        for second in range(2):
            for sample in range(2):
                x = float(1 - first if inverted else first)
                position = np.asarray((x, float(second), float(sample)))
                fields[0, 0, first, second, sample, RAY_STATE_LAYOUT.position] = (
                    position
                )
                fields[0, 0, first, second, sample, RAY_STATE_LAYOUT.ray_power] = (
                    1.0 + 2.0 * position[0] + 3.0 * position[1] + 4.0 * position[2]
                )
                fields[
                    0, 0, first, second, sample, RAY_SHEET_LAYOUT.capped_amplitude
                ] = -2.0 + position[0] - position[1] + 0.5 * position[2]
    if include_collapsed_sheet:
        fields[0, 1, ..., RAY_STATE_LAYOUT.position] = 3.0
        fields[0, 1, ..., RAY_STATE_LAYOUT.ray_power] = 99.0
        fields[0, 1, ..., RAY_SHEET_LAYOUT.capped_amplitude] = 99.0
    return fields


def test_tetrahedral_field_jit_batching_selection_and_value_replacement():
    sheet_fields = _sheet_fields()
    field = tetrahedralise_sheet_fields(
        sheet_fields, fields=("ray_power", "capped_amplitude")
    )
    points = np.asarray(((0.2, 0.3, 0.4), (2.0, 2.0, 2.0), (0.5, 0.5, 0.5)))

    result = jax.jit(interpolate_tetrahedral_fields)(field, points)

    assert field.mesh.connectivity.shape == (6, 4)
    assert field.selection.ray_power == 0
    assert field.selection.capped_amplitude == 1
    with pytest.raises(AttributeError):
        _ = field.selection.position
    assert result.values.shape == (1, 2, 3, 2)
    np.testing.assert_array_equal(result.inside[0, 0], (True, False, True))
    np.testing.assert_array_equal(result.inside[0, 1], False)
    assert result.tet_index[0, 0, 0] >= 0
    np.testing.assert_allclose(result.values[0, 0, 0], (3.9, -1.9), atol=1.0e-13)
    np.testing.assert_allclose(result.values[0, 0, 2], (5.5, -1.75), atol=1.0e-13)
    np.testing.assert_allclose(result.values[:, :, 1], 0.0, atol=0.0)
    np.testing.assert_allclose(result.values[0, 1], 0.0, atol=0.0)

    batched = interpolate_tetrahedral_fields_batched(field, points, point_batch_size=2)
    np.testing.assert_allclose(batched.values, result.values, atol=1.0e-13)
    np.testing.assert_array_equal(batched.inside, result.inside)
    np.testing.assert_array_equal(batched.tet_index, result.tet_index)

    updated_sheet_fields = sheet_fields.copy()
    updated_sheet_fields[..., RAY_STATE_LAYOUT.ray_power] += 10.0
    updated = replace_tetrahedral_field_values(field, updated_sheet_fields)
    updated_result = interpolate_tetrahedral_fields(updated, points[:1])
    assert updated.mesh is field.mesh
    np.testing.assert_allclose(
        updated_result.values[0, 0, 0, updated.selection.ray_power],
        13.9,
        atol=1.0e-13,
    )


def test_inverted_tetrahedra_remain_interpolatable():
    field = tetrahedralise_sheet_fields(
        _sheet_fields(inverted=True, include_collapsed_sheet=False),
        fields="ray_power",
    )
    result = interpolate_tetrahedral_fields(field, np.asarray(((0.25, 0.5, 0.75),)))

    np.testing.assert_array_equal(field.mesh.valid, True)
    assert bool(result.inside[0, 0, 0])
    np.testing.assert_allclose(result.values[0, 0, 0, 0], 6.0, atol=1.0e-13)


def test_adaptive_power_deposition_is_conservative_and_beam_selectable():
    sheet_fields = np.repeat(_sheet_fields(include_collapsed_sheet=False), 2, axis=0)
    sheet_fields[0, ..., RAY_SHEET_LAYOUT.inverse_brems_deposition] = 2.0
    sheet_fields[1, ..., RAY_SHEET_LAYOUT.inverse_brems_deposition] = 3.0
    field = tetrahedralise_sheet_fields(sheet_fields, fields="inverse_brems_deposition")
    grid = Grid.create(
        geom="cartesian",
        extents=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
        ncells=(2, 1, 1),
    )

    coarse = deposit_tetrahedral_power(
        field,
        grid,
        max_subdivision_levels=0,
        relative_tolerance=0.0,
        tetrahedron_batch_size=8,
    )
    total = deposit_tetrahedral_power(
        field,
        grid,
        max_subdivision_levels=2,
        relative_tolerance=0.0,
        tetrahedron_batch_size=8,
    )
    first_beam = deposit_tetrahedral_power(
        field,
        grid,
        beam_index=0,
        max_subdivision_levels=2,
        relative_tolerance=0.0,
        tetrahedron_batch_size=8,
    )

    refined_error = np.max(np.abs(np.asarray(total.power_density) - 5.0))
    coarse_error = np.max(np.abs(np.asarray(coarse.power_density) - 5.0))
    assert refined_error < coarse_error
    np.testing.assert_allclose(total.deposited_power, 5.0, rtol=1.0e-13)
    np.testing.assert_allclose(total.outside_power, 0.0, atol=1.0e-15)
    np.testing.assert_allclose(total.conservation_error, 0.0, atol=1.0e-13)
    np.testing.assert_allclose(
        first_beam.power_density, 0.4 * total.power_density, rtol=1.0e-13
    )
    np.testing.assert_allclose(first_beam.source_power, 2.0, rtol=1.0e-13)


@pytest.mark.parametrize(
    ("geometry", "extents", "expected_volume", "cube_lower", "cube_upper"),
    (
        (
            "cartesian",
            ((0.0, 2.0), (0.0, 3.0), (0.0, 4.0)),
            24.0,
            (0.5, 1.0, 1.5),
            (0.6, 1.1, 1.6),
        ),
        (
            "cylindrical",
            ((1.0, 2.0), (0.0, 0.5 * np.pi), (-1.0, 1.0)),
            1.5 * np.pi,
            (1.45, 0.20, -0.05),
            (1.55, 0.30, 0.05),
        ),
        (
            "spherical",
            ((1.0, 2.0), (0.0, 0.5 * np.pi), (0.0, 0.5 * np.pi)),
            7.0 * np.pi / 6.0,
            (0.85, 0.25, 0.85),
            (0.95, 0.35, 0.95),
        ),
    ),
)
def test_grid_cell_volumes_are_exact_in_supported_geometries(
    geometry, extents, expected_volume, cube_lower, cube_upper
):
    grid = Grid.create(geom=geometry, extents=extents, ncells=(1, 1, 1))
    volumes = jax.jit(grid_cell_volumes)(grid)
    np.testing.assert_allclose(volumes, expected_volume, rtol=1.0e-14)

    sheet_fields = _sheet_fields(include_collapsed_sheet=False)
    lower = np.asarray(cube_lower)
    upper = np.asarray(cube_upper)
    unit_positions = sheet_fields[..., RAY_STATE_LAYOUT.position]
    sheet_fields[..., RAY_STATE_LAYOUT.position] = lower + unit_positions * (
        upper - lower
    )
    sheet_fields[..., RAY_SHEET_LAYOUT.inverse_brems_deposition] = 4.0
    field = tetrahedralise_sheet_fields(sheet_fields, fields="inverse_brems_deposition")
    deposition = deposit_tetrahedral_power(
        field,
        grid,
        max_subdivision_levels=1,
        relative_tolerance=0.0,
        tetrahedron_batch_size=8,
    )
    expected_power = 4.0 * np.prod(upper - lower)
    np.testing.assert_allclose(deposition.deposited_power, expected_power)
    np.testing.assert_allclose(deposition.outside_power, 0.0, atol=1.0e-15)
    np.testing.assert_allclose(
        deposition.power_density, expected_power / expected_volume
    )
