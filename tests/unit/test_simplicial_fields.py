import jax
import numpy as np
import pytest

from pyGATH.fields import (
    interpolate_simplicial_fields,
    interpolate_simplicial_fields_batched,
    replace_simplicial_field_values,
    simplicialise_sheet_fields,
)
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


def test_three_dimensional_field_is_jittable_batched_and_replaceable():
    sheet_fields = _sheet_fields()
    field = simplicialise_sheet_fields(
        sheet_fields,
        dimension=3,
        fields=("ray_power", "capped_amplitude"),
    )
    points = np.asarray(((0.2, 0.3, 0.4), (2.0, 2.0, 2.0), (0.5, 0.5, 0.5)))

    result = jax.jit(interpolate_simplicial_fields)(field, points)

    assert field.mesh.connectivity.shape == (6, 4)
    assert field.mesh.nsimplices == 6
    assert field.selection.ray_power == 0
    with pytest.raises(AttributeError):
        _ = field.selection.position
    np.testing.assert_array_equal(result.inside[0, 0], (True, False, True))
    np.testing.assert_array_equal(result.inside[0, 1], False)
    assert result.simplex_index[0, 0, 0] >= 0
    np.testing.assert_allclose(result.values[0, 0, 0], (3.9, -1.9), atol=1.0e-13)
    np.testing.assert_allclose(result.values[0, 0, 2], (5.5, -1.75), atol=1.0e-13)

    batched = interpolate_simplicial_fields_batched(field, points, point_batch_size=2)
    np.testing.assert_allclose(batched.values, result.values, atol=1.0e-13)
    np.testing.assert_array_equal(batched.simplex_index, result.simplex_index)

    updated_sheet_fields = sheet_fields.copy()
    updated_sheet_fields[..., RAY_STATE_LAYOUT.ray_power] += 10.0
    updated = replace_simplicial_field_values(field, updated_sheet_fields)
    assert updated.mesh is field.mesh
    updated_result = interpolate_simplicial_fields(updated, points[:1])
    np.testing.assert_allclose(
        updated_result.values[0, 0, 0, updated.selection.ray_power], 13.9
    )


def test_inverted_tetrahedra_remain_interpolatable():
    field = simplicialise_sheet_fields(
        _sheet_fields(inverted=True, include_collapsed_sheet=False),
        dimension=3,
        fields="ray_power",
    )
    result = interpolate_simplicial_fields(field, np.asarray(((0.25, 0.5, 0.75),)))

    np.testing.assert_array_equal(field.mesh.valid, True)
    assert bool(result.inside[0, 0, 0])
    np.testing.assert_allclose(result.values[0, 0, 0, 0], 6.0, atol=1.0e-13)
