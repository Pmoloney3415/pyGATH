import numpy as np

from pyGATH.fields import (
    build_cartesian_deposition_mesh_from_grid,
    build_circular_deposition_mesh,
    build_circular_deposition_mesh_from_grid,
    build_geodesic_deposition_mesh,
    build_geodesic_deposition_mesh_from_grid,
    build_linear_deposition_mesh,
    build_linear_deposition_mesh_from_grid,
    deposit_simplicial_power_to_mesh,
    simplicialise_sheet_fields,
)
from pyGATH.grid import Grid
from pyGATH.raytracing import RAY_SHEET_LAYOUT, RAY_STATE_LAYOUT


def _affine_line_field(function):
    fields = np.zeros((1, 1, 1, 1, 3, RAY_SHEET_LAYOUT.n_attributes))
    for sample, x in enumerate((-1.0, 0.0, 1.0)):
        fields[0, 0, 0, 0, sample, RAY_STATE_LAYOUT.position] = (x, 0.0, 0.0)
        fields[0, 0, 0, 0, sample, RAY_SHEET_LAYOUT.inverse_brems_deposition] = (
            function(x)
        )
    return simplicialise_sheet_fields(
        fields, dimension=1, fields="inverse_brems_deposition"
    )


def _affine_square_field(function):
    fields = np.zeros((1, 1, 2, 1, 2, RAY_SHEET_LAYOUT.n_attributes))
    for ray, y in enumerate((-1.0, 1.0)):
        for sample, x in enumerate((-1.0, 1.0)):
            fields[0, 0, ray, 0, sample, RAY_STATE_LAYOUT.position] = (x, y, 0.0)
            fields[0, 0, ray, 0, sample, RAY_SHEET_LAYOUT.inverse_brems_deposition] = (
                function(x, y)
            )
    return simplicialise_sheet_fields(
        fields, dimension=2, fields="inverse_brems_deposition"
    )


def _affine_cube_field(function):
    fields = np.zeros((1, 1, 2, 2, 2, RAY_SHEET_LAYOUT.n_attributes))
    for first, x in enumerate((-1.0, 1.0)):
        for second, y in enumerate((-1.0, 1.0)):
            for sample, z in enumerate((-1.0, 1.0)):
                fields[0, 0, first, second, sample, RAY_STATE_LAYOUT.position] = (
                    x,
                    y,
                    z,
                )
                fields[
                    0,
                    0,
                    first,
                    second,
                    sample,
                    RAY_SHEET_LAYOUT.inverse_brems_deposition,
                ] = function(x, y, z)
    return simplicialise_sheet_fields(
        fields, dimension=3, fields="inverse_brems_deposition"
    )


def test_linear_mesh_uses_grid_boundaries_and_reference_area():
    grid = Grid.create(
        geom="cartesian",
        dimensions=1,
        extents=((-1.0, 1.0),),
        ncells=(4,),
        inactive_axis_lengths_m=(2.0, 3.0),
    )
    mesh = build_linear_deposition_mesh_from_grid(grid)

    np.testing.assert_allclose(mesh.radial_boundaries, grid.xb)
    np.testing.assert_allclose(mesh.cell_volumes, 3.0)
    assert mesh.dimension == 1
    assert mesh.topology == "linear"


def test_circular_mesh_scales_ring_counts_and_stitches_with_triangles():
    mesh = build_circular_deposition_mesh(
        (0.0, 1.0, 2.0), maximum_angular_cells=8, inactive_length_m=3.0
    )

    assert mesh.dimension == 2
    np.testing.assert_array_equal(mesh.surface_element_counts, (0, 4, 8))
    assert mesh.vertex_positions.shape == (13, 3)
    assert mesh.simplex_connectivity.shape == (16, 3)
    np.testing.assert_array_equal(np.bincount(mesh.simplex_radial_layer), (4, 12))
    expected_area = 0.5 * 8 * 2.0**2 * np.sin(2.0 * np.pi / 8)
    np.testing.assert_allclose(mesh.cell_volumes.sum(), 3.0 * expected_area)


def test_geodesic_mesh_scales_surface_faces_and_joins_with_tetrahedra():
    mesh = build_geodesic_deposition_mesh(
        (0.0, 0.25, 0.5, 1.0), maximum_angular_cells=40
    )

    assert mesh.dimension == 3
    np.testing.assert_array_equal(mesh.surface_element_counts, (0, 20, 80, 320))
    np.testing.assert_array_equal(mesh.surface_angular_counts, (0, 10, 20, 40))
    assert np.all(mesh.cell_volumes > 0.0)
    np.testing.assert_allclose(mesh.cell_volumes.sum(), 4.047044679978849)


def test_radial_meshes_can_inherit_grid_geometry():
    circular_grid = Grid.create(
        geom="cylindrical",
        dimensions=2,
        extents=((0.0, 2.0), (-np.pi, np.pi)),
        ncells=(2, 8),
        inactive_axis_lengths_m=(3.0,),
    )
    circular = build_circular_deposition_mesh_from_grid(circular_grid)
    np.testing.assert_array_equal(circular.surface_element_counts, (0, 4, 8))
    assert circular.inactive_measure == 3.0

    spherical_grid = Grid.create(
        geom="spherical",
        extents=((0.0, 2.0), (-np.pi, np.pi), (0.0, np.pi)),
        ncells=(2, 8, 4),
    )
    geodesic = build_geodesic_deposition_mesh_from_grid(
        spherical_grid, maximum_angular_cells=10
    )
    np.testing.assert_array_equal(geodesic.surface_element_counts, (0, 20, 20))


def test_cartesian_grids_build_complete_simplicial_meshes():
    square_grid = Grid.create(
        geom="cartesian",
        dimensions=2,
        extents=((0.0, 2.0), (-1.0, 1.0)),
        ncells=(2, 1),
        inactive_axis_lengths_m=(3.0,),
    )
    square = build_cartesian_deposition_mesh_from_grid(square_grid)
    assert square.simplex_connectivity.shape == (4, 3)
    np.testing.assert_allclose(square.cell_volumes.sum(), 12.0)

    cube_grid = Grid.create(
        geom="cartesian",
        extents=((0.0, 2.0), (-1.0, 1.0), (0.0, 3.0)),
        ncells=(2, 1, 1),
    )
    cube = build_cartesian_deposition_mesh_from_grid(cube_grid)
    assert cube.simplex_connectivity.shape == (12, 4)
    np.testing.assert_allclose(cube.cell_volumes.sum(), 12.0)


def test_exact_one_dimensional_affine_deposition_is_conservative():
    field = _affine_line_field(lambda x: 2.0 + x)
    mesh = build_linear_deposition_mesh((-0.5, 0.0, 0.5), inactive_area_m2=3.0)
    deposition = deposit_simplicial_power_to_mesh(field, mesh, pair_batch_size=4)

    np.testing.assert_allclose(deposition.cell_power, (2.625, 3.375))
    np.testing.assert_allclose(deposition.source_power, 12.0)
    np.testing.assert_allclose(deposition.deposited_power, 6.0)
    np.testing.assert_allclose(deposition.outside_power, 6.0)
    np.testing.assert_allclose(deposition.conservation_error, 0.0)


def test_exact_two_dimensional_affine_deposition_and_resolved_selection():
    affine = lambda x, y: 5.0 + 2.0 * x - 3.0 * y
    field = _affine_square_field(affine)
    mesh = build_circular_deposition_mesh(
        (0.0, 1.0), maximum_angular_cells=4, inactive_length_m=2.0
    )
    resolved = deposit_simplicial_power_to_mesh(
        field, mesh, resolve_beam_sheets=True, pair_batch_size=8
    )
    deposition = resolved.total
    expected_density = affine(mesh.cell_centres[:, 0], mesh.cell_centres[:, 1])
    expected_power = mesh.cell_volumes * expected_density

    np.testing.assert_allclose(deposition.cell_power, expected_power, rtol=2.0e-13)
    np.testing.assert_allclose(resolved.select(beam_index=0).cell_power, expected_power)
    np.testing.assert_allclose(deposition.source_power, 40.0)
    np.testing.assert_allclose(deposition.conservation_error, 0.0, atol=1.0e-14)


def test_exact_three_dimensional_affine_deposition_on_geodesic_layers():
    affine = lambda x, y, z: 4.0 + 0.2 * x - 0.3 * y + 0.1 * z
    field = _affine_cube_field(affine)
    mesh = build_geodesic_deposition_mesh((0.0, 0.5, 1.0), maximum_angular_cells=10)
    deposition = deposit_simplicial_power_to_mesh(field, mesh, source_batch_size=16)
    expected_density = affine(
        mesh.cell_centres[:, 0], mesh.cell_centres[:, 1], mesh.cell_centres[:, 2]
    )

    np.testing.assert_allclose(
        deposition.cell_power,
        mesh.cell_volumes * expected_density,
        rtol=2.0e-11,
        atol=1.0e-13,
    )
    np.testing.assert_allclose(deposition.source_power, 32.0)
    np.testing.assert_allclose(deposition.conservation_error, 0.0, atol=1.0e-14)
