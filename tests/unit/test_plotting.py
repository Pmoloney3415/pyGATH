import jax.numpy as jnp
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pyGATH.fields import tetrahedralise_sheet_fields
from pyGATH.grid import Grid, HydroFields
from pyGATH.plotting import (
    plot_hydro_slice,
    plot_ray_trajectories,
    plot_tetrahedral_mesh,
)
from pyGATH.raytracing import RAY_SHEET_LAYOUT, RAY_STATE_LAYOUT, RayTraceResult


def test_hydro_slice_converts_cylindrical_vertices_and_vectors_to_cartesian():
    def radial_flow(coordinates):
        shape = coordinates.vertex_shape
        return HydroFields(
            ne=2.0 * jnp.ones(shape),
            Te=jnp.ones(shape),
            Ti=jnp.ones(shape),
            velocity=jnp.broadcast_to(jnp.asarray((1.0, 0.0, 0.0)), (*shape, 3)),
        )

    grid = Grid.create(
        geom="cylindrical",
        extents=((1.0, 2.0), (0.0, 0.5 * np.pi), (0.0, 1.0)),
        ncells=(1, 1, 1),
        initial_condition=radial_flow,
    )
    fig, ax = plt.subplots()
    _, mesh = plot_hydro_slice(
        grid,
        "velocity",
        axes=("r", "phi"),
        index=0,
        component="y",
        projection="xy",
        ax=ax,
    )

    coordinates = mesh.get_coordinates()
    np.testing.assert_allclose(coordinates[0, 0], (1.0, 0.0), atol=1.0e-15)
    np.testing.assert_allclose(coordinates[1, 1], (0.0, 2.0), atol=1.0e-15)
    np.testing.assert_allclose(mesh.get_array(), [[0.5]], atol=1.0e-15)
    assert ax.get_xlabel() == "x [m]"
    assert ax.get_ylabel() == "y [m]"
    plt.close(fig)


def _trace_result_for_plotting():
    fields = np.zeros((1, 2, 2, 2, 3, RAY_SHEET_LAYOUT.n_attributes))
    paths = np.linspace(0.0, 1.0, 3)
    for sheet in range(2):
        for first in range(2):
            for second in range(2):
                fields[0, sheet, first, second, :, RAY_STATE_LAYOUT.position] = (
                    np.column_stack(
                        (
                            paths + sheet,
                            np.full(3, first),
                            paths + second,
                        )
                    )
                )
    return RayTraceResult(
        sheet_fields=fields,
        has_caustic=np.asarray([[[True, False], [True, True]]]),
        caustic_path=np.zeros((1, 2, 2)),
        caustic_score=np.zeros((1, 2, 2)),
        terminal_path=np.asarray(1.0),
        terminated=np.asarray(True),
    )


def test_ray_trajectories_color_both_sheets_and_scatter_caustics():
    fig, ax = plt.subplots()
    _, artists = plot_ray_trajectories(
        _trace_result_for_plotting(), projection="xz", ax=ax
    )

    assert len(artists["sheet_1"]) == 4
    assert len(artists["sheet_2"]) == 4
    assert artists["sheet_1"][0].get_color() == "tab:blue"
    assert artists["sheet_2"][0].get_color() == "tab:orange"
    assert artists["caustics"].get_offsets().shape == (3, 2)
    np.testing.assert_allclose(artists["sheet_1"][0].get_xdata(), [0.0, 0.5, 1.0])
    np.testing.assert_allclose(artists["sheet_1"][0].get_ydata(), [0.0, 0.5, 1.0])
    assert ax.get_xlabel() == "x [m]"
    assert ax.get_ylabel() == "z [m]"
    plt.close(fig)


def test_ray_stride_applies_to_both_transverse_axes():
    fig, ax = plt.subplots()
    _, artists = plot_ray_trajectories(
        _trace_result_for_plotting(), ray_stride=2, ax=ax
    )
    assert len(artists["sheet_1"]) == 1
    assert len(artists["sheet_2"]) == 1
    assert artists["caustics"].get_offsets().shape == (1, 2)
    plt.close(fig)


def test_tetrahedral_mesh_plot_draws_background_and_highlight():
    fields = np.zeros((1, 1, 2, 2, 2, RAY_SHEET_LAYOUT.n_attributes))
    first, second, sample = np.meshgrid((0.0, 1.0), (0.0, 1.0), (0.0, 1.0))
    fields[0, 0, ..., RAY_STATE_LAYOUT.position] = np.stack(
        (first, second, sample), axis=-1
    )
    tetrahedral_field = tetrahedralise_sheet_fields(fields, fields="ray_power")
    figure = plt.figure()
    axis = figure.add_subplot(111, projection="3d")

    _, artists = plot_tetrahedral_mesh(
        tetrahedral_field,
        tetrahedron_stride=2,
        highlighted_tetrahedra=(0,),
        ax=axis,
    )

    assert artists["background_edges"] is not None
    assert artists["background_vertices"] is not None
    assert len(artists["highlights"]) == 1
    assert axis.get_xlabel() == "x [m]"
    assert axis.get_zlabel() == "z [m]"
    plt.close(figure)
