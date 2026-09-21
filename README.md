# pyGATH

pyGATH is a JAX-based ray tracer and conservative power-deposition library for
high-power laser simulations. It traces ray tubes through a hydro background,
constructs piecewise-affine simplicial fields on the resulting ray sheets, and
integrates those fields onto a separate deposition mesh.

The numerical tracing, interpolation, candidate traversal, and overlap kernels
use JAX. Configuration, validation, and dynamic mesh/BVH topology construction
remain host-side Python/NumPy operations.

## Installation

pyGATH requires Python 3.11–3.13.

```console
uv sync
uv run python -c "import jax; print(jax.devices())"
```

The normal dependency installs CPU JAX. For supported NVIDIA systems, install
the appropriate accelerator wheel by following the JAX installation guide.
Double precision is enabled when `pyGATH` is imported.

## Canonical workflow

The public workflow has five stages:

```python
from pyGATH.fields import (
    build_geodesic_deposition_mesh_from_grid,
    deposit_simplicial_power_to_mesh,
    simplicialise_sheet_fields,
)
from pyGATH.io import load_simulation_config

simulation = load_simulation_config(
    "configs/example_configs/three_dimensional_geodesic_deposition.toml"
)

with simulation.reporting():
    # 1. Hydro grid
    grid = simulation.build_grid()

    # 2. Ray initialization and tracing
    beams = simulation.load_beams()
    initial_rays = simulation.initialize_rays(grid, beams=beams)
    trace = simulation.trace_rays(initial_rays, grid)

    # 3. Sheet-resolved piecewise-affine source
    source = simplicialise_sheet_fields(
        trace.sheet_fields,
        dimension=grid.dimensions,
        fields="inverse_brems_deposition",
    )

    # 4. Independent target mesh
    target = build_geodesic_deposition_mesh_from_grid(
        grid,
        maximum_angular_cells=20,
    )

    # 5. Exact conservative JAX deposition
    deposition = deposit_simplicial_power_to_mesh(source, target)
```

`deposition.cell_power` is measured in watts and
`deposition.power_density` in W/m³. The conservation identity is exposed by
`source_power`, `deposited_power`, `outside_power`, and `conservation_error`.

Set `resolve_beam_sheets=True` to retain the selected beam and sheet axes. The
returned `ResolvedPowerDeposition` can subsequently be reduced with
`select(beam_index=..., sheet_index=...)` or `.total`.

## Progress reporting

The reporting context timestamps every major workflow stage relative to the
start of the simulation. Configure console output, an optional text log, and
periodic progress in the input deck:

```toml
[logging]
verbosity = 2
console = true
file = "simulation.log"
progress_interval_s = 5.0
```

Verbosity `0` is silent, `1` reports stage boundaries, `2` adds numerical
progress and ETA estimates, and `3` adds detailed kernel diagnostics.
Relative log paths are resolved from the input deck. The file is overwritten
for each reporting context.

## Dimensions and deposition meshes

Ray sheets use one representation in every supported dimension:

- 1-D sheets are divided into line segments.
- 2-D sheets are divided into triangles.
- 3-D sheets are divided into tetrahedra.

The matching target-mesh builders are:

- `build_linear_deposition_mesh_from_grid` for 1-D Cartesian grids;
- `build_cartesian_deposition_mesh_from_grid` for 2-D/3-D Cartesian grids;
- `build_circular_deposition_mesh_from_grid` for full-azimuth 2-D cylindrical grids;
- `build_geodesic_deposition_mesh_from_grid` for full-sphere 3-D spherical grids.

Circular and geodesic targets refine their angular resolution with radius.
They are straight-sided physical meshes, independent of the curvilinear hydro
cells used during tracing.

## Field interpolation

The same `SimplicialField` geometry can be reused for interpolation:

```python
from pyGATH.fields import (
    interpolate_simplicial_fields_batched,
    replace_simplicial_field_values,
)

sampled = interpolate_simplicial_fields_batched(source, cartesian_points)
updated = replace_simplicial_field_values(source, new_sheet_fields)
```

Field construction creates a reusable host-built BVH. Point location and
barycentric interpolation execute with JAX.

## Examples

The example set is intentionally small and output-free:

- `examples/ray_tracing_and_sheet_fields.ipynb`
- `examples/one_dimensional_deposition.ipynb`
- `examples/two_dimensional_deposition.ipynb`
- `examples/three_dimensional_deposition.ipynb`

The three-dimensional kernel benchmark is:

```console
uv run python examples/benchmark_three_dimensional_deposition.py
```

## Tests and quality checks

```console
uv run pytest tests/unit
uv run pytest tests/regression
uv run ruff check .
uv run ruff format --check .
```

Unit tests use analytic affine fields to check exact overlap integration and
power conservation directly. Regression tests exercise configured grid, ray,
sheet, mesh, and deposition workflows without importing example code.

## License

BSD 3-Clause. See [LICENSE](LICENSE).
