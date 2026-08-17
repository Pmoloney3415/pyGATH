# pyGATH

[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue)](https://www.python.org/)
[![CI](https://github.com/Pmoloney3415/pyGATH/actions/workflows/ci.yml/badge.svg)](https://github.com/Pmoloney3415/pyGATH/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-BSD--3--Clause-green)](LICENSE)

pyGATH (***P**ython *G*pu *A*ccelerated ray-*T*racing for *H*igh-power lasers**, from GATH meaning Ray/ Beam in Scottish Gaelic), is a [JAX](https://docs.jax.dev/en/latest/index.html)-based ray-tracing tool to compute power deposition in high-power laser systems in the presence of CBET. JAX utilises an [XLA (Accelerated Linear Algebra) compiler](https://openxla.org/xla/), allowing the code to be JIT-compiled to both CPU and GPU hardware, providing that the function is pure.

## Installation

pyGATH requires Python 3.11, 3.12, or 3.13. To install the CPU version from a
clone in a Conda environment:

```console
conda create --name pygath python=3.13 pip
conda activate pygath
python -m pip install --upgrade pip
python -m pip install -e .
python -m pip install jupyterlab
```

The project dependency installs the CPU JAX wheel. For an NVIDIA GPU on Linux,
or experimentally under WSL2, install JAX's CUDA 13 wheel after installing
pyGATH:

```console
python -m pip install --upgrade "jax[cuda13]"
python -c "import jax; print(jax.devices())"
```

The CUDA wheel includes the CUDA and cuDNN Python dependencies but still
requires a compatible NVIDIA driver. Native Windows GPU wheels are not
available. CUDA 12 and AMD ROCm installations are covered in the
[JAX installation guide](https://docs.jax.dev/en/latest/installation.html).

For development, install `uv`, synchronize the locked development environment,
and enable the repository checks:

```console
python -m pip install uv
uv sync --all-groups
uv run pre-commit install
```

## Running pyGATH

A TOML simulation deck defines the grid, beam sampling, and ray-tracing
controls. Reusable demonstration decks live in `configs/example_configs`, while
decks used specifically by regression tests live in `configs/test_configs`.
Beam CSVs follow the same split under `beam_csvs`. The main workflow is:

```python
from pyGATH.io import load_simulation_config

simulation = load_simulation_config(
    "configs/example_configs/example_simulation.toml"
)
grid = simulation.build_grid()
beams = simulation.load_beams()
initial_rays = simulation.initialize_rays(grid, beams=beams)
result = simulation.trace_rays(initial_rays, grid)

print(result.sheet_fields.shape)
```

The notebooks under `examples` are the primary worked examples. Start with:

```console
uv run jupyter lab examples/linear_gradient_turning.ipynb
```

## Reduced-dimensional simulations

Set `grid.dimensions` to select an explicitly reduced simulation. It defaults
to `3`, so existing decks keep their original meaning. Supported reduced
geometries are 1-D Cartesian `x`, 2-D Cartesian `x-y`, and 2-D cylindrical
`r-phi`. Only active axes are listed in `[grid.axes]`:

```toml
[grid]
geometry = "cartesian"
dimensions = 2
inactive_axis_lengths_m = [1.0]

[grid.axes.x]
min = 0.0
max = 1.0
spacing = "uniform"
ncells = 64

[grid.axes.y]
min = -0.5
max = 0.5
spacing = "uniform"
ncells = 64
```

Each omitted direction is stored internally as one centred reference cell.
Its length defaults to 1 m and may be overridden by
`inactive_axis_lengths_m`, ordered after the active axes. Hydro fields must be
invariant in those directions, and inactive velocity components must be zero.
Arrays retain their existing three-coordinate shapes: a 1-D cell field is
`(nx, 1, 1)` and a 2-D field is `(nx, ny, 1)`.

One primary ray per beam is initialized in 1-D. In 2-D, `nrays_axis1` samples
one in-plane transverse row and `nrays_axis2` must be one. The ray ODE and its
three infinitesimal focusing neighbours remain three-dimensional. Beam powers
remain watts over the configured reference extrusion, while intensity and
deposition density retain units W/m² and W/m³. With peak-intensity input,
changing a reference length scales integrated watts but not the source density;
with fixed total-power input, it instead rescales the peak intensity. The
default 1 m lengths give directly comparable reference extrusions.

Only `width_x_m` controls 2-D transverse sampling; `width_y_m` and
`rotation_pi` are retained in the CSV schema but ignored. In 1-D, both widths
and the rotation are ignored because the single primary ray is on the beam
centre.

The corresponding spatial fields use segments, triangles, or tetrahedra:

```python
from pyGATH.fields import (
    deposit_simplicial_power,
    interpolate_simplicial_fields_to_cells,
    simplicialise_sheet_fields,
)

field = simplicialise_sheet_fields(
    result.sheet_fields,
    dimension=grid.dimensions,
    fields=("intensity", "inverse_brems_deposition"),
)
cell_values = interpolate_simplicial_fields_to_cells(field, grid)
heating = deposit_simplicial_power(field, grid)
```

Cell interpolation remains beam- and sheet-resolved. Conservative deposition
sums sheets and beams by default, as in the tetrahedral API. The existing
tetrahedral names remain available as explicitly 3-D compatibility wrappers.
Example decks are provided as `one_dimensional_cartesian.toml`,
`two_dimensional_cartesian.toml`, and `two_dimensional_cylindrical.toml`.

[`one_dimensional_field_reconstruction.ipynb`](examples/one_dimensional_field_reconstruction.ipynb)
reproduces the field-limiter side of the 1-D reflected-beam test from
Follett et al., *Phys. Plasmas* **29**, 113902 (2022). It coherently sums the
incident and reflected sheets, shows their interference, compares capped and
uncapped caustic amplitudes, and verifies that inverse-bremsstrahlung
deposition is zero.

[`two_dimensional_field_reconstruction.ipynb`](examples/two_dimensional_field_reconstruction.ipynb)
runs the corresponding one-beam $S=1/64$ reflection on both Cartesian
`x-y` and cylindrical `r-phi` grids. It triangulates the two ray sheets,
reconstructs the capped coherent field on each native grid, and evaluates both
solutions on a common Cartesian lineout for comparison.

`trace_rays` returns two path-resolved sheets with shape
`(nbeams, 2, nrays_axis1, nrays_axis2, nsamples, 43)`. The first 38 fields use
`RAY_STATE_LAYOUT`; the final five fields are the uncapped amplitude, capped
amplitude, propagated electric-field strength, propagated intensity, and
instantaneous inverse-bremsstrahlung power deposition in `RAY_SHEET_LAYOUT`.

Beam illumination is selected in `[beams.power]` using one of three modes:
`total_power`, `per_beam_power`, or `peak_intensity`. Total-power mode uses
`total_power_w` from the TOML deck and normalized `power_fraction` values from
the beam CSV. The other modes use the selected CSV column (`beam_power_w` or
`peak_intensity_w_m2`) and ignore the unselected power columns. Ray power is
stored as a dimensionless fraction of total incident power; initial intensity
and electric-field strength are stored separately.

The fully ionized plasma composition is specified by repeated
`[[grid.composition.elements]]` tables containing `name`, `A`, `Z`, and the
ion-number `fraction`. Electron density is authoritative, so total ion density
is derived as `ni = ne / sum(f_i Z_i)` and is not supplied by an initial
condition. Inverse bremsstrahlung can be enabled under
`[physics.inverse_bremsstrahlung]`, with the NRL Coulomb logarithm (minimum 2)
or an explicit override. Validation cases that specify the electron-ion
collision frequency at critical density can instead set
`critical_collision_frequency_hz`; this uses
`dTheta/dtau = 2 nu_ei,c (ne/ncritical)^2 / c` and is mutually exclusive with
`coulomb_log_override`.

The one- and two-dimensional deposition notebooks exercise the complete
inverse-bremsstrahlung and grid-interpolation path. The uniform 1-D case
compares total deposited power with the closed-form attenuated beam. The 2-D
case uses ten equal beams uniformly spaced over the full azimuth, with the
`S=1/64` density, beam width, intensity, temperature, collision frequency, and
40-micrometre box from Table II of Follett et al. CBET remains zero, and both
Cartesian and cylindrical grids are run.

The Table II regression is the 91.9% no-CBET absorption. The validation also
reports raw incoherent sheet deposition and a coherent two-sheet deposition at
hydro-cell centres, making the field-limiter loss at caustics visible rather
than folding it into an energy-normalization correction. The input decks are
`uniform_1d_deposition.toml`,
`paper_s64_ten_beam_cartesian_deposition.toml`, and
`paper_s64_ten_beam_cylindrical_deposition.toml`.

[`one_dimensional_deposition_validation.ipynb`](examples/one_dimensional_deposition_validation.ipynb)
plots local and cumulative heating and independently sweeps sheet samples and
ODE tolerances. [`two_dimensional_deposition_validation.ipynb`](examples/two_dimensional_deposition_validation.ipynb)
compares raw and coherent Cartesian/cylindrical maps, radial profiles, a
quantified error budget, and a three-level coupled ray/sheet/grid sweep.

The single-ray
[`linear_gradient_turning.ipynb`](examples/linear_gradient_turning.ipynb)
checks the analytical turning point and two-sheet caustic split, and plots the
Cartesian electron-density slice, both ray sheets, and the detected caustic.

[`linear_gradient_tolerance_convergence.ipynb`](examples/linear_gradient_tolerance_convergence.ipynb)
then sweeps the dimensionless Diffrax tolerances and plots the caustic-position
error against a tighter numerical reference.

[`tetrahedral_linear_gradient.ipynb`](examples/tetrahedral_linear_gradient.ipynb)
traces a `2 x 2` ray lattice with ten samples per sheet, visualizes all
tetrahedral vertices and edges with highlighted examples, and interpolates
phase and path length onto the Cartesian `z=0` plane.

Ray sheets can be converted to an indexed tetrahedral field with a compact
selection of variables:

```python
from pyGATH.fields import (
    deposit_tetrahedral_power,
    interpolate_tetrahedral_fields_batched,
    replace_tetrahedral_field_values,
    tetrahedralise_sheet_fields,
)

field = tetrahedralise_sheet_fields(
    result.sheet_fields,
    fields=("ray_power", "capped_amplitude"),
)
sampled = interpolate_tetrahedral_fields_batched(field, cartesian_points)
power = sampled.values[..., field.selection.ray_power]

# Reuse the geometry and BVH after changing only field values.
field = replace_tetrahedral_field_values(field, updated_sheet_fields)
```

Interpolation is beam and sheet resolved. Invalid collapsed tetrahedra are
ignored, and values are zero where a point is outside a particular sheet. The
initial BVH topology builder runs on the CPU; repeated point location and
barycentric interpolation use JAX and can run on a GPU.

Inverse-bremsstrahlung heating can instead be conservatively scattered from
the source tetrahedra to cell-centred hydro data:

```python
heating_field = tetrahedralise_sheet_fields(
    result.sheet_fields,
    fields="inverse_brems_deposition",
)
heating = deposit_tetrahedral_power(
    heating_field,
    grid,
    max_subdivision_levels=2,
    relative_tolerance=1e-3,
)
hydro_heating_w_m3 = heating.power_density
```

The default sums both sheets and every beam. Set `beam_index` to one integer
to select a single beam. Adaptive subdivision terminates immediately for a
tetrahedron certified inside one hydro cell and refines uncertain cell-boundary
overlaps. `cell_power`, `outside_power`, and `conservation_error` expose the
corresponding finite-grid power balance.

For example, launch the tetrahedral validation notebook with:

```powershell
uv run jupyter lab examples/tetrahedral_linear_gradient.ipynb
```

[`omega_60_beam_deposition.ipynb`](examples/omega_60_beam_deposition.ipynb)
uses the spherical density and flow fits from Follett et al., *Phys. Plasmas*
**29**, 113902 (2022), the converted OMEGA beam payout, and conservative
inverse-bremsstrahlung deposition. It runs a demonstrated `12 x 12` case with
synchronized stage timings, plots both tetrahedral sheets for beam 1, and
displays Cartesian and radial heating profiles. The resolution constants near
the top of the notebook can be reduced for a quick smoke run or increased for
a convergence study.

[`omega_single_beam_caustic.ipynb`](examples/omega_single_beam_caustic.ipynb)
reduces the same plasma to four near-normal ray tubes. It compares predicted
turning radii with detected sheet splits and plots signed tube area, amplitude,
caustic score, permittivity, momentum, and the resulting tetrahedral sheets.
