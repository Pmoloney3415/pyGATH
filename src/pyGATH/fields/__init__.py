"""Simplicial field construction, interpolation, and conservative deposition."""

from .deposition import PowerDeposition
from .deposition_mesh import (
    SimplicialDepositionMesh,
    build_cartesian_deposition_mesh_from_grid,
    build_circular_deposition_mesh,
    build_circular_deposition_mesh_from_grid,
    build_geodesic_deposition_mesh,
    build_geodesic_deposition_mesh_from_grid,
    build_linear_deposition_mesh,
    build_linear_deposition_mesh_from_grid,
)
from .fieldlayout import FIELD_LAYOUT, FieldLayout, FieldSelection
from .mesh_deposition import (
    ResolvedPowerDeposition,
    deposit_simplicial_power_to_mesh,
)
from .simplicial import (
    InterpolatedSimplicialFields,
    SimplicialField,
    SimplicialMesh,
    interpolate_simplicial_fields,
    interpolate_simplicial_fields_batched,
    interpolate_simplicial_fields_to_cells,
    replace_simplicial_field_values,
    simplicialise_sheet_fields,
)

__all__ = [
    "FIELD_LAYOUT",
    "FieldLayout",
    "FieldSelection",
    "InterpolatedSimplicialFields",
    "PowerDeposition",
    "ResolvedPowerDeposition",
    "SimplicialDepositionMesh",
    "SimplicialField",
    "SimplicialMesh",
    "build_cartesian_deposition_mesh_from_grid",
    "build_circular_deposition_mesh",
    "build_circular_deposition_mesh_from_grid",
    "build_geodesic_deposition_mesh",
    "build_geodesic_deposition_mesh_from_grid",
    "build_linear_deposition_mesh",
    "build_linear_deposition_mesh_from_grid",
    "deposit_simplicial_power_to_mesh",
    "interpolate_simplicial_fields",
    "interpolate_simplicial_fields_batched",
    "interpolate_simplicial_fields_to_cells",
    "replace_simplicial_field_values",
    "simplicialise_sheet_fields",
]
