"""Backward-compatible three-dimensional simplicial field API."""

from __future__ import annotations

from .simplicial import (
    InterpolatedSimplicialFields,
    SimplicialField,
    SimplicialMesh,
    interpolate_simplicial_fields,
    interpolate_simplicial_fields_batched,
    replace_simplicial_field_values,
    simplicialise_sheet_fields,
)

TetrahedralMesh = SimplicialMesh
TetrahedralField = SimplicialField
InterpolatedTetrahedralFields = InterpolatedSimplicialFields


def tetrahedralise_sheet_fields(
    sheet_fields,
    *,
    fields=None,
    volume_tolerance: float = 1.0e-12,
) -> TetrahedralField:
    """Build the legacy 3-D tetrahedral field representation."""
    return simplicialise_sheet_fields(
        sheet_fields,
        dimension=3,
        fields=fields,
        measure_tolerance=volume_tolerance,
    )


def replace_tetrahedral_field_values(
    field: TetrahedralField, sheet_fields
) -> TetrahedralField:
    return replace_simplicial_field_values(field, sheet_fields)


def interpolate_tetrahedral_fields(
    field: TetrahedralField,
    points,
    *,
    barycentric_tolerance: float = 1.0e-10,
) -> InterpolatedTetrahedralFields:
    return interpolate_simplicial_fields(
        field,
        points,
        barycentric_tolerance=barycentric_tolerance,
    )


def interpolate_tetrahedral_fields_batched(
    field: TetrahedralField,
    points,
    *,
    point_batch_size: int = 8192,
    barycentric_tolerance: float = 1.0e-10,
) -> InterpolatedTetrahedralFields:
    return interpolate_simplicial_fields_batched(
        field,
        points,
        point_batch_size=point_batch_size,
        barycentric_tolerance=barycentric_tolerance,
    )


__all__ = [
    "InterpolatedTetrahedralFields",
    "TetrahedralField",
    "TetrahedralMesh",
    "interpolate_tetrahedral_fields",
    "interpolate_tetrahedral_fields_batched",
    "replace_tetrahedral_field_values",
    "tetrahedralise_sheet_fields",
]
