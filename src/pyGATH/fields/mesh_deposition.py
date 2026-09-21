"""Canonical JAX conservative deposition onto simplicial target meshes."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from numbers import Integral
from typing import Any

import numpy as np

from pyGATH.reporting import get_reporter, reported_stage

from .deposition import PowerDeposition
from .deposition_mesh import SimplicialDepositionMesh
from .simplicial import SimplicialField

ProgressCallback = Callable[[dict[str, Any]], None]


@dataclass(frozen=True)
class ResolvedPowerDeposition:
    """Beam- and sheet-resolved power on a simplicial target mesh."""

    cell_power: np.ndarray
    source_power: np.ndarray
    cell_volumes: np.ndarray
    beam_indices: np.ndarray
    sheet_indices: np.ndarray

    @property
    def power_density(self) -> np.ndarray:
        return self.cell_power / self.cell_volumes[None, None, :]

    @property
    def total(self) -> PowerDeposition:
        return self.select()

    def select(
        self,
        *,
        beam_index: int | None = None,
        sheet_index: int | None = None,
    ) -> PowerDeposition:
        """Sum a requested beam/sheet selection into a scalar cell field."""
        beam_mask = np.ones(self.beam_indices.size, dtype=bool)
        sheet_mask = np.ones(self.sheet_indices.size, dtype=bool)
        if beam_index is not None:
            beam_mask = self.beam_indices == beam_index
            if not np.any(beam_mask):
                raise IndexError(f"beam_index {beam_index} was not deposited")
        if sheet_index is not None:
            sheet_mask = self.sheet_indices == sheet_index
            if not np.any(sheet_mask):
                raise IndexError(f"sheet_index {sheet_index} was not deposited")
        selected = self.cell_power[beam_mask][:, sheet_mask].sum(axis=(0, 1))
        source = float(self.source_power[beam_mask][:, sheet_mask].sum())
        deposited = float(np.sum(selected))
        return PowerDeposition(
            power_density=selected / self.cell_volumes,
            cell_power=selected,
            deposited_power=deposited,
            outside_power=source - deposited,
            source_power=source,
        )


@dataclass
class _AabbNode:
    lower: np.ndarray
    upper: np.ndarray
    left: _AabbNode | None = None
    right: _AabbNode | None = None
    indices: np.ndarray | None = None


def _build_aabb_tree(
    lower: np.ndarray,
    upper: np.ndarray,
    indices: np.ndarray | None = None,
    *,
    leaf_size: int = 16,
) -> _AabbNode:
    """Build the host-side topology consumed by JAX BVH traversal kernels."""
    if indices is None:
        indices = np.arange(lower.shape[0], dtype=np.int32)
    node_lower = np.min(lower[indices], axis=0)
    node_upper = np.max(upper[indices], axis=0)
    if indices.size <= leaf_size:
        return _AabbNode(node_lower, node_upper, indices=indices)
    centroids = 0.5 * (lower[indices] + upper[indices])
    split_axis = int(np.argmax(np.ptp(centroids, axis=0)))
    order = np.argsort(centroids[:, split_axis], kind="stable")
    midpoint = order.size // 2
    return _AabbNode(
        node_lower,
        node_upper,
        left=_build_aabb_tree(
            lower, upper, indices[order[:midpoint]], leaf_size=leaf_size
        ),
        right=_build_aabb_tree(
            lower, upper, indices[order[midpoint:]], leaf_size=leaf_size
        ),
    )


def _describe_deposition(result: PowerDeposition | ResolvedPowerDeposition) -> str:
    total = result.total if isinstance(result, ResolvedPowerDeposition) else result
    return (
        f"source {total.source_power:.6e} W, deposited "
        f"{total.deposited_power:.6e} W, outside {total.outside_power:.6e} W"
    )


@reported_stage("deposit simplicial power", describe=_describe_deposition)
def deposit_simplicial_power_to_mesh(
    field: SimplicialField,
    mesh: SimplicialDepositionMesh,
    *,
    field_name: str = "inverse_brems_deposition",
    beam_index: int | None = None,
    sheet_index: int | None = None,
    resolve_beam_sheets: bool = False,
    intersection_tolerance: float = 1.0e-10,
    progress_interval_s: float | None = None,
    progress_callback: ProgressCallback | None = None,
    pair_batch_size: int = 65_536,
    source_batch_size: int = 2048,
    bvh_leaf_size: int = 8,
) -> PowerDeposition | ResolvedPowerDeposition:
    """Exactly deposit an affine simplicial field with the canonical JAX kernels."""
    if not isinstance(field, SimplicialField):
        raise TypeError("field must be a SimplicialField")
    if not isinstance(mesh, SimplicialDepositionMesh):
        raise TypeError("mesh must be a SimplicialDepositionMesh")
    if field.mesh.dimension != mesh.dimension:
        raise ValueError("field and deposition mesh dimensions must match")
    if not isinstance(field_name, str):
        raise TypeError("field_name must be a string")
    if not isinstance(resolve_beam_sheets, bool):
        raise TypeError("resolve_beam_sheets must be a boolean")
    if not np.isfinite(intersection_tolerance) or intersection_tolerance <= 0.0:
        raise ValueError("intersection_tolerance must be finite and positive")
    if progress_interval_s is not None and (
        not np.isfinite(progress_interval_s) or progress_interval_s <= 0.0
    ):
        raise ValueError("progress_interval_s must be finite and positive")
    if progress_callback is not None and not callable(progress_callback):
        raise TypeError("progress_callback must be callable or None")
    for value, name in (
        (pair_batch_size, "pair_batch_size"),
        (source_batch_size, "source_batch_size"),
        (bvh_leaf_size, "bvh_leaf_size"),
    ):
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError(f"{name} must be an integer")
        if value < 1:
            raise ValueError(f"{name} must be positive")

    reporter = get_reporter()
    if reporter is not None and reporter.enabled and reporter.config.verbosity >= 2:
        if progress_interval_s is None:
            progress_interval_s = reporter.config.progress_interval_s
        if progress_callback is None:
            progress_callback = reporter.callback
        else:
            supplied_callback = progress_callback

            def emit_to_both(event):
                supplied_callback(event)
                reporter.callback(event)

            progress_callback = emit_to_both

    from .jax_mesh_deposition import deposit_simplicial_power_to_mesh_jax

    return deposit_simplicial_power_to_mesh_jax(
        field,
        mesh,
        field_name=field_name,
        beam_index=beam_index,
        sheet_index=sheet_index,
        resolve_beam_sheets=resolve_beam_sheets,
        intersection_tolerance=intersection_tolerance,
        progress_interval_s=progress_interval_s,
        progress_callback=progress_callback,
        pair_batch_size=int(pair_batch_size),
        source_batch_size=int(source_batch_size),
        bvh_leaf_size=int(bvh_leaf_size),
    )


__all__ = ["ResolvedPowerDeposition", "deposit_simplicial_power_to_mesh"]
