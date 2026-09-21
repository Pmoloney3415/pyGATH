"""Small repeatable benchmark for exact tetrahedral deposition kernels."""

from __future__ import annotations

import argparse
import time

import numpy as np

from pyGATH.fields import (
    build_geodesic_deposition_mesh,
    deposit_simplicial_power_to_mesh,
    simplicialise_sheet_fields,
)
from pyGATH.raytracing import RAY_SHEET_LAYOUT, RAY_STATE_LAYOUT


def build_problem(nrays: int, nsamples: int, radial_layers: int, angular_cells: int):
    first, second, path = np.meshgrid(
        np.linspace(-0.9, 0.9, nrays),
        np.linspace(-0.9, 0.9, nrays),
        np.linspace(-1.1, 1.1, nsamples),
        indexing="ij",
    )
    positions = np.stack((first, second, path), axis=-1)
    fields = np.zeros((1, 1, nrays, nrays, nsamples, RAY_SHEET_LAYOUT.n_attributes))
    fields[0, 0, ..., RAY_STATE_LAYOUT.position] = positions
    fields[0, 0, ..., RAY_SHEET_LAYOUT.inverse_brems_deposition] = (
        4.0 + 0.2 * first - 0.3 * second + 0.1 * path
    )
    source = simplicialise_sheet_fields(
        fields, dimension=3, fields="inverse_brems_deposition"
    )
    target = build_geodesic_deposition_mesh(
        np.linspace(0.0, 1.0, radial_layers + 1),
        maximum_angular_cells=angular_cells,
    )
    return source, target


def timed_deposition(source, target, *, pair_batch_size: int):
    events = []
    started = time.perf_counter()
    deposition = deposit_simplicial_power_to_mesh(
        source,
        target,
        pair_batch_size=pair_batch_size,
        source_batch_size=256,
        progress_interval_s=1.0,
        progress_callback=events.append,
    )
    elapsed = time.perf_counter() - started
    overlap = [event for event in events if event["stage"] == "jax bvh overlap"]
    return deposition, elapsed, overlap[-1] if overlap else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nrays", type=int, default=6)
    parser.add_argument("--nsamples", type=int, default=6)
    parser.add_argument("--radial-layers", type=int, default=6)
    parser.add_argument("--angular-cells", type=int, default=20)
    parser.add_argument("--pair-batch-size", type=int, default=8192)
    parser.add_argument("--warm-runs", type=int, default=1)
    arguments = parser.parse_args()

    source, target = build_problem(
        arguments.nrays,
        arguments.nsamples,
        arguments.radial_layers,
        arguments.angular_cells,
    )
    print(f"source tetrahedra: {source.mesh.nsimplices:,}")
    print(f"target tetrahedra: {target.ncells:,}")
    deposition, cold_seconds, summary = timed_deposition(
        source,
        target,
        pair_batch_size=arguments.pair_batch_size,
    )
    print(f"cold run: {cold_seconds:.3f} s")
    if summary is not None:
        print(summary["message"])
    warm_seconds = []
    for _ in range(arguments.warm_runs):
        warmed, elapsed, _ = timed_deposition(
            source,
            target,
            pair_batch_size=arguments.pair_batch_size,
        )
        np.testing.assert_allclose(
            warmed.cell_power, deposition.cell_power, rtol=2.0e-10, atol=1.0e-12
        )
        warm_seconds.append(elapsed)
    if warm_seconds:
        print(f"warm median: {np.median(warm_seconds):.3f} s")


if __name__ == "__main__":
    main()
