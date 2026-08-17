"""Load a pyGATH simulation deck and construct its grid, beams, and rays."""

from __future__ import annotations

import argparse
from pathlib import Path

from pyGATH.io import load_simulation_config
from pyGATH.raytracing import RAY_STATE_LAYOUT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="path to a TOML simulation deck")
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    simulation = load_simulation_config(arguments.config)
    grid = simulation.build_grid()
    beams = simulation.load_beams()
    rays = simulation.initialize_rays(grid, beams=beams)

    print(f"Loaded: {simulation.source}")
    print(f"Geometry: {grid.geom.value}")
    print(f"Cells: {grid.ncells}")
    print(f"Uniform axes: {grid.is_uniform}")
    print(f"Actual extents: {grid.extents.tolist()}")
    print(f"Beams: {beams.nbeams}")
    print(f"Ray state shape: {rays.state.shape}")
    total_power = rays.state[..., RAY_STATE_LAYOUT.ray_power].sum()
    print(f"Total normalized ray power: {total_power:.16f}")
    print(f"Rays intersecting grid: {rays.will_hit_grid.sum()}")


if __name__ == "__main__":
    main()
