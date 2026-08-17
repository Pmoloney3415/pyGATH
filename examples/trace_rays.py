"""Load a simulation deck, trace its rays, and construct two ray sheets."""

from __future__ import annotations

import argparse
from pathlib import Path

from pyGATH.io import load_simulation_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="path to a TOML simulation deck")
    return parser.parse_args()


def main() -> None:
    simulation = load_simulation_config(parse_args().config)
    grid = simulation.build_grid()
    initial_rays = simulation.initialize_rays(grid)
    result = simulation.trace_rays(initial_rays, grid)

    print(f"Sheet field shape: {result.sheet_fields.shape}")
    print(f"Global terminal path: {float(result.terminal_path):.8e} m")
    print(f"Terminated because all primaries exited: {bool(result.terminated)}")
    print(f"Rays with a caustic: {int(result.has_caustic.sum())}")


if __name__ == "__main__":
    main()
