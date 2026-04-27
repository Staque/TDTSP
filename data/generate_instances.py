"""
Deterministic TSP benchmark instance generator.

Produces the five Euclidean instances (n = 5, 10, 25, 50, 100) used in the
IEEE Quantum Week 2026 paper. Each instance is written to data/instances/
as a JSON file containing the random seed, city labels, 2D coordinates, and
the rounded NxN distance matrix. Re-running the script reproduces the same
instances bit-for-bit.

Usage:
    python data/generate_instances.py
"""
import json
import os
from pathlib import Path

import numpy as np

SIZES = [5, 10, 25, 50, 100]
SEED = 42
COORD_SCALE = 100.0
DECIMALS = 2

THIS_DIR = Path(__file__).resolve().parent
OUT_DIR = THIS_DIR / "instances"


def generate_instance(n: int, seed: int = SEED) -> dict:
    """Generate a single Euclidean TSP instance with `n` cities."""
    rng_state = np.random.RandomState(seed)
    coords = rng_state.rand(n, 2) * COORD_SCALE

    locations = [f"C{i + 1}" for i in range(n)]

    distance_matrix = []
    for i in range(n):
        row = []
        for j in range(n):
            if i == j:
                row.append(0.0)
            else:
                dx = coords[i][0] - coords[j][0]
                dy = coords[i][1] - coords[j][1]
                row.append(round(float(np.sqrt(dx * dx + dy * dy)), DECIMALS))
        distance_matrix.append(row)

    return {
        "n": n,
        "seed": seed,
        "coord_scale": COORD_SCALE,
        "locations": locations,
        "coords": coords.tolist(),
        "distance_matrix": distance_matrix,
    }


def write_instance(instance: dict, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"tsp_n{instance['n']}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(instance, f, indent=2)
    return path


def load_instance(n: int, instances_dir: Path = OUT_DIR) -> dict:
    """Load a previously generated instance by city count."""
    path = instances_dir / f"tsp_n{n}.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    print(f"Generating TSP instances (seed={SEED}) into {OUT_DIR}")
    for n in SIZES:
        inst = generate_instance(n)
        path = write_instance(inst, OUT_DIR)
        print(f"  n={n:>3}  -> {path.relative_to(THIS_DIR.parent)}")
    print("Done.")


if __name__ == "__main__":
    main()
