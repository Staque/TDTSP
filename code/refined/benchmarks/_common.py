"""Shared helpers for the refined cluster-based benchmark scripts."""
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
INSTANCES_DIR = REPO_ROOT / "data" / "instances"
RESULTS_DIR = REPO_ROOT / "results"

REFINED_DIR = REPO_ROOT / "code" / "refined"
if str(REFINED_DIR) not in sys.path:
    sys.path.insert(0, str(REFINED_DIR))

SIZES = [5, 10, 25, 50, 100]


def load_instance(n: int) -> dict:
    path = INSTANCES_DIR / f"tsp_n{n}.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_results(filename: str, results: list) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / filename
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    return path
