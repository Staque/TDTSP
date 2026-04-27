"""
Canonical time-slot definitions for the Time-Dependent TSP (TD-TSP).

Each slot specifies:
  - name        : machine-readable identifier
  - label       : human-readable label used in tables / plots / paper
  - hour        : representative departure hour (0-23, local time)
  - multiplier  : nominal traffic multiplier from the IEEE Quantum Week 2026 spec

Travel-time scaling rule (per spec):

    D_t[i][j] = D_base[i][j] * m_t

where D_base is a free-flow / off-peak distance matrix (Euclidean for the
synthetic instances, or Google Maps free-flow driving duration for the
real-world instance), and m_t is the multiplier of the chosen slot.

The nominal multipliers can be overridden per-instance with multipliers
derived from real-world data. The TD-TSP instances shipped with this repo
(``data/instances/tdtsp_n{5,10,25,50,100}.json``) carry a per-slot
``computed`` multiplier derived from the Google Maps Distance Matrix API:

    m_t = mean over O-D pairs of [duration_in_traffic_t(i,j) / duration_free(i,j)]

This module is the single source of truth for the slot list; instance
JSONs stamp a copy into ``time_slots`` for full reproducibility, so the
benchmarks read the slot definitions from the instance file rather than
recomputing them.
"""
from typing import Dict, List


TIME_SLOTS: List[Dict] = [
    {"name": "morning_peak", "label": "Morning Peak (8 AM)",  "hour": 8,  "multiplier": 1.5},
    {"name": "midday",       "label": "Midday (12 PM)",       "hour": 12, "multiplier": 1.0},
    {"name": "evening_peak", "label": "Evening Peak (6 PM)",  "hour": 18, "multiplier": 1.6},
    {"name": "night",        "label": "Night (10 PM)",        "hour": 22, "multiplier": 0.8},
]


def get_slot(name: str) -> Dict:
    for slot in TIME_SLOTS:
        if slot["name"] == name:
            return slot
    raise KeyError(f"Unknown TD-TSP time slot: {name!r}")


def scale_matrix(base_matrix: List[List[float]],
                 multiplier: float,
                 decimals: int = 2) -> List[List[float]]:
    """Return D_t = round(D_base * multiplier, decimals)."""
    return [
        [round(d * multiplier, decimals) for d in row]
        for row in base_matrix
    ]
