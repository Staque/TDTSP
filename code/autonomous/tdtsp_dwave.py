"""
Time-Dependent TSP (TD-TSP) solver - D-Wave backend (autonomous, non-clustered).

Solves a single time-slot TD-TSP instance: given a pre-scaled distance matrix

    D_t[i][j] = D_base[i][j] * m_t

for some chosen time slot t (Morning Peak / Midday / Evening Peak / Night),
this solver finds the optimal Hamiltonian tour using a one-hot Constrained
Quadratic Model on the D-Wave LeapHybridCQMSampler.

Binary variables x[i, t] = 1 if city i sits at position t in the tour.
The CQM has two equality constraints (each city visited once, each position
filled once) and an objective summing d[i][j] * x[i, t] * x[j, (t+1) mod n].

In addition to the optimal tour and total distance, the solver constructs a
per-stop schedule: starting from a configurable departure clock time
(default 08:00 in the instance's local timezone), it walks the tour edges
treating their values as seconds and emits per-stop depart/arrive clock
times together with the tour-level start time, end time, and total tour
duration. The result also carries wall-clock instrumentation (UTC-aware
ISO timestamps) for the optimisation run itself.

Designed for small instances (n <= ~30). For larger instances, see the
cluster-decomposed solver in code/refined/solvers/.

Usage:
    from tdtsp_dwave import TDTSPDWaveSolver
    solver = TDTSPDWaveSolver()
    result = solver.solve(
        locations=instance["locations"],
        distance_matrix=slot["distance_matrix"],
        time_slot_label=slot["label"],
        multiplier=slot["multiplier"],
        start_time="08:00",
        start_tz="America/New_York",
    )
"""
import os
import time
from datetime import datetime, time as dtime, timezone
from typing import Dict, List, Optional, Union

from dotenv import load_dotenv

from _tdtsp_common import build_schedule

load_dotenv()


class TDTSPDWaveSolver:
    """Departure-time-dependent TSP solver (single time slot, hybrid CQM)."""

    def __init__(self, api_token: Optional[str] = None):
        self.api_token = api_token or os.getenv("DWAVE_API_TOKEN")
        if not self.api_token:
            raise ValueError("DWAVE_API_TOKEN not found in environment variables")

    def solve(self,
              locations: List[str],
              distance_matrix: List[List[float]],
              time_slot_label: Optional[str] = None,
              multiplier: Optional[float] = None,
              start_time: Union[str, dtime, None] = "08:00",
              start_tz: Optional[str] = None,
              start_date: Optional[str] = None,
              metric: str = "driving_duration_seconds",
              start_index: int = 0,
              time_limit: float = 60.0,
              verbose: bool = False) -> Dict:
        n = len(locations)
        if n < 2:
            raise ValueError("TD-TSP requires at least 2 locations")
        if len(distance_matrix) != n or any(len(r) != n for r in distance_matrix):
            raise ValueError(f"Distance matrix must be {n}x{n}")
        if not (0 <= start_index < n):
            raise ValueError(f"start_index out of range: {start_index}")

        wall_start = datetime.now(timezone.utc)
        t0 = time.perf_counter()

        import dimod
        from dwave.system import LeapHybridCQMSampler

        cqm = dimod.ConstrainedQuadraticModel()
        x = [[dimod.Binary(f"x_{i}_{t}") for t in range(n)] for i in range(n)]

        objective = 0
        for t in range(n):
            next_t = (t + 1) % n
            for i in range(n):
                for j in range(n):
                    if i != j:
                        objective += distance_matrix[i][j] * x[i][t] * x[j][next_t]
        cqm.set_objective(objective)

        for i in range(n):
            cqm.add_constraint(
                sum(x[i][t] for t in range(n)) == 1,
                label=f"visit_city_{i}",
            )
        for t in range(n):
            cqm.add_constraint(
                sum(x[i][t] for i in range(n)) == 1,
                label=f"position_{t}",
            )

        sampler = LeapHybridCQMSampler(token=self.api_token)
        sampleset = sampler.sample_cqm(cqm, time_limit=time_limit)
        feasible = sampleset.filter(lambda s: s.is_feasible)

        solve_time = time.perf_counter() - t0
        wall_end = datetime.now(timezone.utc)
        wall_block = {
            "wall_clock_start_iso": wall_start.isoformat(),
            "wall_clock_end_iso": wall_end.isoformat(),
            "solve_time_seconds": round(solve_time, 6),
        }

        if len(feasible) == 0:
            return {
                "status": "failed",
                "feasible_samples": 0,
                "n": n,
                "time_slot": time_slot_label,
                "multiplier": multiplier,
                **wall_block,
            }

        sample = feasible.first.sample

        position_to_city: Dict[int, int] = {}
        for key, val in sample.items():
            if val == 1 and key.startswith("x_"):
                _, ci, ti = key.split("_")
                position_to_city[int(ti)] = int(ci)
        if len(position_to_city) != n:
            return {
                "status": "failed",
                "reason": "decoded sample has missing positions",
                "n": n,
                "time_slot": time_slot_label,
                "multiplier": multiplier,
                **wall_block,
            }
        tour = [position_to_city[t] for t in range(n)]

        if start_index in tour:
            shift = tour.index(start_index)
            tour = tour[shift:] + tour[:shift]
        tour_with_return = tour + [tour[0]]

        sched = build_schedule(
            tour_with_return, locations, distance_matrix,
            start_time=start_time, start_tz=start_tz, start_date=start_date,
            metric=metric,
        )
        dwave_timing = sampleset.info.get("timing", {}) if hasattr(sampleset, "info") else {}

        return {
            "status": "optimal",
            "n": n,
            "solver": "D-Wave Hybrid CQM (autonomous)",
            "time_slot": time_slot_label,
            "multiplier": multiplier,
            "metric": metric,
            "tour": tour_with_return,
            "tour_names": [locations[i] for i in tour_with_return],
            "route_string": " -> ".join(locations[i] for i in tour_with_return),
            "start_tz": start_tz,
            **sched,
            "feasible_samples": int(len(feasible)),
            "total_samples": int(len(sampleset)),
            "qpu_access_time_us": int(dwave_timing.get("qpu_access_time", 0)),
            **wall_block,
        }


def _self_test():
    """Sanity-check the solver on a 5-city symmetric instance at multiplier 1.5."""
    locations = ["A", "B", "C", "D", "E"]
    base = [
        [0,   2,   9,   10, 7],
        [2,   0,   6,    4, 3],
        [9,   6,   0,    8, 5],
        [10,  4,   8,    0, 6],
        [7,   3,   5,    6, 0],
    ]
    multiplier = 1.5
    scaled = [[round(v * multiplier, 4) for v in row] for row in base]

    solver = TDTSPDWaveSolver()
    result = solver.solve(
        locations=locations,
        distance_matrix=scaled,
        time_slot_label="Morning Peak (8 AM)",
        multiplier=multiplier,
        start_time="08:00",
        metric="units",
        time_limit=10,
    )
    print("\nTD-TSP D-Wave self-test")
    print("-" * 60)
    print(f"  status            : {result['status']}")
    print(f"  time slot         : {result['time_slot']}")
    print(f"  multiplier        : {result['multiplier']}")
    print(f"  route             : {result.get('route_string', '-')}")
    print(f"  tour start (local): {result.get('tour_start_local', '-')}")
    print(f"  tour end   (local): {result.get('tour_end_local', '-')}")
    print(f"  total tour value  : {result.get('total_tour_value', '-')} "
          f"{result.get('total_tour_unit', '')}")
    print(f"  solver wall (s)   : {result['solve_time_seconds']}")


if __name__ == "__main__":
    _self_test()
