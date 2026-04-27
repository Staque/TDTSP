"""
Time-Dependent TSP (TD-TSP) solver - Gurobi backend (autonomous, non-clustered).

Solves a single time-slot TD-TSP instance: given a pre-scaled distance matrix

    D_t[i][j] = D_base[i][j] * m_t

for some chosen time slot t (Morning Peak / Midday / Evening Peak / Night),
this solver finds the optimal Hamiltonian tour using the Miller-Tucker-Zemlin
(MTZ) subtour-elimination formulation in Gurobi.

In addition to the optimal tour and total distance, the solver constructs a
per-stop schedule: starting from a configurable departure clock time (default
08:00 in the instance's local timezone), it walks the tour edges treating
their values as seconds and emits per-stop depart/arrive clock times together
with the tour-level start time, end time, and total tour duration. The result
also carries wall-clock instrumentation (UTC-aware ISO timestamps) for the
optimisation run itself.

Designed for small instances (n <= ~30). For larger instances, see the
cluster-decomposed solver in code/refined/solvers/.

Usage:
    from tdtsp_gurobi import TDTSPGurobiSolver
    solver = TDTSPGurobiSolver()
    result = solver.solve(
        locations=instance["locations"],
        distance_matrix=slot["distance_matrix"],
        time_slot_label=slot["label"],
        multiplier=slot["multiplier"],
        start_time="08:00",
        start_tz="America/New_York",
    )
"""
import time
from datetime import datetime, time as dtime, timezone
from typing import Dict, List, Optional, Union

import gurobipy as gp
from gurobipy import GRB

from _tdtsp_common import build_schedule


class TDTSPGurobiSolver:
    """Departure-time-dependent TSP solver (single time slot, MTZ formulation)."""

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

        model = gp.Model("TDTSP")
        model.setParam("OutputFlag", 1 if verbose else 0)

        x = {(i, j): model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}")
             for i in range(n) for j in range(n) if i != j}
        u = {i: model.addVar(lb=1, ub=n - 1, vtype=GRB.CONTINUOUS, name=f"u_{i}")
             for i in range(n) if i != start_index}

        model.setObjective(
            gp.quicksum(distance_matrix[i][j] * x[i, j]
                        for i in range(n) for j in range(n) if i != j),
            GRB.MINIMIZE,
        )

        for i in range(n):
            model.addConstr(
                gp.quicksum(x[i, j] for j in range(n) if j != i) == 1,
                name=f"out_{i}",
            )
        for j in range(n):
            model.addConstr(
                gp.quicksum(x[i, j] for i in range(n) if i != j) == 1,
                name=f"in_{j}",
            )
        for i in range(n):
            for j in range(n):
                if i != j and i != start_index and j != start_index:
                    model.addConstr(
                        u[j] >= u[i] + 1 - n * (1 - x[i, j]),
                        name=f"mtz_{i}_{j}",
                    )

        model.optimize()
        solve_time = time.perf_counter() - t0
        wall_end = datetime.now(timezone.utc)

        wall_block = {
            "wall_clock_start_iso": wall_start.isoformat(),
            "wall_clock_end_iso": wall_end.isoformat(),
            "solve_time_seconds": round(solve_time, 6),
        }

        if model.status != GRB.OPTIMAL:
            return {
                "status": "failed",
                "model_status": int(model.status),
                "n": n,
                "time_slot": time_slot_label,
                "multiplier": multiplier,
                **wall_block,
            }

        tour = [start_index]
        current = start_index
        for _ in range(n - 1):
            for j in range(n):
                if j != current and x[current, j].X > 0.5:
                    tour.append(j)
                    current = j
                    break
        tour_with_return = tour + [tour[0]]

        sched = build_schedule(
            tour_with_return, locations, distance_matrix,
            start_time=start_time, start_tz=start_tz, start_date=start_date,
            metric=metric,
        )

        return {
            "status": "optimal",
            "n": n,
            "solver": "Gurobi (autonomous, MTZ)",
            "time_slot": time_slot_label,
            "multiplier": multiplier,
            "metric": metric,
            "tour": tour_with_return,
            "tour_names": [locations[i] for i in tour_with_return],
            "route_string": " -> ".join(locations[i] for i in tour_with_return),
            "start_tz": start_tz,
            **sched,
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

    solver = TDTSPGurobiSolver()
    result = solver.solve(
        locations=locations,
        distance_matrix=scaled,
        time_slot_label="Morning Peak (8 AM)",
        multiplier=multiplier,
        start_time="08:00",
        metric="units",
    )
    print("\nTD-TSP Gurobi self-test")
    print("-" * 60)
    print(f"  status            : {result['status']}")
    print(f"  time slot         : {result['time_slot']}")
    print(f"  multiplier        : {result['multiplier']}")
    print(f"  route             : {result['route_string']}")
    print(f"  tour start (local): {result['tour_start_local']}")
    print(f"  tour end   (local): {result['tour_end_local']}")
    print(f"  total tour value  : {result['total_tour_value']} {result['total_tour_unit']}")
    print(f"  solver wall (s)   : {result['solve_time_seconds']}")


if __name__ == "__main__":
    _self_test()
