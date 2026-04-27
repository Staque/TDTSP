"""
Cluster-Gurobi Time-Dependent TSP solver.

Pipeline (single time slot per call):
    1. K-means clustering (real lat/lng if provided, else MDS embedding).
    2. Per-cluster TSP via Gurobi MILP using the Miller-Tucker-Zemlin
       formulation on the slot-scaled (asymmetric) duration matrix.
    3. Greedy nearest-neighbour ordering of cluster centroids computed from
       cluster member coordinates.
    4. Stitch sub-tours by rotating each cluster to its best entry point
       given the previous cluster's exit city.
    5. Asymmetric 2-opt local search using a per-segment delta that accounts
       for direction-dependent edges (correct for real-world driving times).

The backend exposes a single ``solve`` method whose output matches the
canonical TD-TSP result schema produced by every other backend in this
package (per-stop schedule, ISO start/end, multiplier, raw vs. refined
distance, cluster decomposition, and timings).
"""
from __future__ import annotations

import os
import time
from datetime import datetime, time as dtime, timezone
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.cluster import KMeans
from sklearn.manifold import MDS
from dotenv import load_dotenv

from ._tdtsp_cluster_common import (
    assemble_tdtsp_result,
    build_schedule,
    fail_result,
)

load_dotenv()


def _estimate_coordinates(distance_matrix: List[List[float]]) -> np.ndarray:
    n = len(distance_matrix)
    if n <= 2:
        return np.array([[0.0, 0.0], [1.0, 0.0]][:n])
    dm = np.array(distance_matrix, dtype=float)
    dm = (dm + dm.T) / 2
    np.fill_diagonal(dm, 0)
    try:
        mds = MDS(n_components=2, dissimilarity="precomputed",
                  random_state=42, max_iter=300)
        return mds.fit_transform(dm)
    except Exception:
        return np.random.rand(n, 2) * 100


def _kmeans_clusters(coords: np.ndarray,
                     max_cluster_size: int) -> Tuple[List[List[int]], np.ndarray]:
    n = len(coords)
    n_clusters = max(1, int(np.ceil(n / max_cluster_size)))
    if n_clusters >= n:
        return [[i] for i in range(n)], coords.copy()
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(coords)
    clusters = [[] for _ in range(n_clusters)]
    for i, lab in enumerate(labels):
        clusters[lab].append(i)
    clusters = [c for c in clusters if c]
    return clusters, km.cluster_centers_


def _greedy_tsp(indices: List[int], dm: List[List[float]]) -> List[int]:
    if len(indices) <= 1:
        return list(indices)
    tour = [indices[0]]
    remaining = set(indices[1:])
    while remaining:
        cur = tour[-1]
        nxt = min(remaining, key=lambda x: dm[cur][x])
        tour.append(nxt)
        remaining.remove(nxt)
    return tour


def _solve_cluster_mtz(cluster_indices: List[int],
                       full_dm: List[List[float]]) -> List[int]:
    """Solve a single-cluster TSP with Gurobi (MTZ) on the slot-scaled matrix."""
    n = len(cluster_indices)
    if n <= 2:
        return list(cluster_indices)

    sub_dm = [
        [full_dm[cluster_indices[i]][cluster_indices[j]] for j in range(n)]
        for i in range(n)
    ]

    try:
        import gurobipy as gp
        from gurobipy import GRB

        env = gp.Env(empty=True)
        env.setParam("OutputFlag", 0)
        env.start()
        m = gp.Model("tsp_cluster", env=env)
        m.setParam("OutputFlag", 0)
        m.setParam("TimeLimit", 30)

        x = {(i, j): m.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}")
             for i in range(n) for j in range(n) if i != j}
        u = {i: m.addVar(lb=1, ub=n - 1, vtype=GRB.CONTINUOUS, name=f"u_{i}")
             for i in range(1, n)}

        m.setObjective(
            gp.quicksum(sub_dm[i][j] * x[i, j]
                        for i in range(n) for j in range(n) if i != j),
            GRB.MINIMIZE,
        )
        for i in range(n):
            m.addConstr(gp.quicksum(x[i, j] for j in range(n) if i != j) == 1)
            m.addConstr(gp.quicksum(x[j, i] for j in range(n) if i != j) == 1)
        for i in range(1, n):
            for j in range(1, n):
                if i != j:
                    m.addConstr(u[i] - u[j] + (n - 1) * x[i, j] <= n - 2)

        m.optimize()

        if m.status in (GRB.OPTIMAL, GRB.TIME_LIMIT):
            tour = [0]
            visited = {0}
            current = 0
            while len(tour) < n:
                stepped = False
                for j in range(n):
                    if j not in visited and (current, j) in x and x[current, j].X > 0.5:
                        tour.append(j)
                        visited.add(j)
                        current = j
                        stepped = True
                        break
                if not stepped:
                    unv = [j for j in range(n) if j not in visited]
                    if not unv:
                        break
                    nearest = min(unv, key=lambda j: sub_dm[current][j])
                    tour.append(nearest)
                    visited.add(nearest)
                    current = nearest
            m.dispose()
            env.dispose()
            return [cluster_indices[i] for i in tour]

        m.dispose()
        env.dispose()
        return list(cluster_indices)
    except Exception:
        return _greedy_tsp(list(cluster_indices), full_dm)


def _stitch_tours(cluster_tours: List[List[int]],
                  cluster_order: List[int],
                  dm: List[List[float]]) -> List[int]:
    full_tour: List[int] = []
    for cluster_idx in cluster_order:
        tour = cluster_tours[cluster_idx]
        if not tour:
            continue
        if not full_tour:
            full_tour.extend(tour)
            continue
        last = full_tour[-1]
        best_start, best_d = 0, float("inf")
        for i, city in enumerate(tour):
            d = dm[last][city]
            if d < best_d:
                best_d = d
                best_start = i
        rotated = tour[best_start:] + tour[:best_start]
        full_tour.extend(rotated)
    return full_tour


def _closed_distance(tour: List[int], dm: List[List[float]]) -> float:
    if len(tour) <= 1:
        return 0.0
    total = sum(dm[tour[i]][tour[i + 1]] for i in range(len(tour) - 1))
    total += dm[tour[-1]][tour[0]]
    return total


def _two_opt_asymmetric(tour: List[int],
                        dm: List[List[float]]) -> Tuple[List[int], float]:
    """2-opt with explicit per-segment delta (correct for asymmetric matrices)."""
    n = len(tour)
    if n <= 3:
        return tour, _closed_distance(tour, dm)

    best = tour[:]
    improved = True
    while improved:
        improved = False
        for i in range(n - 1):
            for j in range(i + 2, n):
                if j == n - 1 and i == 0:
                    continue
                i1, i2 = best[i], best[i + 1]
                j1 = best[j]
                j2 = best[(j + 1) % n]
                delta = (dm[i1][j1] - dm[i1][i2]
                         + dm[i2][j2] - dm[j1][j2])
                for k in range(i + 1, j):
                    a, b = best[k], best[k + 1]
                    delta += dm[b][a] - dm[a][b]
                if delta < -1e-10:
                    best[i + 1:j + 1] = best[i + 1:j + 1][::-1]
                    improved = True
    return best, _closed_distance(best, dm)


class ClusterTDTSPGurobiSolver:
    """Cluster-based TD-TSP solver using Gurobi MTZ MILP per cluster."""

    def __init__(self,
                 max_cluster_size: int = 15,
                 use_local_search: bool = True,
                 verbose: bool = False):
        self.max_cluster_size = max_cluster_size
        self.use_local_search = use_local_search
        self.verbose = verbose

    def solve(self,
              locations: List[str],
              distance_matrix: List[List[float]],
              time_slot_label: Optional[str] = None,
              multiplier: Optional[float] = None,
              start_time: Union[str, dtime, None] = "08:00",
              start_tz: Optional[str] = None,
              start_date: Optional[str] = None,
              coords: Optional[List[List[float]]] = None,
              metric: str = "driving_duration_seconds") -> Dict:
        n = len(locations)
        if n < 2:
            raise ValueError("TD-TSP requires at least 2 locations")
        if len(distance_matrix) != n or any(len(r) != n for r in distance_matrix):
            raise ValueError(f"Distance matrix must be {n}x{n}")

        wall_start = datetime.now(timezone.utc)
        t0 = time.perf_counter()
        cluster_t0 = time.time()

        timing: Dict[str, float] = {}

        # Step 1: clustering
        ts = time.time()
        if coords is not None:
            coords_arr = np.asarray(coords, dtype=float)
            if coords_arr.shape != (n, 2):
                raise ValueError(
                    f"coords must be shape ({n}, 2); got {coords_arr.shape}"
                )
        else:
            coords_arr = _estimate_coordinates(distance_matrix)
        max_cluster = min(self.max_cluster_size, n)
        clusters, _ = _kmeans_clusters(coords_arr, max_cluster)
        timing["clustering"] = time.time() - ts

        # Step 2: per-cluster MTZ
        ts = time.time()
        cluster_tours: List[List[int]] = []
        for cluster in clusters:
            cluster_tours.append(_solve_cluster_mtz(cluster, distance_matrix))
        timing["gurobi_solving"] = time.time() - ts

        # Step 3: cluster ordering via centroid nearest-neighbour
        ts = time.time()
        if len(clusters) > 1:
            centroids = []
            for tour in cluster_tours:
                if tour:
                    cx = float(np.mean([coords_arr[i][0] for i in tour]))
                    cy = float(np.mean([coords_arr[i][1] for i in tour]))
                    centroids.append((cx, cy))
                else:
                    centroids.append((0.0, 0.0))
            order = [0]
            remaining = set(range(1, len(clusters)))
            while remaining:
                cx, cy = centroids[order[-1]]
                nxt = min(
                    remaining,
                    key=lambda i: (centroids[i][0] - cx) ** 2
                                  + (centroids[i][1] - cy) ** 2,
                )
                order.append(nxt)
                remaining.remove(nxt)
        else:
            order = [0]
        timing["cluster_ordering"] = time.time() - ts

        # Step 4: stitch
        ts = time.time()
        full_tour = _stitch_tours(cluster_tours, order, distance_matrix)
        raw_distance = _closed_distance(full_tour, distance_matrix)
        timing["stitching"] = time.time() - ts

        # Step 5: 2-opt
        ts = time.time()
        if self.use_local_search:
            full_tour, _ = _two_opt_asymmetric(full_tour, distance_matrix)
        timing["local_search"] = time.time() - ts
        timing["total"] = time.time() - cluster_t0

        solve_time = time.perf_counter() - t0
        wall_end = datetime.now(timezone.utc)
        wall_block = {
            "wall_clock_start_iso": wall_start.isoformat(),
            "wall_clock_end_iso": wall_end.isoformat(),
            "solve_time_seconds": round(solve_time, 6),
        }

        if not full_tour or len(full_tour) != n:
            return fail_result(
                n=n,
                time_slot_label=time_slot_label,
                multiplier=multiplier,
                inner_status="error",
                inner_message="cluster-Gurobi did not return a complete tour",
                wall_block=wall_block,
            )

        sched = build_schedule(locations, distance_matrix, full_tour,
                               start_time, metric)

        return assemble_tdtsp_result(
            locations=locations,
            schedule_block=sched,
            n=n,
            solver_label="Cluster-Gurobi (K-means + MTZ + 2-opt)",
            time_slot_label=time_slot_label,
            multiplier=multiplier,
            metric=metric,
            start_tz=start_tz,
            start_date=start_date,
            n_clusters=len(clusters),
            cluster_sizes=[len(c) for c in clusters],
            raw_distance_pre_2opt=raw_distance,
            cluster_timing=timing,
            wall_block=wall_block,
            inner_status="optimal",
        )
