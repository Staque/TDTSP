"""
Cluster-D-Wave Time-Dependent TSP solver.

Pipeline (single time slot per call):
    1. K-means clustering (real lat/lng if provided, else manual MDS).
    2. Per-cluster TSP via D-Wave Leap Hybrid CQM. The CQM uses
       binary x[i,t] = 1 if city i is at position t and encodes the
       slot-scaled (asymmetric) duration matrix directly in the
       quadratic objective; permutation feasibility is enforced as
       linear equalities.
    3. Greedy nearest-neighbour ordering of K-means cluster centers.
    4. Stitch sub-tours by rotating each cluster to its best entry city
       given the previous cluster's last city.
    5. 2-opt local search via full closed-cycle recomputation
       (correct for asymmetric matrices).

Requires DWAVE_API_TOKEN in the environment (see .env.example).
"""
from __future__ import annotations

import os
import sys
import time
from datetime import datetime, time as dtime, timezone
from io import StringIO
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.cluster import KMeans
from dotenv import load_dotenv

from ._tdtsp_cluster_common import (
    assemble_tdtsp_result,
    build_schedule,
    fail_result,
)

load_dotenv()


def _estimate_coordinates(distance_matrix: List[List[float]]) -> np.ndarray:
    """Classical (eigen-decomposition) MDS used by the original solver."""
    n = len(distance_matrix)
    D = np.array(distance_matrix, dtype=float)
    D_sq = D ** 2
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ D_sq @ J
    eigvals, eigvecs = np.linalg.eigh(B)
    idx = np.argsort(eigvals)[::-1][:2]
    eigvals = np.maximum(eigvals[idx], 0)
    eigvecs = eigvecs[:, idx]
    return eigvecs @ np.diag(np.sqrt(eigvals))


def _kmeans_clusters(coords: np.ndarray,
                     max_cluster_size: int) -> Tuple[List[List[int]], np.ndarray]:
    n = len(coords)
    n_clusters = max(1, (n + max_cluster_size - 1) // max_cluster_size)
    if n_clusters <= 1:
        return [list(range(n))], coords.mean(axis=0, keepdims=True)
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(coords)
    clusters = [[] for _ in range(n_clusters)]
    for i, lab in enumerate(labels):
        clusters[lab].append(i)
    clusters = [c for c in clusters if c]
    return clusters, km.cluster_centers_


def _dwave_solve_cluster(cluster_indices: List[int],
                         full_dm: List[List[float]],
                         time_limit: float,
                         api_token: str,
                         _sampler_cache: Dict) -> List[int]:
    """Solve a single-cluster TSP on D-Wave Leap Hybrid CQM."""
    n = len(cluster_indices)
    if n <= 2:
        return list(cluster_indices)

    sub_dm = [
        [full_dm[cluster_indices[i]][cluster_indices[j]] for j in range(n)]
        for i in range(n)
    ]

    try:
        import dimod
        from dwave.system import LeapHybridCQMSampler

        cqm = dimod.ConstrainedQuadraticModel()
        x = [[dimod.Binary(f"x_{i}_{t}") for t in range(n)] for i in range(n)]

        objective = 0
        for t in range(n):
            nt = (t + 1) % n
            for i in range(n):
                for j in range(n):
                    if i != j:
                        objective += sub_dm[i][j] * x[i][t] * x[j][nt]
        cqm.set_objective(objective)

        for i in range(n):
            cqm.add_constraint(sum(x[i][t] for t in range(n)) == 1,
                               label=f"visit_city_{i}")
        for t in range(n):
            cqm.add_constraint(sum(x[i][t] for i in range(n)) == 1,
                               label=f"position_{t}")

        sampler = _sampler_cache.get("sampler")
        if sampler is None:
            sampler = LeapHybridCQMSampler(token=api_token)
            _sampler_cache["sampler"] = sampler

        # Suppress D-Wave SDK chatter; the cluster benchmarks own all logging.
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        try:
            sampleset = sampler.sample_cqm(cqm, time_limit=time_limit)
        finally:
            sys.stdout = old_stdout

        feasible = sampleset.filter(lambda s: s.is_feasible)
        if len(feasible) == 0:
            sample = sampleset.first.sample
        else:
            sample = feasible.first.sample

        tour = [None] * n
        for key, value in sample.items():
            if value == 1 and key.startswith("x_"):
                _, ci, ti = key.split("_")
                ci, ti = int(ci), int(ti)
                if 0 <= ci < n and 0 <= ti < n:
                    tour[ti] = ci
        if None in tour or len(set(tour)) != n:
            return list(cluster_indices)
        return [cluster_indices[i] for i in tour]
    except Exception:
        return list(cluster_indices)


def _stitch_tours(cluster_tours: List[List[int]],
                  cluster_order: List[int],
                  dm: List[List[float]]) -> List[int]:
    if len(cluster_order) == 1:
        return list(cluster_tours[cluster_order[0]])
    full_tour = list(cluster_tours[cluster_order[0]])
    for idx in range(1, len(cluster_order)):
        nxt_tour = cluster_tours[cluster_order[idx]]
        last = full_tour[-1]
        best_d, best_entry = float("inf"), 0
        for j, city in enumerate(nxt_tour):
            d = dm[last][city]
            if d < best_d:
                best_d = d
                best_entry = j
        rotated = nxt_tour[best_entry:] + nxt_tour[:best_entry]
        full_tour.extend(rotated)
    return full_tour


def _closed_distance(tour: List[int], dm: List[List[float]]) -> float:
    n = len(tour)
    return sum(dm[tour[i]][tour[(i + 1) % n]] for i in range(n))


def _two_opt_full_recompute(tour: List[int],
                            dm: List[List[float]]) -> Tuple[List[int], float]:
    """Slow but asymmetric-correct 2-opt with break-on-improvement, capped
    at 100 outer passes - matches the original D-Wave/Quanfluence cluster
    behaviour bit-for-bit."""
    n = len(tour)
    cur = tour[:]
    cur_d = _closed_distance(cur, dm)
    improved = True
    iters = 0
    while improved and iters < 100:
        improved = False
        iters += 1
        for i in range(n - 1):
            stop = False
            for k in range(i + 1, n):
                new_tour = cur[:i] + cur[i:k + 1][::-1] + cur[k + 1:]
                new_d = _closed_distance(new_tour, dm)
                if new_d < cur_d - 1e-10:
                    cur, cur_d = new_tour, new_d
                    improved = True
                    stop = True
                    break
            if stop:
                break
    return cur, cur_d


class ClusterTDTSPDWaveSolver:
    """Cluster-based TD-TSP solver using D-Wave Leap Hybrid CQM per cluster."""

    def __init__(self,
                 max_cluster_size: int = 15,
                 use_local_search: bool = True,
                 time_limit_per_cluster: int = 10,
                 verbose: bool = False):
        self.max_cluster_size = max_cluster_size
        self.use_local_search = use_local_search
        self.time_limit_per_cluster = time_limit_per_cluster
        self.verbose = verbose
        self._sampler_cache: Dict = {}

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

        api_token = os.getenv("DWAVE_API_TOKEN")
        if not api_token:
            raise ValueError("DWAVE_API_TOKEN not found in environment variables")

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
        clusters, centers = _kmeans_clusters(coords_arr, max_cluster)
        timing["cluster_time_s"] = time.time() - ts

        # Step 2: per-cluster CQM
        ts = time.time()
        cluster_tours: List[List[int]] = []
        for cluster in clusters:
            cluster_tours.append(
                _dwave_solve_cluster(
                    cluster, distance_matrix,
                    self.time_limit_per_cluster,
                    api_token, self._sampler_cache,
                )
            )
        timing["dwave_time_s"] = time.time() - ts

        # Step 3: cluster ordering via centroid nearest-neighbour (kmeans centers)
        ts = time.time()
        if len(clusters) > 1:
            order = [0]
            remaining = set(range(1, len(clusters)))
            while remaining:
                cur_center = centers[order[-1]]
                nxt = min(remaining,
                          key=lambda i: float(np.linalg.norm(cur_center - centers[i])))
                order.append(nxt)
                remaining.remove(nxt)
        else:
            order = [0]
        timing["order_time_s"] = time.time() - ts

        # Step 4: stitch
        ts = time.time()
        full_tour = _stitch_tours(cluster_tours, order, distance_matrix)
        raw_distance = _closed_distance(full_tour, distance_matrix)
        timing["stitch_time_s"] = time.time() - ts

        # Step 5: 2-opt
        ts = time.time()
        if self.use_local_search:
            full_tour, _ = _two_opt_full_recompute(full_tour, distance_matrix)
        timing["refine_time_s"] = time.time() - ts
        timing["total_time_s"] = time.time() - cluster_t0

        solve_time = time.perf_counter() - t0
        wall_end = datetime.now(timezone.utc)
        wall_block = {
            "wall_clock_start_iso": wall_start.isoformat(),
            "wall_clock_end_iso": wall_end.isoformat(),
            "solve_time_seconds": round(solve_time, 6),
        }

        if not full_tour or len(full_tour) != n:
            return fail_result(
                n=n, time_slot_label=time_slot_label, multiplier=multiplier,
                inner_status="error",
                inner_message="cluster-D-Wave did not return a complete tour",
                wall_block=wall_block,
            )

        sched = build_schedule(locations, distance_matrix, full_tour,
                               start_time, metric)

        return assemble_tdtsp_result(
            locations=locations,
            schedule_block=sched,
            n=n,
            solver_label="Cluster-D-Wave (K-means + Hybrid CQM + 2-opt)",
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
            extra={"time_limit_per_cluster": self.time_limit_per_cluster},
        )
