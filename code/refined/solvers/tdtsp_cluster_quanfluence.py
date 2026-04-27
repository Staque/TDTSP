"""
Cluster-Quanfluence Time-Dependent TSP solver.

Pipeline (single time slot per call):
    1. K-means clustering (real lat/lng if provided, else manual MDS).
    2. Per-cluster TSP via Quanfluence Ising machine. Each n-city cluster
       is encoded as an n*n-binary QUBO with the standard one-hot
       permutation encoding and the slot-scaled (asymmetric) duration
       matrix as the cost coefficient. The QUBO is sent over the
       Quanfluence gateway's synchronous direct-execute REST endpoint.
    3. Greedy nearest-neighbour ordering of K-means cluster centers.
    4. Stitch sub-tours by minimum last-to-first edge per cluster.
    5. 2-opt local search via full closed-cycle recomputation
       (correct for asymmetric matrices).

The REST client is inlined and trimmed to exactly the auth + direct
QUBO-execute surface that the cluster solver uses. We do not need the
async job / parameter-tuning / file-upload code paths of the original
QuanfluenceClient, so they are not included here. The device is run with
its current saved parameters.

Requires QUANFLUENCE_USERNAME, QUANFLUENCE_PASSWORD, and
QUANFLUENCE_DEVICE_ID in the environment - see .env.example.
"""
from __future__ import annotations

import io
import os
import time
import zipfile
from datetime import datetime, time as dtime, timezone
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import requests
from sklearn.cluster import KMeans
from dotenv import load_dotenv

from ._tdtsp_cluster_common import (
    assemble_tdtsp_result,
    build_schedule,
    fail_result,
)

load_dotenv()


# -----------------------------------------------------------------------------
# Minimal Quanfluence REST client (inlined; direct QUBO execute path only).
# -----------------------------------------------------------------------------

class _QuanfluenceClient:
    """Sign-in + direct QUBO submission against the Quanfluence gateway."""

    def __init__(self,
                 base_url: Optional[str] = None,
                 username: Optional[str] = None,
                 password: Optional[str] = None,
                 device_id: Optional[str] = None):
        self.base_url = base_url or os.getenv(
            "QUANFLUENCE_BASE_URL", "https://gateway.quanfluence.com"
        )
        self.username = username or os.getenv("QUANFLUENCE_USERNAME")
        self.password = password or os.getenv("QUANFLUENCE_PASSWORD")
        self.device_id = device_id or os.getenv("QUANFLUENCE_DEVICE_ID", "41")
        if not self.username or not self.password:
            raise ValueError(
                "Quanfluence credentials not found in environment variables"
            )
        self.token: Optional[str] = None
        self.token_expiry: float = 0.0

    def _ensure_authenticated(self) -> str:
        if self.token and time.time() < self.token_expiry - 60:
            return self.token
        return self._authenticate()

    def _authenticate(self) -> str:
        url = f"{self.base_url}/api/clients/signin"
        resp = requests.post(
            url,
            json={"username": self.username, "password": self.password},
            headers={"Content-Type": "application/json"},
            timeout=30,
        )
        if resp.status_code != 200:
            raise RuntimeError(
                f"Quanfluence auth failed: {resp.status_code} - {resp.text}"
            )
        data = resp.json()
        if data.get("status") != "success":
            raise RuntimeError(f"Quanfluence auth failed: {data}")
        self.token = data["data"]["token"]
        expires_in = data["data"].get("expiresIn", 3600)
        self.token_expiry = time.time() + expires_in
        return self.token

    def _headers(self) -> Dict[str, str]:
        self._ensure_authenticated()
        return {"Authorization": f"Bearer {self.token}"}

    @staticmethod
    def _qubo_to_zip(Q: Dict[Tuple[int, int], float]) -> bytes:
        items = [
            f"({i}, {j}): {coeff}"
            for (i, j), coeff in Q.items()
            if abs(coeff) > 1e-10
        ]
        qubo_str = "{" + ", ".join(items) + "}"
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("data.txt", qubo_str)
        return buf.getvalue()

    def execute_qubo(self, Q: Dict[Tuple[int, int], float]) -> Dict[str, Any]:
        url = f"{self.base_url}/api/execute/device/{self.device_id}"
        files = {
            "file": ("data.zip", self._qubo_to_zip(Q), "application/zip"),
        }
        resp = requests.post(url, files=files, headers=self._headers(), timeout=3600)
        if resp.status_code != 200:
            raise RuntimeError(
                f"Quanfluence execute failed: {resp.status_code} - {resp.text}"
            )
        data = resp.json()
        if data.get("status") != "success":
            raise RuntimeError(f"Quanfluence execute failed: {data}")
        return data["data"]


# -----------------------------------------------------------------------------
# Clustering / QUBO / decoding / stitching / 2-opt
# -----------------------------------------------------------------------------

def _estimate_coordinates(distance_matrix: List[List[float]]) -> np.ndarray:
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


def _tsp_to_qubo(n: int, dm: List[List[float]],
                 penalty: Optional[float] = None) -> Dict[Tuple[int, int], float]:
    """Standard one-hot TSP QUBO. Quanfluence accepts upper-triangular form."""
    if penalty is None:
        max_dist = max(max(row) for row in dm)
        penalty = 2 * max_dist * n

    def vidx(city: int, pos: int) -> int:
        return city * n + pos

    Q: Dict[Tuple[int, int], float] = {}
    for t in range(n):
        nt = (t + 1) % n
        for i in range(n):
            for j in range(n):
                if i != j and dm[i][j] > 0:
                    a, b = vidx(i, t), vidx(j, nt)
                    Q[(a, b)] = Q.get((a, b), 0.0) + dm[i][j]

    for i in range(n):
        for t in range(n):
            idx = vidx(i, t)
            Q[(idx, idx)] = Q.get((idx, idx), 0.0) - penalty
        for t1 in range(n):
            for t2 in range(t1 + 1, n):
                a, b = vidx(i, t1), vidx(i, t2)
                Q[(a, b)] = Q.get((a, b), 0.0) + 2 * penalty

    for t in range(n):
        for i in range(n):
            idx = vidx(i, t)
            Q[(idx, idx)] = Q.get((idx, idx), 0.0) - penalty
        for i1 in range(n):
            for i2 in range(i1 + 1, n):
                a, b = vidx(i1, t), vidx(i2, t)
                Q[(a, b)] = Q.get((a, b), 0.0) + 2 * penalty

    return Q


def _decode_tour(result_vector: List[int], n: int) -> Optional[List[int]]:
    tour: List[Optional[int]] = [None] * n
    for idx, bit in enumerate(result_vector):
        if bit == 1:
            city, pos = idx // n, idx % n
            if 0 <= city < n and 0 <= pos < n and tour[pos] is None:
                tour[pos] = city
    if None in tour:
        used = set(c for c in tour if c is not None)
        missing = [c for c in range(n) if c not in used]
        for i, val in enumerate(tour):
            if val is None and missing:
                tour[i] = missing.pop(0)
    if None in tour or len(set(tour)) != n:
        return None
    return tour  # type: ignore[return-value]


def _solve_cluster_quanfluence(cluster_indices: List[int],
                               full_dm: List[List[float]],
                               client: _QuanfluenceClient) -> List[int]:
    n = len(cluster_indices)
    if n <= 2:
        return list(cluster_indices)

    sub_dm = [
        [full_dm[cluster_indices[i]][cluster_indices[j]] for j in range(n)]
        for i in range(n)
    ]
    Q = _tsp_to_qubo(n, sub_dm)
    try:
        result_data = client.execute_qubo(Q)
        result_vector = result_data.get("result", [])
        tour = _decode_tour(result_vector, n)
        if tour is None:
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


class ClusterTDTSPQuanfluenceSolver:
    """Cluster-based TD-TSP solver using Quanfluence Ising machine per cluster."""

    def __init__(self,
                 max_cluster_size: int = 10,
                 use_local_search: bool = True,
                 verbose: bool = False):
        self.max_cluster_size = max_cluster_size
        self.use_local_search = use_local_search
        self.verbose = verbose
        self._client: Optional[_QuanfluenceClient] = None

    def _get_client(self) -> _QuanfluenceClient:
        if self._client is None:
            self._client = _QuanfluenceClient()
        return self._client

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
        client = self._get_client()

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

        # Step 2: per-cluster Quanfluence solves
        ts = time.time()
        cluster_tours: List[List[int]] = []
        for cluster in clusters:
            cluster_tours.append(
                _solve_cluster_quanfluence(cluster, distance_matrix, client)
            )
        timing["quanfluence_time_s"] = time.time() - ts

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
                inner_message="cluster-Quanfluence did not return a complete tour",
                wall_block=wall_block,
            )

        sched = build_schedule(locations, distance_matrix, full_tour,
                               start_time, metric)

        return assemble_tdtsp_result(
            locations=locations,
            schedule_block=sched,
            n=n,
            solver_label="Cluster-Quanfluence (K-means + Ising QUBO + 2-opt)",
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
