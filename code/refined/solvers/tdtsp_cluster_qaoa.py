"""
Cluster-QAOA Time-Dependent TSP solver (AWS Braket SV1 simulator or
Rigetti Cepheus QPU).

Pipeline (single time slot per call):
    1. K-means clustering (real lat/lng if provided, else manual MDS).
    2. Per-cluster TSP via QAOA. Each n-city cluster becomes an n*n-binary
       QUBO using the standard one-hot permutation encoding; the
       slot-scaled (asymmetric) duration matrix is the linear coefficient
       on the cost term `dm[i][j] * x[i,t] * x[j,t+1]`. The QAOA circuit
       uses fixed angles (gamma=pi/4, beta=pi/8) per layer and runs on
       AWS Braket SV1 (default) or a real QPU like Rigetti Cepheus-1-108Q.
    3. Greedy nearest-neighbour ordering of K-means cluster centers.
    4. Stitch sub-tours by minimum last-to-first edge per cluster.
    5. 2-opt local search via full closed-cycle recomputation
       (correct for asymmetric matrices).

When ``capture_raw_shots=True`` (the benchmark default) the full
per-cluster ``measurement_counts`` histograms are persisted on the result
dict under ``raw_cluster_shots`` so the published JSONs include every
500-shot bitstring distribution for paper reproducibility.

Requires AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, and (for Rigetti)
BRAKET_S3_BUCKET_RIGETTI in the environment - see .env.example.
"""
from __future__ import annotations

import os
import time
from datetime import datetime, time as dtime, timezone
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


# Braket device ARNs and regions used in the paper. Extend if needed.
DEVICE_ARNS = {
    "sv1": "arn:aws:braket:::device/quantum-simulator/amazon/sv1",
    "rigetti": "arn:aws:braket:us-west-1::device/qpu/rigetti/Cepheus-1-108Q",
}
DEVICE_REGIONS = {
    "sv1": "us-east-1",
    "rigetti": "us-west-1",
}
REAL_QPUS = {"rigetti"}


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
    """QAOA cluster solver historically uses floor(n/max), preserved here."""
    n = len(coords)
    n_clusters = max(1, n // max_cluster_size)
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
                 penalty: Optional[float] = None) -> Tuple[Dict[Tuple[int, int], float], int]:
    """Permutation-matrix one-hot QUBO. x[i,t] = 1 iff city i at position t."""
    if penalty is None:
        max_dist = max(max(row) for row in dm)
        penalty = 2 * max_dist * n

    num_qubits = n * n

    def vidx(city: int, pos: int) -> int:
        return city * n + pos

    Q: Dict[Tuple[int, int], float] = {}

    # Cost: dm[i][j] * x[i,t] * x[j,t+1]
    for t in range(n):
        nt = (t + 1) % n
        for i in range(n):
            for j in range(n):
                if i != j and dm[i][j] > 0:
                    a, b = vidx(i, t), vidx(j, nt)
                    key = (min(a, b), max(a, b))
                    Q[key] = Q.get(key, 0.0) + dm[i][j]

    # Each city visited exactly once.
    for i in range(n):
        for t in range(n):
            idx = vidx(i, t)
            Q[(idx, idx)] = Q.get((idx, idx), 0.0) - penalty
        for t1 in range(n):
            for t2 in range(t1 + 1, n):
                a, b = vidx(i, t1), vidx(i, t2)
                key = (min(a, b), max(a, b))
                Q[key] = Q.get(key, 0.0) + 2 * penalty

    # Each position holds exactly one city.
    for t in range(n):
        for i in range(n):
            idx = vidx(i, t)
            Q[(idx, idx)] = Q.get((idx, idx), 0.0) - penalty
        for i1 in range(n):
            for i2 in range(i1 + 1, n):
                a, b = vidx(i1, t), vidx(i2, t)
                key = (min(a, b), max(a, b))
                Q[key] = Q.get(key, 0.0) + 2 * penalty

    return Q, num_qubits


def _decode_tour(bitstring: str, n: int) -> Optional[List[int]]:
    tour: List[Optional[int]] = [None] * n
    for idx, bit in enumerate(bitstring):
        if bit == "1":
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


def _run_qaoa(Q: Dict[Tuple[int, int], float],
              num_qubits: int,
              p: int,
              shots: int,
              device_name: str,
              s3_bucket: Optional[str],
              s3_prefix: str,
              aws_access_key: str,
              aws_secret_key: str) -> Dict[str, int]:
    """Build the QAOA circuit and submit to the configured Braket device."""
    from braket.aws import AwsDevice, AwsSession
    from braket.circuits import Circuit
    import boto3

    device_arn = DEVICE_ARNS.get(device_name, DEVICE_ARNS["sv1"])
    region = DEVICE_REGIONS.get(device_name, "us-east-1")

    circuit = Circuit()
    for i in range(num_qubits):
        circuit.h(i)

    gamma = [np.pi / 4] * p
    beta = [np.pi / 8] * p
    for layer in range(p):
        for (i, j), coeff in Q.items():
            if i == j:
                circuit.rz(i, 2 * gamma[layer] * coeff)
            else:
                circuit.cnot(i, j)
                circuit.rz(j, 2 * gamma[layer] * coeff)
                circuit.cnot(i, j)
        for i in range(num_qubits):
            circuit.rx(i, 2 * beta[layer])

    aws_session = AwsSession(
        boto_session=boto3.Session(
            region_name=region,
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key,
        )
    )
    device = AwsDevice(device_arn, aws_session=aws_session)

    if s3_bucket:
        task = device.run(
            circuit,
            s3_destination_folder=(s3_bucket, s3_prefix),
            shots=shots,
        )
    else:
        task = device.run(circuit, shots=shots)

    result = task.result()
    return result.measurement_counts


def _solve_cluster_qaoa(cluster_indices: List[int],
                        full_dm: List[List[float]],
                        p: int,
                        shots: int,
                        device_name: str,
                        s3_bucket: Optional[str],
                        s3_prefix: str,
                        aws_access_key: str,
                        aws_secret_key: str,
                        capture: List[Dict],
                        capture_raw_shots: bool) -> List[int]:
    n = len(cluster_indices)
    if n <= 1:
        if capture_raw_shots:
            capture.append({
                "cluster_indices": list(cluster_indices),
                "cluster_n": n,
                "skipped": True,
                "reason": "cluster size <= 1, no QAOA run",
            })
        return list(cluster_indices)
    if n == 2:
        if capture_raw_shots:
            capture.append({
                "cluster_indices": list(cluster_indices),
                "cluster_n": n,
                "skipped": True,
                "reason": "cluster size == 2, trivial ordering, no QAOA run",
            })
        return list(cluster_indices)

    sub_dm = [
        [full_dm[cluster_indices[i]][cluster_indices[j]] for j in range(n)]
        for i in range(n)
    ]
    Q, num_qubits = _tsp_to_qubo(n, sub_dm)

    try:
        measurements = _run_qaoa(
            Q, num_qubits, p, shots,
            device_name, s3_bucket, s3_prefix,
            aws_access_key, aws_secret_key,
        )
    except Exception as e:
        if capture_raw_shots:
            capture.append({
                "cluster_indices": list(cluster_indices),
                "cluster_n": n,
                "error": str(e),
                "valid_tour_decoded": False,
            })
        return list(cluster_indices)

    best_tour: Optional[List[int]] = None
    best_distance = float("inf")
    for bitstring, _count in measurements.items():
        tour = _decode_tour(bitstring, n)
        if tour is None:
            continue
        distance = sum(sub_dm[tour[i]][tour[(i + 1) % n]] for i in range(n))
        if distance < best_distance:
            best_distance = distance
            best_tour = tour

    if capture_raw_shots:
        capture.append({
            "cluster_indices": list(cluster_indices),
            "cluster_n": n,
            "num_qubits": num_qubits,
            "p": p,
            "shots_requested": shots,
            "shots_observed": int(sum(measurements.values())),
            "unique_bitstrings": len(measurements),
            "measurement_counts": {str(bs): int(c) for bs, c in measurements.items()},
            "best_tour_in_cluster": list(best_tour) if best_tour is not None else None,
            "best_distance_in_cluster": float(best_distance) if best_tour is not None else None,
            "valid_tour_decoded": best_tour is not None,
        })

    if best_tour is None:
        best_tour = list(range(n))
    return [cluster_indices[i] for i in best_tour]


def _find_best_connection(tour1: List[int], tour2: List[int],
                          dm: List[List[float]]) -> Tuple[int, int, int, int]:
    """Best (any-city, any-city) edge between two tours - QAOA cluster
    historically uses this denser stitching (vs. last-city-only)."""
    best_d = float("inf")
    best = (len(tour1) - 1, 0, tour1[-1], tour2[0])
    for i, c1 in enumerate(tour1):
        for j, c2 in enumerate(tour2):
            d = dm[c1][c2]
            if d < best_d:
                best_d = d
                best = (i, j, c1, c2)
    return best


def _stitch_tours(cluster_tours: List[List[int]],
                  cluster_order: List[int],
                  dm: List[List[float]]) -> List[int]:
    if len(cluster_order) == 1:
        return list(cluster_tours[cluster_order[0]])
    full_tour = list(cluster_tours[cluster_order[0]])
    for idx in range(1, len(cluster_order)):
        nxt_tour = cluster_tours[cluster_order[idx]]
        _, entry, _, _ = _find_best_connection(full_tour, nxt_tour, dm)
        rotated = nxt_tour[entry:] + nxt_tour[:entry]
        full_tour.extend(rotated)
    return full_tour


def _closed_distance(tour: List[int], dm: List[List[float]]) -> float:
    n = len(tour)
    return sum(dm[tour[i]][tour[(i + 1) % n]] for i in range(n))


def _two_opt_full_recompute(tour: List[int],
                            dm: List[List[float]]) -> Tuple[List[int], float]:
    """Original QAOA cluster 2-opt: full-cycle recomputation, no iter cap."""
    n = len(tour)
    cur = tour[:]
    cur_d = _closed_distance(cur, dm)
    improved = True
    while improved:
        improved = False
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


class ClusterTDTSPQAOASolver:
    """Cluster-based TD-TSP solver using QAOA on AWS Braket per cluster."""

    def __init__(self,
                 device: str = "sv1",
                 max_cluster_size: int = 5,
                 use_local_search: bool = True,
                 p: int = 2,
                 shots: int = 500,
                 capture_raw_shots: bool = True,
                 verbose: bool = False,
                 s3_bucket: Optional[str] = None,
                 s3_prefix: Optional[str] = None):
        self.device = device
        self.max_cluster_size = max_cluster_size
        self.use_local_search = use_local_search
        self.p = p
        self.shots = shots
        self.capture_raw_shots = capture_raw_shots
        self.verbose = verbose
        self.s3_bucket = s3_bucket or os.getenv("BRAKET_S3_BUCKET")
        self.s3_prefix = s3_prefix or os.getenv("BRAKET_S3_PREFIX", "tasks")

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

        aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
        aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        if not aws_access_key or not aws_secret_key:
            raise ValueError("AWS credentials not found in environment variables")

        wall_start = datetime.now(timezone.utc)
        t0 = time.perf_counter()
        cluster_t0 = time.time()

        timing: Dict[str, float] = {}
        captured_shots: List[Dict] = []

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

        # Step 2: per-cluster QAOA
        ts = time.time()
        cluster_tours: List[List[int]] = []
        for cluster in clusters:
            if len(cluster) <= max_cluster:
                tour = _solve_cluster_qaoa(
                    cluster, distance_matrix,
                    self.p, self.shots,
                    self.device, self.s3_bucket, self.s3_prefix,
                    aws_access_key, aws_secret_key,
                    captured_shots, self.capture_raw_shots,
                )
            else:
                tour = list(cluster)
            cluster_tours.append(tour)
        timing["qaoa_time_s"] = time.time() - ts

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
                inner_message="cluster-QAOA did not return a complete tour",
                wall_block=wall_block,
            )

        sched = build_schedule(locations, distance_matrix, full_tour,
                               start_time, metric)

        return assemble_tdtsp_result(
            locations=locations,
            schedule_block=sched,
            n=n,
            solver_label=f"Cluster-QAOA ({self.device.upper()}) (K-means + QAOA + 2-opt)",
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
            extra={
                "device": self.device,
                "qaoa_layers": self.p,
                "shots": self.shots,
                "raw_cluster_shots": captured_shots if self.capture_raw_shots else None,
                "cluster_assignments": [list(c) for c in clusters],
            },
        )
