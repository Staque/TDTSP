"""
Time-Dependent TSP (TD-TSP) solver - Quanfluence Ising Machine backend
(autonomous, non-clustered).

Given a pre-scaled distance matrix D_t[i][j] = D_base[i][j] * m_t for one
time slot t, formulate the n-city tour as a one-hot QUBO over n*n binary
variables x[i, t] (city i sits at position t) with a quadratic assignment
penalty, and submit it to the Quanfluence Ising Machine REST API:

    POST /api/clients/signin                    -> bearer token
    POST /api/devices/{device_id}/qubo/upload   -> .qubo filename
    GET  /api/execute/device/{device_id}/qubo/{filename}  -> direct execution

The result vector is decoded into a tour and the starting city is
normalised. The solver also produces a per-stop schedule (depart/arrive
clock times) anchored at a configurable start time (default 08:00) and
optional tz/date for ISO timestamps. Designed for n <= ~30; use the
cluster-decomposed solver for larger instances.
"""
import os
import time
from datetime import datetime, time as dtime, timezone
from typing import Dict, List, Optional, Tuple, Union

import requests
from dotenv import load_dotenv

from _tdtsp_common import build_schedule

load_dotenv()


class TDTSPQuanfluenceSolver:
    """Departure-time-dependent TSP solver (single time slot, Quanfluence)."""

    def __init__(self,
                 base_url: Optional[str] = None,
                 username: Optional[str] = None,
                 password: Optional[str] = None,
                 device_id: Optional[str] = None):
        self.base_url = base_url or os.getenv("QUANFLUENCE_BASE_URL",
                                              "https://gateway.quanfluence.com")
        self.username = username or os.getenv("QUANFLUENCE_USERNAME")
        self.password = password or os.getenv("QUANFLUENCE_PASSWORD")
        self.device_id = device_id or os.getenv("QUANFLUENCE_DEVICE_ID", "41")
        if not self.username or not self.password:
            raise ValueError("Quanfluence credentials not found in environment variables")
        self._token: Optional[str] = None
        self._token_expiry: float = 0.0

    def _authenticate(self) -> str:
        if self._token and time.time() < self._token_expiry - 60:
            return self._token
        r = requests.post(
            f"{self.base_url}/api/clients/signin",
            json={"username": self.username, "password": self.password},
            headers={"Content-Type": "application/json"},
            timeout=30,
        )
        r.raise_for_status()
        data = r.json()
        if data.get("status") != "success":
            raise RuntimeError(f"Quanfluence auth failed: {data}")
        self._token = data["data"]["token"]
        self._token_expiry = time.time() + data["data"].get("expiresIn", 3600)
        return self._token

    def _headers(self) -> Dict[str, str]:
        return {"Authorization": f"Bearer {self._authenticate()}"}

    @staticmethod
    def _tsp_to_qubo(n: int, distance_matrix: List[List[float]],
                     penalty: Optional[float] = None) -> Dict[Tuple[int, int], float]:
        """One-hot QUBO with quadratic row/column penalty."""
        if penalty is None:
            penalty = 2 * max(max(row) for row in distance_matrix) * n

        def idx(city: int, position: int) -> int:
            return city * n + position

        Q: Dict[Tuple[int, int], float] = {}
        for t in range(n):
            nt = (t + 1) % n
            for i in range(n):
                for j in range(n):
                    if i != j and distance_matrix[i][j] > 0:
                        Q[(idx(i, t), idx(j, nt))] = (
                            Q.get((idx(i, t), idx(j, nt)), 0.0) + distance_matrix[i][j]
                        )
        for i in range(n):
            for t in range(n):
                k = idx(i, t)
                Q[(k, k)] = Q.get((k, k), 0.0) - penalty
            for t1 in range(n):
                for t2 in range(t1 + 1, n):
                    Q[(idx(i, t1), idx(i, t2))] = (
                        Q.get((idx(i, t1), idx(i, t2)), 0.0) + 2 * penalty
                    )
        for t in range(n):
            for i in range(n):
                k = idx(i, t)
                Q[(k, k)] = Q.get((k, k), 0.0) - penalty
            for i1 in range(n):
                for i2 in range(i1 + 1, n):
                    Q[(idx(i1, t), idx(i2, t))] = (
                        Q.get((idx(i1, t), idx(i2, t)), 0.0) + 2 * penalty
                    )
        return Q

    @staticmethod
    def _decode_solution(result_vector: List[int], n: int) -> Optional[List[int]]:
        tour: List[Optional[int]] = [None] * n
        for k, bit in enumerate(result_vector):
            if bit == 1:
                city, position = divmod(k, n)
                if position >= n or city >= n:
                    continue
                if tour[position] is None:
                    tour[position] = city
        if any(c is None for c in tour) or len(set(tour)) != n:
            return None
        return tour  # type: ignore[return-value]

    def _upload_qubo(self, Q: Dict[Tuple[int, int], float]) -> str:
        items = [f"({i}, {j}): {coeff}" for (i, j), coeff in Q.items() if abs(coeff) > 1e-10]
        body = "{" + ", ".join(items) + "}"
        r = requests.post(
            f"{self.base_url}/api/devices/{self.device_id}/qubo/upload",
            files={"file": ("problem.qubo", body.encode("utf-8"), "text/plain")},
            headers=self._headers(),
            timeout=60,
        )
        if r.status_code not in (200, 201):
            raise RuntimeError(f"QUBO upload failed: {r.status_code} - {r.text}")
        data = r.json()
        if data.get("status") != "success":
            raise RuntimeError(f"QUBO upload failed: {data}")
        return data["data"]["result"]

    def _execute(self, filename: str) -> Dict:
        r = requests.get(
            f"{self.base_url}/api/execute/device/{self.device_id}/qubo/{filename}",
            headers=self._headers(),
            timeout=3600,
        )
        if r.status_code != 200:
            raise RuntimeError(f"QUBO execution failed: {r.status_code} - {r.text}")
        data = r.json()
        if data.get("status") != "success":
            raise RuntimeError(f"QUBO execution failed: {data}")
        return data["data"]

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

        Q = self._tsp_to_qubo(n, distance_matrix)
        filename = self._upload_qubo(Q)
        run_data = self._execute(filename)

        solve_time = time.perf_counter() - t0
        wall_end = datetime.now(timezone.utc)
        wall_block = {
            "wall_clock_start_iso": wall_start.isoformat(),
            "wall_clock_end_iso": wall_end.isoformat(),
            "solve_time_seconds": round(solve_time, 6),
        }

        result_vector = run_data.get("result", []) or []
        energy = run_data.get("energy", None)
        decoded = self._decode_solution(result_vector, n)

        if decoded is None:
            return {
                "status": "failed",
                "reason": "Quanfluence sample did not decode to a valid tour",
                "n": n,
                "time_slot": time_slot_label,
                "multiplier": multiplier,
                "energy": energy,
                **wall_block,
            }

        tour = decoded
        if start_index in tour:
            shift = tour.index(start_index)
            tour = tour[shift:] + tour[:shift]
        tour_with_return = tour + [tour[0]]

        sched = build_schedule(
            tour_with_return, locations, distance_matrix,
            start_time=start_time, start_tz=start_tz, start_date=start_date,
            metric=metric,
        )

        return {
            "status": "optimal",
            "n": n,
            "solver": "Quanfluence Ising Machine (autonomous)",
            "device_id": self.device_id,
            "time_slot": time_slot_label,
            "multiplier": multiplier,
            "metric": metric,
            "tour": tour_with_return,
            "tour_names": [locations[i] for i in tour_with_return],
            "route_string": " -> ".join(locations[i] for i in tour_with_return),
            "start_tz": start_tz,
            **sched,
            "energy": energy,
            "qubo_terms": len(Q),
            **wall_block,
        }


def _self_test():
    """Sanity-check the solver on a 5-city symmetric instance at multiplier 1.5."""
    locations = ["A", "B", "C", "D", "E"]
    base = [
        [0, 2, 9, 10, 7], [2, 0, 6, 4, 3], [9, 6, 0, 8, 5],
        [10, 4, 8, 0, 6], [7, 3, 5, 6, 0],
    ]
    scaled = [[round(v * 1.5, 4) for v in row] for row in base]
    result = TDTSPQuanfluenceSolver().solve(
        locations=locations, distance_matrix=scaled,
        time_slot_label="Morning Peak (8 AM)", multiplier=1.5,
        start_time="08:00", metric="units",
    )
    print("\nTD-TSP Quanfluence self-test")
    print("-" * 60)
    print(f"  status            : {result['status']}")
    print(f"  route             : {result.get('route_string', '-')}")
    print(f"  tour start->end   : {result.get('tour_start_local', '-')} -> "
          f"{result.get('tour_end_local', '-')}")
    print(f"  total tour value  : {result.get('total_tour_value', '-')} "
          f"{result.get('total_tour_unit', '')}")
    print(f"  energy            : {result.get('energy', '-')}")
    print(f"  solver wall (s)   : {result['solve_time_seconds']}")


if __name__ == "__main__":
    _self_test()
