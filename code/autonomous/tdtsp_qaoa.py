"""
Time-Dependent TSP (TD-TSP) solver - QAOA backend (autonomous, non-clustered).

Given a pre-scaled distance matrix D_t[i][j] = D_base[i][j] * m_t for one
time slot t, formulate the n-city tour as a one-hot QUBO over n*n binary
variables x[i, t] (city i sits at position t) with a quadratic assignment
penalty, run a fixed-angle QAOA circuit on AWS Braket, and return the best
valid tour across all measurement bitstrings.

The encoding mirrors the cluster-QAOA solver in code/refined/solvers/, so
results are directly comparable for small instances. Default device is the
SV1 statevector simulator; the Rigetti QPU ARN is also exposed.

The solver also produces a per-stop schedule (depart/arrive clock times)
anchored at a configurable start time (default 08:00) and optional tz/date
to emit ISO timestamps. Designed for n <= 5 (25 qubits at n=5); use the
cluster-decomposed solver for larger instances.
"""
import math
import os
import time
from datetime import datetime, time as dtime, timezone
from typing import Dict, List, Optional, Tuple, Union

from dotenv import load_dotenv

from _tdtsp_common import build_schedule

load_dotenv()


class TDTSPQAOASolver:
    """Departure-time-dependent TSP solver (single time slot, QAOA on Braket)."""

    DEVICES = {
        "sv1": "arn:aws:braket:::device/quantum-simulator/amazon/sv1",
        "rigetti": "arn:aws:braket:us-west-1::device/qpu/rigetti/Cepheus-1-108Q",
    }

    def __init__(self, device: str = "sv1"):
        self.aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
        self.aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        self.aws_region = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
        if not self.aws_access_key or not self.aws_secret_key:
            raise ValueError("AWS credentials not found in environment variables")
        if device not in self.DEVICES:
            raise ValueError(f"Unknown device {device!r}; expected one of {list(self.DEVICES)}")
        self.device_name = device
        self.device_arn = self.DEVICES[device]

    @staticmethod
    def _tsp_to_qubo(n: int, distance_matrix: List[List[float]],
                     penalty: Optional[float] = None) -> Tuple[Dict[Tuple[int, int], float], int]:
        """One-hot QUBO with quadratic row/column penalty. Returns (Q, num_qubits)."""
        if penalty is None:
            penalty = 2 * max(max(row) for row in distance_matrix)

        def idx(city: int, position: int) -> int:
            return city * n + position

        Q: Dict[Tuple[int, int], float] = {}
        for t in range(n):
            nt = (t + 1) % n
            for i in range(n):
                for j in range(n):
                    if i != j and distance_matrix[i][j] > 0:
                        a, b = idx(i, t), idx(j, nt)
                        key = (min(a, b), max(a, b))
                        Q[key] = Q.get(key, 0.0) + distance_matrix[i][j]
        for i in range(n):
            for t in range(n):
                k = idx(i, t)
                Q[(k, k)] = Q.get((k, k), 0.0) - penalty
            for t1 in range(n):
                for t2 in range(t1 + 1, n):
                    a, b = idx(i, t1), idx(i, t2)
                    Q[(min(a, b), max(a, b))] = Q.get((min(a, b), max(a, b)), 0.0) + 2 * penalty
        for t in range(n):
            for i in range(n):
                k = idx(i, t)
                Q[(k, k)] = Q.get((k, k), 0.0) - penalty
            for i1 in range(n):
                for i2 in range(i1 + 1, n):
                    a, b = idx(i1, t), idx(i2, t)
                    Q[(min(a, b), max(a, b))] = Q.get((min(a, b), max(a, b)), 0.0) + 2 * penalty
        return Q, n * n

    @staticmethod
    def _decode_bitstring(bitstring: str, n: int) -> Optional[List[int]]:
        tour: List[Optional[int]] = [None] * n
        for k, bit in enumerate(bitstring):
            if bit == "1":
                city, position = divmod(k, n)
                if tour[position] is not None:
                    return None
                tour[position] = city
        if any(c is None for c in tour) or len(set(tour)) != n:
            return None
        return tour  # type: ignore[return-value]

    def _build_circuit(self, Q: Dict[Tuple[int, int], float], num_qubits: int,
                       gammas: List[float], betas: List[float]):
        from braket.circuits import Circuit
        circuit = Circuit()
        for q in range(num_qubits):
            circuit.h(q)
        for layer in range(len(gammas)):
            gamma, beta = gammas[layer], betas[layer]
            for (i, j), coeff in Q.items():
                if i == j:
                    circuit.rz(i, 2 * gamma * coeff)
                else:
                    circuit.cnot(i, j)
                    circuit.rz(j, 2 * gamma * coeff)
                    circuit.cnot(i, j)
            for q in range(num_qubits):
                circuit.rx(q, 2 * beta)
        return circuit

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
              p: int = 2,
              shots: int = 500,
              s3_bucket: Optional[str] = None,
              s3_prefix: Optional[str] = None,
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

        import boto3
        from braket.aws import AwsDevice, AwsSession

        Q, num_qubits = self._tsp_to_qubo(n, distance_matrix)
        gammas = [math.pi / 4] * p
        betas = [math.pi / 8] * p
        circuit = self._build_circuit(Q, num_qubits, gammas, betas)

        boto_session = boto3.Session(
            region_name=self.aws_region,
            aws_access_key_id=self.aws_access_key,
            aws_secret_access_key=self.aws_secret_key,
        )
        aws_session = AwsSession(boto_session=boto_session)
        device = AwsDevice(self.device_arn, aws_session=aws_session)

        s3_bucket = s3_bucket or os.getenv("BRAKET_S3_BUCKET")
        s3_prefix = s3_prefix or os.getenv("BRAKET_S3_PREFIX", "tasks/tdtsp")
        run_kwargs = {"shots": shots}
        if s3_bucket:
            run_kwargs["s3_destination_folder"] = (s3_bucket, s3_prefix)

        task = device.run(circuit, **run_kwargs)
        task_result = task.result()
        measurements = task_result.measurement_counts

        solve_time = time.perf_counter() - t0
        wall_end = datetime.now(timezone.utc)
        wall_block = {
            "wall_clock_start_iso": wall_start.isoformat(),
            "wall_clock_end_iso": wall_end.isoformat(),
            "solve_time_seconds": round(solve_time, 6),
        }

        best_tour: Optional[List[int]] = None
        best_distance = float("inf")
        valid_count = 0
        for bitstring, count in measurements.items():
            decoded = self._decode_bitstring(bitstring, n)
            if decoded is None:
                continue
            valid_count += count
            d = sum(distance_matrix[decoded[i]][decoded[(i + 1) % n]] for i in range(n))
            if d < best_distance:
                best_distance = d
                best_tour = decoded

        if best_tour is None:
            return {
                "status": "failed",
                "reason": "no valid tour in measurements",
                "n": n,
                "time_slot": time_slot_label,
                "multiplier": multiplier,
                "shots": shots,
                "qaoa_layers": p,
                **wall_block,
            }

        tour = best_tour
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
            "solver": f"QAOA ({self.device_name.upper()}, autonomous)",
            "device_arn": self.device_arn,
            "time_slot": time_slot_label,
            "multiplier": multiplier,
            "metric": metric,
            "tour": tour_with_return,
            "tour_names": [locations[i] for i in tour_with_return],
            "route_string": " -> ".join(locations[i] for i in tour_with_return),
            "start_tz": start_tz,
            **sched,
            "qaoa_layers": p,
            "shots": shots,
            "valid_shots": valid_count,
            "num_qubits": num_qubits,
            "qubo_terms": len(Q),
            **wall_block,
        }


def _self_test():
    """Sanity-check the solver on a 4-city symmetric instance at multiplier 1.5."""
    locations = ["A", "B", "C", "D"]
    base = [[0, 10, 15, 20], [10, 0, 35, 25], [15, 35, 0, 30], [20, 25, 30, 0]]
    scaled = [[round(v * 1.5, 4) for v in row] for row in base]
    result = TDTSPQAOASolver(device="sv1").solve(
        locations=locations, distance_matrix=scaled,
        time_slot_label="Morning Peak (8 AM)", multiplier=1.5,
        start_time="08:00", metric="units", p=1, shots=200,
    )
    print("\nTD-TSP QAOA (SV1) self-test")
    print("-" * 60)
    print(f"  status            : {result['status']}")
    print(f"  route             : {result.get('route_string', '-')}")
    print(f"  tour start->end   : {result.get('tour_start_local', '-')} -> "
          f"{result.get('tour_end_local', '-')}")
    print(f"  total tour value  : {result.get('total_tour_value', '-')} "
          f"{result.get('total_tour_unit', '')}")
    print(f"  valid/total shots : {result.get('valid_shots', 0)}/{result.get('shots', 0)}")
    print(f"  solver wall (s)   : {result['solve_time_seconds']}")


if __name__ == "__main__":
    _self_test()
