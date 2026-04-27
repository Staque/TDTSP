"""
TD-TSP benchmark for the cluster-based QAOA backend on AWS Braket
Rigetti Cepheus-1-108Q (real superconducting QPU, us-west-1).

This is the Rigetti-QPU companion to bench_tdtsp_qaoa_sv1.py. It runs the
exact same ClusterTDTSPQAOASolver wrapper across all four time slots
(Morning Peak / Midday / Evening Peak / Night) for every real-world
TD-TSP instance present in data/instances/tdtsp_n{N}.json (N in {5, 10,
25, 50, 100} by default; missing files are skipped automatically).

Key differences vs the SV1 bench:
  - device='rigetti'  -> arn:aws:braket:us-west-1::device/qpu/rigetti/Cepheus-1-108Q
  - Task results are written to a us-west-1 S3 bucket (same region as the
    QPU); the bucket name is read from BRAKET_S3_BUCKET_RIGETTI in .env.
  - Each cluster QAOA call is a real billable task on a real QPU; expect
    several minutes of queue + wall time per task and a per-cluster cost
    of (task_fee + shots * shot_fee). Rigetti Cepheus pricing on Braket
    is $0.30 per task + $0.00035 per shot (so 500 shots ~= $0.475/task).
  - Results are saved INCREMENTALLY to disk after every (size, slot)
    combination so partial progress is preserved if the QPU goes offline
    or the run is interrupted.

For each (size, slot) pair the benchmark prints the same per-stop
schedule, route, multiplier, cluster decomposition, and timing summary
as the SV1 bench. The QUBO is built directly from the slot-scaled
asymmetric driving-time matrix, so direction-dependent durations are
encoded natively.

Two artefacts are written under results/:
  - results_tdtsp_qaoa_rigetti.json  : the standard summary (compact);
  - raw_tdtsp_qaoa_rigetti.json      : the full per-cluster raw 500-shot
    measurement histograms for every QAOA call. Required for paper
    reproducibility on real-QPU shot data.

Cluster size is capped at 5 cities (= 25 qubits per QUBO), well within
Cepheus' 108-qubit budget.

Usage:
    python code/refined/benchmarks/bench_tdtsp_qaoa_rigetti.py
"""
import copy
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[3]
REFINED_DIR = REPO_ROOT / "code" / "refined"
if str(REFINED_DIR) not in sys.path:
    sys.path.insert(0, str(REFINED_DIR))

load_dotenv(REPO_ROOT / ".env")

from solvers import ClusterTDTSPQAOASolver  # noqa: E402

INSTANCES_DIR = REPO_ROOT / "data" / "instances"
RESULTS_PATH = REPO_ROOT / "results" / "results_tdtsp_qaoa_rigetti.json"
RAW_PATH = REPO_ROOT / "results" / "raw_tdtsp_qaoa_rigetti.json"
SIZES = [5, 10, 25, 50, 100]
NAME_COL_WIDTH = 36
DEVICE = "rigetti"
MAX_CLUSTER_SIZE = 5
# Rigetti Cepheus-1-108Q has a hard 20,000-gate-per-task validation limit.
# Empirically a 25-qubit (5-city) QAOA QUBO at p=2 transpiles to ~21,239
# native gates after Rigetti's compiler inserts SWAP networks for hardware
# connectivity, so it fails validation. Dropping to p=1 cuts the cost-layer
# gate count roughly in half, fitting comfortably under the limit on real
# hardware. The SV1 simulator run at p=2 is preserved separately for
# comparison; the paper notes this real-QPU constraint.
QAOA_LAYERS = 1
QAOA_SHOTS = 500
S3_BUCKET = os.getenv("BRAKET_S3_BUCKET_RIGETTI") or os.getenv("BRAKET_S3_BUCKET")
S3_PREFIX = os.getenv("BRAKET_S3_PREFIX", "tasks/tdtsp")


def _slot_start_time(hour: int) -> str:
    return f"{hour:02d}:00"


def _truncate(label: str, width: int = NAME_COL_WIDTH) -> str:
    if len(label) <= width:
        return label
    return label[: width - 3] + "..."


def _print_header(instance: dict) -> None:
    city = instance.get("city", "synthetic")
    tz = instance.get("tz", "n/a")
    metric = instance.get("metric", "units")
    origins = instance.get("origin_selection", "n/a")
    nominal = instance.get("multipliers", {}).get("nominal", {})
    computed = instance.get("multipliers", {}).get("computed", {})

    print("\n" + "=" * 100)
    print(f"TD-TSP Cluster-QAOA ({DEVICE.upper()} / Cepheus-1-108Q)  -  "
          f"n={instance['n']}  city={city}  tz={tz}  metric={metric}  "
          f"origins={origins}")
    if "anchor_monday" in instance:
        print(f"Anchor Monday: {instance['anchor_monday']}    "
              f"Baseline: {instance.get('baseline', 'n/a')}")
    print(f"QAOA: p={QAOA_LAYERS}, shots={QAOA_SHOTS}, "
          f"max_cluster_size={MAX_CLUSTER_SIZE}, "
          f"s3={S3_BUCKET}/{S3_PREFIX}")
    print("=" * 100)
    if nominal and computed:
        print(f"{'Slot':<22} {'Nominal':>9} {'Computed':>10}")
        print("-" * 100)
        for slot in instance["time_slots"]:
            n_val = nominal.get(slot["name"], slot.get("nominal_multiplier"))
            c_val = computed.get(slot["name"], slot.get("multiplier"))
            print(f"{slot['label']:<22} {n_val:>9.4f} {c_val:>10.4f}")
        print("=" * 100)


def _print_schedule(result: dict, slot: dict) -> None:
    cluster_sizes = result.get("cluster_sizes", [])
    cs_str = "/".join(str(c) for c in cluster_sizes) if cluster_sizes else "-"
    print(f"\n--- {slot['label']}  (start {result['tour_start_local']} local, "
          f"multiplier {slot['multiplier']:.4f}, "
          f"clusters={result.get('n_clusters', 1)} [{cs_str}], "
          f"inner_status={result.get('inner_status')}) ---")
    print(f"Route: {result['route_string']}")
    if result.get("tour_start_iso"):
        print(f"Tour start: {result['tour_start_iso']}")
        print(f"Tour end  : {result['tour_end_iso']}")
    else:
        print(f"Tour start: {result['tour_start_local']} local")
        print(f"Tour end  : {result['tour_end_local']} local")
    if result.get("total_tour_human"):
        print(f"Total tour duration: {result['total_tour_value']:.2f} "
              f"{result['total_tour_unit']}  ({result['total_tour_human']})")
    else:
        print(f"Total tour value: {result['total_tour_value']:.2f} "
              f"{result['total_tour_unit']}")
    raw = result.get("raw_distance_pre_2opt")
    if raw is not None and abs(raw - result["total_tour_value"]) > 1e-6:
        improvement = raw - result["total_tour_value"]
        pct = 100.0 * improvement / raw if raw > 0 else 0.0
        print(f"2-opt improvement: {improvement:.2f} ({pct:.1f}% over raw {raw:.2f})")

    name_w = NAME_COL_WIDTH
    print(f"\n{'Step':>4}  {'From':<{name_w}} {'To':<{name_w}} "
          f"{'Depart':>9} {'Arrive':>9} {'EdgeTime':>14}")
    print("-" * (4 + 2 + name_w + 1 + name_w + 1 + 9 + 1 + 9 + 1 + 14))
    for s in result["schedule"]:
        edge_disp = s.get("edge_human") or f"{s['edge_value']:.2f}"
        print(f"{s['step']:>4}  "
              f"{_truncate(s['from'], name_w):<{name_w}} "
              f"{_truncate(s['to'], name_w):<{name_w}} "
              f"{s['depart']:>9} {s['arrive']:>9} {edge_disp:>14}")


def _print_per_slot_summary(rows: list, metric: str) -> None:
    print(f"\n{'-' * 100}")
    print("Per-slot summary (time dependence)")
    print("-" * 100)
    unit = "seconds" if metric == "driving_duration_seconds" else "units"
    print(f"{'Slot':<22} {'Mult':>7} {'Start':>8} {'End':>8} "
          f"{'Total ' + unit:>14} {'Human':>14} {'Solve(s)':>10} {'Clusters':>9}")
    print("-" * 100)
    success = [r for r in rows if r["status"] == "optimal"]
    best = min(success, key=lambda r: r["total_tour_value"]) if success else None
    for r in rows:
        marker = "  <- best" if r is best else ""
        human = r.get("total_tour_human") or "-"
        clusters = r.get("n_clusters", "-")
        print(f"{r['label']:<22} {r['multiplier']:>7.4f} "
              f"{r['tour_start_local']:>8} {r['tour_end_local']:>8} "
              f"{r['total_tour_value']:>14.2f} {human:>14} "
              f"{r['solve_time_seconds']:>10.4f} {str(clusters):>9}{marker}")
    if best is not None:
        worst_value = max(r["total_tour_value"] for r in success)
        delta_pct = worst_value / best["total_tour_value"] - 1
        print("-" * 100)
        print(f"Best slot to depart: {best['label']}  "
              f"({best['total_tour_value']:.2f} {unit})  "
              f"|  worst-vs-best delta = {delta_pct * 100:.1f}%")


def _print_cross_size_summary(by_size: dict) -> None:
    print(f"\n{'=' * 100}")
    print(f"Cross-size summary (real-world NYC TD-TSP, cluster-QAOA "
          f"Rigetti Cepheus-1-108Q, 4 time slots each)")
    print("=" * 100)
    print(f"{'n':>4} {'Clu':>4} {'Best slot':<22} {'Best (s)':>12} {'Best (h:m:s)':>14} "
          f"{'Worst slot':<22} {'Worst (s)':>12} {'Spread %':>9}")
    print("-" * 100)
    for n in sorted(int(k) for k in by_size):
        rec = by_size[str(n)]
        runs = [r for r in rec["runs"] if r["status"] == "optimal"]
        if not runs:
            print(f"{n:>4} (no successful runs)")
            continue
        best = min(runs, key=lambda r: r["total_tour_value"])
        worst = max(runs, key=lambda r: r["total_tour_value"])
        spread = worst["total_tour_value"] / best["total_tour_value"] - 1
        clusters = best.get("n_clusters", "-")
        print(f"{n:>4} {str(clusters):>4} {best['label']:<22} "
              f"{best['total_tour_value']:>12.2f} "
              f"{best.get('total_tour_human') or '-':>14} "
              f"{worst['label']:<22} {worst['total_tour_value']:>12.2f} "
              f"{spread * 100:>8.1f}%")
    print("=" * 100)


def _instance_meta(instance: dict) -> dict:
    return {
        "n": instance["n"],
        "city": instance.get("city"),
        "tz": instance.get("tz"),
        "anchor_monday": instance.get("anchor_monday"),
        "metric": instance.get("metric"),
        "origin_selection": instance.get("origin_selection"),
        "multipliers": instance.get("multipliers"),
        "locations": instance["locations"],
        "coords": instance.get("coords"),
        "fetch_stats": instance.get("fetch_stats"),
    }


def _save_summary(by_size: dict) -> None:
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {
                "solver": f"Cluster-QAOA ({DEVICE.upper()} / Cepheus-1-108Q) "
                          f"(K-means + QAOA + 2-opt)",
                "device": DEVICE,
                "device_arn": "arn:aws:braket:us-west-1::device/qpu/rigetti/"
                              "Cepheus-1-108Q",
                "max_cluster_size": MAX_CLUSTER_SIZE,
                "qaoa_layers": QAOA_LAYERS,
                "shots": QAOA_SHOTS,
                "s3_bucket": S3_BUCKET,
                "s3_prefix": S3_PREFIX,
                "raw_shots_file": RAW_PATH.name,
                "by_size": by_size,
            },
            f,
            indent=2,
        )


def _save_raw(raw_by_size: dict) -> None:
    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RAW_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {
                "solver": f"Cluster-QAOA ({DEVICE.upper()} / Cepheus-1-108Q) "
                          f"(K-means + QAOA + 2-opt)",
                "device": DEVICE,
                "device_arn": "arn:aws:braket:us-west-1::device/qpu/rigetti/"
                              "Cepheus-1-108Q",
                "max_cluster_size": MAX_CLUSTER_SIZE,
                "qaoa_layers": QAOA_LAYERS,
                "shots": QAOA_SHOTS,
                "s3_bucket": S3_BUCKET,
                "s3_prefix": S3_PREFIX,
                "by_size": raw_by_size,
            },
            f,
            indent=2,
        )


def _run_one_instance(instance: dict, by_size: dict, raw_by_size: dict) -> tuple:
    """
    Run all 4 time slots for one instance. Updates `by_size` and
    `raw_by_size` and persists a snapshot to disk after every slot.

    Returns (rows_for_summary, raw_shots_by_slot).
    """
    n = instance["n"]
    metric = instance.get("metric", "units")
    locations = instance["locations"]
    coords = instance.get("coords")
    start_tz = instance.get("start_time_tz") or instance.get("tz")
    anchor = instance.get("anchor_monday")

    solver = ClusterTDTSPQAOASolver(
        device=DEVICE,
        max_cluster_size=MAX_CLUSTER_SIZE,
        use_local_search=True,
        p=QAOA_LAYERS,
        shots=QAOA_SHOTS,
        capture_raw_shots=True,
        verbose=False,
        s3_bucket=S3_BUCKET,
        s3_prefix=S3_PREFIX,
    )
    rows = []
    raw_by_slot = {}

    # Initialise the per-instance entry up front so partial snapshots are
    # well-formed even if a slot fails / the run is interrupted.
    by_size[str(n)] = {
        "instance": _instance_meta(instance),
        "runs": [],
    }
    raw_by_size[str(n)] = {
        "instance_meta": {
            "n": n,
            "city": instance.get("city"),
            "anchor_monday": instance.get("anchor_monday"),
            "locations": instance["locations"],
        },
        "slots": {},
    }

    for slot in instance["time_slots"]:
        print(f"  -> running {slot['label']} on Rigetti Cepheus ...", flush=True)
        try:
            result = solver.solve(
                locations=locations,
                distance_matrix=slot["distance_matrix"],
                time_slot_label=slot["label"],
                multiplier=slot["multiplier"],
                start_time=_slot_start_time(slot["hour"]),
                start_tz=start_tz,
                start_date=anchor,
                coords=coords,
                metric=metric,
            )
        except Exception as exc:  # noqa: BLE001 - bench wants to keep going
            print(f"     ! slot {slot['name']} failed: {exc}", flush=True)
            result = {
                "status": "failed",
                "inner_status": "exception",
                "inner_message": str(exc),
                "n": n,
                "time_slot": slot["label"],
                "multiplier": slot["multiplier"],
                "tour_start_local": _slot_start_time(slot["hour"]),
                "tour_end_local": _slot_start_time(slot["hour"]),
                "total_tour_value": float("nan"),
                "total_tour_unit": ("seconds" if metric == "driving_duration_seconds"
                                    else "units"),
                "solve_time_seconds": 0.0,
                "schedule": [],
                "n_clusters": None,
                "cluster_sizes": [],
                "raw_cluster_shots": [],
            }

        result["name"] = slot["name"]
        result["label"] = slot["label"]
        result["hour"] = slot["hour"]

        raw_shots = result.pop("raw_cluster_shots", None)
        raw_by_slot[slot["name"]] = {
            "label": slot["label"],
            "hour": slot["hour"],
            "multiplier": slot["multiplier"],
            "n_clusters": result.get("n_clusters"),
            "cluster_sizes": result.get("cluster_sizes"),
            "cluster_assignments": result.get("cluster_assignments"),
            "tour": result.get("tour"),
            "qaoa_layers": QAOA_LAYERS,
            "shots": QAOA_SHOTS,
            "device": DEVICE,
            "raw_cluster_shots": raw_shots,
        }
        rows.append(result)

        # incremental snapshot persistence
        clean_rows = []
        for r in rows:
            r2 = copy.copy(r)
            r2.pop("cluster_assignments", None)
            clean_rows.append(r2)
        by_size[str(n)]["runs"] = clean_rows
        raw_by_size[str(n)]["slots"] = raw_by_slot
        _save_summary(by_size)
        _save_raw(raw_by_size)

        if result.get("status") == "optimal":
            _print_schedule(result, slot)

    _print_per_slot_summary(rows, metric)
    return rows, raw_by_slot


def main():
    if not S3_BUCKET:
        raise SystemExit(
            "BRAKET_S3_BUCKET_RIGETTI not set in .env. Rigetti Cepheus is in "
            "us-west-1 and Braket task results need a same-region bucket."
        )

    by_size = {}
    raw_by_size = {}
    skipped = []
    for n in SIZES:
        path = INSTANCES_DIR / f"tdtsp_n{n}.json"
        if not path.exists():
            skipped.append(n)
            print(f"[skip] {path.relative_to(REPO_ROOT)} not found")
            continue
        with open(path, "r", encoding="utf-8") as f:
            instance = json.load(f)
        _print_header(instance)
        _run_one_instance(instance, by_size, raw_by_size)

    if not by_size:
        raise SystemExit(
            "No TD-TSP instances found in data/instances/. The frozen tdtsp_n{5,10,25,50,100}.json files ship with the repository - re-clone or re-checkout if missing."
        )

    _print_cross_size_summary(by_size)
    if skipped:
        print(f"\nSkipped sizes (instance file missing): {skipped}")

    _save_summary(by_size)
    _save_raw(raw_by_size)
    print(f"\nSaved summary: {RESULTS_PATH.relative_to(REPO_ROOT)}")
    print(f"Saved raw {QAOA_SHOTS}-shot output: {RAW_PATH.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
