"""
TD-TSP benchmark for the AUTONOMOUS Gurobi backend (single MTZ solve per slot,
no clustering). This is the small-instance reference; the canonical
cluster-based TD-TSP Gurobi benchmark is in bench_tdtsp_gurobi.py.

Runs the autonomous Gurobi TD-TSP solver across all four time slots
(Morning Peak / Midday / Evening Peak / Night) for every real-world TD-TSP
instance currently present in data/instances/tdtsp_n{N}.json (N in
{5, 10, 25, 50, 100} by default; missing files are skipped automatically).

For each (size, slot) pair the benchmark prints:
  - the optimal route (Hamiltonian cycle returning to the start)
  - tour start / end clock times in the instance's local timezone
  - a per-stop schedule (step, from, to, depart, arrive, edge time)
  - the slot's distance multiplier and the total tour duration
  - solver wall-clock instrumentation

After each size, a per-slot summary table is printed showing time dependence.
Finally, a cross-size summary highlights the best slot and worst-vs-best
duration spread for every size. Results are saved to
results/results_tdtsp_gurobi_autonomous.json keyed by size.

Usage:
    python code/refined/benchmarks/bench_tdtsp_gurobi_autonomous.py
"""
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "code" / "autonomous"))

from tdtsp_gurobi import TDTSPGurobiSolver  # noqa: E402

INSTANCES_DIR = REPO_ROOT / "data" / "instances"
RESULTS_PATH = REPO_ROOT / "results" / "results_tdtsp_gurobi_autonomous.json"
SIZES = [5, 10, 25, 50, 100]
NAME_COL_WIDTH = 36


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
    print(f"TD-TSP Gurobi  -  n={instance['n']}  city={city}  tz={tz}  "
          f"metric={metric}  origins={origins}")
    if "anchor_monday" in instance:
        print(f"Anchor Monday: {instance['anchor_monday']}    "
              f"Baseline: {instance.get('baseline', 'n/a')}")
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
    print(f"\n--- {slot['label']}  (start {result['tour_start_local']} local, "
          f"multiplier {slot['multiplier']:.4f}) ---")
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
          f"{'Total ' + unit:>14} {'Human':>14} {'Solve(s)':>10}")
    print("-" * 100)
    optimal = [r for r in rows if r["status"] == "optimal"]
    best = min(optimal, key=lambda r: r["total_tour_value"]) if optimal else None
    for r in rows:
        marker = "  <- best" if r is best else ""
        human = r.get("total_tour_human") or "-"
        print(f"{r['label']:<22} {r['multiplier']:>7.4f} "
              f"{r['tour_start_local']:>8} {r['tour_end_local']:>8} "
              f"{r['total_tour_value']:>14.2f} {human:>14} "
              f"{r['solve_time_seconds']:>10.4f}{marker}")
    if best is not None:
        worst_value = max(r["total_tour_value"] for r in optimal)
        delta_pct = worst_value / best["total_tour_value"] - 1
        print("-" * 100)
        print(f"Best slot to depart: {best['label']}  "
              f"({best['total_tour_value']:.2f} {unit})  "
              f"|  worst-vs-best delta = {delta_pct * 100:.1f}%")


def _print_cross_size_summary(by_size: dict) -> None:
    print(f"\n{'=' * 100}")
    print("Cross-size summary (real-world NYC TD-TSP, 4 time slots each)")
    print("=" * 100)
    print(f"{'n':>4} {'Best slot':<22} {'Best (s)':>12} {'Best (h:m:s)':>14} "
          f"{'Worst slot':<22} {'Worst (s)':>12} {'Spread %':>10}")
    print("-" * 100)
    for n in sorted(int(k) for k in by_size):
        rec = by_size[str(n)]
        runs = [r for r in rec["runs"] if r["status"] == "optimal"]
        if not runs:
            print(f"{n:>4} {'(no optimal runs)':<22}")
            continue
        best = min(runs, key=lambda r: r["total_tour_value"])
        worst = max(runs, key=lambda r: r["total_tour_value"])
        spread = worst["total_tour_value"] / best["total_tour_value"] - 1
        print(f"{n:>4} {best['label']:<22} {best['total_tour_value']:>12.2f} "
              f"{best.get('total_tour_human') or '-':>14} "
              f"{worst['label']:<22} {worst['total_tour_value']:>12.2f} "
              f"{spread * 100:>9.1f}%")
    print("=" * 100)


def _run_one_instance(instance: dict) -> list:
    metric = instance.get("metric", "units")
    locations = instance["locations"]
    start_tz = instance.get("start_time_tz") or instance.get("tz")
    anchor = instance.get("anchor_monday")

    solver = TDTSPGurobiSolver()
    rows = []
    for slot in instance["time_slots"]:
        result = solver.solve(
            locations=locations,
            distance_matrix=slot["distance_matrix"],
            time_slot_label=slot["label"],
            multiplier=slot["multiplier"],
            start_time=_slot_start_time(slot["hour"]),
            start_tz=start_tz,
            start_date=anchor,
            metric=metric,
        )
        result["name"] = slot["name"]
        result["label"] = slot["label"]
        result["hour"] = slot["hour"]
        rows.append(result)
        _print_schedule(result, slot)
    _print_per_slot_summary(rows, metric)
    return rows


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


def main():
    by_size = {}
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
        rows = _run_one_instance(instance)
        by_size[str(n)] = {
            "instance": _instance_meta(instance),
            "runs": rows,
        }

    if not by_size:
        raise SystemExit(
            "No TD-TSP instances found in data/instances/. The frozen "
            "tdtsp_n{5,10,25,50}.json files ship with the repository - "
            "re-clone or re-checkout if missing."
        )

    _print_cross_size_summary(by_size)
    if skipped:
        print(f"\nSkipped sizes (instance file missing): {skipped}")

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump({"by_size": by_size}, f, indent=2)
    print(f"\nSaved: {RESULTS_PATH.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
