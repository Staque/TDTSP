"""
Shared helpers for the four cluster-based TD-TSP solvers.

What is shared:
    - Clock / duration formatting and start-time parsing.
    - Schedule construction from an open tour + slot-scaled distance matrix.
    - ISO timestamp anchoring (local date + tz -> tz-aware ISO strings).
    - Final TD-TSP result-dict assembly with the canonical field set every
      benchmark consumes.

What is NOT shared:
    Each backend (Gurobi MTZ, D-Wave Hybrid CQM, AWS-Braket QAOA, Quanfluence
    Ising) inlines its own K-means clustering, per-cluster sub-solver,
    stitching, and 2-opt local search. Those are deliberately kept in each
    backend's file because each has small algorithmic quirks (n_clusters
    formula, stitching strategy, 2-opt iteration policy, asymmetric-delta vs
    full-cycle recomputation) that we preserve verbatim so the published
    result JSONs reproduce bit-identically.
"""
from __future__ import annotations

from datetime import datetime, time as dtime, timedelta, timezone
from typing import Dict, List, Optional, Tuple, Union
from zoneinfo import ZoneInfo


def format_clock(seconds_since_midnight: float) -> str:
    """Render `HH:MM:SS` for a wall-clock time given as seconds since 00:00."""
    s = int(round(seconds_since_midnight))
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    h = h % 24
    return f"{h:02d}:{m:02d}:{s:02d}"


def format_duration(seconds: float) -> str:
    """Render `[Hh] MMm SSs` for a duration given in seconds."""
    s = int(round(seconds))
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    parts = []
    if h:
        parts.append(f"{h}h")
    if m or h:
        parts.append(f"{m:02d}m")
    parts.append(f"{s:02d}s")
    return " ".join(parts)


def parse_start_time(start_time: Union[str, dtime, None]) -> dtime:
    """Accept `None` / `datetime.time` / `'HH:MM'` / `'HH:MM:SS'`. Default 08:00."""
    if start_time is None:
        return dtime(8, 0)
    if isinstance(start_time, dtime):
        return start_time
    parts = str(start_time).split(":")
    if len(parts) == 2:
        return dtime(int(parts[0]), int(parts[1]))
    if len(parts) == 3:
        return dtime(int(parts[0]), int(parts[1]), int(parts[2]))
    raise ValueError(f"Cannot parse start_time={start_time!r}")


def build_schedule(
    locations: List[str],
    distance_matrix: List[List[float]],
    tour_open: List[int],
    start_time: Union[str, dtime, None],
    metric: str,
) -> Dict:
    """
    Build the schedule block for a TD-TSP solution.

    Parameters
    ----------
    locations : full list of location names indexed by ``tour_open``.
    distance_matrix : the slot-scaled NxN matrix used for this run.
    tour_open : Hamiltonian path as a list of city indices, *without* the
                returning copy of the start city.
    start_time : local clock time of the first departure.
    metric : ``"driving_duration_seconds"`` for real-world TD-TSP, otherwise
             the tag is preserved verbatim and edge_human is omitted.

    Returns
    -------
    dict with keys:
        ``tour_with_return``, ``total_value``, ``schedule``,
        ``tour_start_clock``, ``tour_end_clock``, ``start_time_obj``.
    """
    tour_with_return = list(tour_open) + [tour_open[0]]
    total_value = sum(
        distance_matrix[tour_with_return[i]][tour_with_return[i + 1]]
        for i in range(len(tour_with_return) - 1)
    )

    start_t = parse_start_time(start_time)
    cursor = float(start_t.hour * 3600 + start_t.minute * 60 + start_t.second)
    tour_start_clock = format_clock(cursor)

    use_seconds = metric == "driving_duration_seconds"
    schedule = []
    for step in range(len(tour_with_return) - 1):
        i = tour_with_return[step]
        j = tour_with_return[step + 1]
        edge_value = float(distance_matrix[i][j])
        depart_clock = format_clock(cursor)
        cursor += edge_value
        arrive_clock = format_clock(cursor)
        schedule.append({
            "step": step + 1,
            "from_index": i,
            "from": locations[i],
            "to_index": j,
            "to": locations[j],
            "depart": depart_clock,
            "arrive": arrive_clock,
            "edge_value": round(edge_value, 4),
            "edge_unit": "seconds" if use_seconds else "units",
            "edge_human": format_duration(edge_value) if use_seconds else None,
        })

    return {
        "tour_with_return": tour_with_return,
        "total_value": total_value,
        "schedule": schedule,
        "tour_start_clock": tour_start_clock,
        "tour_end_clock": format_clock(cursor),
        "start_time_obj": start_t,
    }


def make_iso_timestamps(
    start_t: dtime,
    total_value: float,
    start_tz: Optional[str],
    start_date: Optional[str],
) -> Tuple[Optional[str], Optional[str]]:
    """Return tz-aware ISO `(start, end)` strings, or `(None, None)` on failure."""
    if not (start_tz and start_date):
        return None, None
    try:
        tz = ZoneInfo(start_tz)
        anchor = datetime.fromisoformat(start_date).replace(tzinfo=tz)
        start_dt = anchor.replace(
            hour=start_t.hour, minute=start_t.minute, second=start_t.second
        )
        end_dt = start_dt + timedelta(seconds=total_value)
        return start_dt.isoformat(), end_dt.isoformat()
    except Exception:
        return None, None


def assemble_tdtsp_result(
    *,
    locations: List[str],
    schedule_block: Dict,
    n: int,
    solver_label: str,
    time_slot_label: Optional[str],
    multiplier: Optional[float],
    metric: str,
    start_tz: Optional[str],
    start_date: Optional[str],
    n_clusters: int,
    cluster_sizes: List[int],
    raw_distance_pre_2opt: float,
    cluster_timing: Dict,
    wall_block: Dict,
    inner_status: Optional[str],
    extra: Optional[Dict] = None,
) -> Dict:
    """Assemble the canonical TD-TSP result dict from a schedule block."""
    tour_with_return = schedule_block["tour_with_return"]
    total_value = schedule_block["total_value"]
    start_t = schedule_block["start_time_obj"]

    start_iso, end_iso = make_iso_timestamps(
        start_t, total_value, start_tz, start_date
    )
    use_seconds = metric == "driving_duration_seconds"

    out = {
        "status": "optimal",
        "inner_status": inner_status,
        "n": n,
        "solver": solver_label,
        "time_slot": time_slot_label,
        "multiplier": multiplier,
        "metric": metric,
        "tour": tour_with_return,
        "tour_names": [locations[i] for i in tour_with_return],
        "route_string": " -> ".join(locations[i] for i in tour_with_return),
        "tour_start_local": schedule_block["tour_start_clock"],
        "tour_end_local": schedule_block["tour_end_clock"],
        "tour_start_iso": start_iso,
        "tour_end_iso": end_iso,
        "start_tz": start_tz,
        "total_tour_value": round(total_value, 4),
        "total_tour_unit": "seconds" if use_seconds else "units",
        "total_tour_human": format_duration(total_value) if use_seconds else None,
        "schedule": schedule_block["schedule"],
        "n_clusters": n_clusters,
        "cluster_sizes": list(cluster_sizes),
        "raw_distance_pre_2opt": round(raw_distance_pre_2opt, 4),
        "cluster_timing": cluster_timing,
        **wall_block,
    }
    if extra:
        out.update(extra)
    return out


def fail_result(
    *,
    n: int,
    time_slot_label: Optional[str],
    multiplier: Optional[float],
    inner_status: Optional[str],
    inner_message: Optional[str],
    wall_block: Dict,
) -> Dict:
    """Failure record returned by every backend when the inner solve fails."""
    return {
        "status": "failed",
        "inner_status": inner_status,
        "inner_message": inner_message,
        "n": n,
        "time_slot": time_slot_label,
        "multiplier": multiplier,
        **wall_block,
    }
