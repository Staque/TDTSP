"""
Shared helpers for the autonomous TD-TSP solvers (single time slot).

All four backends (Gurobi, D-Wave, QAOA, Quanfluence) accept the same
inputs (locations, pre-scaled distance matrix, time-slot metadata) and
emit the same per-stop schedule shape. This module centralises the
clock / duration formatters, the start-time parser, and the schedule
builder so each solver file stays small and focused on its backend.
"""
from datetime import datetime, time as dtime, timedelta, timezone
from typing import Dict, List, Optional, Union
from zoneinfo import ZoneInfo


def format_clock(seconds_since_midnight: float) -> str:
    s = int(round(seconds_since_midnight))
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    h = h % 24
    return f"{h:02d}:{m:02d}:{s:02d}"


def format_duration(seconds: float) -> str:
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


def build_schedule(tour_with_return: List[int],
                   locations: List[str],
                   distance_matrix: List[List[float]],
                   start_time: Union[str, dtime, None] = "08:00",
                   start_tz: Optional[str] = None,
                   start_date: Optional[str] = None,
                   metric: str = "driving_duration_seconds") -> Dict:
    """Walk the tour edges (treated as seconds) and emit the per-stop schedule.

    Returns a dict with `schedule`, `tour_start_local`, `tour_end_local`,
    `tour_start_iso`, `tour_end_iso`, `total_tour_value`,
    `total_tour_unit`, and `total_tour_human`.
    """
    start_t = parse_start_time(start_time)
    cursor = float(start_t.hour * 3600 + start_t.minute * 60 + start_t.second)
    tour_start_clock = format_clock(cursor)
    is_seconds = (metric == "driving_duration_seconds")

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
            "edge_unit": "seconds" if is_seconds else "units",
            "edge_human": format_duration(edge_value) if is_seconds else None,
        })

    tour_end_clock = format_clock(cursor)
    total_value = sum(
        float(distance_matrix[tour_with_return[i]][tour_with_return[i + 1]])
        for i in range(len(tour_with_return) - 1)
    )

    tour_start_iso = None
    tour_end_iso = None
    if start_tz and start_date:
        try:
            tz = ZoneInfo(start_tz)
            anchor = datetime.fromisoformat(start_date).replace(tzinfo=tz)
            start_dt = anchor.replace(
                hour=start_t.hour, minute=start_t.minute, second=start_t.second
            )
            end_dt = start_dt + timedelta(seconds=total_value)
            tour_start_iso = start_dt.isoformat()
            tour_end_iso = end_dt.isoformat()
        except Exception:
            tour_start_iso = None
            tour_end_iso = None

    return {
        "schedule": schedule,
        "tour_start_local": tour_start_clock,
        "tour_end_local": tour_end_clock,
        "tour_start_iso": tour_start_iso,
        "tour_end_iso": tour_end_iso,
        "total_tour_value": round(total_value, 4),
        "total_tour_unit": "seconds" if is_seconds else "units",
        "total_tour_human": format_duration(total_value) if is_seconds else None,
    }


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
