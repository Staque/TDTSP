"""
Visualization utilities for TD-TSP instances and solved tours.

Pipeline (single figure per call):
    1. Load the instance JSON or the results JSON.
    2. Look up the per-time-slot distance matrix from the corresponding
       data/instances/tdtsp_n*.json (used as edge duration labels on routes).
    3. Render markers (start as a square, other stops as circles) and an
       optional Hamiltonian tour with arrows.
    4. Apply shared title/axes styling from PlotConfig and save / show.

Also exposes batch drivers that iterate every instance size and time slot
and dump PNGs into separate sub-folders inside `output_dir`:
    - plots/network/   (instance points only)
    - plots/routes/    (one PNG per (n, time_slot) for any results_tdtsp_*.json)

Edge labels on routes use the matching `time_slots[*].distance_matrix` from
the instance JSON, so labels stay consistent across solver back-ends.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D


@dataclass
class PlotConfig:
    """Centralized styling for TD-TSP plots."""

    fig_size: Tuple[float, float] = (8, 8)

    # Fonts
    title_fsize: int = 9
    axis_label_fsize: int = 11
    axis_label_fsize_small: int = 9
    tick_label_fsize: int = 8
    tick_label_fsize_small: int = 6
    legend_title_fsize: int = 12
    legend_fsize: int = 8
    annotation_fsize: int = 8
    edge_label_fsize: int = 6
    axis_label_fontweight: str = "bold"
    title_fontweight: str = "bold"

    # Markers / lines
    point_marker_size: int = 80
    start_marker_size: int = 90
    route_linewidth: float = 2.5
    arrow_linewidth: float = 1.5

    # Legend
    legend_marker_size: int = 8
    legend_start_marker_size: float = 8.5
    legend_labelspacing: float = 1.0
    legend_edgecolor: str = "none"

    # Axes / saving
    x_lim: Tuple[float, float] = (-74.03, -73.89)
    y_lim: Tuple[float, float] = (40.690, 40.890)
    save_dpi: int = 300
    save_bbox_inches: str = "tight"

    # Palette: 3 colors -> (start, stops, route lines)
    palette_name: str = "gist_earth"
    edge_darken_factor: float = 0.7


CONFIG = PlotConfig()


def darken_color(color, factor: float = 0.7) -> Tuple[float, float, float]:
    """Multiply RGB channels by ``factor`` (<1 darker, >1 lighter)."""
    return tuple(c * factor for c in mcolors.to_rgb(color))


def _slug(value: str) -> str:
    """Filesystem-safe slug for figure names."""
    return re.sub(r"[^\w\-]+", "_", value.strip()).strip("_") or "plot"


def _load_json(path: Union[str, Path]) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _slot_distance_matrix(instance_data: dict,
                          run: dict) -> List[List[float]]:
    """Return the time-slot distance_matrix in ``instance_data`` matching ``run``."""
    for slot in instance_data.get("time_slots", []):
        if (
            slot.get("label") == run.get("time_slot")
            or slot.get("label") == run.get("label")
            or slot.get("name") == run.get("name")
        ):
            return slot.get("distance_matrix") or []
    return []


def _draw(ax,
          coords: Sequence[Sequence[float]],
          tour: Optional[Sequence[int]],
          cfg: PlotConfig,
          edge_labels: Sequence[str] = (),
          locations: Sequence[str] = (),
          show_labels: bool = False) -> None:
    """Render nodes (start as a square, others as circles) and an optional tour."""
    color_start, color_stops, color_lines = sns.color_palette(
        cfg.palette_name, n_colors=3
    )
    edge_start = darken_color(color_start, cfg.edge_darken_factor)
    edge_stops = darken_color(color_stops, cfg.edge_darken_factor)

    # Step 1: scatter the non-start stops then the square start marker.
    start_idx = tour[0] if tour else 0
    others = [i for i in range(len(coords)) if i != start_idx]
    if others:
        ax.scatter(
            [coords[i][1] for i in others],
            [coords[i][0] for i in others],
            s=cfg.point_marker_size,
            c=[color_stops] * len(others),
            edgecolors=edge_stops,
            zorder=3,
        )
    ax.scatter(
        [coords[start_idx][1]],
        [coords[start_idx][0]],
        s=cfg.start_marker_size,
        c=[color_start],
        marker="s",
        edgecolors=edge_start,
        zorder=5,
    )

    # Step 2: optional per-node text labels.
    if show_labels:
        for i, (lat, lng) in enumerate(coords):
            label = locations[i] if i < len(locations) else f"Node {i}"
            ax.annotate(
                f"{i}: {label}",
                (lng, lat),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=cfg.annotation_fsize,
            )

    # Step 3: legend handles for start / stops; route handle added later.
    handles = [
        Line2D(
            [0], [0],
            linestyle="None", marker="s",
            markersize=cfg.legend_start_marker_size,
            markerfacecolor=color_start, markeredgecolor=edge_start,
            label=f"Start (node {start_idx})",
        ),
        Line2D(
            [0], [0],
            linestyle="None", marker="o",
            markersize=cfg.legend_marker_size,
            markerfacecolor=color_stops, markeredgecolor=edge_stops,
            label="Stops",
        ),
    ]

    # Step 4: route edges with optional duration text at midpoints.
    if tour and len(tour) >= 2:
        for idx, (a, b) in enumerate(zip(tour[:-1], tour[1:])):
            la, ga = coords[a]
            lb, gb = coords[b]
            ax.plot(
                [ga, gb], [la, lb],
                color=color_lines,
                linewidth=cfg.route_linewidth,
                zorder=2,
            )
            if idx < len(edge_labels):
                ax.text(
                    (ga + gb) / 2,
                    (la + lb) / 2,
                    edge_labels[idx],
                    fontsize=cfg.edge_label_fsize,
                    ha="center",
                    va="center",
                    zorder=6,
                )
            ax.annotate(
                "", xy=(gb, lb), xytext=(ga, la),
                arrowprops=dict(arrowstyle="->", color=color_lines,
                                lw=cfg.arrow_linewidth),
                zorder=2,
            )
        handles.append(
            Line2D([0, 1], [0, 0], color=color_lines,
                   linewidth=cfg.route_linewidth, label="Route")
        )

    ax.legend(
        handles=handles,
        loc="best",
        fontsize=cfg.legend_fsize,
        labelspacing=cfg.legend_labelspacing,
        edgecolor=cfg.legend_edgecolor,
    )


def _finalize(fig,
              ax,
              title: str,
              cfg: PlotConfig,
              save_path: Optional[Union[str, Path]],
              show: bool) -> None:
    """Apply title / axis styling, save the figure, then close or show it."""
    ax.set_title(title, fontsize=cfg.title_fsize, fontweight=cfg.title_fontweight)
    ax.set_xlabel("Longitude", fontsize=cfg.axis_label_fsize,
                  fontweight=cfg.axis_label_fontweight)
    ax.set_ylabel("Latitude", fontsize=cfg.axis_label_fsize,
                  fontweight=cfg.axis_label_fontweight)
    ax.set_xlim(cfg.x_lim)
    ax.set_ylim(cfg.y_lim)
    ax.tick_params(labelsize=cfg.tick_label_fsize)
    ax.grid(False)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=cfg.save_dpi, bbox_inches=cfg.save_bbox_inches)

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_instance_points(instance_json_path: Union[str, Path],
                         show_labels: bool = False,
                         save_path: Optional[Union[str, Path]] = None,
                         show: bool = True,
                         cfg: PlotConfig = CONFIG):
    """Plot the points of a TD-TSP instance JSON (e.g. data/instances/tdtsp_n5.json)."""
    instance = _load_json(instance_json_path)
    coords = instance.get("coords") or []
    if not coords:
        raise ValueError(f"No 'coords' in {instance_json_path}")

    fig, ax = plt.subplots(figsize=cfg.fig_size)
    _draw(
        ax, coords, tour=None, cfg=cfg,
        locations=instance.get("locations", []),
        show_labels=show_labels,
    )
    title = (
        f"TD-TSP Instance Points (n={instance.get('n', len(coords))}, "
        f"city={instance.get('city', 'Unknown')})"
    )
    _finalize(fig, ax, title, cfg, save_path, show)
    return fig, ax


def plot_route(results_json_path: Union[str, Path],
               n: int = 5,
               time_slot: Optional[str] = None,
               run_index: int = 0,
               instances_dir: Union[str, Path] = "data/instances",
               show_labels: bool = False,
               save_path: Optional[Union[str, Path]] = None,
               show: bool = True,
               cfg: PlotConfig = CONFIG):
    """Plot one solved Hamiltonian tour from any results_tdtsp_*.json file."""
    results = _load_json(results_json_path)
    solver_name = results.get("solver", Path(results_json_path).stem)

    block = (results.get("by_size") or {}).get(str(n))
    if block is None:
        sizes = ", ".join(sorted((results.get("by_size") or {}).keys()))
        raise ValueError(f"n={n} not found. Available sizes: {sizes}")

    coords = (block.get("instance") or {}).get("coords") or []
    locations = (block.get("instance") or {}).get("locations") or []
    runs = block.get("runs") or []
    if not coords or not runs:
        raise ValueError(f"Missing coords or runs for n={n} in {results_json_path}")

    # Step 1: select the run by ``time_slot`` if given, else by ``run_index``.
    if time_slot is not None:
        matches = [r for r in runs if r.get("time_slot") == time_slot]
        if not matches:
            known = sorted({r.get("time_slot", "") for r in runs})
            raise ValueError(
                f"time_slot={time_slot!r} not found for n={n}. Known: {known}"
            )
        run = matches[0]
    else:
        if not 0 <= run_index < len(runs):
            raise ValueError(f"run_index must be in [0, {len(runs) - 1}]")
        run = runs[run_index]

    tour = run.get("tour") or []
    if len(tour) < 2:
        raise ValueError("Selected run has no valid 'tour'.")

    # Step 2: edge labels from the matching time-slot distance_matrix.
    instance_path = Path(instances_dir) / f"tdtsp_n{n}.json"
    instance_data = _load_json(instance_path)
    slot_matrix = _slot_distance_matrix(instance_data, run)
    if not slot_matrix:
        raise ValueError(
            f"No matching time-slot distance_matrix found for "
            f"{run.get('time_slot')!r} in {instance_path}"
        )
    edge_labels = [
        f"{slot_matrix[a][b]:.0f}" for a, b in zip(tour[:-1], tour[1:])
    ]

    # Step 3: draw and finalize.
    fig, ax = plt.subplots(figsize=cfg.fig_size)
    _draw(
        ax, coords, tour=tour, cfg=cfg,
        edge_labels=edge_labels,
        locations=locations,
        show_labels=show_labels,
    )
    title = (
        f"{solver_name}\n"
        f"(n={n}, {run.get('time_slot', 'Unknown slot')}, "
        f"total={run.get('total_tour_human', run.get('total_tour_value', 'N/A'))})"
    )
    _finalize(fig, ax, title, cfg, save_path, show)
    return fig, ax


def plot_all_instance_points(instances_dir: Union[str, Path] = "data/instances",
                             output_dir: Union[str, Path] = "plots",
                             show_labels: bool = False,
                             cfg: PlotConfig = CONFIG) -> List[Path]:
    """Save one network PNG per ``tdtsp_n*.json`` under ``plots/network/``."""
    files = sorted(Path(instances_dir).glob("tdtsp_n*.json"))
    if not files:
        raise ValueError(f"No tdtsp_n*.json files found in {instances_dir}")

    network_dir = Path(output_dir) / "network"
    saved: List[Path] = []
    for instance_path in files:
        out = network_dir / f"instance_{instance_path.stem}.png"
        plot_instance_points(
            instance_path,
            show_labels=show_labels,
            save_path=out,
            show=False,
            cfg=cfg,
        )
        saved.append(out)
    return saved


def plot_all_routes(results_json_path: Union[str, Path],
                    output_dir: Union[str, Path] = "plots",
                    instances_dir: Union[str, Path] = "data/instances",
                    time_slots: Optional[Iterable[str]] = None,
                    show_labels: bool = False,
                    cfg: PlotConfig = CONFIG) -> List[Path]:
    """Save one route PNG per (n, time_slot) entry under ``plots/routes/``.

    Works for any results_tdtsp_*.json (Gurobi, D-Wave, QAOA, Quanfluence).
    """
    results = _load_json(results_json_path)
    by_size = results.get("by_size") or {}
    if not by_size:
        raise ValueError(f"No 'by_size' block in {results_json_path}")

    keep = set(time_slots) if time_slots is not None else None
    result_name = Path(results_json_path).stem.removeprefix("results_tdtsp_")
    route_dir = Path(output_dir) / "routes"
    saved: List[Path] = []

    for n_str in sorted(by_size, key=int):
        for run in by_size[n_str].get("runs") or []:
            slot = run.get("time_slot", "unknown_slot")
            if keep is not None and slot not in keep:
                continue
            out = route_dir / f"{_slug(result_name)}_route_n{n_str}_{_slug(slot)}.png"
            plot_route(
                results_json_path,
                n=int(n_str),
                time_slot=slot,
                instances_dir=instances_dir,
                show_labels=show_labels,
                save_path=out,
                show=False,
                cfg=cfg,
            )
            saved.append(out)
    return saved


# Backward-compatible aliases for earlier Gurobi-only names.
def plot_gurobi_route(*args, **kwargs):
    """Backward-compatible alias for ``plot_route``."""
    return plot_route(*args, **kwargs)


def plot_all_gurobi_routes(*args, **kwargs):
    """Backward-compatible alias for ``plot_all_routes``."""
    return plot_all_routes(*args, **kwargs)

