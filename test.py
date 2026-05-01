# pyright: reportMissingImports=false

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "code"))

from utilities.visualization import (
    PlotConfig,
    plot_all_instance_points,
    plot_all_routes,
)

cfg = PlotConfig()

plot_all_instance_points("data/instances", "plots", cfg=cfg)
plot_all_routes("results/results_tdtsp_gurobi.json", "plots", cfg=cfg)
plot_all_routes("results/results_tdtsp_dwave.json", "plots", cfg=cfg)
plot_all_routes("results/results_tdtsp_qaoa_sv1.json", "plots", cfg=cfg)
