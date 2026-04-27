"""
Cluster-based Time-Dependent TSP solvers used in the IEEE Quantum Week 2026
paper.

Each backend lives in a single, self-contained file:

    tdtsp_cluster_gurobi.py        - Gurobi MILP (MTZ) per cluster
    tdtsp_cluster_dwave.py         - D-Wave Leap Hybrid CQM per cluster
    tdtsp_cluster_qaoa.py          - AWS Braket SV1 / Rigetti QAOA per cluster
    tdtsp_cluster_quanfluence.py   - Quanfluence Ising machine per cluster

Shared schedule / formatting helpers live in ``_tdtsp_cluster_common.py``.
The canonical time-slot list (departure hour + nominal multiplier) lives in
``tdtsp_time_slots.py``; instance JSONs stamp a copy of this list for full
reproducibility.
"""
from .tdtsp_cluster_gurobi import ClusterTDTSPGurobiSolver
from .tdtsp_cluster_dwave import ClusterTDTSPDWaveSolver
from .tdtsp_cluster_qaoa import ClusterTDTSPQAOASolver
from .tdtsp_cluster_quanfluence import ClusterTDTSPQuanfluenceSolver
from .tdtsp_time_slots import TIME_SLOTS, get_slot, scale_matrix

__all__ = [
    "ClusterTDTSPGurobiSolver",
    "ClusterTDTSPDWaveSolver",
    "ClusterTDTSPQAOASolver",
    "ClusterTDTSPQuanfluenceSolver",
    "TIME_SLOTS",
    "get_slot",
    "scale_matrix",
]
