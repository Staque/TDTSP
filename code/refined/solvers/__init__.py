"""
Refined cluster-based TSP solvers used in the IEEE Quantum Week 2026 paper.

Each cluster solver decomposes the input via K-means, solves intra-cluster
sub-problems on its target backend (Gurobi, D-Wave, QAOA, Quanfluence),
stitches the sub-tours together, and refines the result with 2-opt.
"""

from .tsp_cluster_gurobi_solver import ClusterGurobiTSPSolver

try:
    from .tsp_cluster_dwave_solver import ClusterDWaveTSPSolver
except ImportError:
    ClusterDWaveTSPSolver = None

try:
    from .tsp_cluster_qaoa_solver import ClusterQAOATSPSolver
except ImportError:
    ClusterQAOATSPSolver = None

try:
    from .tsp_cluster_quanfluence_solver import ClusterQuanfluenceTSPSolver
except ImportError:
    ClusterQuanfluenceTSPSolver = None

__all__ = [
    "ClusterGurobiTSPSolver",
    "ClusterDWaveTSPSolver",
    "ClusterQAOATSPSolver",
    "ClusterQuanfluenceTSPSolver",
]
