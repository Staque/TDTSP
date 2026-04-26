"""
Solvers Package
Contains optimization solvers for various problem types
"""

from .solver import UniversalSolver
from .tsp_solver import TSPSolver
from .time_dependent_tsp_solver import TimeDependentTSPSolver

# Quantum/Quantum-inspired solvers (optional imports)
try:
    from .tsp_dwave_solver import DWaveTSPSolver
except ImportError:
    DWaveTSPSolver = None

try:
    from .tsp_qaoa_solver import QAOATSPSolver
except ImportError:
    QAOATSPSolver = None

try:
    from .tsp_quanfluence_solver import QuanfluenceTSPSolver
except ImportError:
    QuanfluenceTSPSolver = None

try:
    from .tsp_cluster_qaoa_solver import ClusterQAOATSPSolver
except ImportError:
    ClusterQAOATSPSolver = None

try:
    from .tsp_cluster_quanfluence_solver import ClusterQuanfluenceTSPSolver
except ImportError:
    ClusterQuanfluenceTSPSolver = None

try:
    from .tsp_cluster_dwave_solver import ClusterDWaveTSPSolver
except ImportError:
    ClusterDWaveTSPSolver = None

try:
    from .tsp_cluster_gurobi_solver import ClusterGurobiTSPSolver
except ImportError:
    ClusterGurobiTSPSolver = None

__all__ = [
    'UniversalSolver',
    'TSPSolver',
    'TimeDependentTSPSolver',
    'DWaveTSPSolver',
    'QAOATSPSolver',
    'QuanfluenceTSPSolver',
    'ClusterQAOATSPSolver',
    'ClusterQuanfluenceTSPSolver',
    'ClusterDWaveTSPSolver',
    'ClusterGurobiTSPSolver',
]
