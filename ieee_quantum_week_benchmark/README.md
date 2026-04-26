# IEEE Quantum Week 2026 - TSP Benchmark Suite

**Benchmarking Quantum vs Classical Approaches for Time-Dependent TSP**

This repository contains the complete benchmark suite for comparing classical, quantum, and quantum-inspired solvers on the Traveling Salesperson Problem (TSP).

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Solvers](#solvers)
- [Running Benchmarks](#running-benchmarks)
- [Results](#results)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)

## Overview

This benchmark suite evaluates four optimization approaches:

| Solver | Type | Platform |
|--------|------|----------|
| **Gurobi** | Classical MILP | Local (WLS License) |
| **D-Wave** | Quantum Annealer | D-Wave Leap Cloud |
| **QAOA** | Gate-based Quantum | AWS Braket (SV1/Rigetti) |
| **Quanfluence** | Quantum-Inspired Ising Machine | REST API |

All solvers use a **cluster-based decomposition** strategy for scalability:
1. K-means clustering divides cities into smaller groups
2. Each cluster's TSP is solved independently
3. Tours are stitched together via greedy nearest-neighbor
4. 2-opt local search refines the final solution

## Installation

### Prerequisites

- Python 3.10+
- Gurobi Optimizer 13.0+ with valid license
- AWS Account with Braket access
- D-Wave Leap account
- Quanfluence API credentials

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Required packages:
```
gurobipy
numpy
scikit-learn>=1.3.0
python-dotenv
dwave-ocean-sdk
amazon-braket-sdk
boto3
requests
```

### Environment Variables

Create a `.env` file in the project root:

```env
# AWS Credentials (for Braket)
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_DEFAULT_REGION=us-east-1

# D-Wave
DWAVE_API_TOKEN=your_dwave_token

# Quanfluence
QUANFLUENCE_BASE_URL=https://api.quanfluence.com
QUANFLUENCE_USERNAME=your_username
QUANFLUENCE_PASSWORD=your_password
QUANFLUENCE_DEVICE_ID=41

# Gurobi (if using WLS)
GRB_LICENSE_FILE=path/to/gurobi.lic
```

## Project Structure

```
ieee_quantum_week_benchmark/
├── README.md                 # This file
├── solvers/                  # Solver implementations
│   ├── __init__.py
│   ├── tsp_solver.py                    # Base Gurobi TSP solver
│   ├── tsp_cluster_gurobi_solver.py     # Clustered Gurobi
│   ├── tsp_dwave_solver.py              # D-Wave CQM solver
│   ├── tsp_cluster_dwave_solver.py      # Clustered D-Wave
│   ├── tsp_qaoa_solver.py               # QAOA on Braket
│   ├── tsp_cluster_qaoa_solver.py       # Clustered QAOA
│   ├── tsp_quanfluence_solver.py        # Quanfluence solver
│   └── tsp_cluster_quanfluence_solver.py # Clustered Quanfluence
├── benchmarks/               # Benchmark scripts
│   ├── bench_gurobi.py       # Run Gurobi benchmark
│   ├── bench_dwave.py        # Run D-Wave benchmark
│   ├── bench_quanfluence.py  # Run Quanfluence benchmark
│   ├── bench_qaoa.py         # Run QAOA (SV1) benchmark
│   ├── bench_qaoa_rigetti.py # Run QAOA (Rigetti QPU) benchmark
│   ├── check_braket_devices.py  # Check available QPUs
│   └── estimate_gate_counts.py  # Estimate QAOA gate counts
└── results/                  # Benchmark results (JSON)
    ├── results_gurobi.json
    ├── results_dwave.json
    ├── results_quanfluence.json
    ├── results_qaoa.json
    └── results_qaoa_rigetti.json
```

## Solvers

### Gurobi (Classical Baseline)
- **Formulation**: Miller-Tucker-Zemlin (MTZ) for subtour elimination
- **Cluster size**: Up to 15 cities per cluster
- **Usage**:
```python
from solvers import ClusterGurobiTSPSolver
solver = ClusterGurobiTSPSolver(max_cluster_size=15, use_local_search=True)
result = solver.solve_tsp(locations, distance_matrix)
```

### D-Wave (Quantum Annealing)
- **Formulation**: Constrained Quadratic Model (CQM)
- **Cluster size**: Up to 15 cities per cluster
- **Usage**:
```python
from solvers import ClusterDWaveTSPSolver
solver = ClusterDWaveTSPSolver(max_cluster_size=15, use_local_search=True)
result = solver.solve_tsp(locations, distance_matrix)
```

### QAOA (Gate-based Quantum)
- **Formulation**: QUBO encoded as cost Hamiltonian
- **Cluster size**: 4-5 cities max (due to gate count limits)
- **Devices**: SV1 simulator, Rigetti Cepheus-1-108Q
- **Usage**:
```python
from solvers import ClusterQAOATSPSolver
solver = ClusterQAOATSPSolver(device='sv1', max_cluster_size=5, use_local_search=True)
result = solver.solve_tsp(locations, distance_matrix, p=2, shots=500)
```

### Quanfluence (Quantum-Inspired)
- **Formulation**: QUBO for Ising machine
- **Cluster size**: Up to 10 cities per cluster
- **Usage**:
```python
from solvers import ClusterQuanfluenceTSPSolver
solver = ClusterQuanfluenceTSPSolver(max_cluster_size=10, use_local_search=True)
result = solver.solve_tsp(locations, distance_matrix)
```

## Running Benchmarks

### Quick Start

Run all benchmarks sequentially:

```bash
# 1. Check Gurobi license
python benchmarks/bench_gurobi.py

# 2. Run D-Wave benchmark
python benchmarks/bench_dwave.py

# 3. Run Quanfluence benchmark
python benchmarks/bench_quanfluence.py

# 4. Run QAOA on SV1 simulator
python benchmarks/bench_qaoa.py

# 5. (Optional) Run on Rigetti QPU
python benchmarks/bench_qaoa_rigetti.py
```

### Check Available Quantum Hardware

Before running on real QPUs:

```bash
python benchmarks/check_braket_devices.py
```

### Estimate Gate Counts

To understand QAOA circuit limitations:

```bash
python benchmarks/estimate_gate_counts.py
```

### Problem Sizes

All benchmarks test 5 problem sizes: **5, 10, 25, 50, 100 cities**

Each uses a fixed random seed (42) for reproducibility.

## Results

Results are saved as JSON files in the `results/` folder.

### Example Result Format

```json
{
  "n": 100,
  "distance": 781.56,
  "raw_distance": 963.31,
  "n_clusters": 7,
  "time": 151.85,
  "status": "optimal",
  "solver": "Cluster-D-Wave"
}
```

### Summary of Our Results

| Cities | Gurobi | D-Wave | Quanfluence | QAOA (SV1) | Rigetti |
|--------|--------|--------|-------------|------------|---------|
| 5 | **227.35** | **227.35** | **227.35** | **227.35** | **227.35** |
| 10 | **290.31** | **290.31** | **290.31** | **290.31** | **290.31** |
| 25 | 418.58 | 415.96 | **407.30** | 449.06 | **407.30** |
| 50 | 643.05 | 591.58 | **589.93** | 591.35 | 609.40 |
| 100 | 824.33 | **781.56** | 934.98 | 782.62 | 790.13 |

## Configuration

### QAOA Parameters

| Parameter | SV1 (Simulator) | Rigetti (QPU) |
|-----------|-----------------|---------------|
| QAOA layers (p) | 2 | 1 |
| Shots | 500 | 100 |
| Max cluster size | 5 cities | 4 cities |
| Max qubits/cluster | 25 | 16 |

### Gate Count Limits

Rigetti Cepheus-1-108Q has a **20,000 gate limit**:
- 5 cities (25 qubits), p=2 → ~21,000 gates (FAILS)
- 4 cities (16 qubits), p=1 → ~4,300 gates (WORKS)

## Troubleshooting

### Gurobi License Issues

```
Error: Single-use license. Another Gurobi process running.
```

**Solution**: Kill existing Python processes:
```powershell
Get-Process python* | Stop-Process -Force
```

### AWS Braket S3 Bucket Error

```
Error: Caller doesn't have access to bucket
```

**Solution**: Let Braket use its default bucket (don't specify one).

### Rigetti Gate Count Exceeded

```
Error: Total number of gates is higher than 20,000
```

**Solution**: Reduce cluster size to 4 and use p=1.

### D-Wave Timeout

```
Error: Problem too large for sampler
```

**Solution**: Reduce cluster size or use the hybrid CQM solver.

## Citation

If you use this benchmark suite, please cite:

```bibtex
@inproceedings{khan2026benchmarking,
  title={Benchmarking Quantum vs Classical Approaches for Time-Dependent TSP},
  author={Khan, Muhammad Ali and Ganesh, Krishna and Hossain, Shahadat},
  booktitle={IEEE International Conference on Quantum Computing and Engineering (QCE)},
  year={2026},
  organization={IEEE}
}
```

## License

This project is provided for research purposes. See individual solver licenses for commercial use restrictions.

## Authors

- **Muhammad Ali Khan** - SuperQ Quantum Computing Inc., Calgary, Canada
- **Krishna Ganesh** - SuperQ Quantum Computing Inc., Dubai, UAE
- **Shahadat Hossain** - University of Northern British Columbia, Canada

## Acknowledgments

- SuperQ Quantum Computing Inc. for the Super autonomous optimization platform
- AWS for Amazon Braket access
- D-Wave Systems for Leap platform access
- Quanfluence for Ising machine API access
- Gurobi Optimization for academic licensing
