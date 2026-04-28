# Benchmarking Quantum vs Classical Approaches for Time-Dependent TSP

Companion repository for the IEEE Quantum Week 2026 (IEEE QCE) paper
*Benchmarking Quantum vs Classical Approaches for Time-Dependent TSP* by
M. A. Khan, K. Ganesh, and S. Hossain.

The repository is scoped to the **Time-Dependent Traveling Salesperson Problem
(TD-TSP)**: the same Hamiltonian-tour objective as classical TSP, but with
travel times that depend on the departure time-of-day, evaluated against four
fixed time slots:

| Slot          | Departure (local) | Nominal traffic multiplier `m_t` |
|---------------|-------------------|----------------------------------|
| Morning Peak  | 08:00             | 1.5                              |
| Midday        | 12:00             | 1.0                              |
| Evening Peak  | 18:00             | 1.6                              |
| Night         | 22:00             | 0.8                              |

Each benchmark runs a tour starting at the slot's representative hour in the
instance's local timezone and computes per-stop depart/arrive clock times by
walking the tour edges as durations in seconds. Distance matrices are taken
from the **Google Maps Distance Matrix API** (`driving_duration_seconds`,
`traffic_model=best_guess`) for **New York City, NYC**, anchored to Monday
20 April 2026; the resulting per-slot *computed* multipliers are stamped into
each instance JSON (see `data/instances/tdtsp_n*.json`).

## Repository layout

```
.
├── data/
│   └── instances/
│       ├── tdtsp_n5.json          # 5  nodes
│       ├── tdtsp_n10.json         # 10 nodes
│       ├── tdtsp_n25.json         # 25 nodes
│       ├── tdtsp_n50.json         # 50 nodes
│       └── tdtsp_n100.json        # 100 100 nodes
├── code/
│   ├── autonomous/                # single-file TD-TSP solvers, no clustering
│   │   ├── _tdtsp_common.py       # shared schedule/format helpers
│   │   ├── tdtsp_gurobi.py
│   │   ├── tdtsp_dwave.py
│   │   ├── tdtsp_qaoa.py
│   │   └── tdtsp_quanfluence.py
│   └── refined/
│       ├── solvers/               # cluster TD-TSP solvers (one file each)
│       │   ├── _tdtsp_cluster_common.py
│       │   ├── tdtsp_time_slots.py
│       │   ├── tdtsp_cluster_gurobi.py
│       │   ├── tdtsp_cluster_dwave.py
│       │   ├── tdtsp_cluster_qaoa.py
│       │   └── tdtsp_cluster_quanfluence.py
│       └── benchmarks/            # benchmark drivers (one per backend)
├── results/                       # per-solver TD-TSP result JSONs + raw shots
├── TDTSP.tex                      # IEEE conference paper source
├── requirements.txt
├── .env.example                   # template for backend credentials
└── README.md
```

## Solvers

Four optimization paradigms are evaluated on every instance, across all four
time slots:

| Solver        | Type                            | Platform                              |
|---------------|---------------------------------|---------------------------------------|
| Gurobi        | Classical MILP (MTZ)            | Local (license required)              |
| D-Wave        | Quantum annealing (Hybrid CQM)  | D-Wave Leap cloud                     |
| QAOA          | Gate-based quantum (QUBO)       | AWS Braket SV1, Rigetti Cepheus-1-108Q|
| Quanfluence   | Quantum-inspired Ising machine  | Quanfluence REST API                  |

### Autonomous vs cluster

The repository contains two distinct families of solvers:

**`code/autonomous/`** — single-file TD-TSP solvers intended as the direct
artefact of the SuperQ "Super" autonomous optimization platform. Each script
accepts the slot-scaled distance matrix `D_t = m_t * D_base` and solves the
n-node tour end-to-end on its backend with **no decomposition and no 2-opt
post-processing** — a direct QUBO/MILP translation of the canonical TSP
formulation. The autonomous solvers target small instances (n = 5).

**`code/refined/`** — the human-refined cluster pipeline used to scale to 100
nodes. All four backends share the same decomposition strategy:

1. K-means on the instance's real lat/lng coordinates.
2. Partition the nodes into clusters sized to the backend's hardware limit.
3. Solve each cluster's intra-cluster TSP on the target backend, using the
   slot-scaled distance matrix.
4. Stitch the cluster tours together with a greedy nearest-neighbour
   connection.
5. Apply asymmetry-aware **2-opt** local search on the full stitched tour.

Per-backend cluster size limits used in the paper:

| Backend     | Max cluster size | Notes                                                |
|-------------|------------------|------------------------------------------------------|
| Gurobi      | 15               | Limited by per-cluster MILP solve time.              |
| D-Wave      | 15               | Limited by Hybrid CQM submission size.               |
| QAOA (SV1)  | 5  (25 qubits)   | Simulator memory and runtime; QAOA depth `p = 2`.    |
| QAOA (QPU)  | 5  (25 qubits)   | Rigetti Cepheus 20,000-gate compile limit; `p = 1`.  |
| Quanfluence | 10 (100 vars)    | API throughput.                                      |

The Rigetti runs use **`p = 1`** because the 25-qubit `p = 2` circuit
transpiles to ≈21k native gates, just over Cepheus-1-108Q's 20,000-gate
per-task budget. The SV1 runs use **`p = 2`** as in the original SuperQ
configuration.

## Installation

Python 3.10+ is required.

```bash
pip install -r requirements.txt
cp .env.example .env     # then fill in the credentials for whatever backends you want to run
```

Backend-specific prerequisites:

- **Gurobi 13.0+** with a valid license. Place your `gurobi.lic` in your home
  directory or set `GRB_LICENSE_FILE`.
- **AWS account with Amazon Braket access.** Rigetti Cepheus-1-108Q lives in
  `us-west-1`; you need a Braket task-result S3 bucket in the same region. SV1
  results can go to a `us-east-1` bucket.
- **D-Wave Leap** account and API token.
- **Quanfluence** API credentials.

The Google Maps API key is **not** required — the five TD-TSP instance JSONs
ship pre-fetched and frozen with this repository.

## Reproducing the paper

### 1. Run the cluster TD-TSP benchmarks

From the repository root:

```bash
python code/refined/benchmarks/bench_tdtsp_gurobi.py
python code/refined/benchmarks/bench_tdtsp_dwave.py
python code/refined/benchmarks/bench_tdtsp_quanfluence.py
python code/refined/benchmarks/bench_tdtsp_qaoa_sv1.py        # p = 2, 500 shots
python code/refined/benchmarks/bench_tdtsp_qaoa_rigetti.py    # p = 1, 500 shots
```

Each script loads the instances from `data/instances/`, runs the matching
`ClusterTDTSP*Solver` across all five sizes and all four time slots, and
writes its results to `results/results_tdtsp_<solver>.json`. The QAOA scripts
additionally write the **raw 500-shot measurement histograms per cluster** to
`results/raw_tdtsp_qaoa_sv1.json` and `results/raw_tdtsp_qaoa_rigetti.json`
for paper reproducibility.

### 2. Run the autonomous TD-TSP reference

```bash
python code/refined/benchmarks/bench_tdtsp_gurobi_autonomous.py
```

This drives `code/autonomous/tdtsp_gurobi.py` over the small instances and
writes `results/results_tdtsp_gurobi_autonomous.json`. The other three
autonomous solvers (`tdtsp_dwave.py`, `tdtsp_qaoa.py`,
`tdtsp_quanfluence.py`) each ship a `_self_test()` entry-point and can be
executed directly as `python code/autonomous/tdtsp_<backend>.py`.

## Result format

Each benchmark writes a JSON keyed by node count `n`, with one entry per
time slot. A typical entry looks like:

```json
{
  "n": 25,
  "time_slot": "Morning Peak (8 AM)",
  "multiplier": 1.5,
  "solver": "Cluster-Gurobi (K-means + MTZ + 2-opt)",
  "tour": [0, 4, 11, 7, 23, /* ... */],
  "tour_names": ["Times Square, ...", "..."],
  "tour_start_local": "08:00:00",
  "tour_end_local":   "10:14:32",
  "total_tour_value": 8072,
  "total_tour_unit":  "seconds",
  "total_tour_human": "2h 14m 32s",
  "schedule": [
    {"step": 1, "from": "Times Square, ...",
     "to":   "Wall Street, ...",
     "depart": "08:00:00", "arrive": "08:18:43",
     "edge_value": 1123, "edge_unit": "seconds"}
  ],
  "n_clusters": 2,
  "raw_distance_pre_2opt": 8214,
  "solve_time_seconds": 12.4,
  "wall_clock_start_iso": "2026-04-26T...",
  "wall_clock_end_iso":   "2026-04-26T..."
}
```

`raw_distance_pre_2opt` is the stitched tour length before 2-opt;
`total_tour_value` is the final tour length after 2-opt refinement.

## Citation

```bibtex
@inproceedings{khan2026benchmarking,
  title     = {Benchmarking Quantum vs Classical Approaches for Time-Dependent TSP},
  author    = {Ganesh, Krishna and Hossain, Shahadat and Khan, Muhammad Ali},
  booktitle = {Preprint},
  year      = {2026},
  organization = {SuperQ Quantum}
}
```

## Authors

- Krishna Ganesh, SuperQ Quantum Computing Inc., Dubai, UAE
- Shahadat Hossain, University of Northern British Columbia, Canada
- Muhammad Ali Khan, SuperQ Quantum Computing Inc., Calgary, Canada

## Acknowledgements

SuperQ Quantum Computing Inc. and University of Northern British Columbia
