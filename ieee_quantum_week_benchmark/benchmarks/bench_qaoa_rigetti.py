"""Cluster-QAOA benchmark on Rigetti Cepheus-1-108Q QPU for paper"""
import sys, os, json, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if sys.platform == 'win32': 
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from dotenv import load_dotenv
load_dotenv()

from solvers.tsp_cluster_qaoa_solver import ClusterQAOATSPSolver

SIZES = [5, 10, 25, 50, 100]
results = []

def generate_problem(n, seed=42):
    """Generate reproducible random TSP instance"""
    np.random.seed(seed)
    coords = np.random.rand(n, 2) * 100
    locations = [f"C{i+1}" for i in range(n)]
    dm = [[0 if i==j else round(np.sqrt((coords[i][0]-coords[j][0])**2 + (coords[i][1]-coords[j][1])**2), 2) 
           for j in range(n)] for i in range(n)]
    return locations, dm

print("=" * 70)
print("CLUSTER-QAOA BENCHMARK ON RIGETTI CEPHEUS-1-108Q")
print("=" * 70)
print("Device: Rigetti Cepheus-1-108Q (108 qubits)")
print("Region: us-west-1")
print("Problem sizes:", SIZES)
print("=" * 70)

for n in SIZES:
    print(f"\n{'='*70}")
    print(f"[{n} CITIES] Starting benchmark...")
    print("="*70)
    
    locations, dm = generate_problem(n)
    
    try:
        # Use Rigetti QPU with cluster approach
        # Max cluster = 4 cities = 16 qubits with p=1 = ~4,352 gates (safe)
        # Rigetti has 20,000 gate limit
        # 5 cities (25 qubits, p=2) = 21,239 gates (FAILED)
        # 4 cities (16 qubits, p=1) = ~4,352 gates (SAFE)
        max_cluster = min(4, n)
        
        solver = ClusterQAOATSPSolver(
            device='rigetti',  # Use Rigetti Cepheus-1-108Q
            max_cluster_size=max_cluster, 
            use_local_search=True, 
            verbose=True
        )
        
        print(f"  Solver initialized (max_cluster={max_cluster})")
        print(f"  Submitting to Rigetti QPU...")
        
        start = time.time()
        # Use p=1 (single QAOA layer) to reduce gate count for real QPU
        r = solver.solve_tsp(locations, dm, p=1, shots=100)
        elapsed = time.time() - start
        
        result = {
            "n": n, 
            "distance": r['total_distance'], 
            "raw_distance": r.get('raw_distance', r['total_distance']),
            "n_clusters": r.get('n_clusters', 1),
            "time": elapsed, 
            "status": r.get('status', 'unknown'),
            "solver": "Cluster-QAOA (Rigetti Cepheus-1-108Q)",
            "device": "rigetti",
            "shots": 100,
            "p_layers": 1,
            "max_cluster_size": max_cluster
        }
        results.append(result)
        
        print(f"\n  RESULT for {n} cities:")
        print(f"    Clusters: {r.get('n_clusters', 1)}")
        print(f"    Raw distance: {r.get('raw_distance', 'N/A'):.2f}")
        print(f"    Final distance (after 2-opt): {r['total_distance']:.2f}")
        print(f"    Time: {elapsed:.2f}s")
        print(f"    Status: {r.get('status', 'unknown')}")
        
        # Save intermediate results after each size
        with open(os.path.join(os.path.dirname(__file__), "results_qaoa_rigetti.json"), "w") as f:
            json.dump(results, f, indent=2)
        print(f"  [Saved intermediate results]")
        
    except Exception as e:
        result = {"n": n, "status": "error", "message": str(e), "solver": "Cluster-QAOA (Rigetti)"}
        results.append(result)
        print(f"\n  ERROR for {n} cities: {e}")
        import traceback
        traceback.print_exc()
        
        # Save even on error
        with open(os.path.join(os.path.dirname(__file__), "results_qaoa_rigetti.json"), "w") as f:
            json.dump(results, f, indent=2)

print("\n" + "="*70)
print("BENCHMARK COMPLETE")
print("="*70)

# Final summary
print("\nSUMMARY:")
print(f"{'Cities':>8} | {'Clusters':>8} | {'Raw Dist':>10} | {'Final Dist':>10} | {'Time':>10} | Status")
print("-" * 70)
for r in results:
    if r.get('status') != 'error':
        print(f"{r['n']:>8} | {r.get('n_clusters', 'N/A'):>8} | {r.get('raw_distance', 0):>10.2f} | {r['distance']:>10.2f} | {r['time']:>9.1f}s | {r.get('status', 'ok')}")
    else:
        print(f"{r['n']:>8} | {'N/A':>8} | {'N/A':>10} | {'N/A':>10} | {'N/A':>10} | ERROR: {r.get('message', '')[:30]}")

print("\n[OK] Results saved to results_qaoa_rigetti.json")
