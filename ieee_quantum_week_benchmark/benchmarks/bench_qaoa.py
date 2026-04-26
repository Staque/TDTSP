"""Cluster-QAOA benchmark for paper - Using Cluster Approach for ALL sizes"""
import sys, os, json, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if sys.platform == 'win32': sys.stdout.reconfigure(encoding='utf-8', errors='replace')
from dotenv import load_dotenv
load_dotenv()
from solvers.tsp_cluster_qaoa_solver import ClusterQAOATSPSolver

SIZES = [5, 10, 25, 50, 100]
results = []

def generate_problem(n, seed=42):
    np.random.seed(seed)
    coords = np.random.rand(n, 2) * 100
    locations = [f"C{i+1}" for i in range(n)]
    dm = [[0 if i==j else round(np.sqrt((coords[i][0]-coords[j][0])**2 + (coords[i][1]-coords[j][1])**2), 2) for j in range(n)] for i in range(n)]
    return locations, dm

print("CLUSTER-QAOA (SV1) BENCHMARK")
print("="*50)

for n in SIZES:
    print(f"\n[{n} cities]...", flush=True)
    locations, dm = generate_problem(n)
    try:
        # Use Cluster-QAOA for ALL sizes
        # Max cluster size = 5 (25 qubits per cluster, fits on SV1)
        max_cluster = min(5, n)
        solver = ClusterQAOATSPSolver(device='sv1', max_cluster_size=max_cluster, use_local_search=True, verbose=False)
        start = time.time()
        r = solver.solve_tsp(locations, dm, p=2, shots=500)
        elapsed = time.time() - start
        
        results.append({
            "n": n, 
            "distance": r['total_distance'], 
            "raw_distance": r.get('raw_distance', r['total_distance']),
            "n_clusters": r.get('n_clusters', 1),
            "time": elapsed, 
            "status": r.get('status', 'unknown'),
            "solver": "Cluster-QAOA (SV1)"
        })
        print(f"  Clusters: {r.get('n_clusters', 1)}, Raw: {r.get('raw_distance', 'N/A'):.2f}, Final: {r['total_distance']:.2f}, Time: {elapsed:.2f}s")
    except Exception as e:
        results.append({"n": n, "status": "error", "message": str(e)})
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()

with open(os.path.join(os.path.dirname(__file__), "results_qaoa.json"), "w") as f:
    json.dump(results, f, indent=2)
print("\n[OK] Saved to results_qaoa.json")
