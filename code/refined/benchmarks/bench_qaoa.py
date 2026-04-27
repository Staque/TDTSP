"""Cluster-QAOA benchmark on AWS Braket SV1 simulator."""
import time

from dotenv import load_dotenv

from _common import SIZES, load_instance, save_results

load_dotenv()

from solvers import ClusterQAOATSPSolver


def main():
    print("CLUSTER-QAOA (SV1) BENCHMARK")
    print("=" * 50)

    results = []

    for n in SIZES:
        print(f"\n[{n} cities]...", flush=True)
        inst = load_instance(n)
        locations = inst["locations"]
        dm = inst["distance_matrix"]

        try:
            max_cluster = min(5, n)
            solver = ClusterQAOATSPSolver(
                device="sv1",
                max_cluster_size=max_cluster,
                use_local_search=True,
                verbose=False,
            )
            start = time.time()
            r = solver.solve_tsp(locations, dm, p=2, shots=500)
            elapsed = time.time() - start

            results.append({
                "n": n,
                "distance": r["total_distance"],
                "raw_distance": r.get("raw_distance", r["total_distance"]),
                "n_clusters": r.get("n_clusters", 1),
                "time": elapsed,
                "status": r.get("status", "unknown"),
                "solver": "Cluster-QAOA (SV1)",
            })
            print(
                f"  Clusters: {r.get('n_clusters', 1)}, "
                f"Raw: {r.get('raw_distance', float('nan')):.2f}, "
                f"Final: {r['total_distance']:.2f}, "
                f"Time: {elapsed:.2f}s"
            )
        except Exception as e:
            results.append({"n": n, "status": "error", "message": str(e)})
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()

    out = save_results("results_qaoa.json", results)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
