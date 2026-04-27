"""Cluster-D-Wave benchmark over the standard 5/10/25/50/100-city instances."""
import time

from dotenv import load_dotenv

from _common import SIZES, load_instance, save_results

load_dotenv()

from solvers import ClusterDWaveTSPSolver


def main():
    print("CLUSTER-DWAVE BENCHMARK")
    print("=" * 50)

    results = []

    for n in SIZES:
        print(f"\n[{n} cities]...", flush=True)
        inst = load_instance(n)
        locations = inst["locations"]
        dm = inst["distance_matrix"]

        try:
            max_cluster = min(15, n)
            solver = ClusterDWaveTSPSolver(
                max_cluster_size=max_cluster,
                use_local_search=True,
                time_limit_per_cluster=10,
                verbose=False,
            )
            start = time.time()
            r = solver.solve_tsp(locations, dm)
            elapsed = time.time() - start

            results.append({
                "n": n,
                "distance": r["total_distance"],
                "raw_distance": r.get("raw_distance", r["total_distance"]),
                "time": elapsed,
                "status": r.get("status", "unknown"),
                "solver": "Cluster-D-Wave",
                "n_clusters": r.get("n_clusters", 1),
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

    out = save_results("results_dwave.json", results)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
