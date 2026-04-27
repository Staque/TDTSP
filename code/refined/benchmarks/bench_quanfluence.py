"""Cluster-Quanfluence benchmark over the standard 5/10/25/50/100-city instances."""
import time

from dotenv import load_dotenv

from _common import SIZES, load_instance, save_results

load_dotenv()

from solvers import ClusterQuanfluenceTSPSolver


def main():
    print("CLUSTER-QUANFLUENCE BENCHMARK")
    print("=" * 50)

    results = []

    for n in SIZES:
        print(f"\n[{n} cities]...", flush=True)
        inst = load_instance(n)
        locations = inst["locations"]
        dm = inst["distance_matrix"]

        try:
            max_cluster = min(10, n)
            solver = ClusterQuanfluenceTSPSolver(
                max_cluster_size=max_cluster,
                use_local_search=True,
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
                "solver": "Cluster-Quanfluence",
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

    out = save_results("results_quanfluence.json", results)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
