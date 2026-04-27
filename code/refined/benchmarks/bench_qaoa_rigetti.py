"""Cluster-QAOA benchmark on Rigetti Cepheus-1-108Q QPU.

Uses max cluster size 4 (16 qubits) and p=1 to stay below the device's
20,000-gate compilation limit. Intermediate results are flushed to disk
after every problem size.
"""
import time

from dotenv import load_dotenv

from _common import SIZES, load_instance, save_results

load_dotenv()

from solvers import ClusterQAOATSPSolver


def main():
    print("=" * 70)
    print("CLUSTER-QAOA BENCHMARK ON RIGETTI CEPHEUS-1-108Q")
    print("=" * 70)
    print("Device: Rigetti Cepheus-1-108Q (108 qubits)")
    print("Region: us-west-1")
    print("Problem sizes:", SIZES)
    print("=" * 70)

    results = []

    for n in SIZES:
        print(f"\n{'=' * 70}")
        print(f"[{n} CITIES] Starting benchmark...")
        print("=" * 70)

        inst = load_instance(n)
        locations = inst["locations"]
        dm = inst["distance_matrix"]

        try:
            max_cluster = min(4, n)
            solver = ClusterQAOATSPSolver(
                device="rigetti",
                max_cluster_size=max_cluster,
                use_local_search=True,
                verbose=True,
            )

            print(f"  Solver initialized (max_cluster={max_cluster})")
            print("  Submitting to Rigetti QPU...")

            start = time.time()
            r = solver.solve_tsp(locations, dm, p=1, shots=100)
            elapsed = time.time() - start

            results.append({
                "n": n,
                "distance": r["total_distance"],
                "raw_distance": r.get("raw_distance", r["total_distance"]),
                "n_clusters": r.get("n_clusters", 1),
                "time": elapsed,
                "status": r.get("status", "unknown"),
                "solver": "Cluster-QAOA (Rigetti Cepheus-1-108Q)",
                "device": "rigetti",
                "shots": 100,
                "p_layers": 1,
                "max_cluster_size": max_cluster,
            })

            print(f"\n  RESULT for {n} cities:")
            print(f"    Clusters: {r.get('n_clusters', 1)}")
            print(f"    Raw distance: {r.get('raw_distance', float('nan')):.2f}")
            print(f"    Final distance (after 2-opt): {r['total_distance']:.2f}")
            print(f"    Time: {elapsed:.2f}s")
            print(f"    Status: {r.get('status', 'unknown')}")

        except Exception as e:
            results.append({
                "n": n,
                "status": "error",
                "message": str(e),
                "solver": "Cluster-QAOA (Rigetti)",
            })
            print(f"\n  ERROR for {n} cities: {e}")
            import traceback
            traceback.print_exc()

        save_results("results_qaoa_rigetti.json", results)
        print("  [Saved intermediate results]")

    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)

    print("\nSUMMARY:")
    print(f"{'Cities':>8} | {'Clusters':>8} | {'Raw Dist':>10} | "
          f"{'Final Dist':>10} | {'Time':>10} | Status")
    print("-" * 70)
    for r in results:
        if r.get("status") != "error":
            print(
                f"{r['n']:>8} | {r.get('n_clusters', 'N/A'):>8} | "
                f"{r.get('raw_distance', 0):>10.2f} | {r['distance']:>10.2f} | "
                f"{r['time']:>9.1f}s | {r.get('status', 'ok')}"
            )
        else:
            print(
                f"{r['n']:>8} | {'N/A':>8} | {'N/A':>10} | {'N/A':>10} | "
                f"{'N/A':>10} | ERROR: {r.get('message', '')[:30]}"
            )


if __name__ == "__main__":
    main()
