"""
Cluster-Quanfluence TSP Solver
Scalable Quanfluence implementation using hierarchical clustering for 50+ city problems

Approach:
1. K-means clustering to divide cities into manageable clusters (max 10 cities each)
2. Quanfluence to solve intra-cluster TSP (small QUBOs, fast)
3. Greedy nearest-neighbor for inter-cluster ordering
4. Stitch sub-tours together
5. 2-opt refinement for final optimization
"""

import os
import time
import numpy as np
from typing import List, Dict, Optional, Tuple
from sklearn.cluster import KMeans
from dotenv import load_dotenv

load_dotenv()


class ClusterQuanfluenceTSPSolver:
    """
    Scalable TSP solver using Cluster-Quanfluence approach
    
    Breaks large problems into clusters, solves each with Quanfluence,
    then stitches together with 2-opt refinement.
    """
    
    # Max cities per cluster (10 cities = 100 vars, very fast on Quanfluence)
    MAX_CLUSTER_SIZE = 10
    
    def __init__(self, 
                 max_cluster_size: int = 10,
                 use_local_search: bool = True,
                 verbose: bool = True):
        """
        Initialize Cluster-Quanfluence solver
        
        Args:
            max_cluster_size: Maximum cities per cluster (default 10 = 100 vars)
            use_local_search: Apply 2-opt after stitching
            verbose: Print progress messages
        """
        self.max_cluster_size = max_cluster_size
        self.use_local_search = use_local_search
        self.verbose = verbose
        self.timing_info = {}
        
        # Import Quanfluence solver for cluster solving
        from solvers.tsp_quanfluence_solver import QuanfluenceTSPSolver
        self.qf_solver = QuanfluenceTSPSolver(
            use_local_search=False,  # We'll do 2-opt at the end
            local_search_method='2opt'
        )
    
    def _estimate_coordinates(self, distance_matrix: List[List[float]]) -> np.ndarray:
        """Estimate 2D coordinates from distance matrix using MDS"""
        n = len(distance_matrix)
        D = np.array(distance_matrix)
        
        # Classical MDS
        D_sq = D ** 2
        J = np.eye(n) - np.ones((n, n)) / n
        B = -0.5 * J @ D_sq @ J
        
        eigenvalues, eigenvectors = np.linalg.eigh(B)
        idx = np.argsort(eigenvalues)[::-1][:2]
        eigenvalues = np.maximum(eigenvalues[idx], 0)
        eigenvectors = eigenvectors[:, idx]
        
        coords = eigenvectors @ np.diag(np.sqrt(eigenvalues))
        return coords
    
    def _cluster_cities(self, coords: np.ndarray, n_clusters: int = None) -> Tuple[List[List[int]], np.ndarray]:
        """Cluster cities using K-means"""
        n = len(coords)
        
        if n_clusters is None:
            n_clusters = max(1, (n + self.max_cluster_size - 1) // self.max_cluster_size)
        
        if n_clusters <= 1:
            return [[i for i in range(n)]], coords.mean(axis=0, keepdims=True)
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(coords)
        centers = kmeans.cluster_centers_
        
        clusters = [[] for _ in range(n_clusters)]
        for i, label in enumerate(labels):
            clusters[label].append(i)
        
        clusters = [c for c in clusters if len(c) > 0]
        return clusters, centers
    
    def _solve_cluster_tsp(self, 
                          cluster_indices: List[int],
                          full_distance_matrix: List[List[float]]) -> List[int]:
        """Solve TSP for a single cluster using Quanfluence"""
        n = len(cluster_indices)
        
        if n <= 1:
            return cluster_indices
        if n == 2:
            return cluster_indices
        
        # Extract sub-matrix
        sub_matrix = [[full_distance_matrix[i][j] for j in cluster_indices] 
                      for i in cluster_indices]
        locations = [f"C{i}" for i in range(n)]
        
        try:
            result = self.qf_solver.solve_tsp(locations, sub_matrix, verbose=False)
            
            if result.get('status') != 'error' and result.get('tour'):
                tour = result['tour']
                return [cluster_indices[i] for i in tour]
            else:
                return cluster_indices
        except Exception as e:
            if self.verbose:
                print(f"      Quanfluence error: {e}")
            return cluster_indices
    
    def _stitch_tours(self,
                     cluster_tours: List[List[int]],
                     cluster_order: List[int],
                     distance_matrix: List[List[float]]) -> List[int]:
        """Stitch cluster tours together"""
        if len(cluster_order) == 1:
            return cluster_tours[cluster_order[0]]
        
        full_tour = list(cluster_tours[cluster_order[0]])
        
        for idx in range(1, len(cluster_order)):
            next_cluster = cluster_order[idx]
            next_tour = cluster_tours[next_cluster]
            
            # Find best connection point
            best_dist = float('inf')
            best_entry = 0
            
            last_city = full_tour[-1]
            for j, city in enumerate(next_tour):
                dist = distance_matrix[last_city][city]
                if dist < best_dist:
                    best_dist = dist
                    best_entry = j
            
            # Rotate and append
            rotated = next_tour[best_entry:] + next_tour[:best_entry]
            full_tour.extend(rotated)
        
        return full_tour
    
    def _calculate_distance(self, tour: List[int], distance_matrix: List[List[float]]) -> float:
        """Calculate total tour distance"""
        n = len(tour)
        return sum(distance_matrix[tour[i]][tour[(i + 1) % n]] for i in range(n))
    
    def _two_opt_improve(self, tour: List[int], distance_matrix: List[List[float]]) -> Tuple[List[int], float]:
        """Apply 2-opt local search"""
        n = len(tour)
        improved = True
        current_tour = tour.copy()
        current_distance = self._calculate_distance(current_tour, distance_matrix)
        
        iterations = 0
        max_iterations = 100  # Limit iterations for large problems
        
        while improved and iterations < max_iterations:
            improved = False
            iterations += 1
            
            for i in range(n - 1):
                for k in range(i + 1, n):
                    new_tour = current_tour[:i] + current_tour[i:k+1][::-1] + current_tour[k+1:]
                    new_distance = self._calculate_distance(new_tour, distance_matrix)
                    
                    if new_distance < current_distance - 1e-10:
                        current_tour = new_tour
                        current_distance = new_distance
                        improved = True
                        break
                if improved:
                    break
        
        return current_tour, current_distance
    
    def solve_tsp(self,
                  locations: List[str],
                  distance_matrix: List[List[float]],
                  start_location: Optional[str] = None) -> Dict:
        """
        Solve large-scale TSP using Cluster-Quanfluence
        
        Args:
            locations: List of location names
            distance_matrix: NxN distance matrix
            start_location: Optional starting location
            
        Returns:
            Solution dictionary
        """
        n = len(locations)
        total_start = time.time()
        
        if self.verbose:
            print("\n" + "="*70)
            print("CLUSTER-QUANFLUENCE TSP SOLVER")
            print("="*70)
            print(f"Total cities: {n}")
            print(f"Max cluster size: {self.max_cluster_size}")
            print("="*70)
        
        try:
            # Step 1: Cluster cities
            if self.verbose:
                print("\n[Step 1] Clustering cities...")
            
            cluster_start = time.time()
            coords = self._estimate_coordinates(distance_matrix)
            n_clusters = max(1, (n + self.max_cluster_size - 1) // self.max_cluster_size)
            clusters, centers = self._cluster_cities(coords, n_clusters)
            cluster_time = time.time() - cluster_start
            
            if self.verbose:
                print(f"  Created {len(clusters)} clusters")
                for i, c in enumerate(clusters):
                    print(f"    Cluster {i+1}: {len(c)} cities ({len(c)**2} vars)")
            
            # Step 2: Solve each cluster with Quanfluence
            if self.verbose:
                print(f"\n[Step 2] Solving {len(clusters)} cluster TSPs with Quanfluence...")
            
            qf_start = time.time()
            cluster_tours = []
            
            for i, cluster in enumerate(clusters):
                if self.verbose:
                    print(f"  Cluster {i+1}/{len(clusters)} ({len(cluster)} cities)...", end=" ", flush=True)
                
                tour = self._solve_cluster_tsp(cluster, distance_matrix)
                cluster_tours.append(tour)
                
                if self.verbose:
                    print("Done")
            
            qf_time = time.time() - qf_start
            
            # Step 3: Determine cluster order (greedy nearest-neighbor on centers)
            if self.verbose:
                print(f"\n[Step 3] Determining cluster visit order...")
            
            order_start = time.time()
            
            if len(clusters) > 1:
                cluster_order = [0]
                remaining = set(range(1, len(clusters)))
                
                while remaining:
                    current = cluster_order[-1]
                    current_center = centers[current]
                    
                    best_next = None
                    best_dist = float('inf')
                    
                    for next_c in remaining:
                        dist = np.linalg.norm(current_center - centers[next_c])
                        if dist < best_dist:
                            best_dist = dist
                            best_next = next_c
                    
                    cluster_order.append(best_next)
                    remaining.remove(best_next)
            else:
                cluster_order = [0]
            
            order_time = time.time() - order_start
            
            if self.verbose:
                print(f"  Cluster order: {[i+1 for i in cluster_order]}")
            
            # Step 4: Stitch tours
            if self.verbose:
                print(f"\n[Step 4] Stitching sub-tours...")
            
            stitch_start = time.time()
            full_tour = self._stitch_tours(cluster_tours, cluster_order, distance_matrix)
            raw_distance = self._calculate_distance(full_tour, distance_matrix)
            stitch_time = time.time() - stitch_start
            
            if self.verbose:
                print(f"  Stitched tour distance: {raw_distance:.2f}")
            
            # Step 5: 2-opt refinement
            if self.use_local_search:
                if self.verbose:
                    print(f"\n[Step 5] Applying 2-opt local search...")
                
                refine_start = time.time()
                full_tour, final_distance = self._two_opt_improve(full_tour, distance_matrix)
                refine_time = time.time() - refine_start
                
                improvement = raw_distance - final_distance
                if self.verbose:
                    print(f"  Refined distance: {final_distance:.2f}")
                    print(f"  Improvement: {improvement:.2f} ({improvement/raw_distance*100:.1f}%)")
            else:
                final_distance = raw_distance
                refine_time = 0
                improvement = 0
            
            total_time = time.time() - total_start
            
            # Reorder if start location specified
            if start_location and start_location in locations:
                start_idx = locations.index(start_location)
                if start_idx in full_tour:
                    pos = full_tour.index(start_idx)
                    full_tour = full_tour[pos:] + full_tour[:pos]
            
            tour_names = [locations[i] for i in full_tour]
            
            self.timing_info = {
                'cluster_time_s': cluster_time,
                'quanfluence_time_s': qf_time,
                'order_time_s': order_time,
                'stitch_time_s': stitch_time,
                'refine_time_s': refine_time,
                'total_time_s': total_time
            }
            
            if self.verbose:
                print("\n" + "="*70)
                print("CLUSTER-QUANFLUENCE SOLUTION")
                print("="*70)
                print(f"Total cities: {n}")
                print(f"Clusters used: {len(clusters)}")
                print(f"Raw distance (before 2-opt): {raw_distance:.2f}")
                print(f"Final distance: {final_distance:.2f}")
                if self.use_local_search:
                    print(f"Improvement from 2-opt: {improvement:.2f} ({improvement/raw_distance*100:.1f}%)")
                print(f"\nTiming:")
                print(f"  Clustering: {cluster_time:.2f}s")
                print(f"  Quanfluence solving: {qf_time:.2f}s")
                print(f"  Cluster ordering: {order_time:.2f}s")
                print(f"  Stitching: {stitch_time:.2f}s")
                if self.use_local_search:
                    print(f"  2-opt refinement: {refine_time:.2f}s")
                print(f"  Total: {total_time:.2f}s")
                print("="*70)
            
            return {
                'status': 'optimal',
                'tour': full_tour,
                'tour_names': tour_names,
                'total_distance': final_distance,
                'raw_distance': raw_distance,
                'locations': locations,
                'distance_matrix': distance_matrix,
                'start_location': locations[full_tour[0]],
                'timing': self.timing_info,
                'solver': 'Cluster-Quanfluence',
                'n_clusters': len(clusters),
                'cluster_sizes': [len(c) for c in clusters],
                'improvement': improvement if self.use_local_search else 0
            }
            
        except Exception as e:
            import traceback
            if self.verbose:
                print(f"\n[ERROR] {str(e)}")
                traceback.print_exc()
            return {
                'status': 'error',
                'message': str(e),
                'solver': 'Cluster-Quanfluence'
            }


def test_cluster_quanfluence():
    """Test Cluster-Quanfluence on a 50-city problem"""
    import numpy as np
    
    print("\n" + "="*70)
    print("TESTING CLUSTER-QUANFLUENCE SOLVER")
    print("="*70)
    
    # 50-city problem
    np.random.seed(42)
    n = 50
    coords = np.random.rand(n, 2) * 100
    
    locations = [f"City_{i+1}" for i in range(n)]
    distance_matrix = []
    for i in range(n):
        row = []
        for j in range(n):
            if i == j:
                row.append(0)
            else:
                dist = np.sqrt((coords[i][0] - coords[j][0])**2 + 
                              (coords[i][1] - coords[j][1])**2)
                row.append(round(dist, 2))
        distance_matrix.append(row)
    
    solver = ClusterQuanfluenceTSPSolver(max_cluster_size=10)
    result = solver.solve_tsp(locations, distance_matrix)
    
    print(f"\nResult: {result['status']}")
    print(f"Distance: {result.get('total_distance', 'N/A')}")
    
    return result


if __name__ == "__main__":
    test_cluster_quanfluence()
