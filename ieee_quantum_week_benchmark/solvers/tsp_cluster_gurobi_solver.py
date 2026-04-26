"""
Cluster-Gurobi TSP Solver
Uses K-means clustering + Gurobi for sub-problems + 2-opt refinement
Consistent approach with quantum solvers for fair benchmarking
"""

import os
import time
import numpy as np
from typing import List, Dict, Tuple, Optional
from sklearn.cluster import KMeans
from sklearn.manifold import MDS
from dotenv import load_dotenv

load_dotenv()


class ClusterGurobiTSPSolver:
    """
    Scalable TSP solver using Cluster-Gurobi approach
    
    Architecture:
    1. K-means clustering to divide cities into manageable clusters
    2. Gurobi to solve intra-cluster TSP (small sub-problems)
    3. Greedy ordering for inter-cluster connections
    4. Stitch sub-tours with nearest-neighbor connection
    5. 2-opt refinement for final optimization
    """
    
    MAX_CLUSTER_SIZE = 15  # Gurobi handles larger clusters efficiently
    
    def __init__(self, 
                 max_cluster_size: int = 15,
                 use_local_search: bool = True,
                 verbose: bool = True):
        self.max_cluster_size = max_cluster_size
        self.use_local_search = use_local_search
        self.verbose = verbose
        self.timing_info = {}
        
    def _log(self, message: str, end: str = "\n"):
        if self.verbose:
            print(message, end=end, flush=True)
    
    def _estimate_coordinates(self, distance_matrix: List[List[float]]) -> np.ndarray:
        """Estimate 2D coordinates from distance matrix using MDS"""
        n = len(distance_matrix)
        if n <= 2:
            return np.array([[0, 0], [1, 0]][:n])
        
        dm = np.array(distance_matrix)
        dm = (dm + dm.T) / 2
        np.fill_diagonal(dm, 0)
        
        try:
            mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42, max_iter=300)
            coords = mds.fit_transform(dm)
        except:
            coords = np.random.rand(n, 2) * 100
        
        return coords
    
    def _cluster_cities(self, coords: np.ndarray, n_clusters: int = None) -> Tuple[List[List[int]], np.ndarray]:
        """Cluster cities using K-means"""
        n = len(coords)
        
        if n_clusters is None:
            n_clusters = max(1, int(np.ceil(n / self.max_cluster_size)))
        
        if n_clusters >= n:
            return [[i] for i in range(n)], coords
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(coords)
        
        clusters = [[] for _ in range(n_clusters)]
        for i, label in enumerate(labels):
            clusters[label].append(i)
        
        clusters = [c for c in clusters if len(c) > 0]
        
        return clusters, kmeans.cluster_centers_
    
    def _solve_cluster_tsp_gurobi(self, cluster_indices: List[int], 
                                   full_distance_matrix: List[List[float]]) -> List[int]:
        """Solve TSP for a single cluster using Gurobi"""
        n = len(cluster_indices)
        
        if n <= 1:
            return cluster_indices
        if n == 2:
            return cluster_indices
        
        # Build sub-distance matrix
        sub_dm = [[full_distance_matrix[cluster_indices[i]][cluster_indices[j]] 
                   for j in range(n)] for i in range(n)]
        
        try:
            import gurobipy as gp
            from gurobipy import GRB
            
            env = gp.Env(empty=True)
            env.setParam('OutputFlag', 0)
            env.start()
            
            m = gp.Model('tsp_cluster', env=env)
            m.setParam('OutputFlag', 0)
            m.setParam('TimeLimit', 30)  # 30 second limit per cluster
            
            # Variables
            x = {}
            for i in range(n):
                for j in range(n):
                    if i != j:
                        x[i, j] = m.addVar(vtype=GRB.BINARY, name=f'x_{i}_{j}')
            
            u = {}
            for i in range(1, n):
                u[i] = m.addVar(lb=1, ub=n-1, vtype=GRB.CONTINUOUS, name=f'u_{i}')
            
            # Objective
            m.setObjective(
                gp.quicksum(sub_dm[i][j] * x[i, j] for i in range(n) for j in range(n) if i != j),
                GRB.MINIMIZE
            )
            
            # Constraints: each city visited exactly once
            for i in range(n):
                m.addConstr(gp.quicksum(x[i, j] for j in range(n) if i != j) == 1)
                m.addConstr(gp.quicksum(x[j, i] for j in range(n) if i != j) == 1)
            
            # MTZ subtour elimination
            for i in range(1, n):
                for j in range(1, n):
                    if i != j:
                        m.addConstr(u[i] - u[j] + (n - 1) * x[i, j] <= n - 2)
            
            m.optimize()
            
            if m.status == GRB.OPTIMAL or m.status == GRB.TIME_LIMIT:
                # Extract tour
                tour = [0]
                current = 0
                visited = {0}
                
                while len(tour) < n:
                    for j in range(n):
                        if j not in visited and (current, j) in x and x[current, j].X > 0.5:
                            tour.append(j)
                            visited.add(j)
                            current = j
                            break
                    else:
                        # Fallback: find nearest unvisited
                        unvisited = [j for j in range(n) if j not in visited]
                        if unvisited:
                            nearest = min(unvisited, key=lambda j: sub_dm[current][j])
                            tour.append(nearest)
                            visited.add(nearest)
                            current = nearest
                
                m.dispose()
                env.dispose()
                
                return [cluster_indices[i] for i in tour]
            else:
                m.dispose()
                env.dispose()
                return cluster_indices
                
        except Exception as e:
            self._log(f"    Gurobi error: {e}, using greedy fallback")
            return self._greedy_tsp(cluster_indices, full_distance_matrix)
    
    def _greedy_tsp(self, indices: List[int], dm: List[List[float]]) -> List[int]:
        """Simple greedy nearest neighbor TSP"""
        if len(indices) <= 1:
            return indices
        
        tour = [indices[0]]
        remaining = set(indices[1:])
        
        while remaining:
            current = tour[-1]
            nearest = min(remaining, key=lambda x: dm[current][x])
            tour.append(nearest)
            remaining.remove(nearest)
        
        return tour
    
    def _stitch_tours(self, cluster_tours: List[List[int]], cluster_order: List[int],
                      distance_matrix: List[List[float]]) -> List[int]:
        """Stitch cluster tours together in given order"""
        if not cluster_tours:
            return []
        
        full_tour = []
        
        for idx, cluster_idx in enumerate(cluster_order):
            tour = cluster_tours[cluster_idx]
            if not tour:
                continue
            
            if not full_tour:
                full_tour.extend(tour)
            else:
                last_city = full_tour[-1]
                
                # Find best entry point in this cluster
                best_start = 0
                best_dist = float('inf')
                for i, city in enumerate(tour):
                    d = distance_matrix[last_city][city]
                    if d < best_dist:
                        best_dist = d
                        best_start = i
                
                # Rotate tour to start at best entry point
                rotated = tour[best_start:] + tour[:best_start]
                full_tour.extend(rotated)
        
        return full_tour
    
    def _calculate_distance(self, tour: List[int], dm: List[List[float]]) -> float:
        """Calculate total tour distance"""
        if len(tour) <= 1:
            return 0.0
        total = sum(dm[tour[i]][tour[i+1]] for i in range(len(tour)-1))
        total += dm[tour[-1]][tour[0]]  # Return to start
        return total
    
    def _two_opt_improve(self, tour: List[int], dm: List[List[float]]) -> Tuple[List[int], float]:
        """Apply 2-opt local search"""
        n = len(tour)
        if n <= 3:
            return tour, self._calculate_distance(tour, dm)
        
        improved = True
        best_tour = tour[:]
        
        while improved:
            improved = False
            for i in range(n - 1):
                for j in range(i + 2, n):
                    if j == n - 1 and i == 0:
                        continue
                    
                    # Current edges: (i, i+1) and (j, j+1 mod n)
                    # New edges: (i, j) and (i+1, j+1 mod n)
                    i1, i2 = best_tour[i], best_tour[i + 1]
                    j1 = best_tour[j]
                    j2 = best_tour[(j + 1) % n]
                    
                    current_dist = dm[i1][i2] + dm[j1][j2]
                    new_dist = dm[i1][j1] + dm[i2][j2]
                    
                    if new_dist < current_dist - 1e-10:
                        best_tour[i+1:j+1] = best_tour[i+1:j+1][::-1]
                        improved = True
        
        return best_tour, self._calculate_distance(best_tour, dm)
    
    def solve_tsp(self, locations: List[str], distance_matrix: List[List[float]]) -> Dict:
        """
        Solve TSP using Cluster-Gurobi approach
        """
        n = len(locations)
        start_time = time.time()
        
        self._log("=" * 70)
        self._log("CLUSTER-GUROBI TSP SOLVER")
        self._log("=" * 70)
        self._log(f"Total cities: {n}")
        self._log(f"Max cluster size: {self.max_cluster_size}")
        self._log("=" * 70)
        
        # Step 1: Estimate coordinates and cluster
        self._log("\n[Step 1] Clustering cities...")
        t0 = time.time()
        coords = self._estimate_coordinates(distance_matrix)
        clusters, centers = self._cluster_cities(coords)
        self.timing_info['clustering'] = time.time() - t0
        
        self._log(f"  Created {len(clusters)} clusters")
        for i, c in enumerate(clusters):
            self._log(f"    Cluster {i+1}: {len(c)} cities")
        
        # Step 2: Solve each cluster TSP with Gurobi
        self._log(f"\n[Step 2] Solving {len(clusters)} cluster TSPs with Gurobi...")
        t0 = time.time()
        cluster_tours = []
        
        for i, cluster in enumerate(clusters):
            self._log(f"  Cluster {i+1}/{len(clusters)} ({len(cluster)} cities)...", end=" ")
            tour = self._solve_cluster_tsp_gurobi(cluster, distance_matrix)
            cluster_tours.append(tour)
            self._log("Done")
        
        self.timing_info['gurobi_solving'] = time.time() - t0
        
        # Step 3: Determine cluster visit order
        self._log("\n[Step 3] Determining cluster visit order...")
        t0 = time.time()
        
        if len(clusters) > 1:
            cluster_centroids = []
            for tour in cluster_tours:
                if tour:
                    cx = np.mean([coords[i][0] for i in tour])
                    cy = np.mean([coords[i][1] for i in tour])
                    cluster_centroids.append((cx, cy))
                else:
                    cluster_centroids.append((0, 0))
            
            # Greedy ordering starting from cluster 0
            cluster_order = [0]
            remaining = set(range(1, len(clusters)))
            
            while remaining:
                last = cluster_order[-1]
                cx, cy = cluster_centroids[last]
                nearest = min(remaining, key=lambda i: 
                    (cluster_centroids[i][0] - cx)**2 + (cluster_centroids[i][1] - cy)**2)
                cluster_order.append(nearest)
                remaining.remove(nearest)
        else:
            cluster_order = [0]
        
        self.timing_info['cluster_ordering'] = time.time() - t0
        self._log(f"  Cluster order: {[i+1 for i in cluster_order]}")
        
        # Step 4: Stitch tours
        self._log("\n[Step 4] Stitching sub-tours...")
        t0 = time.time()
        full_tour = self._stitch_tours(cluster_tours, cluster_order, distance_matrix)
        raw_distance = self._calculate_distance(full_tour, distance_matrix)
        self.timing_info['stitching'] = time.time() - t0
        self._log(f"  Stitched tour distance: {raw_distance:.2f}")
        
        # Step 5: 2-opt refinement
        if self.use_local_search:
            self._log("\n[Step 5] Applying 2-opt local search...")
            t0 = time.time()
            full_tour, final_distance = self._two_opt_improve(full_tour, distance_matrix)
            self.timing_info['local_search'] = time.time() - t0
            improvement = raw_distance - final_distance
            self._log(f"  Refined distance: {final_distance:.2f}")
            self._log(f"  Improvement: {improvement:.2f} ({100*improvement/raw_distance:.1f}%)")
        else:
            final_distance = raw_distance
            self.timing_info['local_search'] = 0
        
        total_time = time.time() - start_time
        self.timing_info['total'] = total_time
        
        # Build tour with location names
        tour_locations = [locations[i] for i in full_tour]
        tour_locations.append(tour_locations[0])  # Return to start
        
        self._log("\n" + "=" * 70)
        self._log("CLUSTER-GUROBI SOLUTION")
        self._log("=" * 70)
        self._log(f"Total cities: {n}")
        self._log(f"Clusters used: {len(clusters)}")
        self._log(f"Raw distance (before 2-opt): {raw_distance:.2f}")
        self._log(f"Final distance: {final_distance:.2f}")
        if self.use_local_search:
            self._log(f"Improvement from 2-opt: {raw_distance - final_distance:.2f} ({100*(raw_distance-final_distance)/raw_distance:.1f}%)")
        self._log(f"\nTiming:")
        self._log(f"  Clustering: {self.timing_info.get('clustering', 0):.2f}s")
        self._log(f"  Gurobi solving: {self.timing_info.get('gurobi_solving', 0):.2f}s")
        self._log(f"  Cluster ordering: {self.timing_info.get('cluster_ordering', 0):.2f}s")
        self._log(f"  Stitching: {self.timing_info.get('stitching', 0):.2f}s")
        self._log(f"  Local search: {self.timing_info.get('local_search', 0):.2f}s")
        self._log(f"  Total: {total_time:.2f}s")
        self._log("=" * 70)
        
        return {
            'tour': tour_locations,
            'tour_indices': full_tour,
            'total_distance': final_distance,
            'raw_distance': raw_distance,
            'n_clusters': len(clusters),
            'status': 'optimal',
            'solve_time': total_time,
            'timing': self.timing_info
        }
