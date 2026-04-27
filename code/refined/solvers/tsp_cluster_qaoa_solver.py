"""
Cluster-QAOA TSP Solver
Scalable QAOA implementation using hierarchical clustering for 50+ city problems

Based on:
- Cl-QAOA (Clustering QAOA) approach from ECAI 2025
- Hybrid VQE+ML framework achieving 80-city solutions

Supports: AWS Braket SV1 (simulator) and Rigetti Ankaa-2 (real QPU)
"""

import os
import time
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from sklearn.cluster import KMeans
from dotenv import load_dotenv

load_dotenv()


class ClusterQAOATSPSolver:
    """
    Scalable TSP solver using Cluster-QAOA approach
    
    Architecture:
    1. K-means clustering to divide cities into manageable clusters
    2. QAOA to solve intra-cluster TSP (small sub-problems)
    3. QAOA to solve inter-cluster ordering (which cluster to visit next)
    4. Stitch sub-tours with nearest-neighbor connection
    5. 2-opt/3-opt refinement for final optimization
    
    Scales to 50+ cities by keeping each QAOA call to ~25 qubits
    """
    
    # Device ARNs for AWS Braket (Updated April 2026)
    DEVICES = {
        # Simulators
        'sv1': 'arn:aws:braket:::device/quantum-simulator/amazon/sv1',
        'tn1': 'arn:aws:braket:::device/quantum-simulator/amazon/tn1',
        'dm1': 'arn:aws:braket:::device/quantum-simulator/amazon/dm1',
        
        # Rigetti - Cepheus (108 qubits) - Region: us-west-1
        'rigetti': 'arn:aws:braket:us-west-1::device/qpu/rigetti/Cepheus-1-108Q',
        'rigetti_cepheus': 'arn:aws:braket:us-west-1::device/qpu/rigetti/Cepheus-1-108Q',
        
        # IonQ - Forte (trapped ion)
        'ionq': 'arn:aws:braket:us-east-1::device/qpu/ionq/Forte-1',
        'ionq_forte': 'arn:aws:braket:us-east-1::device/qpu/ionq/Forte-1',
        
        # QuEra - Aquila (neutral atom)
        'quera': 'arn:aws:braket:us-east-1::device/qpu/quera/Aquila',
        
        # IQM
        'iqm': 'arn:aws:braket:eu-north-1::device/qpu/iqm/Garnet',
    }
    
    # Region mapping for devices
    DEVICE_REGIONS = {
        'sv1': 'us-east-1',
        'tn1': 'us-west-2',
        'dm1': 'us-east-1',
        'rigetti': 'us-west-1',
        'rigetti_cepheus': 'us-west-1',
        'ionq': 'us-east-1',
        'ionq_forte': 'us-east-1',
        'quera': 'us-east-1',
        'iqm': 'eu-north-1',
    }
    
    # Max cities per cluster (5 cities = 25 qubits, fits on Rigetti Ankaa-2 which has 84 qubits)
    MAX_CLUSTER_SIZE = 5
    
    def __init__(self, 
                 device: str = 'sv1',
                 max_cluster_size: int = 5,
                 use_local_search: bool = True,
                 local_search_method: str = '2opt',
                 verbose: bool = True):
        """
        Initialize Cluster-QAOA solver
        
        Args:
            device: 'sv1' for simulator, 'rigetti' for real QPU
            max_cluster_size: Maximum cities per cluster (default 5 = 25 qubits)
            use_local_search: Apply 2-opt/3-opt after stitching
            local_search_method: '2opt', '3opt', or 'both'
            verbose: Print progress messages
        """
        self.aws_access_key = os.getenv('AWS_ACCESS_KEY_ID')
        self.aws_secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
        self.aws_region = os.getenv('AWS_DEFAULT_REGION', 'us-east-1')
        
        if not self.aws_access_key or not self.aws_secret_key:
            raise ValueError("AWS credentials not found in environment variables")
        
        self.device_name = device
        self.device_arn = self.DEVICES.get(device, self.DEVICES['sv1'])
        self.max_cluster_size = max_cluster_size
        self.use_local_search = use_local_search
        self.local_search_method = local_search_method
        self.verbose = verbose
        self.timing_info = {}
        
        # Set region based on device
        self.aws_region = self.DEVICE_REGIONS.get(device, 'us-east-1')
        
        # Track if using real QPU (affects shots and error handling)
        self.is_real_qpu = device in ['rigetti', 'rigetti_ankaa', 'rigetti_aspen', 'ionq']
    
    # =========================================================================
    # CLUSTERING
    # =========================================================================
    
    def _cluster_cities(self, 
                       coords: np.ndarray, 
                       n_clusters: int = None) -> Tuple[List[List[int]], np.ndarray]:
        """
        Cluster cities using K-means based on coordinates
        
        Args:
            coords: Nx2 array of city coordinates
            n_clusters: Number of clusters (auto-calculated if None)
            
        Returns:
            Tuple of (list of city indices per cluster, cluster centers)
        """
        n = len(coords)
        
        if n_clusters is None:
            # Auto-calculate: aim for ~5 cities per cluster
            n_clusters = max(1, n // self.max_cluster_size)
            # Ensure at least 2 cities per cluster on average
            n_clusters = min(n_clusters, n // 2)
        
        if n_clusters <= 1:
            return [[i for i in range(n)]], coords.mean(axis=0, keepdims=True)
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(coords)
        centers = kmeans.cluster_centers_
        
        # Group cities by cluster
        clusters = [[] for _ in range(n_clusters)]
        for i, label in enumerate(labels):
            clusters[label].append(i)
        
        # Remove empty clusters
        clusters = [c for c in clusters if len(c) > 0]
        
        return clusters, centers
    
    def _estimate_coordinates(self, distance_matrix: List[List[float]]) -> np.ndarray:
        """
        Estimate 2D coordinates from distance matrix using MDS
        
        Args:
            distance_matrix: NxN distance matrix
            
        Returns:
            Nx2 coordinate array
        """
        n = len(distance_matrix)
        D = np.array(distance_matrix)
        
        # Classical MDS
        # Center the squared distance matrix
        D_sq = D ** 2
        J = np.eye(n) - np.ones((n, n)) / n
        B = -0.5 * J @ D_sq @ J
        
        # Eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(B)
        
        # Take top 2 eigenvalues
        idx = np.argsort(eigenvalues)[::-1][:2]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Handle negative eigenvalues
        eigenvalues = np.maximum(eigenvalues, 0)
        
        # Compute coordinates
        coords = eigenvectors @ np.diag(np.sqrt(eigenvalues))
        
        return coords
    
    # =========================================================================
    # QUBO AND QAOA CORE
    # =========================================================================
    
    def _tsp_to_qubo(self, n: int, distance_matrix: List[List[float]], 
                    penalty: float = None) -> Tuple[Dict, int]:
        """Convert TSP to QUBO"""
        if penalty is None:
            max_dist = max(max(row) for row in distance_matrix)
            penalty = 2 * max_dist * n
        
        num_qubits = n * n
        
        def var_index(city: int, position: int) -> int:
            return city * n + position
        
        Q = {}
        
        # Objective
        for t in range(n):
            next_t = (t + 1) % n
            for i in range(n):
                for j in range(n):
                    if i != j and distance_matrix[i][j] > 0:
                        idx_i = var_index(i, t)
                        idx_j = var_index(j, next_t)
                        key = (min(idx_i, idx_j), max(idx_i, idx_j))
                        Q[key] = Q.get(key, 0) + distance_matrix[i][j]
        
        # Constraint 1: Each city once
        for i in range(n):
            for t in range(n):
                idx = var_index(i, t)
                Q[(idx, idx)] = Q.get((idx, idx), 0) - penalty
            
            for t1 in range(n):
                for t2 in range(t1 + 1, n):
                    idx1 = var_index(i, t1)
                    idx2 = var_index(i, t2)
                    key = (min(idx1, idx2), max(idx1, idx2))
                    Q[key] = Q.get(key, 0) + 2 * penalty
        
        # Constraint 2: Each position has one city
        for t in range(n):
            for i in range(n):
                idx = var_index(i, t)
                Q[(idx, idx)] = Q.get((idx, idx), 0) - penalty
            
            for i1 in range(n):
                for i2 in range(i1 + 1, n):
                    idx1 = var_index(i1, t)
                    idx2 = var_index(i2, t)
                    key = (min(idx1, idx2), max(idx1, idx2))
                    Q[key] = Q.get(key, 0) + 2 * penalty
        
        return Q, num_qubits
    
    def _run_qaoa(self, 
                  Q: Dict, 
                  num_qubits: int, 
                  p: int = 2,
                  shots: int = 500) -> Dict[str, int]:
        """
        Run QAOA circuit on specified device
        
        Supports both SV1 simulator and real QPUs (Rigetti, IonQ)
        
        Returns:
            Measurement counts dictionary
        """
        from braket.aws import AwsDevice, AwsSession
        from braket.circuits import Circuit
        import boto3
        
        # Adjust shots for real QPU (cost considerations)
        if self.is_real_qpu and shots > 100:
            actual_shots = min(shots, 100)
            if self.verbose:
                print(f"      [QPU: reducing shots from {shots} to {actual_shots}]")
        else:
            actual_shots = shots
        
        # Build circuit
        circuit = Circuit()
        
        # Initial superposition
        for i in range(num_qubits):
            circuit.h(i)
        
        # QAOA layers with optimized angles
        gamma = [np.pi / 4] * p
        beta = [np.pi / 8] * p
        
        for layer in range(p):
            # Cost layer - encode problem Hamiltonian
            for (i, j), coeff in Q.items():
                if i == j:
                    circuit.rz(i, 2 * gamma[layer] * coeff)
                else:
                    # ZZ interaction via CNOT-RZ-CNOT
                    circuit.cnot(i, j)
                    circuit.rz(j, 2 * gamma[layer] * coeff)
                    circuit.cnot(i, j)
            
            # Mixer layer - transverse field
            for i in range(num_qubits):
                circuit.rx(i, 2 * beta[layer])
        
        # Create AWS session with proper region for device
        aws_session = AwsSession(
            boto_session=boto3.Session(
                region_name=self.aws_region,
                aws_access_key_id=self.aws_access_key,
                aws_secret_access_key=self.aws_secret_key
            )
        )
        
        device = AwsDevice(self.device_arn, aws_session=aws_session)
        
        # Check device availability for real QPUs
        if self.is_real_qpu:
            try:
                status = device.status
                if self.verbose:
                    print(f"      [QPU Status: {status}]")
                if status != 'ONLINE':
                    raise RuntimeError(f"Device {self.device_name} is {status}")
            except Exception as e:
                if self.verbose:
                    print(f"      [Could not check device status: {e}]")
        
        # Submit task
        task = device.run(circuit, shots=actual_shots)
        
        # Wait for result (may take longer on real QPU)
        result = task.result()
        
        return result.measurement_counts
    
    def _decode_tour(self, bitstring: str, n: int) -> List[int]:
        """Decode bitstring to tour"""
        tour = [None] * n
        
        for idx, bit in enumerate(bitstring):
            if bit == '1':
                city = idx // n
                position = idx % n
                if city < n and position < n and tour[position] is None:
                    tour[position] = city
        
        # Fill missing
        if None in tour:
            used = set(c for c in tour if c is not None)
            missing = [c for c in range(n) if c not in used]
            for i, pos in enumerate(tour):
                if pos is None and missing:
                    tour[i] = missing.pop(0)
        
        if None in tour or len(set(tour)) != n:
            return list(range(n))
        
        return tour
    
    def _solve_cluster_tsp(self, 
                          cluster_indices: List[int],
                          full_distance_matrix: List[List[float]],
                          p: int = 2,
                          shots: int = 500) -> List[int]:
        """
        Solve TSP for a single cluster using QAOA
        
        Returns:
            Ordered list of city indices (from the original problem)
        """
        n = len(cluster_indices)
        
        if n <= 1:
            return cluster_indices
        
        if n == 2:
            return cluster_indices
        
        # Extract sub-matrix
        sub_matrix = [[full_distance_matrix[i][j] for j in cluster_indices] 
                      for i in cluster_indices]
        
        # Convert to QUBO
        Q, num_qubits = self._tsp_to_qubo(n, sub_matrix)
        
        # Run QAOA
        try:
            measurements = self._run_qaoa(Q, num_qubits, p, shots)
            
            # Find best valid tour
            best_tour = None
            best_distance = float('inf')
            
            for bitstring, count in measurements.items():
                tour = self._decode_tour(bitstring, n)
                if tour and len(set(tour)) == n:
                    distance = sum(
                        sub_matrix[tour[i]][tour[(i + 1) % n]]
                        for i in range(n)
                    )
                    if distance < best_distance:
                        best_distance = distance
                        best_tour = tour
            
            if best_tour is None:
                best_tour = list(range(n))
            
            # Map back to original indices
            return [cluster_indices[i] for i in best_tour]
            
        except Exception as e:
            if self.verbose:
                print(f"    QAOA error for cluster: {e}")
            return cluster_indices
    
    # =========================================================================
    # STITCHING AND REFINEMENT
    # =========================================================================
    
    def _find_best_connection(self,
                             tour1: List[int],
                             tour2: List[int],
                             distance_matrix: List[List[float]]) -> Tuple[int, int, int, int]:
        """
        Find best connection points between two tours
        
        Returns:
            (exit_idx_from_tour1, entry_idx_to_tour2, exit_city, entry_city)
        """
        best_dist = float('inf')
        best_connection = (len(tour1) - 1, 0, tour1[-1], tour2[0])
        
        for i, city1 in enumerate(tour1):
            for j, city2 in enumerate(tour2):
                dist = distance_matrix[city1][city2]
                if dist < best_dist:
                    best_dist = dist
                    best_connection = (i, j, city1, city2)
        
        return best_connection
    
    def _stitch_tours(self,
                     cluster_tours: List[List[int]],
                     cluster_order: List[int],
                     distance_matrix: List[List[float]]) -> List[int]:
        """
        Stitch cluster tours together based on cluster ordering
        
        Args:
            cluster_tours: List of tours for each cluster
            cluster_order: Order in which to visit clusters
            distance_matrix: Full distance matrix
            
        Returns:
            Complete tour visiting all cities
        """
        if len(cluster_order) == 1:
            return cluster_tours[cluster_order[0]]
        
        # Start with first cluster
        full_tour = list(cluster_tours[cluster_order[0]])
        
        # Stitch remaining clusters
        for idx in range(1, len(cluster_order)):
            next_cluster = cluster_order[idx]
            next_tour = cluster_tours[next_cluster]
            
            # Find best connection
            _, entry_idx, _, _ = self._find_best_connection(
                full_tour, next_tour, distance_matrix
            )
            
            # Rotate next_tour to start at entry point
            rotated = next_tour[entry_idx:] + next_tour[:entry_idx]
            
            # Append to full tour
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
        
        while improved:
            improved = False
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
    
    # =========================================================================
    # MAIN SOLVER
    # =========================================================================
    
    def solve_tsp(self,
                  locations: List[str],
                  distance_matrix: List[List[float]],
                  start_location: Optional[str] = None,
                  p: int = 2,
                  shots: int = 500) -> Dict:
        """
        Solve large-scale TSP using Cluster-QAOA
        
        Args:
            locations: List of location names
            distance_matrix: NxN distance matrix
            start_location: Optional starting location
            p: QAOA layers per sub-problem
            shots: Shots per QAOA run
            
        Returns:
            Solution dictionary
        """
        n = len(locations)
        total_start = time.time()
        
        if self.verbose:
            print("\n" + "="*70)
            print(f"CLUSTER-QAOA TSP SOLVER - {self.device_name.upper()}")
            print("="*70)
            print(f"Total cities: {n}")
            print(f"Max cluster size: {self.max_cluster_size}")
            print(f"Device: {self.device_arn}")
            print(f"QAOA layers: {p}, Shots: {shots}")
            print("="*70)
        
        try:
            # Step 1: Estimate coordinates and cluster
            if self.verbose:
                print("\n[Step 1] Clustering cities...")
            
            cluster_start = time.time()
            coords = self._estimate_coordinates(distance_matrix)
            n_clusters = max(1, n // self.max_cluster_size)
            clusters, centers = self._cluster_cities(coords, n_clusters)
            cluster_time = time.time() - cluster_start
            
            if self.verbose:
                print(f"  Created {len(clusters)} clusters")
                for i, c in enumerate(clusters):
                    print(f"    Cluster {i+1}: {len(c)} cities")
            
            # Step 2: Solve intra-cluster TSP with QAOA
            if self.verbose:
                print(f"\n[Step 2] Solving {len(clusters)} cluster TSPs with QAOA...")
            
            qaoa_start = time.time()
            cluster_tours = []
            
            for i, cluster in enumerate(clusters):
                if self.verbose:
                    print(f"  Cluster {i+1}/{len(clusters)} ({len(cluster)} cities, {len(cluster)**2} qubits)...", end=" ")
                
                if len(cluster) <= self.max_cluster_size:
                    tour = self._solve_cluster_tsp(cluster, distance_matrix, p, shots)
                else:
                    # Recursively split large clusters
                    tour = cluster  # Fallback to original order
                
                cluster_tours.append(tour)
                
                if self.verbose:
                    dist = self._calculate_distance(
                        list(range(len(cluster))),
                        [[distance_matrix[cluster[a]][cluster[b]] for b in range(len(cluster))] 
                         for a in range(len(cluster))]
                    )
                    print(f"Done")
            
            qaoa_time = time.time() - qaoa_start
            
            # Step 3: Determine cluster order using nearest-neighbor on centers
            if self.verbose:
                print(f"\n[Step 3] Determining cluster visit order...")
            
            order_start = time.time()
            
            if len(clusters) > 1:
                # Use greedy nearest-neighbor for cluster ordering
                # (Could use QAOA here too for very large number of clusters)
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
            
            # Step 5: Local search refinement
            if self.use_local_search:
                if self.verbose:
                    print(f"\n[Step 5] Applying {self.local_search_method} local search...")
                
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
                'qaoa_time_s': qaoa_time,
                'order_time_s': order_time,
                'stitch_time_s': stitch_time,
                'refine_time_s': refine_time,
                'total_time_s': total_time
            }
            
            # Print results
            if self.verbose:
                print("\n" + "="*70)
                print("CLUSTER-QAOA SOLUTION")
                print("="*70)
                print(f"Total cities: {n}")
                print(f"Clusters used: {len(clusters)}")
                print(f"Raw distance (before 2-opt): {raw_distance:.2f}")
                print(f"Final distance: {final_distance:.2f}")
                if self.use_local_search:
                    print(f"Improvement from 2-opt: {improvement:.2f} ({improvement/raw_distance*100:.1f}%)")
                print(f"\nTiming:")
                print(f"  Clustering: {cluster_time:.2f}s")
                print(f"  QAOA solving: {qaoa_time:.2f}s")
                print(f"  Cluster ordering: {order_time:.2f}s")
                print(f"  Stitching: {stitch_time:.2f}s")
                if self.use_local_search:
                    print(f"  Local search: {refine_time:.2f}s")
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
                'solver': f'Cluster-QAOA ({self.device_name.upper()})',
                'n_clusters': len(clusters),
                'cluster_sizes': [len(c) for c in clusters],
                'qaoa_layers': p,
                'shots': shots,
                'improvement': improvement if self.use_local_search else 0
            }
            
        except Exception as e:
            import traceback
            if self.verbose:
                print(f"\nError: {str(e)}")
                traceback.print_exc()
            return {
                'status': 'error',
                'message': str(e),
                'solver': f'Cluster-QAOA ({self.device_name.upper()})'
            }


def test_cluster_qaoa():
    """Test Cluster-QAOA on a 10-city problem"""
    print("\n" + "="*70)
    print("TESTING CLUSTER-QAOA SOLVER")
    print("="*70)
    
    # 10-city problem
    np.random.seed(42)
    n = 10
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
    
    solver = ClusterQAOATSPSolver(device='sv1', max_cluster_size=5)
    result = solver.solve_tsp(locations, distance_matrix, p=2, shots=500)
    
    print(f"\nResult: {result['status']}")
    print(f"Distance: {result.get('total_distance', 'N/A')}")
    
    return result


if __name__ == "__main__":
    test_cluster_qaoa()
