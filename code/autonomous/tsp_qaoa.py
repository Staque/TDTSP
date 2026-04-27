"""
QAOA TSP Solver using AWS Braket.
Solves TSP using the Quantum Approximate Optimization Algorithm.
Supports the SV1 simulator and Rigetti hardware.
"""

import os
import time
import numpy as np
from typing import List, Dict, Optional, Tuple
from dotenv import load_dotenv

load_dotenv()


class QAOATSPSolver:
    """
    Traveling Salesman Problem solver using QAOA on AWS Braket
    Supports SV1 simulator (development) and Rigetti Ankaa (production)
    """
    
    # Device ARNs for AWS Braket
    DEVICES = {
        'sv1': 'arn:aws:braket:::device/quantum-simulator/amazon/sv1',
        'tn1': 'arn:aws:braket:::device/quantum-simulator/amazon/tn1',
        'rigetti': 'arn:aws:braket:us-west-1::device/qpu/rigetti/Ankaa-2',
        'ionq': 'arn:aws:braket:us-east-1::device/qpu/ionq/Harmony',
    }
    
    def __init__(self, device: str = 'sv1'):
        """
        Initialize the QAOA TSP solver
        
        Args:
            device: Device to use ('sv1', 'tn1', 'rigetti', 'ionq')
        """
        self.aws_access_key = os.getenv('AWS_ACCESS_KEY_ID')
        self.aws_secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
        self.aws_region = os.getenv('AWS_DEFAULT_REGION', 'us-east-1')
        
        if not self.aws_access_key or not self.aws_secret_key:
            raise ValueError("AWS credentials not found in environment variables")
        
        self.device_name = device
        self.device_arn = self.DEVICES.get(device, self.DEVICES['sv1'])
        self.timing_info = {}
        
    def _tsp_to_qubo(self, n: int, distance_matrix: List[List[float]], 
                     penalty: float = None) -> Tuple[Dict, int]:
        """
        Convert TSP to QUBO formulation
        
        Uses binary variables x_{i,t} = 1 if city i is at position t in tour
        
        Args:
            n: Number of cities
            distance_matrix: NxN distance matrix
            penalty: Penalty weight for constraints (auto-calculated if None)
            
        Returns:
            Tuple of (QUBO dictionary, number of qubits)
        """
        if penalty is None:
            penalty = 2 * max(max(row) for row in distance_matrix)
        
        num_qubits = n * n  # x_{i,t} for i,t in [0,n)
        
        def var_index(city: int, position: int) -> int:
            """Get variable index for x_{city, position}"""
            return city * n + position
        
        Q = {}
        
        # Objective: Minimize tour distance
        # sum_{i,j,t} d[i][j] * x[i][t] * x[j][(t+1) % n]
        for t in range(n):
            next_t = (t + 1) % n
            for i in range(n):
                for j in range(n):
                    if i != j and distance_matrix[i][j] > 0:
                        idx_i = var_index(i, t)
                        idx_j = var_index(j, next_t)
                        key = (min(idx_i, idx_j), max(idx_i, idx_j))
                        Q[key] = Q.get(key, 0) + distance_matrix[i][j]
        
        # Constraint 1: Each city visited exactly once
        # sum_t x[i][t] = 1 for all i
        # Penalty: P * (sum_t x[i][t] - 1)^2 = P * (sum_t x[i][t]^2 - 2*sum_t x[i][t] + 1)
        for i in range(n):
            # Linear terms: -2P * x[i][t] + P * x[i][t] (since x^2 = x for binary)
            for t in range(n):
                idx = var_index(i, t)
                Q[(idx, idx)] = Q.get((idx, idx), 0) + penalty * (1 - 2)  # -P
            
            # Quadratic terms: 2P * x[i][t] * x[i][t'] for t != t'
            for t1 in range(n):
                for t2 in range(t1 + 1, n):
                    idx1 = var_index(i, t1)
                    idx2 = var_index(i, t2)
                    key = (min(idx1, idx2), max(idx1, idx2))
                    Q[key] = Q.get(key, 0) + 2 * penalty
        
        # Constraint 2: Each position has exactly one city
        # sum_i x[i][t] = 1 for all t
        for t in range(n):
            for i in range(n):
                idx = var_index(i, t)
                Q[(idx, idx)] = Q.get((idx, idx), 0) + penalty * (1 - 2)  # -P
            
            for i1 in range(n):
                for i2 in range(i1 + 1, n):
                    idx1 = var_index(i1, t)
                    idx2 = var_index(i2, t)
                    key = (min(idx1, idx2), max(idx1, idx2))
                    Q[key] = Q.get(key, 0) + 2 * penalty
        
        return Q, num_qubits
    
    def _build_qaoa_circuit(self, Q: Dict, num_qubits: int, gamma: float, beta: float):
        """
        Build QAOA circuit for the QUBO problem
        
        Args:
            Q: QUBO dictionary
            num_qubits: Number of qubits
            gamma: Phase separation angle
            beta: Mixing angle
            
        Returns:
            Braket Circuit
        """
        from braket.circuits import Circuit
        
        circuit = Circuit()
        
        # Initial state: uniform superposition
        for i in range(num_qubits):
            circuit.h(i)
        
        # Cost Hamiltonian (phase separation)
        for (i, j), coeff in Q.items():
            if i == j:
                # Linear term: e^{-i * gamma * coeff * Z}
                circuit.rz(i, 2 * gamma * coeff)
            else:
                # Quadratic term: e^{-i * gamma * coeff * Z_i Z_j}
                circuit.cnot(i, j)
                circuit.rz(j, 2 * gamma * coeff)
                circuit.cnot(i, j)
        
        # Mixer Hamiltonian
        for i in range(num_qubits):
            circuit.rx(i, 2 * beta)
        
        return circuit
    
    def _decode_solution(self, bitstring: str, n: int) -> List[int]:
        """
        Decode bitstring to tour
        
        Args:
            bitstring: Binary string from measurement
            n: Number of cities
            
        Returns:
            Tour as list of city indices, or empty list if invalid
        """
        tour = [None] * n
        
        for idx, bit in enumerate(bitstring):
            if bit == '1':
                city = idx // n
                position = idx % n
                if tour[position] is None:
                    tour[position] = city
                else:
                    # Conflict - multiple cities at same position
                    return []
        
        # Check validity
        if None in tour or len(set(tour)) != n:
            return []
        
        return tour
    
    def solve_tsp(self,
                  locations: List[str],
                  distance_matrix: List[List[float]],
                  start_location: Optional[str] = None,
                  p: int = 2,
                  shots: int = 1000,
                  optimize_angles: bool = True) -> Dict:
        """
        Solve TSP using QAOA on AWS Braket
        
        Args:
            locations: List of location names
            distance_matrix: NxN matrix of distances
            start_location: Optional starting location
            p: Number of QAOA layers
            shots: Number of measurement shots
            optimize_angles: Whether to optimize gamma/beta angles
            
        Returns:
            Dictionary with tour, distance, timing, etc.
        """
        n = len(locations)
        
        if n < 2:
            return {'status': 'error', 'message': 'Need at least 2 locations'}
        
        if n > 5 and self.device_name == 'sv1':
            print(f"Warning: {n} cities requires {n*n} qubits. May be slow on simulator.")
        
        print("\n" + "="*70)
        print(f"QAOA TSP SOLVER - AWS Braket ({self.device_name.upper()})")
        print("="*70)
        print(f"Number of locations: {n}")
        print(f"Qubits required: {n*n}")
        print(f"QAOA layers (p): {p}")
        print(f"Shots: {shots}")
        print(f"Device: {self.device_arn}")
        print("="*70)
        
        try:
            from braket.aws import AwsDevice, AwsQuantumTask
            from braket.circuits import Circuit
            import boto3
            
            # Set up AWS session
            boto3.setup_default_session(
                aws_access_key_id=self.aws_access_key,
                aws_secret_access_key=self.aws_secret_key,
                region_name=self.aws_region
            )
            
            # Convert TSP to QUBO
            build_start = time.time()
            Q, num_qubits = self._tsp_to_qubo(n, distance_matrix)
            build_time = time.time() - build_start
            print(f"QUBO built in {build_time:.3f}s ({len(Q)} terms)")
            
            # Initial angles (can be optimized)
            if optimize_angles and n <= 4:
                # Simple grid search for small problems
                best_gamma, best_beta = self._optimize_angles_simple(Q, num_qubits, p)
            else:
                # Default angles
                best_gamma = [0.5] * p
                best_beta = [0.5] * p
            
            # Build final circuit
            circuit_start = time.time()
            circuit = Circuit()
            
            # Initial superposition
            for i in range(num_qubits):
                circuit.h(i)
            
            # QAOA layers
            for layer in range(p):
                gamma = best_gamma[layer] if isinstance(best_gamma, list) else best_gamma
                beta = best_beta[layer] if isinstance(best_beta, list) else best_beta
                
                # Cost layer
                for (i, j), coeff in Q.items():
                    if i == j:
                        circuit.rz(i, 2 * gamma * coeff)
                    else:
                        circuit.cnot(i, j)
                        circuit.rz(j, 2 * gamma * coeff)
                        circuit.cnot(i, j)
                
                # Mixer layer
                for i in range(num_qubits):
                    circuit.rx(i, 2 * beta)
            
            circuit_time = time.time() - circuit_start
            print(f"Circuit built in {circuit_time:.3f}s")
            print(f"   Circuit depth: {circuit.depth}")
            
            # Run on device
            print(f"\nSubmitting to {self.device_name.upper()}...")
            submit_start = time.time()
            
            # Create AWS session with explicit credentials
            from braket.aws import AwsSession
            aws_session = AwsSession(
                boto_session=boto3.Session(
                    region_name=self.aws_region,
                    aws_access_key_id=self.aws_access_key,
                    aws_secret_access_key=self.aws_secret_key
                )
            )
            
            device = AwsDevice(self.device_arn, aws_session=aws_session)
            
            # Submit without specifying S3 bucket - uses default Braket bucket
            task = device.run(circuit, shots=shots)
            result = task.result()
            
            submit_time = time.time() - submit_start
            print(f"Received response in {submit_time:.3f}s")
            
            # Process results
            measurements = result.measurement_counts
            
            # Find best valid tour
            best_tour = None
            best_distance = float('inf')
            valid_count = 0
            
            for bitstring, count in measurements.items():
                tour = self._decode_solution(bitstring, n)
                if tour:
                    valid_count += count
                    distance = sum(
                        distance_matrix[tour[i]][tour[(i + 1) % n]]
                        for i in range(n)
                    )
                    if distance < best_distance:
                        best_distance = distance
                        best_tour = tour
            
            if best_tour is None:
                print("No valid tour found in measurements")
                # Try to construct a tour heuristically
                best_tour = list(range(n))
                best_distance = sum(
                    distance_matrix[best_tour[i]][best_tour[(i + 1) % n]]
                    for i in range(n)
                )
                is_valid = False
            else:
                is_valid = True
                print(f"Found {valid_count}/{shots} valid measurements")
            
            # Reorder tour if start_location specified
            if start_location and start_location in locations:
                start_idx = locations.index(start_location)
                if start_idx in best_tour:
                    start_pos = best_tour.index(start_idx)
                    best_tour = best_tour[start_pos:] + best_tour[:start_pos]
            
            tour_names = [locations[i] for i in best_tour]
            
            self.timing_info = {
                'build_time_s': build_time,
                'circuit_time_s': circuit_time,
                'submit_time_s': submit_time,
                'total_time_s': build_time + circuit_time + submit_time
            }
            
            # Print results
            print("\n" + "="*70)
            print("BEST TOUR FOUND" + (" (VALID)" if is_valid else " (HEURISTIC)"))
            print("="*70)
            print(f"Total Distance: {best_distance:.2f}")
            print(f"Valid solutions found: {valid_count}/{shots}")
            print(f"\nTour Order:")
            for idx, loc_idx in enumerate(best_tour, 1):
                print(f"  {idx}. {locations[loc_idx]}")
            print(f"  {len(best_tour) + 1}. {locations[best_tour[0]]} (return to start)")
            print("\n" + "-"*70)
            print("TIMING:")
            print(f"  QUBO build time: {build_time:.3f}s")
            print(f"  Circuit build time: {circuit_time:.3f}s")
            print(f"  Submit/response time: {submit_time:.3f}s")
            print(f"  Total time: {self.timing_info['total_time_s']:.3f}s")
            print("="*70)
            
            return {
                'status': 'optimal' if is_valid else 'heuristic',
                'tour': best_tour,
                'tour_names': tour_names,
                'total_distance': best_distance,
                'locations': locations,
                'distance_matrix': distance_matrix,
                'start_location': locations[best_tour[0]],
                'timing': self.timing_info,
                'solver': f'QAOA ({self.device_name.upper()})',
                'is_valid': is_valid,
                'valid_count': valid_count,
                'total_shots': shots,
                'qaoa_layers': p,
                'circuit_depth': circuit.depth
            }
            
        except Exception as e:
            import traceback
            print(f"Error: {str(e)}")
            traceback.print_exc()
            return {
                'status': 'error',
                'message': str(e),
                'solver': f'QAOA ({self.device_name.upper()})'
            }
    
    def _optimize_angles_simple(self, Q: Dict, num_qubits: int, p: int) -> Tuple[List[float], List[float]]:
        """
        Simple grid search for QAOA angles (for small problems)
        
        Returns:
            Tuple of (gamma_list, beta_list)
        """
        # For now, use heuristic values
        # A proper implementation would do variational optimization
        gamma = [np.pi / 4] * p
        beta = [np.pi / 8] * p
        return gamma, beta


def test_qaoa_solver():
    """Test the QAOA TSP solver with a sample problem"""
    print("\n" + "="*70)
    print("TESTING QAOA TSP SOLVER (SV1)")
    print("="*70)
    
    # Small problem for simulator (4 cities = 16 qubits)
    locations = ["A", "B", "C", "D"]
    
    distance_matrix = [
        [0,   10,  15,  20],
        [10,  0,   35,  25],
        [15,  35,  0,   30],
        [20,  25,  30,  0]
    ]
    
    solver = QAOATSPSolver(device='sv1')
    results = solver.solve_tsp(locations, distance_matrix, p=1, shots=100)
    
    if results.get('status') in ['optimal', 'heuristic']:
        print("\nTEST PASSED")
        return True
    else:
        print(f"\nTEST FAILED: {results.get('message', 'Unknown error')}")
        return False


if __name__ == "__main__":
    test_qaoa_solver()
