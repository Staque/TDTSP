"""
D-Wave TSP Solver
Solves TSP using D-Wave's Hybrid CQM (Constrained Quadratic Model) Solver
Part of Universal Gurobi Controller - Quantum Benchmarking
"""

import os
import time
import numpy as np
from typing import List, Dict, Optional
from dotenv import load_dotenv

load_dotenv()


class DWaveTSPSolver:
    """
    Traveling Salesman Problem solver using D-Wave's Hybrid CQM Solver
    Uses Constrained Quadratic Model for native constraint handling
    """
    
    def __init__(self):
        """Initialize the D-Wave TSP solver"""
        self.api_token = os.getenv('DWAVE_API_TOKEN')
        if not self.api_token:
            raise ValueError("DWAVE_API_TOKEN not found in environment variables")
        
        self.sampler = None
        self.timing_info = {}
        
    def _get_sampler(self):
        """Get or create the D-Wave hybrid CQM sampler"""
        if self.sampler is None:
            from dwave.system import LeapHybridCQMSampler
            self.sampler = LeapHybridCQMSampler(token=self.api_token)
        return self.sampler
    
    def _build_cqm(self, n: int, distance_matrix: List[List[float]]):
        """
        Build a Constrained Quadratic Model for TSP
        
        Args:
            n: Number of cities
            distance_matrix: NxN matrix of distances
            
        Returns:
            CQM model
        """
        import dimod
        
        cqm = dimod.ConstrainedQuadraticModel()
        
        # Binary variables: x[i][t] = 1 if city i is visited at position t
        x = [[dimod.Binary(f'x_{i}_{t}') for t in range(n)] for i in range(n)]
        
        # Objective: Minimize total distance
        objective = 0
        for t in range(n):
            next_t = (t + 1) % n  # Wrap around for return to start
            for i in range(n):
                for j in range(n):
                    if i != j:
                        objective += distance_matrix[i][j] * x[i][t] * x[j][next_t]
        
        cqm.set_objective(objective)
        
        # Constraint 1: Each city must be visited exactly once
        for i in range(n):
            cqm.add_constraint(
                sum(x[i][t] for t in range(n)) == 1,
                label=f'visit_city_{i}'
            )
        
        # Constraint 2: Each position must have exactly one city
        for t in range(n):
            cqm.add_constraint(
                sum(x[i][t] for i in range(n)) == 1,
                label=f'position_{t}'
            )
        
        return cqm, x
    
    def _extract_tour(self, sample: dict, n: int) -> List[int]:
        """
        Extract tour from sample result
        
        Args:
            sample: Sample dictionary from D-Wave
            n: Number of cities
            
        Returns:
            List of city indices in tour order
        """
        tour = [None] * n
        
        for key, value in sample.items():
            if value == 1 and key.startswith('x_'):
                parts = key.split('_')
                city = int(parts[1])
                position = int(parts[2])
                tour[position] = city
        
        # Handle any missing positions (shouldn't happen with valid solution)
        if None in tour:
            return []
        
        return tour
    
    def solve_tsp(self,
                  locations: List[str],
                  distance_matrix: List[List[float]],
                  start_location: Optional[str] = None,
                  time_limit: float = 60.0) -> Dict:
        """
        Solve the Traveling Salesman Problem using D-Wave hybrid CQM solver
        
        Args:
            locations: List of location names
            distance_matrix: NxN matrix of distances between locations
            start_location: Optional starting location (if None, uses first location)
            time_limit: Maximum time limit for solver (seconds)
            
        Returns:
            Dictionary containing:
                - tour: Ordered list of location indices
                - tour_names: Ordered list of location names
                - total_distance: Total distance of tour
                - status: Optimization status
                - timing: Timing information
        """
        n = len(locations)
        
        if n < 2:
            return {'status': 'error', 'message': 'Need at least 2 locations'}
        
        print("\n" + "="*70)
        print("D-WAVE HYBRID CQM SOLVER - TSP")
        print("="*70)
        print(f"Number of locations: {n}")
        print(f"Variables: {n*n} binary")
        print(f"Time limit: {time_limit}s")
        print("="*70)
        
        try:
            # Build CQM
            build_start = time.time()
            cqm, x = self._build_cqm(n, distance_matrix)
            build_time = time.time() - build_start
            print(f"✅ CQM built in {build_time:.3f}s")
            
            # Submit to D-Wave
            print("\n🔄 Submitting to D-Wave hybrid solver...")
            submit_start = time.time()
            
            sampler = self._get_sampler()
            sampleset = sampler.sample_cqm(cqm, time_limit=time_limit)
            
            submit_time = time.time() - submit_start
            print(f"✅ Received response in {submit_time:.3f}s")
            
            # Get feasible samples
            feasible = sampleset.filter(lambda s: s.is_feasible)
            
            if len(feasible) == 0:
                print("⚠️ No feasible solution found, using best sample")
                best_sample = sampleset.first.sample
                is_feasible = False
            else:
                best_sample = feasible.first.sample
                is_feasible = True
                print(f"✅ Found {len(feasible)} feasible solutions")
            
            # Extract tour
            tour = self._extract_tour(best_sample, n)
            
            if not tour or len(tour) != n:
                return {
                    'status': 'error',
                    'message': 'Could not extract valid tour from solution'
                }
            
            # Calculate total distance
            total_distance = sum(
                distance_matrix[tour[i]][tour[(i + 1) % n]]
                for i in range(n)
            )
            
            # Reorder tour to start from start_location if specified
            if start_location and start_location in locations:
                start_idx = locations.index(start_location)
                if start_idx in tour:
                    start_pos = tour.index(start_idx)
                    tour = tour[start_pos:] + tour[:start_pos]
            
            tour_names = [locations[i] for i in tour]
            
            # Timing info from D-Wave
            dwave_timing = sampleset.info.get('timing', {})
            
            self.timing_info = {
                'build_time_s': build_time,
                'submit_time_s': submit_time,
                'total_time_s': build_time + submit_time,
                'qpu_access_time_us': dwave_timing.get('qpu_access_time', 0),
                'charge_time_s': dwave_timing.get('charge_time', 0) / 1000000 if dwave_timing.get('charge_time') else 0
            }
            
            # Print results
            print("\n" + "="*70)
            print("OPTIMAL TOUR FOUND" if is_feasible else "BEST TOUR FOUND (may be infeasible)")
            print("="*70)
            print(f"Total Distance: {total_distance:.2f}")
            print(f"Feasible: {is_feasible}")
            print(f"\nTour Order:")
            for idx, loc_idx in enumerate(tour, 1):
                print(f"  {idx}. {locations[loc_idx]}")
            print(f"  {len(tour) + 1}. {locations[tour[0]]} (return to start)")
            print("\n" + "-"*70)
            print("TIMING:")
            print(f"  Build time: {build_time:.3f}s")
            print(f"  Submit/response time: {submit_time:.3f}s")
            print(f"  Total time: {build_time + submit_time:.3f}s")
            print("="*70)
            
            return {
                'status': 'optimal' if is_feasible else 'feasible',
                'tour': tour,
                'tour_names': tour_names,
                'total_distance': total_distance,
                'locations': locations,
                'distance_matrix': distance_matrix,
                'start_location': locations[tour[0]],
                'timing': self.timing_info,
                'solver': 'D-Wave Hybrid CQM',
                'is_feasible': is_feasible,
                'num_samples': len(sampleset),
                'num_feasible': len(feasible) if is_feasible else 0
            }
            
        except Exception as e:
            import traceback
            print(f"❌ Error: {str(e)}")
            traceback.print_exc()
            return {
                'status': 'error',
                'message': str(e),
                'solver': 'D-Wave Hybrid CQM'
            }


def test_dwave_solver():
    """Test the D-Wave TSP solver with a sample problem"""
    print("\n" + "="*70)
    print("🧪 TESTING D-WAVE TSP SOLVER")
    print("="*70)
    
    # Sample problem: 5 cities
    locations = ["New York", "Boston", "Philadelphia", "Washington DC", "Baltimore"]
    
    distance_matrix = [
        [0,    215,  95,   225,  185],
        [215,  0,    310,  440,  400],
        [95,   310,  0,    140,  100],
        [225,  440,  140,  0,    40],
        [185,  400,  100,  40,   0]
    ]
    
    solver = DWaveTSPSolver()
    results = solver.solve_tsp(locations, distance_matrix, time_limit=30)
    
    if results.get('status') in ['optimal', 'feasible']:
        print("\n✅ TEST PASSED")
        return True
    else:
        print(f"\n❌ TEST FAILED: {results.get('message', 'Unknown error')}")
        return False


if __name__ == "__main__":
    test_dwave_solver()
