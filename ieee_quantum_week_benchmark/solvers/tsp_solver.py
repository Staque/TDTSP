"""
TSP (Traveling Salesman Problem) Solver
Solves the Traveling Salesman Problem using Gurobi optimization
Part of Universal Gurobi Controller
"""

import gurobipy as gp
from gurobipy import GRB
import pandas as pd
from typing import List, Dict, Tuple, Optional
import numpy as np


class TSPSolver:
    """
    Traveling Salesman Problem solver using Gurobi
    Uses Miller-Tucker-Zemlin (MTZ) formulation to prevent subtours
    """
    
    def __init__(self):
        self.model = None
        self.variables = {}
        self.results = {}
    
    def solve_tsp(self,
                  locations: List[str],
                  distance_matrix: List[List[float]],
                  start_location: Optional[str] = None) -> Dict:
        """
        Solve the Traveling Salesman Problem
        
        Args:
            locations: List of location names (e.g., ["Times Square", "Central Park", ...])
            distance_matrix: NxN matrix of distances between locations
                            distance_matrix[i][j] = distance from location i to location j
            start_location: Optional starting location (if None, uses first location)
        
        Returns:
            Dictionary containing:
                - tour: Ordered list of location indices in optimal tour
                - tour_names: Ordered list of location names in optimal tour
                - total_distance: Total distance of optimal tour
                - status: Optimization status
        """
        try:
            n = len(locations)
            
            if n < 2:
                print("❌ Error: Need at least 2 locations for TSP")
                return {}
            
            if len(distance_matrix) != n or any(len(row) != n for row in distance_matrix):
                print(f"❌ Error: Distance matrix must be {n}x{n}")
                return {}
            
            # Determine start index
            if start_location:
                if start_location not in locations:
                    print(f"❌ Error: Start location '{start_location}' not in locations list")
                    return {}
                start_idx = locations.index(start_location)
            else:
                start_idx = 0
            
            print("\n" + "="*70)
            print("TSP OPTIMIZATION - GUROBI SOLVER")
            print("="*70)
            print(f"Number of locations: {n}")
            print(f"Start location: {locations[start_idx]}")
            print(f"Optimization method: Miller-Tucker-Zemlin (MTZ) formulation")
            print("="*70)
            
            # Create model
            self.model = gp.Model("TSP")
            self.model.setParam('OutputFlag', 0)  # Suppress Gurobi output
            
            # Decision variables: x[i,j] = 1 if we go from location i to location j
            x = {}
            for i in range(n):
                for j in range(n):
                    if i != j:
                        x[i, j] = self.model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}")
            
            # MTZ variables: u[i] represents the position of location i in the tour
            # Used to eliminate subtours
            u = {}
            for i in range(n):
                if i != start_idx:
                    u[i] = self.model.addVar(lb=1, ub=n-1, vtype=GRB.CONTINUOUS, name=f"u_{i}")
            
            # Objective: Minimize total distance
            objective = gp.quicksum(
                distance_matrix[i][j] * x[i, j]
                for i in range(n) for j in range(n) if i != j
            )
            self.model.setObjective(objective, GRB.MINIMIZE)
            
            # Constraint 1: Each location must be left exactly once
            for i in range(n):
                self.model.addConstr(
                    gp.quicksum(x[i, j] for j in range(n) if j != i) == 1,
                    name=f"leave_{i}"
                )
            
            # Constraint 2: Each location must be entered exactly once
            for j in range(n):
                self.model.addConstr(
                    gp.quicksum(x[i, j] for i in range(n) if i != j) == 1,
                    name=f"enter_{j}"
                )
            
            # Constraint 3: MTZ subtour elimination constraints
            # If we go from i to j (and neither is the start), then u[j] >= u[i] + 1
            for i in range(n):
                for j in range(n):
                    if i != j and i != start_idx and j != start_idx:
                        self.model.addConstr(
                            u[j] >= u[i] + 1 - n * (1 - x[i, j]),
                            name=f"mtz_{i}_{j}"
                        )
            
            # Optimize
            print("\n🔄 Optimizing...")
            self.model.optimize()
            
            # Extract results
            if self.model.status == GRB.OPTIMAL:
                print("✅ Optimal solution found!")
                
                # Build tour by following the x variables
                tour = [start_idx]
                current = start_idx
                
                for _ in range(n - 1):
                    for j in range(n):
                        if j != current and x[current, j].X > 0.5:
                            tour.append(j)
                            current = j
                            break
                
                # Calculate total distance
                total_distance = sum(
                    distance_matrix[tour[i]][tour[i + 1]]
                    for i in range(len(tour) - 1)
                )
                # Add distance back to start
                total_distance += distance_matrix[tour[-1]][tour[0]]
                
                # Get location names in tour order
                tour_names = [locations[i] for i in tour]
                
                # Print results
                print("\n" + "="*70)
                print("OPTIMAL TOUR FOUND")
                print("="*70)
                print(f"Total Distance: {total_distance:.2f} units")
                print(f"\nTour Order:")
                for idx, loc_idx in enumerate(tour, 1):
                    print(f"  {idx}. {locations[loc_idx]}")
                print(f"  {len(tour) + 1}. {locations[tour[0]]} (return to start)")
                
                print("\n" + "-"*70)
                print("DISTANCE BREAKDOWN:")
                for i in range(len(tour)):
                    from_loc = locations[tour[i]]
                    to_loc = locations[tour[(i + 1) % len(tour)]]
                    dist = distance_matrix[tour[i]][tour[(i + 1) % len(tour)]]
                    print(f"  {from_loc} → {to_loc}: {dist:.2f}")
                
                print("="*70)
                
                # Store results
                results = {
                    'tour': tour,
                    'tour_names': tour_names,
                    'total_distance': total_distance,
                    'status': 'optimal',
                    'distance_matrix': distance_matrix,
                    'locations': locations,
                    'start_location': locations[start_idx]
                }
                
                return results
            
            else:
                print(f"❌ Optimization failed. Status: {self.model.status}")
                return {
                    'status': 'failed',
                    'model_status': self.model.status
                }
        
        except Exception as e:
            print(f"❌ Error in TSP optimization: {str(e)}")
            import traceback
            traceback.print_exc()
            return {}
    
    def create_tour_dataframe(self, results: Dict) -> pd.DataFrame:
        """
        Create a pandas DataFrame from TSP results for easy viewing
        
        Args:
            results: Results dictionary from solve_tsp()
        
        Returns:
            DataFrame with tour details
        """
        if not results or results.get('status') != 'optimal':
            return pd.DataFrame()
        
        tour = results['tour']
        locations = results['locations']
        distance_matrix = results['distance_matrix']
        
        tour_data = []
        for i in range(len(tour)):
            from_idx = tour[i]
            to_idx = tour[(i + 1) % len(tour)]
            
            tour_data.append({
                'Step': i + 1,
                'From': locations[from_idx],
                'To': locations[to_idx],
                'Distance': round(distance_matrix[from_idx][to_idx], 2)
            })
        
        df = pd.DataFrame(tour_data)
        return df


def test_tsp_solver():
    """
    Test the TSP solver with a sample problem
    """
    print("\n" + "="*70)
    print("🧪 TESTING TSP SOLVER")
    print("="*70)
    
    # Sample problem: 5 cities
    locations = ["New York", "Boston", "Philadelphia", "Washington DC", "Baltimore"]
    
    # Sample distance matrix (approximate distances in miles)
    # This is a symmetric matrix for simplicity
    distance_matrix = [
        [0,    215,  95,   225,  185],  # From New York
        [215,  0,    310,  440,  400],  # From Boston
        [95,   310,  0,    140,  100],  # From Philadelphia
        [225,  440,  140,  0,    40],   # From Washington DC
        [185,  400,  100,  40,   0]     # From Baltimore
    ]
    
    print("\nTest Problem:")
    print(f"Locations: {locations}")
    print(f"Distance Matrix:")
    for i, row in enumerate(distance_matrix):
        print(f"  {locations[i]}: {row}")
    
    # Solve TSP
    solver = TSPSolver()
    results = solver.solve_tsp(locations, distance_matrix)
    
    if results and results.get('status') == 'optimal':
        print("\n✅ TEST PASSED: TSP solver working correctly!")
        
        # Create and display DataFrame
        df = solver.create_tour_dataframe(results)
        print("\n📊 Tour DataFrame:")
        print(df.to_string(index=False))
        
        return True
    else:
        print("\n❌ TEST FAILED: TSP solver did not find optimal solution")
        return False


if __name__ == "__main__":
    # Run test when script is executed directly
    success = test_tsp_solver()
    
    if success:
        print("\n" + "="*70)
        print("🎉 TSP SOLVER READY FOR STEP 3!")
        print("="*70)
        print("\n📋 Next: Integrate with Google Maps API for real distances")
        print("   Tell the assistant: 'Start Step 3'")
    else:
        print("\n⚠️  Please check the errors above")

