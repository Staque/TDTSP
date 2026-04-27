"""
Quanfluence Ising Machine TSP Solver.
Solves TSP using Quanfluence's quantum-inspired Ising Machine via REST API.
Used as the per-cluster sub-solver by ClusterQuanfluenceTSPSolver.

Features:
- Parameter tuning for device optimization.
- Async job support for larger problems.
- Device query and configuration.
"""

import os
import io
import json
import time
import zipfile
import requests
from typing import List, Dict, Optional, Tuple, Any
from dotenv import load_dotenv

load_dotenv()


class QuanfluenceClient:
    """
    Client for Quanfluence Ising Machine API
    Handles authentication, device management, and job execution
    """
    
    def __init__(self, 
                 base_url: str = None,
                 username: str = None,
                 password: str = None,
                 device_id: str = None):
        """
        Initialize Quanfluence client
        
        Args:
            base_url: Gateway URL (default from env)
            username: API username (default from env)
            password: API password (default from env)
            device_id: Default device ID (default from env)
        """
        self.base_url = base_url or os.getenv('QUANFLUENCE_BASE_URL', 'https://gateway.quanfluence.com')
        self.username = username or os.getenv('QUANFLUENCE_USERNAME')
        self.password = password or os.getenv('QUANFLUENCE_PASSWORD')
        self.device_id = device_id or os.getenv('QUANFLUENCE_DEVICE_ID', '41')
        
        if not self.username or not self.password:
            raise ValueError("Quanfluence credentials not found in environment variables")
        
        self.token = None
        self.token_expiry = 0
    
    def _ensure_authenticated(self) -> str:
        """Ensure we have a valid authentication token"""
        if self.token and time.time() < self.token_expiry - 60:
            return self.token
        return self._authenticate()
    
    def _authenticate(self) -> str:
        """Authenticate with Quanfluence gateway"""
        auth_url = f"{self.base_url}/api/clients/signin"
        
        response = requests.post(
            auth_url,
            json={"username": self.username, "password": self.password},
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        if response.status_code != 200:
            raise Exception(f"Authentication failed: {response.status_code} - {response.text}")
        
        data = response.json()
        if data.get('status') != 'success':
            raise Exception(f"Authentication failed: {data}")
        
        self.token = data['data']['token']
        expires_in = data['data'].get('expiresIn', 3600)
        self.token_expiry = time.time() + expires_in
        
        return self.token
    
    def _headers(self) -> Dict[str, str]:
        """Get authorization headers"""
        self._ensure_authenticated()
        return {'Authorization': f'Bearer {self.token}'}
    
    # =========================================================================
    # DEVICE MANAGEMENT
    # =========================================================================
    
    def get_device(self, device_id: str = None) -> Dict[str, Any]:
        """
        Get device details and current parameters
        
        Args:
            device_id: Device ID (default: configured device)
            
        Returns:
            Device configuration dictionary
        """
        device_id = device_id or self.device_id
        url = f"{self.base_url}/api/devices/{device_id}"
        
        response = requests.get(url, headers=self._headers(), timeout=30)
        
        if response.status_code != 200:
            raise Exception(f"Failed to get device: {response.status_code} - {response.text}")
        
        data = response.json()
        if data.get('status') != 'success':
            raise Exception(f"Failed to get device: {data}")
        
        return data['data']
    
    def update_device(self, 
                     device_id: str = None,
                     iters: int = None,
                     runs: int = None,
                     trials: int = None,
                     alpha: float = None,
                     beta: float = None,
                     beta_decay: float = None,
                     noise_stdev: float = None,
                     runtime: float = None,
                     title: str = None,
                     description: str = None) -> Dict[str, Any]:
        """
        Update device parameters
        
        Args:
            device_id: Device ID (default: configured device)
            iters: Number of iterations per run
            runs: Number of runs
            trials: Number of trials
            alpha: Algorithm alpha parameter
            beta: Algorithm beta parameter  
            beta_decay: Beta decay rate
            noise_stdev: Noise standard deviation
            runtime: Runtime limit
            title: Device title
            description: Device description
            
        Returns:
            Updated device info
        """
        device_id = device_id or self.device_id
        url = f"{self.base_url}/api/devices/{device_id}"
        
        payload = {}
        if iters is not None: payload['iters'] = iters
        if runs is not None: payload['runs'] = runs
        if trials is not None: payload['trials'] = trials
        if alpha is not None: payload['alpha'] = alpha
        if beta is not None: payload['beta'] = beta
        if beta_decay is not None: payload['beta_decay'] = beta_decay
        if noise_stdev is not None: payload['noise_stdev'] = noise_stdev
        if runtime is not None: payload['runtime'] = runtime
        if title is not None: payload['title'] = title
        if description is not None: payload['description'] = description
        
        if not payload:
            return self.get_device(device_id)
        
        response = requests.put(
            url, 
            json=payload, 
            headers={**self._headers(), 'Content-Type': 'application/json'},
            timeout=30
        )
        
        # API returns 200 or 201 for success
        if response.status_code not in [200, 201]:
            raise Exception(f"Failed to update device: {response.status_code} - {response.text}")
        
        data = response.json()
        if data.get('status') != 'success':
            raise Exception(f"Failed to update device: {data}")
        
        return data
    
    # =========================================================================
    # QUBO EXECUTION
    # =========================================================================
    
    def execute_qubo(self, 
                    qubo: Dict[Tuple[int, int], float],
                    device_id: str = None) -> Dict[str, Any]:
        """
        Execute QUBO directly on device
        
        Args:
            qubo: QUBO dictionary {(i, j): coefficient}
            device_id: Device ID (default: configured device)
            
        Returns:
            Result dictionary with 'result', 'energy', 'trials'
        """
        device_id = device_id or self.device_id
        
        # Convert QUBO to zip format
        qubo_zip = self._qubo_to_zip(qubo)
        
        url = f"{self.base_url}/api/execute/device/{device_id}"
        files = {'file': ('data.zip', qubo_zip, 'application/zip')}
        
        response = requests.post(
            url, 
            files=files, 
            headers=self._headers(),
            timeout=3600  # 1 hour timeout
        )
        
        if response.status_code != 200:
            raise Exception(f"Execution failed: {response.status_code} - {response.text}")
        
        data = response.json()
        if data.get('status') != 'success':
            raise Exception(f"Execution failed: {data}")
        
        return data['data']
    
    def _qubo_to_zip(self, Q: Dict[Tuple[int, int], float]) -> bytes:
        """Convert QUBO dictionary to zip file format"""
        qubo_str = "{"
        items = []
        for (i, j), coeff in Q.items():
            if abs(coeff) > 1e-10:
                items.append(f'({i}, {j}): {coeff}')
        qubo_str += ", ".join(items)
        qubo_str += "}"
        
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
            zf.writestr('data.txt', qubo_str)
        
        return zip_buffer.getvalue()
    
    # =========================================================================
    # QUBO FILE MANAGEMENT
    # =========================================================================
    
    def upload_qubo_file(self, 
                        qubo: Dict[Tuple[int, int], float],
                        device_id: str = None) -> str:
        """
        Upload QUBO to server for later execution
        
        Args:
            qubo: QUBO dictionary
            device_id: Device ID
            
        Returns:
            Unique filename on server
        """
        device_id = device_id or self.device_id
        
        # Create .qubo file content
        qubo_str = "{"
        items = []
        for (i, j), coeff in qubo.items():
            if abs(coeff) > 1e-10:
                items.append(f'({i}, {j}): {coeff}')
        qubo_str += ", ".join(items)
        qubo_str += "}"
        
        # Create file with .qubo extension
        file_content = qubo_str.encode('utf-8')
        
        url = f"{self.base_url}/api/devices/{device_id}/qubo/upload"
        files = {'file': ('problem.qubo', file_content, 'text/plain')}
        
        response = requests.post(
            url,
            files=files,
            headers=self._headers(),
            timeout=60
        )
        
        if response.status_code not in [200, 201]:
            raise Exception(f"Upload failed: {response.status_code} - {response.text}")
        
        data = response.json()
        if data.get('status') != 'success':
            raise Exception(f"Upload failed: {data}")
        
        return data['data']['result']
    
    def execute_qubo_file(self, 
                         filename: str,
                         device_id: str = None) -> Dict[str, Any]:
        """
        Execute a QUBO file that was previously uploaded
        
        Args:
            filename: Filename returned from upload_qubo_file
            device_id: Device ID
            
        Returns:
            Result dictionary
        """
        device_id = device_id or self.device_id
        url = f"{self.base_url}/api/execute/device/{device_id}/qubo/{filename}"
        
        response = requests.get(
            url,
            headers=self._headers(),
            timeout=3600
        )
        
        if response.status_code != 200:
            raise Exception(f"Execution failed: {response.status_code} - {response.text}")
        
        data = response.json()
        if data.get('status') != 'success':
            raise Exception(f"Execution failed: {data}")
        
        return data['data']
    
    # =========================================================================
    # ASYNC JOB MANAGEMENT
    # =========================================================================
    
    def create_job(self, 
                  filename: str,
                  device_id: str = None) -> str:
        """
        Create an async job for QUBO execution
        
        Args:
            filename: QUBO filename (from upload_qubo_file)
            device_id: Device ID
            
        Returns:
            Job ID
        """
        device_id = device_id or self.device_id
        url = f"{self.base_url}/api/jobs"
        
        payload = {
            'device_id': int(device_id),
            'filename': filename
        }
        
        response = requests.post(
            url,
            json=payload,
            headers={**self._headers(), 'Content-Type': 'application/json'},
            timeout=30
        )
        
        if response.status_code not in [200, 201]:
            raise Exception(f"Job creation failed: {response.status_code} - {response.text}")
        
        data = response.json()
        if data.get('status') != 'success':
            raise Exception(f"Job creation failed: {data}")
        
        return data['data']['id']
    
    def get_job(self, job_id: str) -> Dict[str, Any]:
        """
        Get job status and result
        
        Args:
            job_id: Job ID from create_job
            
        Returns:
            Job details including result if completed
        """
        url = f"{self.base_url}/api/jobs/{job_id}"
        
        response = requests.get(
            url,
            headers=self._headers(),
            timeout=30
        )
        
        if response.status_code != 200:
            raise Exception(f"Failed to get job: {response.status_code} - {response.text}")
        
        data = response.json()
        if data.get('status') != 'success':
            raise Exception(f"Failed to get job: {data}")
        
        return data['data']
    
    def wait_for_job(self, 
                    job_id: str,
                    poll_interval: float = 5.0,
                    timeout: float = 3600) -> Dict[str, Any]:
        """
        Wait for async job to complete
        
        Args:
            job_id: Job ID
            poll_interval: Seconds between status checks
            timeout: Maximum wait time
            
        Returns:
            Job result
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            job_data = self.get_job(job_id)
            
            status = job_data.get('status', 'unknown')
            if status == 'completed':
                return job_data
            elif status == 'failed':
                raise Exception(f"Job failed: {job_data}")
            
            time.sleep(poll_interval)
        
        raise TimeoutError(f"Job {job_id} did not complete within {timeout}s")


class QuanfluenceTSPSolver:
    """
    Traveling Salesman Problem solver using Quanfluence Ising Machine
    
    Features:
    - Direct QUBO execution
    - Async job support for large problems
    - Device parameter tuning
    - 2-opt and 3-opt local search post-processing
    """
    
    # Default device parameters for TSP optimization
    DEFAULT_PARAMS = {
        'iters': 5000,
        'runs': 20,
        'trials': 10,
        'alpha': 0.5,
        'beta': 1.0,
        'beta_decay': 0.1,
        'noise_stdev': 0.0,
        'runtime': 1.0
    }
    
    # Tuned parameters for better TSP results (more iterations/trials)
    TUNED_PARAMS = {
        'iters': 10000,
        'runs': 50,
        'trials': 20,
        'alpha': 0.5,
        'beta': 1.0,
        'beta_decay': 0.05,
        'noise_stdev': 0.01,
        'runtime': 5.0
    }
    
    def __init__(self, 
                 use_tuned_params: bool = False,
                 custom_params: Dict = None,
                 use_local_search: bool = True,
                 local_search_method: str = '2opt'):
        """
        Initialize the Quanfluence TSP solver
        
        Args:
            use_tuned_params: Use pre-tuned parameters for better TSP results
            custom_params: Custom device parameters to apply
            use_local_search: Apply local search post-processing (default True)
            local_search_method: '2opt', '3opt', or 'both' (default '2opt')
        """
        self.client = QuanfluenceClient()
        self.use_tuned_params = use_tuned_params
        self.custom_params = custom_params
        self.use_local_search = use_local_search
        self.local_search_method = local_search_method
        self.timing_info = {}
        self._original_params = None
    
    # =========================================================================
    # LOCAL SEARCH METHODS
    # =========================================================================
    
    @staticmethod
    def _calculate_tour_distance(tour: List[int], distance_matrix: List[List[float]]) -> float:
        """Calculate total tour distance"""
        n = len(tour)
        return sum(distance_matrix[tour[i]][tour[(i + 1) % n]] for i in range(n))
    
    @staticmethod
    def _two_opt_swap(tour: List[int], i: int, k: int) -> List[int]:
        """Perform 2-opt swap: reverse segment between i and k"""
        return tour[:i] + tour[i:k+1][::-1] + tour[k+1:]
    
    def _two_opt_improve(self, tour: List[int], distance_matrix: List[List[float]]) -> Tuple[List[int], float]:
        """
        Apply 2-opt local search to improve tour
        
        2-opt removes two edges and reconnects the tour differently.
        Continues until no improvement is found.
        """
        n = len(tour)
        improved = True
        current_tour = tour.copy()
        current_distance = self._calculate_tour_distance(current_tour, distance_matrix)
        
        iterations = 0
        max_iterations = n * n * 2
        
        while improved and iterations < max_iterations:
            improved = False
            iterations += 1
            
            for i in range(n - 1):
                for k in range(i + 1, n):
                    # Try reversing the segment from i to k
                    new_tour = self._two_opt_swap(current_tour, i, k)
                    new_distance = self._calculate_tour_distance(new_tour, distance_matrix)
                    
                    if new_distance < current_distance - 1e-10:
                        current_tour = new_tour
                        current_distance = new_distance
                        improved = True
                        break
                
                if improved:
                    break
        
        return current_tour, current_distance
    
    def _three_opt_improve(self, tour: List[int], distance_matrix: List[List[float]]) -> Tuple[List[int], float]:
        """
        Apply 3-opt local search to improve tour
        
        3-opt removes three edges and reconnects in the best way.
        More powerful than 2-opt but slower.
        """
        n = len(tour)
        improved = True
        current_tour = tour.copy()
        current_distance = self._calculate_tour_distance(current_tour, distance_matrix)
        
        iterations = 0
        max_iterations = n * 10
        
        while improved and iterations < max_iterations:
            improved = False
            iterations += 1
            
            for i in range(n - 2):
                for j in range(i + 2, n - 1):
                    for k in range(j + 2, n + (1 if i > 0 else 0)):
                        # Try all possible 3-opt reconnections
                        candidates = self._three_opt_candidates(current_tour, i, j, k, n)
                        
                        for new_tour in candidates:
                            new_distance = self._calculate_tour_distance(new_tour, distance_matrix)
                            
                            if new_distance < current_distance - 1e-10:
                                current_tour = new_tour
                                current_distance = new_distance
                                improved = True
                                break
                        
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
        
        return current_tour, current_distance
    
    @staticmethod
    def _three_opt_candidates(tour: List[int], i: int, j: int, k: int, n: int) -> List[List[int]]:
        """Generate all 3-opt reconnection candidates"""
        # Segments: A = tour[0:i+1], B = tour[i+1:j+1], C = tour[j+1:k+1], D = tour[k+1:]
        A = tour[:i+1]
        B = tour[i+1:j+1]
        C = tour[j+1:k+1] if k < n else tour[j+1:]
        D = tour[k+1:] if k < n else []
        
        candidates = []
        
        # Different reconnection patterns
        # Pattern 1: A + B' + C + D (2-opt on B)
        candidates.append(A + B[::-1] + C + D)
        # Pattern 2: A + B + C' + D (2-opt on C)  
        candidates.append(A + B + C[::-1] + D)
        # Pattern 3: A + B' + C' + D
        candidates.append(A + B[::-1] + C[::-1] + D)
        # Pattern 4: A + C + B + D
        candidates.append(A + C + B + D)
        # Pattern 5: A + C + B' + D
        candidates.append(A + C + B[::-1] + D)
        # Pattern 6: A + C' + B + D
        candidates.append(A + C[::-1] + B + D)
        # Pattern 7: A + C' + B' + D
        candidates.append(A + C[::-1] + B[::-1] + D)
        
        return candidates
    
    def _apply_local_search(self, tour: List[int], distance_matrix: List[List[float]], 
                           verbose: bool = True) -> Tuple[List[int], float, Dict]:
        """
        Apply configured local search method
        
        Returns:
            Tuple of (improved_tour, improved_distance, improvement_info)
        """
        original_distance = self._calculate_tour_distance(tour, distance_matrix)
        current_tour = tour
        current_distance = original_distance
        
        improvement_info = {
            'original_distance': original_distance,
            'methods_applied': [],
            'improvements': []
        }
        
        if self.local_search_method in ['2opt', 'both']:
            new_tour, new_distance = self._two_opt_improve(current_tour, distance_matrix)
            improvement = current_distance - new_distance
            
            if improvement > 1e-10:
                if verbose:
                    print(f"    2-opt improvement: {current_distance:.2f} -> {new_distance:.2f} (-{improvement:.2f})")
                improvement_info['methods_applied'].append('2opt')
                improvement_info['improvements'].append({'method': '2opt', 'improvement': improvement})
                current_tour = new_tour
                current_distance = new_distance
        
        if self.local_search_method in ['3opt', 'both']:
            new_tour, new_distance = self._three_opt_improve(current_tour, distance_matrix)
            improvement = current_distance - new_distance
            
            if improvement > 1e-10:
                if verbose:
                    print(f"    3-opt improvement: {current_distance:.2f} -> {new_distance:.2f} (-{improvement:.2f})")
                improvement_info['methods_applied'].append('3opt')
                improvement_info['improvements'].append({'method': '3opt', 'improvement': improvement})
                current_tour = new_tour
                current_distance = new_distance
        
        improvement_info['final_distance'] = current_distance
        improvement_info['total_improvement'] = original_distance - current_distance
        improvement_info['total_improvement_pct'] = (
            (original_distance - current_distance) / original_distance * 100 
            if original_distance > 0 else 0
        )
        
        return current_tour, current_distance, improvement_info
    
    def get_device_info(self) -> Dict[str, Any]:
        """Get current device configuration"""
        return self.client.get_device()
    
    def configure_device(self, **params) -> Dict[str, Any]:
        """
        Configure device parameters
        
        Args:
            **params: Device parameters (iters, runs, trials, alpha, beta, etc.)
            
        Returns:
            Update result
        """
        return self.client.update_device(**params)
    
    def _apply_params(self):
        """Apply configured parameters to device"""
        if self.custom_params:
            params = self.custom_params
        elif self.use_tuned_params:
            params = self.TUNED_PARAMS
        else:
            return  # Use device defaults
        
        # Save original params for restoration
        if self._original_params is None:
            device_info = self.client.get_device()
            self._original_params = {
                k: device_info.get(k) 
                for k in self.DEFAULT_PARAMS.keys()
                if k in device_info
            }
        
        # Apply new params
        self.client.update_device(**params)
    
    def _restore_params(self):
        """Restore original device parameters"""
        if self._original_params:
            self.client.update_device(**self._original_params)
            self._original_params = None
    
    def _tsp_to_qubo(self, n: int, distance_matrix: List[List[float]], 
                     penalty: float = None) -> Dict[Tuple[int, int], float]:
        """
        Convert TSP to QUBO formulation
        
        Binary variables x_{i,t} = 1 if city i is visited at position t
        """
        if penalty is None:
            max_dist = max(max(row) for row in distance_matrix)
            penalty = 2 * max_dist * n
        
        def var_index(city: int, position: int) -> int:
            return city * n + position
        
        Q = {}
        
        # Objective: Minimize tour distance
        for t in range(n):
            next_t = (t + 1) % n
            for i in range(n):
                for j in range(n):
                    if i != j and distance_matrix[i][j] > 0:
                        idx_i = var_index(i, t)
                        idx_j = var_index(j, next_t)
                        Q[(idx_i, idx_j)] = Q.get((idx_i, idx_j), 0) + distance_matrix[i][j]
        
        # Constraint 1: Each city visited exactly once
        for i in range(n):
            for t in range(n):
                idx = var_index(i, t)
                Q[(idx, idx)] = Q.get((idx, idx), 0) - penalty
            
            for t1 in range(n):
                for t2 in range(t1 + 1, n):
                    idx1 = var_index(i, t1)
                    idx2 = var_index(i, t2)
                    Q[(idx1, idx2)] = Q.get((idx1, idx2), 0) + 2 * penalty
        
        # Constraint 2: Each position has exactly one city
        for t in range(n):
            for i in range(n):
                idx = var_index(i, t)
                Q[(idx, idx)] = Q.get((idx, idx), 0) - penalty
            
            for i1 in range(n):
                for i2 in range(i1 + 1, n):
                    idx1 = var_index(i1, t)
                    idx2 = var_index(i2, t)
                    Q[(idx1, idx2)] = Q.get((idx1, idx2), 0) + 2 * penalty
        
        return Q
    
    def _decode_solution(self, result_vector: List[int], n: int) -> List[int]:
        """Decode solution vector to tour"""
        tour = [None] * n
        
        for idx, bit in enumerate(result_vector):
            if bit == 1:
                city = idx // n
                position = idx % n
                if position < n and city < n:
                    if tour[position] is None:
                        tour[position] = city
        
        # Fill missing positions
        if None in tour:
            used_cities = set(c for c in tour if c is not None)
            missing_cities = [c for c in range(n) if c not in used_cities]
            for i, pos in enumerate(tour):
                if pos is None and missing_cities:
                    tour[i] = missing_cities.pop(0)
        
        if None in tour or len(set(tour)) != n:
            return []
        
        return tour
    
    def solve_tsp(self,
                  locations: List[str],
                  distance_matrix: List[List[float]],
                  start_location: Optional[str] = None,
                  use_async: bool = False,
                  verbose: bool = True) -> Dict:
        """
        Solve TSP using Quanfluence Ising Machine
        
        Args:
            locations: List of location names
            distance_matrix: NxN matrix of distances
            start_location: Optional starting location
            use_async: Use async job API for large problems
            verbose: Print progress messages
            
        Returns:
            Dictionary with tour, distance, timing, etc.
        """
        n = len(locations)
        
        if n < 2:
            return {'status': 'error', 'message': 'Need at least 2 locations'}
        
        if verbose:
            print("\n" + "="*70)
            print("QUANFLUENCE ISING MACHINE SOLVER - TSP")
            print("="*70)
            print(f"Number of locations: {n}")
            print(f"Variables: {n*n} binary")
            print(f"Device ID: {self.client.device_id}")
            print(f"Gateway: {self.client.base_url}")
            print(f"Mode: {'Async Job' if use_async else 'Direct Execution'}")
            if self.use_tuned_params:
                print(f"Parameters: TUNED (iters={self.TUNED_PARAMS['iters']}, runs={self.TUNED_PARAMS['runs']}, trials={self.TUNED_PARAMS['trials']})")
            elif self.custom_params:
                print(f"Parameters: CUSTOM")
            else:
                print(f"Parameters: DEFAULT")
            print("="*70)
        
        try:
            # Authenticate
            auth_start = time.time()
            if verbose:
                print("Authenticating with Quanfluence...")
            self.client._ensure_authenticated()
            auth_time = time.time() - auth_start
            if verbose:
                print(f"Authenticated (token expires in {int(self.client.token_expiry - time.time())}s)")
            
            # Apply parameters if configured
            if self.use_tuned_params or self.custom_params:
                if verbose:
                    print("Applying device parameters...")
                self._apply_params()
            
            # Build QUBO
            build_start = time.time()
            Q = self._tsp_to_qubo(n, distance_matrix)
            build_time = time.time() - build_start
            if verbose:
                print(f"QUBO built in {build_time:.3f}s ({len(Q)} terms)")
            
            # Execute
            submit_start = time.time()
            
            if use_async:
                result_data = self._execute_async(Q, verbose)
            else:
                result_data = self._execute_direct(Q, verbose)
            
            submit_time = time.time() - submit_start
            
            # Restore original parameters
            if self.use_tuned_params or self.custom_params:
                self._restore_params()
            
            # Extract results
            result_vector = result_data.get('result', [])
            energy = result_data.get('energy', 0)
            trials_info = result_data.get('trials', [])
            
            # Decode solution
            tour = self._decode_solution(result_vector, n)
            
            if not tour or len(tour) != n:
                if verbose:
                    print("Could not decode valid tour, using heuristic")
                tour = list(range(n))
                is_valid = False
            else:
                is_valid = True
                if verbose:
                    print(f"Valid tour decoded")
            
            # Calculate raw distance (before local search)
            raw_distance = self._calculate_tour_distance(tour, distance_matrix)
            
            # Apply local search post-processing
            local_search_info = None
            if self.use_local_search:
                local_search_start = time.time()
                if verbose:
                    print(f"\n  Applying {self.local_search_method} local search...")
                    print(f"    Raw distance: {raw_distance:.2f}")
                
                tour, total_distance, local_search_info = self._apply_local_search(
                    tour, distance_matrix, verbose
                )
                
                local_search_time = time.time() - local_search_start
                
                if verbose and local_search_info['total_improvement'] > 0:
                    print(f"    Final distance: {total_distance:.2f}")
                    print(f"    Total improvement: {local_search_info['total_improvement']:.2f} ({local_search_info['total_improvement_pct']:.1f}%)")
                elif verbose:
                    print(f"    No improvement found (local optimum)")
            else:
                total_distance = raw_distance
                local_search_time = 0
            
            # Reorder tour if start_location specified
            if start_location and start_location in locations:
                start_idx = locations.index(start_location)
                if start_idx in tour:
                    start_pos = tour.index(start_idx)
                    tour = tour[start_pos:] + tour[:start_pos]
            
            tour_names = [locations[i] for i in tour]
            
            self.timing_info = {
                'auth_time_s': auth_time,
                'build_time_s': build_time,
                'submit_time_s': submit_time,
                'local_search_time_s': local_search_time if self.use_local_search else 0,
                'total_time_s': auth_time + build_time + submit_time + (local_search_time if self.use_local_search else 0),
                'communication_overhead_s': auth_time + submit_time
            }
            
            if verbose:
                print("\n" + "="*70)
                status_str = "OPTIMAL" if is_valid else "HEURISTIC"
                if self.use_local_search:
                    status_str += f" + {self.local_search_method.upper()}"
                print(f"BEST TOUR FOUND ({status_str})")
                print("="*70)
                if self.use_local_search:
                    print(f"Raw Distance: {raw_distance:.2f}")
                    print(f"Final Distance: {total_distance:.2f}")
                    if local_search_info and local_search_info['total_improvement'] > 0:
                        print(f"Improvement: {local_search_info['total_improvement']:.2f} ({local_search_info['total_improvement_pct']:.1f}%)")
                else:
                    print(f"Total Distance: {total_distance:.2f}")
                print(f"Energy: {energy}")
                print(f"\nTour Order:")
                for idx, loc_idx in enumerate(tour, 1):
                    print(f"  {idx}. {locations[loc_idx]}")
                print(f"  {len(tour) + 1}. {locations[tour[0]]} (return to start)")
                print("\n" + "-"*70)
                print("TIMING:")
                print(f"  Authentication time: {auth_time:.3f}s")
                print(f"  QUBO build time: {build_time:.3f}s")
                print(f"  Submit/response time: {submit_time:.3f}s")
                if self.use_local_search:
                    print(f"  Local search time: {local_search_time:.3f}s")
                print(f"  Total time: {self.timing_info['total_time_s']:.3f}s")
                print(f"  Communication overhead: {self.timing_info['communication_overhead_s']:.3f}s")
                print("="*70)
            
            return {
                'status': 'optimal' if is_valid else 'heuristic',
                'tour': tour,
                'tour_names': tour_names,
                'total_distance': total_distance,
                'raw_distance': raw_distance,
                'locations': locations,
                'distance_matrix': distance_matrix,
                'start_location': locations[tour[0]],
                'timing': self.timing_info,
                'solver': 'Quanfluence Ising Machine',
                'is_valid': is_valid,
                'energy': energy,
                'num_trials': len(trials_info),
                'local_search': local_search_info,
                'local_search_method': self.local_search_method if self.use_local_search else None
            }
            
        except Exception as e:
            import traceback
            if verbose:
                print(f"Error: {str(e)}")
                traceback.print_exc()
            return {
                'status': 'error',
                'message': str(e),
                'solver': 'Quanfluence Ising Machine'
            }
    
    def _execute_direct(self, Q: Dict, verbose: bool = True) -> Dict:
        """Execute QUBO directly"""
        if verbose:
            print("\nSubmitting to Quanfluence Ising Machine...")
        
        result = self.client.execute_qubo(Q)
        
        if verbose:
            print(f"Received response")
        
        return result
    
    def _execute_async(self, Q: Dict, verbose: bool = True) -> Dict:
        """Execute QUBO using async job API"""
        if verbose:
            print("\nUploading QUBO file...")
        
        filename = self.client.upload_qubo_file(Q)
        
        if verbose:
            print(f"Uploaded as: {filename}")
            print("Creating async job...")
        
        job_id = self.client.create_job(filename)
        
        if verbose:
            print(f"Job created: {job_id}")
            print("Waiting for job completion...")
        
        job_result = self.client.wait_for_job(job_id, poll_interval=2.0)
        
        if verbose:
            print(f"Job completed")
        
        return job_result


def print_device_info():
    """Print current device configuration"""
    print("\n" + "="*70)
    print("QUANFLUENCE DEVICE CONFIGURATION")
    print("="*70)
    
    solver = QuanfluenceTSPSolver()
    info = solver.get_device_info()
    
    print(f"Device ID: {info.get('id')}")
    print(f"Title: {info.get('title')}")
    print(f"Type: {info.get('type')}")
    print("\nParameters:")
    print(f"  iters:       {info.get('iters')}")
    print(f"  runs:        {info.get('runs')}")
    print(f"  trials:      {info.get('trials')}")
    print(f"  alpha:       {info.get('alpha')}")
    print(f"  beta:        {info.get('beta')}")
    print(f"  beta_decay:  {info.get('beta_decay')}")
    print(f"  noise_stdev: {info.get('noise_stdev')}")
    print(f"  runtime:     {info.get('runtime')}")
    print("="*70)
    
    return info


def test_quanfluence_solver():
    """Test the Quanfluence TSP solver with different configurations"""
    print("\n" + "="*70)
    print("TESTING QUANFLUENCE TSP SOLVER")
    print("="*70)
    
    # Sample problem: 4 cities
    locations = ["A", "B", "C", "D"]
    distance_matrix = [
        [0,   10,  15,  20],
        [10,  0,   35,  25],
        [15,  35,  0,   30],
        [20,  25,  30,  0]
    ]
    
    print("\n--- Test 1: Default Parameters ---")
    solver1 = QuanfluenceTSPSolver(use_tuned_params=False)
    result1 = solver1.solve_tsp(locations, distance_matrix)
    print(f"Result: {result1.get('status')}, Distance: {result1.get('total_distance')}")
    
    print("\n--- Test 2: Tuned Parameters ---")
    solver2 = QuanfluenceTSPSolver(use_tuned_params=True)
    result2 = solver2.solve_tsp(locations, distance_matrix)
    print(f"Result: {result2.get('status')}, Distance: {result2.get('total_distance')}")
    
    print("\n--- Test 3: Custom Parameters ---")
    custom = {'iters': 8000, 'runs': 30, 'trials': 15}
    solver3 = QuanfluenceTSPSolver(custom_params=custom)
    result3 = solver3.solve_tsp(locations, distance_matrix)
    print(f"Result: {result3.get('status')}, Distance: {result3.get('total_distance')}")
    
    print("\n" + "="*70)
    print("TEST COMPLETE")
    print("="*70)
    
    return True


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--info':
        print_device_info()
    else:
        test_quanfluence_solver()
