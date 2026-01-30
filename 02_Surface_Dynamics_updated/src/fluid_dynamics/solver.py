"""
Main solver for fluid surface dynamics.
Based on: Galeano-Rios et al. (2017), J. Fluid Mech. 826, pp. 97-127
"""
import numpy as np
from typing import Dict, Tuple, Optional
import time
from dataclasses import dataclass


@dataclass
class SimulationState:
    """Container for simulation state"""
    eta: np.ndarray          # Surface elevation
    phi: np.ndarray          # Velocity potential
    w3: np.ndarray           # Vertical vortical component
    w1: np.ndarray           # Horizontal vortical component
    time: float = 0.0        # Current time
    step: int = 0            # Current step
    
    def copy(self):
        """Deep copy of state"""
        return SimulationState(
            eta=self.eta.copy(),
            phi=self.phi.copy(),
            w3=self.w3.copy(),
            w1=self.w1.copy(),
            time=self.time,
            step=self.step
        )


class FluidSolver:
    """
    유체 표면 역학 솔버 (Fluid surface dynamics solver)
    
    Solves the coupled system:
    - Δφ = 0 (Laplace for potential)
    - w_t = (1/Re) Δw (heat equation for vortical components)
    - η_t = φ_z + w³ + (2/Re) Δ_H η (surface evolution)
    - φ_t = -(1/Fr)η + (1/We)κ[η] + (2/Re)Δ_H φ (dynamic BC)
    
    Uses implicit Euler for unconditional stability.
    """
    
    def __init__(self, config, backend="auto"):
        """
        Initialize solver
        
        Args:
            config: SimulationConfig object
            backend: "cpu", "gpu", or "auto"
        """
        from .config import SimulationConfig
        from .operators import build_implicit_matrices, get_backend, compute_curvature
        
        self.config = config
        
        # Determine backend
        self.xp, self.sp, self.device = get_backend(backend)
        if config.compute.verbose:
            print(f"Using device: {self.device}")
        
        # Build operators
        if config.compute.verbose:
            print("Building matrices...")
        self.matrices = build_implicit_matrices(config, self.device)
        self.compute_curvature = compute_curvature
        
        # Get parameters
        self.grid = config.grid_spacing
        self.dims = config.dimensionless_numbers
        self.nx = config.numerical.nx
        self.nz = config.numerical.nz
        self.nt = config.numerical.nt
        
        # Initialize state
        self.state = self._initialize_state()
        
        # History for visualization
        self.history = {
            'eta': [],
            'phi': [],
            'w3_surf': [],
            'time': []
        }
        
        # Timing
        self.timings = {}
        
        if config.compute.verbose:
            print("Solver initialized.")
    
    def _initialize_state(self) -> SimulationState:
        """Initialize simulation state with IC"""
        xp = self.xp
        
        # Create grid
        x_vec = xp.linspace(0, self.grid['L'], self.nx, endpoint=False)
        
        # Initialize arrays
        eta = xp.zeros(self.nx)
        phi = xp.zeros((self.nz, self.nx))
        w3 = xp.zeros((self.nz, self.nx))
        w1 = xp.zeros((self.nz, self.nx))
        
        # Apply initial condition for eta
        ic = self.config.initial
        if ic.type == "gaussian":
            x0 = self.grid['L'] / 2
            sigma_x = self.grid['L'] * ic.width_factor
            for j in range(self.nx):
                eta[j] = ic.amplitude * xp.exp(-(x_vec[j] - x0)**2 / (2 * sigma_x**2))
        
        elif ic.type == "sine":
            for j in range(self.nx):
                eta[j] = ic.amplitude * xp.sin(2 * xp.pi * x_vec[j] / self.grid['L'])
        
        elif ic.type == "custom" and ic.custom_function is not None:
            eta = ic.custom_function(x_vec)
        
        return SimulationState(eta=eta, phi=phi, w3=w3, w1=w1, time=0.0, step=0)
    
    def _solve_sparse(self, A, b):
        """Solve sparse linear system with appropriate backend"""
        if self.device == "gpu":
            import cupyx.scipy.sparse.linalg as cp_linalg
            return cp_linalg.spsolve(A, b)
        else:
            from scipy.sparse.linalg import spsolve
            return spsolve(A, b)
    
    def step(self):
        """
        Advance one time step using implicit Euler
        
        Returns:
            Current state
        """
        xp = self.xp
        state = self.state
        dx, dz, dt = self.grid['dx'], self.grid['dz'], self.grid['dt']
        Re, Fr, We = self.dims['Re'], self.dims['Fr'], self.dims['We']
        
        # Unpack matrices
        Delta_H = self.matrices['Delta_H']
        A_laplace = self.matrices['A_laplace']
        A_phi_surf = self.matrices['A_phi_surf']
        A_eta = self.matrices['A_eta']
        A_heat = self.matrices['A_heat']
        
        # Step 1: Evolve w³ via heat equation
        w3_flat = state.w3.flatten()
        w3_new_flat = self._solve_sparse(A_heat, w3_flat)
        w3_new = w3_new_flat.reshape((self.nz, self.nx))
        
        # Apply BCs for w³
        # Top: w³ = (2/Re) Δ_H η (from eq 2.17)
        w3_new[-1, :] = (2.0 / Re) * Delta_H.dot(state.eta)
        
        # Bottom: non-slip u_z = 0, so w³ = -∂φ/∂z
        phi_z_bottom = (state.phi[1, :] - state.phi[0, :]) / dz
        w3_new[0, :] = -phi_z_bottom
        
        # Step 2: Evolve w¹ via heat equation
        w1_flat = state.w1.flatten()
        w1_new_flat = self._solve_sparse(A_heat, w1_flat)
        w1_new = w1_new_flat.reshape((self.nz, self.nx))
        
        # Bottom BC for w¹: non-slip u_x = 0, so w¹ = -∂φ/∂x
        for j in range(self.nx):
            phi_x_bottom = (state.phi[0, (j + 1) % self.nx] - 
                           state.phi[0, (j - 1) % self.nx]) / (2 * dx)
            w1_new[0, j] = -phi_x_bottom
        
        # Step 3: Update φ at surface
        kappa_eta = self.compute_curvature(state.eta, dx, self.device)
        phi_surf = state.phi[-1, :].copy()
        
        b_phi = phi_surf - (dt / Fr) * state.eta + (dt / We) * kappa_eta
        phi_surf_new = self._solve_sparse(A_phi_surf, b_phi)
        
        # Step 4: Solve Laplace for interior φ
        b_laplace = xp.zeros(self.nx * self.nz)
        
        # Top: Dirichlet
        for j in range(self.nx):
            b_laplace[(self.nz - 1) * self.nx + j] = -phi_surf_new[j]
        
        # Bottom: non-slip contribution
        for j in range(self.nx):
            b_laplace[j] += -2.0 * dz * w3_new[0, j]
        
        phi_vec = self._solve_sparse(A_laplace, b_laplace)
        phi_new = phi_vec.reshape((self.nz, self.nx))
        
        # Step 5: Update η
        phi_z_surf = (phi_new[-1, :] - phi_new[-2, :]) / dz
        w3_surf = w3_new[-1, :]
        
        b_eta = state.eta + dt * (phi_z_surf + w3_surf)
        eta_new = self._solve_sparse(A_eta, b_eta)
        
        # Update state
        state.eta = eta_new
        state.phi = phi_new
        state.w3 = w3_new
        state.w1 = w1_new
        state.time += dt * self.config.numerical.unit_time
        state.step += 1
        
        return state
    
    def run(self, save_history=True):
        """
        Run full simulation
        
        Args:
            save_history: Whether to save states for visualization
        
        Returns:
            Final state
        """
        start_time = time.time()
        save_interval = max(1, self.nt // self.config.compute.save_interval)
        
        if self.config.compute.verbose:
            print(f"\nRunning simulation ({self.nt} steps)...")
        
        for n in range(self.nt):
            self.step()
            
            # Save history
            if save_history and n % save_interval == 0:
                # Convert to CPU if on GPU
                if self.device == "gpu":
                    import cupy as cp
                    eta_cpu = cp.asnumpy(self.state.eta)
                    phi_cpu = cp.asnumpy(self.state.phi)
                    w3_surf_cpu = cp.asnumpy(self.state.w3[-1, :])
                else:
                    eta_cpu = self.state.eta.copy()
                    phi_cpu = self.state.phi.copy()
                    w3_surf_cpu = self.state.w3[-1, :].copy()
                
                self.history['eta'].append(eta_cpu)
                self.history['phi'].append(phi_cpu)
                self.history['w3_surf'].append(w3_surf_cpu)
                self.history['time'].append(self.state.time)
            
            # Progress
            if self.config.compute.verbose and (n + 1) % (self.nt // 10) == 0:
                eta_max = float(np.max(np.abs(self.state.eta)))
                print(f"  {100 * (n + 1) / self.nt:3.0f}% | "
                      f"t = {self.state.time:.3f}s | "
                      f"max|η| = {eta_max:.6f}")
        
        elapsed = time.time() - start_time
        self.timings['simulation'] = elapsed
        
        if self.config.compute.verbose:
            print(f"\nSimulation complete in {elapsed:.2f}s")
            print(f"Performance: {self.nt/elapsed:.1f} steps/sec")
        
        return self.state
    
    def compute_velocity_field(self):
        """
        Compute velocity field u = ∇φ + w
        
        Returns:
            Tuple of (u_x, u_z) arrays
        """
        xp = self.xp
        state = self.state
        dx, dz = self.grid['dx'], self.grid['dz']
        
        u_x = xp.zeros((self.nz, self.nx))
        u_z = xp.zeros((self.nz, self.nx))
        
        for i in range(self.nz):
            for j in range(self.nx):
                # u_x = ∂φ/∂x + w¹
                u_x[i, j] = (state.phi[i, (j + 1) % self.nx] - 
                            state.phi[i, (j - 1) % self.nx]) / (2 * dx) + state.w1[i, j]
                
                # u_z = ∂φ/∂z + w³
                if i == 0:
                    u_z[i, j] = (state.phi[i + 1, j] - state.phi[i, j]) / dz + state.w3[i, j]
                elif i == self.nz - 1:
                    u_z[i, j] = (state.phi[i, j] - state.phi[i - 1, j]) / dz + state.w3[i, j]
                else:
                    u_z[i, j] = (state.phi[i + 1, j] - state.phi[i - 1, j]) / (2 * dz) + state.w3[i, j]
        
        # Convert to CPU if on GPU
        if self.device == "gpu":
            import cupy as cp
            u_x = cp.asnumpy(u_x)
            u_z = cp.asnumpy(u_z)
        
        return u_x, u_z
    
    def verify_boundary_conditions(self):
        """
        Verify that boundary conditions are satisfied
        
        Returns:
            Dictionary of BC errors
        """
        u_x, u_z = self.compute_velocity_field()
        
        # Check non-slip at bottom
        u_x_bottom_max = np.max(np.abs(u_x[0, :]))
        u_z_bottom_max = np.max(np.abs(u_z[0, :]))
        
        return {
            'bottom_u_x': u_x_bottom_max,
            'bottom_u_z': u_z_bottom_max,
            'passes': u_x_bottom_max < 1e-8 and u_z_bottom_max < 1e-6
        }
    
    def reset(self):
        """Reset solver to initial conditions"""
        self.state = self._initialize_state()
        self.history = {'eta': [], 'phi': [], 'w3_surf': [], 'time': []}
        self.timings = {}


def run_parameter_sweep(base_config, param_name, param_values, backend="cpu"):
    """
    Run simulations with varying parameter values
    
    Args:
        base_config: Base SimulationConfig
        param_name: Parameter to vary (e.g., "physical.nu")
        param_values: List of values to try
        backend: "cpu" or "gpu"
    
    Returns:
        List of (param_value, final_state, solver) tuples
    """
    results = []
    
    for value in param_values:
        # Create new config with modified parameter
        import copy
        config = copy.deepcopy(base_config)
        
        # Set parameter (handle nested attributes)
        parts = param_name.split('.')
        obj = config
        for part in parts[:-1]:
            obj = getattr(obj, part)
        setattr(obj, parts[-1], value)
        
        # Run simulation
        print(f"\n{'='*60}")
        print(f"Running with {param_name} = {value}")
        print(f"{'='*60}")
        
        solver = FluidSolver(config, backend=backend)
        final_state = solver.run()
        
        results.append((value, final_state, solver))
    
    return results


if __name__ == "__main__":
    from .config import SimulationConfig, get_preset
    
    # Test solver
    config = get_preset("fast")
    config.summary()
    
    solver = FluidSolver(config, backend="cpu")
    final_state = solver.run()
    
    # Verify BCs
    bc_check = solver.verify_boundary_conditions()
    print(f"\nBoundary condition check:")
    print(f"  Bottom u_x: {bc_check['bottom_u_x']:.2e}")
    print(f"  Bottom u_z: {bc_check['bottom_u_z']:.2e}")
    print(f"  Passes: {bc_check['passes']}")
