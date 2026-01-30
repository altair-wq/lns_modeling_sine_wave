"""
Operator construction for fluid dynamics simulations.
Supports both CPU (numpy/scipy) and GPU (cupy) backends.
"""
import numpy as np
from scipy.sparse import lil_matrix, csr_matrix, eye


def get_backend(device="cpu"):
    """
    Get appropriate backend (numpy/cupy) based on device
    
    Args:
        device: "cpu", "gpu", or "auto"
    
    Returns:
        Tuple of (np_module, sp_module, device_name)
    """
    if device == "auto":
        try:
            import cupy as cp
            import cupyx.scipy.sparse as cp_sparse
            # Test if GPU is available
            cp.cuda.Device(0).compute_capability
            return cp, cp_sparse, "gpu"
        except:
            return np, None, "cpu"
    
    elif device == "gpu":
        try:
            import cupy as cp
            import cupyx.scipy.sparse as cp_sparse
            return cp, cp_sparse, "gpu"
        except ImportError:
            raise ImportError("CuPy not installed. Install with: pip install cupy-cuda11x")
    
    else:  # cpu
        return np, None, "cpu"


def build_horizontal_laplacian(n, dx, backend="cpu"):
    """
    Build 1D horizontal Laplacian with periodic BC
    
    Args:
        n: Number of grid points
        dx: Grid spacing
        backend: "cpu" or "gpu"
    
    Returns:
        Sparse matrix for horizontal Laplacian
    """
    D = lil_matrix((n, n))
    coeff = 1.0 / (dx**2)
    
    for j in range(n):
        D[j, j] = -2.0 * coeff
        D[j, (j + 1) % n] = coeff
        D[j, (j - 1) % n] = coeff
    
    D_csr = D.tocsr()
    
    if backend == "gpu":
        try:
            import cupyx.scipy.sparse as cp_sparse
            return cp_sparse.csr_matrix(D_csr)
        except:
            return D_csr
    
    return D_csr


def build_laplace_matrix_nonslip(nx, nz, dx, dz, backend="cpu"):
    """
    Build Laplace equation matrix Δφ = 0 with:
    - Periodic BC in x
    - Non-slip BC at bottom: ∂φ/∂z = -w³
    - Dirichlet BC at top: φ = φ_surface
    
    Args:
        nx, nz: Number of grid points
        dx, dz: Grid spacing
        backend: "cpu" or "gpu"
    
    Returns:
        Sparse matrix for Laplace equation
    """
    N = nx * nz
    A = lil_matrix((N, N))
    alpha = (dz / dx)**2
    
    for i in range(nz):
        for j in range(nx):
            idx = i * nx + j
            
            # Laplacian coefficients
            A[idx, idx] = -2 * alpha - 2
            
            # x-neighbors (periodic)
            A[idx, i * nx + (j + 1) % nx] = alpha
            A[idx, i * nx + (j - 1) % nx] = alpha
            
            # z-neighbors
            if i > 0:
                A[idx, (i - 1) * nx + j] = 1
            else:
                # Bottom: non-slip BC (ghost point)
                A[idx, (i + 1) * nx + j] = 2
            
            if i < nz - 1:
                A[idx, (i + 1) * nx + j] = 1
            # Top: Dirichlet handled in RHS
    
    A_csr = A.tocsr()
    
    if backend == "gpu":
        try:
            import cupyx.scipy.sparse as cp_sparse
            return cp_sparse.csr_matrix(A_csr)
        except:
            return A_csr
    
    return A_csr


def build_heat_matrix_2d(nx, nz, dx, dz, backend="cpu"):
    """
    Build 2D Laplacian for heat equation with:
    - Periodic BC in x
    - Neumann BC at top and bottom
    
    Args:
        nx, nz: Number of grid points
        dx, dz: Grid spacing
        backend: "cpu" or "gpu"
    
    Returns:
        Sparse matrix for 2D Laplacian
    """
    N = nx * nz
    A = lil_matrix((N, N))
    
    coeff_x = 1.0 / (dx**2)
    coeff_z = 1.0 / (dz**2)
    
    for i in range(nz):
        for j in range(nx):
            idx = i * nx + j
            
            A[idx, idx] = -2.0 * coeff_x - 2.0 * coeff_z
            
            # x-neighbors (periodic)
            A[idx, i * nx + (j + 1) % nx] = coeff_x
            A[idx, i * nx + (j - 1) % nx] = coeff_x
            
            # z-neighbors with Neumann BC
            if i > 0:
                A[idx, (i - 1) * nx + j] = coeff_z
            else:
                A[idx, (i + 1) * nx + j] += coeff_z
            
            if i < nz - 1:
                A[idx, (i + 1) * nx + j] = coeff_z
            else:
                A[idx, (i - 1) * nx + j] += coeff_z
    
    A_csr = A.tocsr()
    
    if backend == "gpu":
        try:
            import cupyx.scipy.sparse as cp_sparse
            return cp_sparse.csr_matrix(A_csr)
        except:
            return A_csr
    
    return A_csr


def compute_curvature(eta, dx, backend="cpu"):
    """
    Compute surface curvature κ[η] = ∂²η/∂x² with periodic BC
    
    Args:
        eta: Surface elevation array
        dx: Grid spacing
        backend: "cpu" or "gpu"
    
    Returns:
        Curvature array
    """
    if backend == "gpu":
        try:
            import cupy as cp
            xp = cp
        except:
            xp = np
    else:
        xp = np
    
    n = len(eta)
    kappa = xp.zeros_like(eta)
    coeff = 1.0 / (dx**2)
    
    for j in range(n):
        kappa[j] = (eta[(j + 1) % n] - 2 * eta[j] + eta[(j - 1) % n]) * coeff
    
    return kappa


def build_implicit_matrices(config, backend="cpu"):
    """
    Build all matrices needed for implicit Euler time stepping
    
    Args:
        config: SimulationConfig object
        backend: "cpu" or "gpu"
    
    Returns:
        Dictionary of matrices
    """
    from .config import SimulationConfig
    
    # Get grid parameters
    grid = config.grid_spacing
    dims = config.dimensionless_numbers
    nx, nz = config.numerical.nx, config.numerical.nz
    dx, dz, dt = grid['dx'], grid['dz'], grid['dt']
    Re = dims['Re']
    
    # Build base operators
    Delta_H = build_horizontal_laplacian(nx, dx, backend)
    Laplacian_2d = build_heat_matrix_2d(nx, nz, dx, dz, backend)
    A_laplace = build_laplace_matrix_nonslip(nx, nz, dx, dz, backend)
    
    # Identity matrices
    if backend == "gpu":
        try:
            import cupyx.scipy.sparse as cp_sparse
            I_1d = cp_sparse.eye(nx, format='csr')
            I_2d = cp_sparse.eye(nx * nz, format='csr')
        except:
            I_1d = eye(nx, format='csr')
            I_2d = eye(nx * nz, format='csr')
    else:
        I_1d = eye(nx, format='csr')
        I_2d = eye(nx * nz, format='csr')
    
    # Build implicit Euler matrices
    # For surface φ: (I - 2*dt/Re * Δ_H) φ^{n+1} = RHS
    A_phi_surf = I_1d - (2.0 * dt / Re) * Delta_H
    
    # For η: (I - 2*dt/Re * Δ_H) η^{n+1} = RHS
    A_eta = I_1d - (2.0 * dt / Re) * Delta_H
    
    # For heat equation (w³, w¹): (I - dt/Re * Δ) w^{n+1} = RHS
    A_heat = I_2d - (dt / Re) * Laplacian_2d
    
    return {
        'Delta_H': Delta_H,
        'A_laplace': A_laplace,
        'A_phi_surf': A_phi_surf,
        'A_eta': A_eta,
        'A_heat': A_heat,
        'I_1d': I_1d,
        'I_2d': I_2d
    }


class OperatorCache:
    """
    Cache for operators to avoid rebuilding
    Useful for parameter sweeps
    """
    def __init__(self):
        self.cache = {}
    
    def get_or_build(self, config, backend="cpu"):
        """Get cached operators or build new ones"""
        # Create cache key from relevant parameters
        key = (
            config.numerical.nx,
            config.numerical.nz,
            config.numerical.nt,
            config.numerical.unit_length,
            config.numerical.unit_time,
            config.physical.nu,
            backend
        )
        
        if key not in self.cache:
            self.cache[key] = build_implicit_matrices(config, backend)
        
        return self.cache[key]
    
    def clear(self):
        """Clear cache"""
        self.cache = {}


if __name__ == "__main__":
    # Test operator construction
    from .config import SimulationConfig
    
    config = SimulationConfig()
    print("Building operators...")
    
    matrices = build_implicit_matrices(config, backend="cpu")
    print(f"Delta_H shape: {matrices['Delta_H'].shape}")
    print(f"A_laplace shape: {matrices['A_laplace'].shape}")
    print(f"A_heat shape: {matrices['A_heat'].shape}")
    
    print("\nTesting cache...")
    cache = OperatorCache()
    ops1 = cache.get_or_build(config)
    ops2 = cache.get_or_build(config)  # Should be cached
    print(f"Cache hit: {ops1 is ops2}")
