import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix, eye, diags, kron
from scipy.sparse.linalg import spsolve
import time

# =============================================================================
# 1. Configuration: "Silicone Oil 20 cSt" (From Galeano-Rios et al., 2017)
# =============================================================================

class Config:
    def __init__(self):
        # --- Physical Parameters ---
        # Density of Silicone Oil (20 cSt)
        self.rho = 0.949        # g/cm^3
        
        # Surface Tension
        self.sigma = 20.6       # dyne/cm
        
        # Gravity
        self.g = 980.0          # cm/s^2
        
        # Viscosity Correction
        # nu* = 0.8025 * nu for Silicone Oil
        nu_raw = 0.20           # cm^2/s
        self.nu = 0.8025 * nu_raw 
        
        # --- Domain Setup ---
        # Faraday wavelength lambda_F ~ 0.5 cm
        # We set L to contain roughly 20 wavelengths
        self.L = 10.0           # cm
        
        # "infinite depth", but numerical grids need a bottom.
        # We set D large enough relative to lambda_F.
        self.D = 2.0            # cm
        
        # --- Numerical Discretization ---
        self.nx = 100           # Grid points in x
        self.nz = 40            # Grid points in z
        
        # Time Stepping
        self.dt = 0.001         # s
        self.nt = 1500          # Total steps (1.5 seconds)
        
        # Derived
        self.dx = self.L / self.nx
        self.dz = self.D / self.nz

# =============================================================================
# 2. Solver Implementation (LNS System)
# =============================================================================

class LNSSolver:
    def __init__(self, cfg):
        self.cfg = cfg
        self.eta = np.zeros(cfg.nx)
        self.phi = np.zeros(cfg.nx * cfg.nz)
        self.w3 = np.zeros(cfg.nx * cfg.nz)
        
        self._build_operators()
        self._build_implicit_matrices()
        
    def _build_operators(self):
        nx, nz, dx, dz = self.cfg.nx, self.cfg.nz, self.cfg.dx, self.cfg.dz
        
        # 1. Horizontal Laplacian (D2x) - Periodic
        e = np.ones(nx)
        data = np.vstack([e, -2*e, e]) / dx**2
        self.D2x = diags(data, [-1, 0, 1], shape=(nx, nx)).tolil()
        self.D2x[0, -1] = 1.0/dx**2; self.D2x[-1, 0] = 1.0/dx**2
        self.D2x = self.D2x.tocsr()
        
        # 2. Full 2D Laplacian (Lap2D)
        I_z, I_x = eye(nz), eye(nx)
        e_z = np.ones(nz)
        data_z = np.vstack([e_z, -2*e_z, e_z]) / dz**2
        D2z_1d = diags(data_z, [-1, 0, 1], shape=(nz, nz))
        self.Lap2D = (kron(I_z, self.D2x) + kron(D2z_1d, I_x)).tocsr()

    def _build_implicit_matrices(self):
        nx, nz = self.cfg.nx, self.cfg.nz
        
        # --- M_heat: Diffusion of Vorticity (w3) ---
        # Matches Eq (2.15b): w_t = (1/Re) * Laplacian(w)
        self.M_heat = (eye(nx*nz) - self.cfg.dt * self.cfg.nu * self.Lap2D).tolil()
        
        # BC Rows
        for i in range(nx):
            # Bottom (i): w3 = -phi_z (Non-slip condition)
            self.M_heat[i, :] = 0; self.M_heat[i, i] = 1.0
            
            # Top (i_top): w3 = 2*nu*eta_xx 
            # Matches Eq (2.17): w3 = (2/Re) * Delta_H(eta)
            idx_top = (nz-1)*nx + i
            self.M_heat[idx_top, :] = 0; self.M_heat[idx_top, idx_top] = 1.0
        self.M_heat = self.M_heat.tocsr()
        
        # --- M_lap: Pressure/Potential (phi) ---
        # Matches Eq (2.20a): Laplacian(phi) = 0
        self.M_lap = self.Lap2D.tolil()
        
        for i in range(nx):
            # Bottom (i): phi_z = -w3 
            self.M_lap[i, :] = 0
            self.M_lap[i, i] = -1.0; self.M_lap[i, i+nx] = 1.0
            
            # Top (i_top): phi = phi_surf
            idx_top = (nz-1)*nx + i
            self.M_lap[idx_top, :] = 0; self.M_lap[idx_top, idx_top] = 1.0
        self.M_lap = self.M_lap.tocsr()

    def step(self):
        nx, nz, dt = self.cfg.nx, self.cfg.nz, self.cfg.dt
        
        # 1. Update w3 (Heat Eq) [Eq 2.15b]
        eta_xx = self.D2x.dot(self.eta)
        w3_top = 2 * self.cfg.nu * eta_xx # Eq (2.17)
        
        # Bottom BC: w3 = -phi_z
        phi_bot_z = (self.phi[nx:2*nx] - self.phi[:nx]) / self.cfg.dz
        w3_bot = -phi_bot_z
        
        rhs_w3 = self.w3.copy()
        rhs_w3[:nx] = w3_bot
        rhs_w3[(nz-1)*nx:] = w3_top
        self.w3 = spsolve(self.M_heat, rhs_w3)
        
        # 2. Update Surface Potential (Explicit Dynamic BC)
        # Matches Eq (2.24b)
        phi_surf = self.phi[(nz-1)*nx:]
        kappa = self.D2x.dot(self.eta)
        lap_phi_surf = self.D2x.dot(phi_surf)
        
        phi_t = -self.cfg.g * self.eta + (self.cfg.sigma/self.cfg.rho)*kappa + 2*self.cfg.nu*lap_phi_surf
        phi_surf_new = phi_surf + dt * phi_t
        
        # 3. Update Bulk Potential (Laplace Eq)
        rhs_phi = np.zeros(nx*nz)
        rhs_phi[:nx] = -self.cfg.dz * self.w3[:nx] # Bottom BC
        rhs_phi[(nz-1)*nx:] = phi_surf_new         # Top BC
        self.phi = spsolve(self.M_lap, rhs_phi)
        
        # 4. Update Surface (Kinematic BC)
        # Matches Eq (2.24a): eta_t = phi_z + w3
        phi_z_surf = (self.phi[(nz-1)*nx:] - self.phi[(nz-2)*nx:(nz-1)*nx]) / self.cfg.dz
        w3_surf = self.w3[(nz-1)*nx:]
        self.eta = self.eta + dt * (phi_z_surf + w3_surf)

# =============================================================================
# 3. Verification Routine
# =============================================================================

def verify_with_paper_params():
    cfg = Config()
    solver = LNSSolver(cfg)
    
    # Setup Sine Wave
    k_mode = 2
    k = 2 * np.pi * k_mode / cfg.L
    x = np.linspace(0, cfg.L, cfg.nx, endpoint=False)
    
    # Small amplitude (0.05 cm) to remain linear
    solver.eta = 0.05 * np.cos(k * x) 
    
    print("="*60)
    print(f"VERIFICATION: Following Galeano-Rios et al. (2017) Parameters")
    print(f"Fluid: Silicone Oil (20 cSt) with Correction")
    print(f"  rho   = {cfg.rho} g/cm^3")
    print(f"  sigma = {cfg.sigma} dyne/cm")
    print(f"  nu    = {cfg.nu:.5f} cm^2/s (corrected from 0.20)")
    print(f"  Grid  = {cfg.nx}x{cfg.nz}, Size={cfg.L}x{cfg.D}cm")
    print("="*60)
    
    times, amps = [], []
    
    # Run
    start_t = time.time()
    for i in range(cfg.nt):
        solver.step()
        if i % 10 == 0:
            times.append(i * cfg.dt)
            amps.append(np.max(np.abs(solver.eta)))
            
    print(f"Simulation finished in {time.time()-start_t:.2f}s")
    
    # Theoretical Check
    times = np.array(times)
    
    # Dispersion Relation
    omega_sq = (cfg.g * k + cfg.sigma * k**3 / cfg.rho) * np.tanh(k * cfg.D)
    omega = np.sqrt(omega_sq)
    
    # Decay Rate
    gamma_bulk = 2 * cfg.nu * k**2
    gamma_bot = (k / (2 * np.sinh(2 * k * cfg.D))) * np.sqrt(cfg.nu * omega / 2)
    gamma_total = gamma_bulk + gamma_bot
    
    theory_decay = amps[0] * np.exp(-gamma_total * times)
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(times, amps, 'b-', label='Simulation (Paper Params)', lw=1.5)
    plt.plot(times, theory_decay, 'r--', label=f'Theory (gamma={gamma_total:.3f})', lw=2)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude (cm)')
    plt.title(f'Verification using Silicone Oil Parameters (nu*={cfg.nu:.3f})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Error Check
    if len(amps) > 0:
        final_error = abs(amps[-1] - theory_decay[-1]) / theory_decay[-1] * 100
        print(f"Final Amplitude Error: {final_error:.2f}%")

if __name__ == "__main__":
    verify_with_paper_params()
