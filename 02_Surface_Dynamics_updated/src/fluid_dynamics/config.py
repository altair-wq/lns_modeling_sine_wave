"""
Configuration module for fluid dynamics simulations.
"""
from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class PhysicalParameters:
    """물리적 파라미터 (Physical parameters)"""
    L_cm: float = 50.0          # Domain length in x (cm)
    D_cm: float = 5.0           # Domain depth in z (cm)
    g: float = 980.0            # Gravitational acceleration (cm/s²)
    sigma: float = 74.9         # Surface tension (dyne/cm)
    rho: float = 1.0            # Density (g/cm³)
    nu: float = 0.00894         # Kinematic viscosity (cm²/s)
    
    def __post_init__(self):
        """Validate parameters"""
        assert self.L_cm > 0, "Length must be positive"
        assert self.D_cm > 0, "Depth must be positive"
        assert self.g > 0, "Gravity must be positive"
        assert self.sigma > 0, "Surface tension must be positive"
        assert self.rho > 0, "Density must be positive"
        assert self.nu > 0, "Viscosity must be positive"


@dataclass
class NumericalParameters:
    """수치 파라미터 (Numerical parameters)"""
    nx: int = 200               # Grid points in x
    nz: int = 40                # Grid points in z
    nt: int = 1000              # Number of time steps
    total_time: float = 5.0     # Total simulation time (seconds)
    
    # Non-dimensionalization scales
    unit_length: float = 5.0    # Length scale (cm)
    unit_time: float = 0.1      # Time scale (s)
    
    def __post_init__(self):
        """Validate parameters"""
        assert self.nx > 0, "nx must be positive"
        assert self.nz > 0, "nz must be positive"
        assert self.nt > 0, "nt must be positive"
        assert self.total_time > 0, "total_time must be positive"
        assert self.unit_length > 0, "unit_length must be positive"
        assert self.unit_time > 0, "unit_time must be positive"
    
    @property
    def unit_velocity(self):
        """Velocity scale"""
        return self.unit_length / self.unit_time


@dataclass
class InitialCondition:
    """초기 조건 (Initial conditions)"""
    type: str = "gaussian"      # Type: "gaussian", "sine", "custom"
    amplitude: float = 0.05     # Amplitude
    width_factor: float = 0.1   # Width as fraction of domain
    custom_function: Optional[callable] = None  # Custom IC function
    
    def __post_init__(self):
        """Validate initial condition"""
        valid_types = ["gaussian", "sine", "custom"]
        assert self.type in valid_types, f"type must be one of {valid_types}"
        if self.type == "custom":
            assert self.custom_function is not None, "custom_function must be provided"


@dataclass
class ComputeConfig:
    """계산 설정 (Compute configuration)"""
    backend: str = "cpu"        # "cpu", "gpu", "auto"
    use_mpi: bool = False       # Use MPI for parallel runs
    num_workers: int = 1        # Number of parallel workers
    save_interval: int = 100    # Save every N steps
    verbose: bool = True        # Print progress
    
    def __post_init__(self):
        """Validate compute config"""
        valid_backends = ["cpu", "gpu", "auto"]
        assert self.backend in valid_backends, f"backend must be one of {valid_backends}"
        assert self.num_workers > 0, "num_workers must be positive"
        assert self.save_interval > 0, "save_interval must be positive"


@dataclass
class SimulationConfig:
    """전체 시뮬레이션 설정 (Complete simulation configuration)"""
    physical: PhysicalParameters = field(default_factory=PhysicalParameters)
    numerical: NumericalParameters = field(default_factory=NumericalParameters)
    initial: InitialCondition = field(default_factory=InitialCondition)
    compute: ComputeConfig = field(default_factory=ComputeConfig)
    
    @property
    def dimensionless_numbers(self):
        """Calculate dimensionless numbers"""
        V = self.numerical.unit_velocity
        L = self.numerical.unit_length
        
        Fr = V**2 / (self.physical.g * L)
        We = self.physical.rho * V**2 * L / self.physical.sigma
        Re = V * L / self.physical.nu
        
        return {"Fr": Fr, "We": We, "Re": Re}
    
    @property
    def grid_spacing(self):
        """Calculate grid spacing"""
        L = self.physical.L_cm / self.numerical.unit_length
        D = self.physical.D_cm / self.numerical.unit_length
        
        dx = L / self.numerical.nx
        dz = D / self.numerical.nz
        dt = (self.numerical.total_time / self.numerical.unit_time) / self.numerical.nt
        
        return {"L": L, "D": D, "dx": dx, "dz": dz, "dt": dt}
    
    def summary(self):
        """Print configuration summary"""
        print("=" * 60)
        print("Simulation Configuration")
        print("=" * 60)
        print("\nPhysical Parameters:")
        print(f"  Domain: {self.physical.L_cm} cm × {self.physical.D_cm} cm")
        print(f"  Gravity: {self.physical.g} cm/s²")
        print(f"  Surface tension: {self.physical.sigma} dyne/cm")
        print(f"  Viscosity: {self.physical.nu} cm²/s")
        
        print("\nNumerical Parameters:")
        print(f"  Grid: {self.numerical.nx} × {self.numerical.nz}")
        print(f"  Time steps: {self.numerical.nt}")
        print(f"  Total time: {self.numerical.total_time} s")
        
        dims = self.dimensionless_numbers
        print("\nDimensionless Numbers:")
        print(f"  Fr = {dims['Fr']:.6f}")
        print(f"  We = {dims['We']:.4f}")
        print(f"  Re = {dims['Re']:.2f}")
        
        grid = self.grid_spacing
        print("\nGrid Spacing:")
        print(f"  dx = {grid['dx']:.6f}, dz = {grid['dz']:.6f}, dt = {grid['dt']:.6f}")
        
        print("\nCompute Configuration:")
        print(f"  Backend: {self.compute.backend}")
        print(f"  MPI: {self.compute.use_mpi}")
        print(f"  Workers: {self.compute.num_workers}")
        print("=" * 60)


# Preset configurations
def get_preset(name: str) -> SimulationConfig:
    """
    Get preset configurations
    
    Presets:
    - "default": Standard configuration
    - "fast": Quick test run
    - "highres": High resolution
    - "viscous": High viscosity
    """
    if name == "default":
        return SimulationConfig()
    
    elif name == "fast":
        return SimulationConfig(
            numerical=NumericalParameters(nx=100, nz=20, nt=100, total_time=1.0),
            compute=ComputeConfig(save_interval=10)
        )
    
    elif name == "highres":
        return SimulationConfig(
            numerical=NumericalParameters(nx=400, nz=80, nt=2000, total_time=10.0),
            compute=ComputeConfig(save_interval=200)
        )
    
    elif name == "viscous":
        return SimulationConfig(
            physical=PhysicalParameters(nu=0.1),  # 10x higher viscosity
            numerical=NumericalParameters(nt=500, total_time=2.0)
        )
    
    else:
        raise ValueError(f"Unknown preset: {name}. Available: default, fast, highres, viscous")


if __name__ == "__main__":
    # Test configurations
    config = SimulationConfig()
    config.summary()
    
    print("\n" + "="*60)
    print("Testing presets...")
    print("="*60)
    
    fast = get_preset("fast")
    print(f"\nFast preset: {fast.numerical.nx}×{fast.numerical.nz}, {fast.numerical.nt} steps")
