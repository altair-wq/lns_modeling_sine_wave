"""
Fluid Surface Dynamics Package
Based on: Galeano-Rios et al. (2017), J. Fluid Mech. 826, pp. 97-127

유체 표면 역학 시뮬레이션 패키지
"""

__version__ = "1.0.0"
__author__ = "Wongyung"

# Main classes and functions
from .config import (
    SimulationConfig,
    PhysicalParameters,
    NumericalParameters,
    InitialCondition,
    ComputeConfig,
    get_preset
)

from .solver import (
    FluidSolver,
    SimulationState,
    run_parameter_sweep
)

from .visualization import (
    plot_surface_evolution,
    plot_velocity_field,
    plot_phase_space,
    plot_energy,
    plot_diagnostics,
    create_animation,
    plot_parameter_sweep
)

from .operators import (
    build_implicit_matrices,
    OperatorCache,
    get_backend
)

__all__ = [
    # Config
    'SimulationConfig',
    'PhysicalParameters',
    'NumericalParameters',
    'InitialCondition',
    'ComputeConfig',
    'get_preset',
    
    # Solver
    'FluidSolver',
    'SimulationState',
    'run_parameter_sweep',
    
    # Visualization
    'plot_surface_evolution',
    'plot_velocity_field',
    'plot_phase_space',
    'plot_energy',
    'plot_diagnostics',
    'create_animation',
    'plot_parameter_sweep',
    
    # Operators
    'build_implicit_matrices',
    'OperatorCache',
    'get_backend',
]

# Quick start example
def quick_run(preset="fast", backend="auto", show_plots=True):
    """
    Quick start: run simulation with preset
    
    Args:
        preset: "fast", "default", "highres", or "viscous"
        backend: "cpu", "gpu", or "auto"
        show_plots: Whether to display plots
    
    Returns:
        solver with results
    
    Example:
        >>> import fluid_dynamics as fd
        >>> solver = fd.quick_run("fast")
    """
    config = get_preset(preset)
    config.summary()
    
    solver = FluidSolver(config, backend=backend)
    solver.run()
    
    if show_plots:
        import matplotlib.pyplot as plt
        plot_diagnostics(solver)
        plt.show()
    
    return solver


# Package info
def info():
    """Print package information"""
    print(f"""
╔═══════════════════════════════════════════════════════════════╗
║           Fluid Surface Dynamics Package                      ║
║           Version: {__version__}                                       ║
╚═══════════════════════════════════════════════════════════════╝

Based on:
  Galeano-Rios, Milewski, and Vanden-Broeck (2017)
  "Non-wetting impact of a sphere onto a bath and its 
   application to bouncing droplets"
  J. Fluid Mech. 826, pp. 97-127

Features:
  ✓ Implicit Euler time stepping (unconditionally stable)
  ✓ Non-slip boundary conditions at solid bottom
  ✓ Full 2D vortical component evolution (w³, w¹)
  ✓ CPU and GPU support (NVIDIA via CuPy)
  ✓ HPC-ready (SLURM integration)
  ✓ Configurable via dataclasses

Quick Start:
  >>> import fluid_dynamics as fd
  >>> solver = fd.quick_run("fast")
  
For more examples:
  - examples/demo.ipynb
  - examples/run_simulation.py
  - examples/slurm_job.sh

Documentation:
  >>> help(fd.FluidSolver)
  >>> help(fd.SimulationConfig)
    """)


if __name__ == "__main__":
    info()
