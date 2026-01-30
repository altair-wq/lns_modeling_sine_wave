# Fluid Surface Dynamics Simulation Package

**유체 표면 역학 시뮬레이션 패키지**

Based on: Galeano-Rios, Milewski, and Vanden-Broeck (2017), *J. Fluid Mech.* **826**, pp. 97-127

---

## Features

✅ **Implicit Euler time stepping** - Unconditionally stable  
✅ **Non-slip boundary conditions** - Physically accurate solid boundaries  
✅ **Full 2D vortical evolution** - Both w³ and w¹ components  
✅ **GPU acceleration** - NVIDIA (CuPy) and Mac (MPS planned)  
✅ **HPC-ready** - SLURM integration with MPI parallelization  
✅ **Easy configuration** - Dataclass-based settings  
✅ **Comprehensive visualization** - Publication-quality plots  

---

## Installation

### Basic Installation (CPU only)

```bash
cd fluid_dynamics_package
pip install -e .
```

### With GPU Support (NVIDIA)

```bash
pip install -e ".[gpu]"
```

### With MPI for Parallel Runs

```bash
pip install -e ".[mpi]"
```

### For Development

```bash
pip install -e ".[dev]"
```

---

## Quick Start

### 1. Python Script

```python
import fluid_dynamics as fd

# Quick run with preset
solver = fd.quick_run("fast")

# Or customize
config = fd.SimulationConfig(
    physical=fd.PhysicalParameters(
        L_cm=50.0,
        D_cm=5.0,
        nu=0.00894  # viscosity
    ),
    numerical=fd.NumericalParameters(
        nx=200,
        nz=40,
        nt=1000,
        total_time=5.0
    )
)

solver = fd.FluidSolver(config, backend="auto")
solver.run()

# Visualize
fd.plot_diagnostics(solver)
```

### 2. Jupyter Notebook

```bash
jupyter notebook examples/demo.ipynb
```

### 3. Command Line

```bash
# Quick test
python examples/run_simulation.py --preset fast --output output/

# Full simulation
python examples/run_simulation.py --preset default --output output/

# Parameter sweep
python examples/run_simulation.py --sweep viscosity --output output/
```

### 4. HPC (SLURM)

```bash
# Single simulation
sbatch examples/slurm_job.sh

# Parallel parameter sweep (MPI)
sbatch examples/slurm_parallel.sh
```

---

## Configuration

### Preset Configurations

- `"fast"` - Quick test (100×20 grid, 100 steps)
- `"default"` - Standard resolution (200×40, 1000 steps)
- `"highres"` - High resolution (400×80, 2000 steps)
- `"viscous"` - High viscosity test

### Custom Configuration

```python
config = fd.SimulationConfig(
    physical=fd.PhysicalParameters(
        L_cm=50.0,          # Domain length (cm)
        D_cm=5.0,           # Domain depth (cm)
        g=980.0,            # Gravity (cm/s²)
        sigma=74.9,         # Surface tension (dyne/cm)
        rho=1.0,            # Density (g/cm³)
        nu=0.00894          # Viscosity (cm²/s)
    ),
    numerical=fd.NumericalParameters(
        nx=200,             # Grid points in x
        nz=40,              # Grid points in z
        nt=1000,            # Time steps
        total_time=5.0,     # Total time (s)
        unit_length=5.0,    # Length scale (cm)
        unit_time=0.1       # Time scale (s)
    ),
    initial=fd.InitialCondition(
        type="gaussian",    # "gaussian", "sine", or "custom"
        amplitude=0.05,
        width_factor=0.1
    ),
    compute=fd.ComputeConfig(
        backend="auto",     # "cpu", "gpu", or "auto"
        verbose=True,
        save_interval=100
    )
)
```

---

## GPU Support

### Mac (Apple Silicon)

```python
# Automatic detection
solver = fd.FluidSolver(config, backend="auto")
```

*Note: Mac GPU (MPS) support planned for future release*

### NVIDIA GPU (CuPy)

```bash
# Install CuPy (adjust CUDA version)
pip install cupy-cuda11x

# Use GPU
solver = fd.FluidSolver(config, backend="gpu")
```

### Performance Comparison

```python
import time

# CPU
t0 = time.time()
solver_cpu = fd.FluidSolver(config, backend="cpu")
solver_cpu.run()
t_cpu = time.time() - t0

# GPU
t0 = time.time()
solver_gpu = fd.FluidSolver(config, backend="gpu")
solver_gpu.run()
t_gpu = time.time() - t0

print(f"Speedup: {t_cpu/t_gpu:.2f}x")
```

---

## HPC Usage (Stanford SCG)

### Setup on SCG

```bash
# Login to SCG
ssh your_sunetid@login.scg.stanford.edu

# Load modules
module load python/3.9
module load cuda/11.7  # for GPU
module load openmpi/4.1.2  # for MPI

# Create virtual environment
python -m venv ~/venvs/fluid_dynamics
source ~/venvs/fluid_dynamics/bin/activate

# Install package
cd ~/fluid_dynamics_package
pip install -e .
pip install cupy-cuda11x  # for GPU
pip install mpi4py  # for parallel runs
```

### Single Simulation

```bash
# Edit slurm_job.sh to choose simulation type
sbatch examples/slurm_job.sh
```

### Parallel Parameter Sweep (MPI)

```bash
# Edit slurm_parallel.sh
sbatch examples/slurm_parallel.sh
```

The parallel script will:
- Distribute parameter values across MPI ranks
- Run simulations in parallel
- Combine results automatically
- Generate summary plots

---

## Parameter Sweeps

### Viscosity Sweep

```python
results = fd.run_parameter_sweep(
    base_config,
    "physical.nu",
    [0.00894, 0.02, 0.05, 0.1],
    backend="auto"
)

# Plot results
fd.plot_parameter_sweep(results, "Viscosity", metric="max_eta")
```

### Reynolds Number Sweep

```bash
python examples/run_simulation.py --sweep reynolds --output output/
```

### Custom Parameter Sweep

```python
# Any parameter can be swept
results = fd.run_parameter_sweep(
    config,
    "initial.amplitude",  # Nested parameter
    [0.01, 0.03, 0.05, 0.07, 0.1],
    backend="cpu"
)
```

---

## Visualization

### Diagnostic Plot

```python
fig = fd.plot_diagnostics(solver, save_path="diagnostics.png")
```

Includes:
- Surface evolution
- Velocity potential field
- Velocity vectors
- Energy components
- Phase space trajectory
- w³ evolution
- Curvature

### Animation

```python
anim = fd.create_animation(solver, fps=10, save_path="animation.gif")
```

### Custom Plots

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

fd.plot_surface_evolution(solver, ax=axes[0, 0])
fd.plot_velocity_field(solver, ax=axes[0, 1])
fd.plot_energy(solver, ax=axes[1, 0])
fd.plot_phase_space(solver, ax=axes[1, 1])

plt.tight_layout()
plt.show()
```

---

## Examples

### Example 1: Basic Simulation

```python
import fluid_dynamics as fd

config = fd.get_preset("default")
solver = fd.FluidSolver(config)
solver.run()

# Verify non-slip BC
bc = solver.verify_boundary_conditions()
print(f"Non-slip BC satisfied: {bc['passes']}")

# Visualize
fd.plot_diagnostics(solver)
```

### Example 2: Custom Initial Condition

```python
import numpy as np

def two_bumps(x):
    L = 10  # domain length
    bump1 = 0.05 * np.exp(-(x - L/3)**2 / 0.5)
    bump2 = 0.03 * np.exp(-(x - 2*L/3)**2 / 0.3)
    return bump1 + bump2

config = fd.get_preset("fast")
config.initial = fd.InitialCondition(
    type="custom",
    custom_function=two_bumps
)

solver = fd.FluidSolver(config)
solver.run()
```

### Example 3: High Viscosity Flow

```python
config = fd.get_preset("viscous")  # nu = 0.1 cm²/s
solver = fd.FluidSolver(config)
solver.run()

# Compare energy dissipation
E0 = np.sum(solver.history['eta'][0]**2)
Ef = np.sum(solver.history['eta'][-1]**2)
print(f"Energy dissipated: {100*(1-Ef/E0):.1f}%")
```

---

## Theory

### Governing Equations

The simulation solves the coupled system:

**Laplace equation** (velocity potential):
```
Δφ = 0
```

**Heat equation** (vortical components):
```
∂w/∂t = (1/Re) Δw
```

**Surface evolution**:
```
∂η/∂t = φ_z + w³ + (2/Re) Δ_H η
```

**Dynamic boundary condition**:
```
∂φ/∂t = -(1/Fr)η + (1/We)κ[η] + (2/Re)Δ_H φ
```

### Boundary Conditions

- **Top (free surface)**: Dynamic BCs from equations above
- **Bottom (solid)**: Non-slip (u = 0)
- **Lateral**: Periodic

### Dimensionless Numbers

- **Froude number**: Fr = V²/(gL)
- **Weber number**: We = ρV²L/σ
- **Reynolds number**: Re = VL/ν

---

## Package Structure

```
fluid_dynamics_package/
├── fluid_dynamics/
│   ├── __init__.py          # Main package
│   ├── config.py            # Configuration classes
│   ├── solver.py            # FluidSolver class
│   ├── operators.py         # Matrix builders
│   └── visualization.py     # Plotting functions
├── examples/
│   ├── demo.ipynb           # Jupyter demo
│   ├── run_simulation.py    # CLI script
│   ├── run_parallel.py      # MPI parallel script
│   ├── slurm_job.sh         # SLURM single job
│   └── slurm_parallel.sh    # SLURM parallel job
├── tests/                   # Unit tests
├── setup.py                 # Installation script
├── requirements.txt         # Dependencies
└── README.md               # This file
```

---

## Troubleshooting

### GPU Not Detected

```python
# Check GPU availability
from fluid_dynamics.operators import get_backend
xp, sp, device = get_backend("auto")
print(f"Device: {device}")

# Force CPU if GPU issues
solver = fd.FluidSolver(config, backend="cpu")
```

### Memory Issues on HPC

```bash
# Increase memory in SLURM script
#SBATCH --mem=32GB

# Or reduce resolution
config.numerical.nx = 100
config.numerical.nz = 20
```

### MPI Not Working

```bash
# Check MPI installation
mpirun --version

# Test simple MPI
mpirun -np 2 python -c "from mpi4py import MPI; print(MPI.COMM_WORLD.Get_rank())"
```

---

## Citation

If you use this code in your research, please cite:

```bibtex
@article{galeano2017nonwetting,
  title={Non-wetting impact of a sphere onto a bath and its application to bouncing droplets},
  author={Galeano-Rios, Carlos A and Milewski, Paul A and Vanden-Broeck, Jean-Marc},
  journal={Journal of Fluid Mechanics},
  volume={826},
  pages={97--127},
  year={2017},
  publisher={Cambridge University Press}
}
```

---

## License

MIT License - see LICENSE file for details

---

## Contact

- **Author**: Wongyung
- **Email**: your.email@stanford.edu
- **Institution**: Minerva University / Stanford University

---

## Changelog

### v1.0.0 (2024-11-24)
- Initial release
- Implicit Euler solver with non-slip BC
- GPU support via CuPy
- MPI parallelization
- Comprehensive visualization tools
- SLURM integration for Stanford SCG
