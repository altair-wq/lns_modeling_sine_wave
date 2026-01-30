"""
Basic tests for fluid dynamics package
Run with: pytest tests/
"""
import sys
sys.path.append('..')

import numpy as np
from fluid_dynamics import (
    SimulationConfig,
    FluidSolver,
    get_preset,
    PhysicalParameters,
    NumericalParameters
)


def test_config_creation():
    """Test configuration creation"""
    config = SimulationConfig()
    assert config.physical.L_cm > 0
    assert config.numerical.nx > 0
    print("✓ Config creation test passed")


def test_presets():
    """Test preset configurations"""
    for preset in ["fast", "default", "highres", "viscous"]:
        config = get_preset(preset)
        assert config.numerical.nx > 0
        assert config.numerical.nt > 0
    print("✓ Presets test passed")


def test_dimensionless_numbers():
    """Test dimensionless number calculation"""
    config = SimulationConfig()
    dims = config.dimensionless_numbers
    
    assert "Fr" in dims
    assert "We" in dims
    assert "Re" in dims
    assert all(v > 0 for v in dims.values())
    print(f"✓ Dimensionless numbers test passed: {dims}")


def test_solver_initialization():
    """Test solver initialization"""
    config = get_preset("fast")
    config.compute.verbose = False
    
    solver = FluidSolver(config, backend="cpu")
    assert solver.state.eta is not None
    assert solver.state.phi is not None
    print("✓ Solver initialization test passed")


def test_single_step():
    """Test single time step"""
    config = get_preset("fast")
    config.compute.verbose = False
    config.numerical.nt = 1
    
    solver = FluidSolver(config, backend="cpu")
    state = solver.step()
    
    assert state.step == 1
    assert not np.any(np.isnan(state.eta))
    assert not np.any(np.isnan(state.phi))
    print("✓ Single step test passed")


def test_full_simulation():
    """Test full simulation"""
    config = get_preset("fast")
    config.compute.verbose = False
    config.numerical.nt = 10  # Very short
    
    solver = FluidSolver(config, backend="cpu")
    final_state = solver.run(save_history=True)
    
    assert final_state.step == 10
    assert len(solver.history['eta']) > 0
    assert not np.any(np.isnan(final_state.eta))
    print("✓ Full simulation test passed")


def test_boundary_conditions():
    """Test non-slip boundary conditions"""
    config = get_preset("fast")
    config.compute.verbose = False
    config.numerical.nt = 50
    
    solver = FluidSolver(config, backend="cpu")
    solver.run(save_history=False)
    
    bc_check = solver.verify_boundary_conditions()
    
    print(f"  Bottom u_x: {bc_check['bottom_u_x']:.2e}")
    print(f"  Bottom u_z: {bc_check['bottom_u_z']:.2e}")
    
    # Allow some numerical error (implicit solver)
    assert bc_check['bottom_u_x'] < 1e-3
    assert bc_check['bottom_u_z'] < 1e-2
    print("✓ Boundary conditions test passed")


def test_energy_conservation():
    """Test energy dissipation (should decrease due to viscosity)"""
    config = get_preset("fast")
    config.compute.verbose = False
    config.numerical.nt = 100
    
    solver = FluidSolver(config, backend="cpu")
    solver.run(save_history=True)
    
    # Energy should decrease
    E0 = np.sum(solver.history['eta'][0]**2)
    Ef = np.sum(solver.history['eta'][-1]**2)
    
    print(f"  E0 = {E0:.6f}, Ef = {Ef:.6f}")
    print(f"  Energy dissipated: {100*(1-Ef/E0):.1f}%")
    
    assert Ef < E0, "Energy should decrease due to viscosity"
    print("✓ Energy dissipation test passed")


def test_reset():
    """Test solver reset"""
    config = get_preset("fast")
    config.compute.verbose = False
    
    solver = FluidSolver(config, backend="cpu")
    solver.run(save_history=True)
    
    initial_max = np.max(np.abs(solver.history['eta'][0]))
    
    solver.reset()
    
    assert solver.state.step == 0
    assert len(solver.history['eta']) == 0
    # Check that eta is restored to initial condition
    restored_max = np.max(np.abs(solver.state.eta))
    print(f"  Initial max: {initial_max:.6f}, Restored max: {restored_max:.6f}")
    assert np.abs(restored_max - initial_max) / initial_max < 0.1  # Within 10%
    print("✓ Reset test passed")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("Running Fluid Dynamics Package Tests")
    print("="*60 + "\n")
    
    test_config_creation()
    test_presets()
    test_dimensionless_numbers()
    test_solver_initialization()
    test_single_step()
    test_full_simulation()
    test_boundary_conditions()
    test_energy_conservation()
    test_reset()
    
    print("\n" + "="*60)
    print("All tests passed! ✓")
    print("="*60)
