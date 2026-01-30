#!/usr/bin/env python
"""
Run fluid dynamics simulation from command line
Supports parameter sweeps and batch processing

Usage:
    python run_simulation.py --preset fast
    python run_simulation.py --config config.json
    python run_simulation.py --sweep viscosity
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import fluid_dynamics as fd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # For headless operation
import matplotlib.pyplot as plt
import argparse
import json
import time
from pathlib import Path


def run_single_simulation(config, output_dir, backend="auto"):
    """Run a single simulation"""
    print(f"\n{'='*60}")
    print("Starting Simulation")
    print(f"{'='*60}")
    
    config.summary()
    
    # Create solver
    solver = fd.FluidSolver(config, backend=backend)
    
    # Run
    start = time.time()
    final_state = solver.run()
    elapsed = time.time() - start
    
    print(f"\n{'='*60}")
    print(f"Simulation completed in {elapsed:.2f}s")
    print(f"Performance: {config.numerical.nt/elapsed:.1f} steps/sec")
    print(f"{'='*60}")
    
    # Verify BCs
    bc_check = solver.verify_boundary_conditions()
    print(f"\nBoundary Condition Check:")
    print(f"  Bottom u_x: {bc_check['bottom_u_x']:.2e}")
    print(f"  Bottom u_z: {bc_check['bottom_u_z']:.2e}")
    print(f"  Passes: {bc_check['passes']}")
    
    # Save results
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Diagnostic plot
    print("\nGenerating diagnostic plot...")
    fig = fd.plot_diagnostics(solver, save_path=output_dir / "diagnostics.png")
    plt.close(fig)
    
    # Animation
    print("Creating animation...")
    anim = fd.create_animation(solver, fps=10, 
                               save_path=output_dir / "animation.gif")
    plt.close('all')
    
    # Save state
    import pickle
    with open(output_dir / 'results.pkl', 'wb') as f:
        pickle.dump({
            'config': config,
            'history': solver.history,
            'final_state': final_state,
            'timings': solver.timings,
            'bc_check': bc_check
        }, f)
    print(f"\nResults saved to {output_dir}")
    
    return solver


def run_viscosity_sweep(base_preset="default", output_dir="output_sweep", 
                       backend="auto"):
    """Run parameter sweep over viscosity"""
    print(f"\n{'='*60}")
    print("Parameter Sweep: Viscosity")
    print(f"{'='*60}")
    
    base_config = fd.get_preset(base_preset)
    base_config.compute.verbose = True
    
    # Viscosity values to test (cm²/s)
    viscosities = [0.00894, 0.02, 0.05, 0.1, 0.2]
    
    print(f"\nTesting {len(viscosities)} viscosity values:")
    print(f"  {viscosities}")
    
    # Run sweep
    results = fd.run_parameter_sweep(
        base_config,
        "physical.nu",
        viscosities,
        backend=backend
    )
    
    # Plot results
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    fd.plot_parameter_sweep(results, "Viscosity (cm²/s)", 
                           metric="max_eta", ax=axes[0])
    fd.plot_parameter_sweep(results, "Viscosity (cm²/s)", 
                           metric="final_eta_center", ax=axes[1])
    fd.plot_parameter_sweep(results, "Viscosity (cm²/s)", 
                           metric="energy_decay", ax=axes[2])
    
    plt.tight_layout()
    plt.savefig(output_dir / "sweep_results.png", dpi=150)
    plt.close()
    
    # Save results
    import pickle
    with open(output_dir / 'sweep_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    print(f"\nSweep results saved to {output_dir}")
    
    return results


def run_reynolds_sweep(base_preset="default", output_dir="output_reynolds",
                      backend="auto"):
    """Run parameter sweep over Reynolds number (via viscosity)"""
    print(f"\n{'='*60}")
    print("Parameter Sweep: Reynolds Number")
    print(f"{'='*60}")
    
    base_config = fd.get_preset(base_preset)
    base_config.compute.verbose = True
    
    # Target Reynolds numbers
    Re_targets = [1000, 5000, 10000, 20000, 30000]
    
    # Convert to viscosities
    # Re = VL/ν, so ν = VL/Re
    V = base_config.numerical.unit_velocity
    L = base_config.numerical.unit_length
    viscosities = [V * L / Re for Re in Re_targets]
    
    print(f"\nTarget Reynolds numbers: {Re_targets}")
    print(f"Corresponding viscosities: {[f'{v:.6f}' for v in viscosities]}")
    
    results = fd.run_parameter_sweep(
        base_config,
        "physical.nu",
        viscosities,
        backend=backend
    )
    
    # Plot with Re on x-axis
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Calculate actual Re for each result
    Re_actual = []
    for nu, _, solver in results:
        dims = solver.config.dimensionless_numbers
        Re_actual.append(dims['Re'])
    
    max_etas = [np.max(np.abs(r[1].eta)) for r in results]
    final_centers = [r[1].eta[len(r[1].eta)//2] for r in results]
    
    # Energy decay
    energy_decays = []
    for _, _, solver in results:
        E0 = np.sum(solver.history['eta'][0]**2)
        Ef = np.sum(solver.history['eta'][-1]**2)
        energy_decays.append(1 - Ef/E0 if E0 > 0 else 0)
    
    axes[0].semilogx(Re_actual, max_etas, 'o-', linewidth=2, markersize=8)
    axes[0].set_xlabel('Reynolds Number')
    axes[0].set_ylabel('max|η|')
    axes[0].set_title('Maximum Surface Elevation vs Re')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].semilogx(Re_actual, final_centers, 'o-', linewidth=2, markersize=8)
    axes[1].set_xlabel('Reynolds Number')
    axes[1].set_ylabel('η (center)')
    axes[1].set_title('Final Center Elevation vs Re')
    axes[1].grid(True, alpha=0.3)
    
    axes[2].semilogx(Re_actual, energy_decays, 'o-', linewidth=2, markersize=8)
    axes[2].set_xlabel('Reynolds Number')
    axes[2].set_ylabel('Energy Decay Fraction')
    axes[2].set_title('Energy Dissipation vs Re')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "reynolds_sweep.png", dpi=150)
    plt.close()
    
    # Save
    import pickle
    with open(output_dir / 'reynolds_sweep.pkl', 'wb') as f:
        pickle.dump({
            'Re_values': Re_actual,
            'viscosities': viscosities,
            'results': results
        }, f)
    
    print(f"\nReynolds sweep results saved to {output_dir}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Fluid Surface Dynamics Simulation"
    )
    parser.add_argument('--preset', type=str, default=None,
                       help='Preset configuration: fast, default, highres, viscous')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to JSON config file')
    parser.add_argument('--sweep', type=str, default=None,
                       help='Run parameter sweep: viscosity, reynolds')
    parser.add_argument('--output', type=str, default='output',
                       help='Output directory')
    parser.add_argument('--backend', type=str, default='auto',
                       help='Compute backend: cpu, gpu, auto')
    parser.add_argument('--no-plots', action='store_true',
                       help='Skip plot generation')
    
    args = parser.parse_args()
    
    # Print info
    fd.info()
    
    # Run sweep
    if args.sweep:
        if args.sweep == "viscosity":
            run_viscosity_sweep(
                base_preset=args.preset or "default",
                output_dir=args.output,
                backend=args.backend
            )
        elif args.sweep == "reynolds":
            run_reynolds_sweep(
                base_preset=args.preset or "default",
                output_dir=args.output,
                backend=args.backend
            )
        else:
            print(f"Unknown sweep type: {args.sweep}")
            print("Available: viscosity, reynolds")
            sys.exit(1)
    
    # Run single simulation
    else:
        if args.config:
            # Load from JSON
            with open(args.config, 'r') as f:
                config_dict = json.load(f)
            # TODO: Implement JSON to config conversion
            print("JSON config loading not yet implemented")
            sys.exit(1)
        else:
            # Use preset
            preset = args.preset or "default"
            config = fd.get_preset(preset)
        
        solver = run_single_simulation(
            config,
            output_dir=args.output,
            backend=args.backend
        )
    
    print("\n" + "="*60)
    print("All done!")
    print("="*60)


if __name__ == "__main__":
    main()
