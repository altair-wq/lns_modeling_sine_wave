#!/usr/bin/env python
"""
Run multiple simulations in parallel using MPI
Useful for large parameter sweeps on HPC clusters

Usage:
    mpirun -np 4 python run_parallel.py --sweep viscosity
    
On SLURM:
    sbatch slurm_parallel.sh
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import fluid_dynamics as fd
import numpy as np
import argparse
import pickle
from pathlib import Path


def setup_mpi():
    """Setup MPI if available"""
    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        return comm, rank, size, True
    except ImportError:
        print("Warning: mpi4py not installed. Running in serial mode.")
        return None, 0, 1, False


def run_parallel_sweep(param_name, param_values, base_config, 
                      output_dir, backend="cpu"):
    """
    Run parameter sweep in parallel
    
    Each MPI rank handles a subset of parameter values
    """
    comm, rank, size, has_mpi = setup_mpi()
    
    if rank == 0:
        print(f"\n{'='*60}")
        print(f"Parallel Parameter Sweep")
        print(f"  Parameter: {param_name}")
        print(f"  Values: {len(param_values)}")
        print(f"  MPI ranks: {size}")
        print(f"{'='*60}\n")
    
    # Divide work among ranks
    my_indices = range(rank, len(param_values), size)
    my_values = [param_values[i] for i in my_indices]
    
    if rank == 0:
        print(f"Rank distribution:")
        for r in range(size):
            r_indices = range(r, len(param_values), size)
            r_values = [param_values[i] for i in r_indices]
            print(f"  Rank {r}: {len(r_values)} simulations")
    
    # Each rank runs its simulations
    my_results = []
    for i, value in enumerate(my_values):
        global_idx = my_indices[i]
        
        print(f"\nRank {rank}: Running simulation {i+1}/{len(my_values)}")
        print(f"  {param_name} = {value}")
        
        # Create config
        import copy
        config = copy.deepcopy(base_config)
        config.compute.verbose = False  # Reduce output
        
        # Set parameter
        parts = param_name.split('.')
        obj = config
        for part in parts[:-1]:
            obj = getattr(obj, part)
        setattr(obj, parts[-1], value)
        
        # Run simulation
        solver = fd.FluidSolver(config, backend=backend)
        final_state = solver.run(save_history=True)
        
        # Save individual result
        output_path = Path(output_dir) / f"result_rank{rank}_idx{global_idx}.pkl"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'wb') as f:
            pickle.dump({
                'param_value': value,
                'param_index': global_idx,
                'final_state': final_state,
                'history': solver.history,
                'config': config,
                'rank': rank
            }, f)
        
        my_results.append((value, final_state, solver))
        
        print(f"  Complete. max|η| = {np.max(np.abs(final_state.eta)):.6f}")
    
    # Gather results at rank 0
    if has_mpi:
        all_results = comm.gather(my_results, root=0)
    else:
        all_results = [my_results]
    
    # Rank 0 processes combined results
    if rank == 0:
        # Flatten results
        combined = []
        for rank_results in all_results:
            combined.extend(rank_results)
        
        # Sort by parameter value
        combined.sort(key=lambda x: x[0])
        
        print(f"\n{'='*60}")
        print(f"All simulations complete!")
        print(f"  Total: {len(combined)} simulations")
        print(f"{'='*60}")
        
        # Save combined results
        output_path = Path(output_dir) / "combined_results.pkl"
        with open(output_path, 'wb') as f:
            pickle.dump(combined, f)
        
        print(f"\nResults saved to {output_dir}/")
        
        # Create summary plot
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        fd.plot_parameter_sweep(combined, param_name, 
                               metric="max_eta", ax=axes[0])
        fd.plot_parameter_sweep(combined, param_name, 
                               metric="final_eta_center", ax=axes[1])
        fd.plot_parameter_sweep(combined, param_name, 
                               metric="energy_decay", ax=axes[2])
        
        plt.tight_layout()
        plt.savefig(Path(output_dir) / "parallel_sweep_results.png", dpi=150)
        plt.close()
        
        print(f"Summary plot saved to {output_dir}/parallel_sweep_results.png")
    
    return my_results if not has_mpi or rank != 0 else combined


def main():
    parser = argparse.ArgumentParser(
        description="Parallel Fluid Surface Dynamics Simulations"
    )
    parser.add_argument('--sweep', type=str, required=True,
                       help='Parameter to sweep: viscosity, reynolds, amplitude')
    parser.add_argument('--preset', type=str, default='default',
                       help='Base preset configuration')
    parser.add_argument('--output', type=str, default='output_parallel',
                       help='Output directory')
    parser.add_argument('--backend', type=str, default='cpu',
                       help='Compute backend: cpu, gpu')
    parser.add_argument('--n-points', type=int, default=10,
                       help='Number of parameter values to test')
    
    args = parser.parse_args()
    
    comm, rank, size, has_mpi = setup_mpi()
    
    # Only rank 0 prints header
    if rank == 0:
        fd.info()
    
    # Load base configuration
    base_config = fd.get_preset(args.preset)
    
    # Define parameter sweep
    if args.sweep == "viscosity":
        param_name = "physical.nu"
        param_values = np.logspace(-3, -0.5, args.n_points).tolist()  # 0.001 to 0.316 cm²/s
    
    elif args.sweep == "reynolds":
        # Convert Re to viscosity
        V = base_config.numerical.unit_velocity
        L = base_config.numerical.unit_length
        Re_values = np.logspace(2, 4.5, args.n_points)  # 100 to ~31623
        param_name = "physical.nu"
        param_values = [V * L / Re for Re in Re_values]
    
    elif args.sweep == "amplitude":
        param_name = "initial.amplitude"
        param_values = np.linspace(0.01, 0.1, args.n_points).tolist()
    
    else:
        if rank == 0:
            print(f"Unknown sweep: {args.sweep}")
            print("Available: viscosity, reynolds, amplitude")
        sys.exit(1)
    
    # Run parallel sweep
    results = run_parallel_sweep(
        param_name,
        param_values,
        base_config,
        args.output,
        backend=args.backend
    )
    
    if rank == 0:
        print("\n" + "="*60)
        print("Parallel sweep complete!")
        print("="*60)


if __name__ == "__main__":
    main()
