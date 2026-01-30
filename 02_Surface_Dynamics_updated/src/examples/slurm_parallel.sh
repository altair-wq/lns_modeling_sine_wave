#!/bin/bash
#SBATCH --job-name=fluid_parallel
#SBATCH --output=logs/parallel_%j.out
#SBATCH --error=logs/parallel_%j.err
#SBATCH --time=04:00:00
#SBATCH --ntasks=10
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4GB
#SBATCH --partition=normal

# ============================================================================
# Parallel Fluid Surface Dynamics Simulation (MPI)
# SLURM Job Script for Stanford SCG
# ============================================================================

echo "=========================================="
echo "Parallel job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_NODELIST"
echo "Tasks: $SLURM_NTASKS"
echo "CPUs per task: $SLURM_CPUS_PER_TASK"
echo "=========================================="

# Load modules
module purge
module load python/3.9
module load openmpi/4.1.2  # For MPI

# Activate environment
# source ~/venvs/fluid_dynamics/bin/activate

# Create directories
mkdir -p logs
mkdir -p output_parallel_${SLURM_JOB_ID}

cd $SLURM_SUBMIT_DIR

# Print environment
echo ""
echo "MPI configuration:"
mpirun --version
echo ""

# ============================================================================
# Run parallel parameter sweep
# ============================================================================

# Viscosity sweep with 10 MPI ranks
echo "Running parallel viscosity sweep..."
mpirun -np $SLURM_NTASKS python examples/run_parallel.py \
    --sweep viscosity \
    --preset default \
    --n-points 20 \
    --output output_parallel_${SLURM_JOB_ID}/viscosity \
    --backend cpu

# Or Reynolds number sweep
# echo "Running parallel Reynolds sweep..."
# mpirun -np $SLURM_NTASKS python examples/run_parallel.py \
#     --sweep reynolds \
#     --preset default \
#     --n-points 20 \
#     --output output_parallel_${SLURM_JOB_ID}/reynolds \
#     --backend cpu

# ============================================================================

echo ""
echo "=========================================="
echo "Parallel job completed at: $(date)"
echo "Results in: output_parallel_${SLURM_JOB_ID}/"
echo "=========================================="
