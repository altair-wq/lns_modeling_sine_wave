#!/bin/bash
#SBATCH --job-name=fluid_dynamics
#SBATCH --output=logs/fluid_%j.out
#SBATCH --error=logs/fluid_%j.err
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --partition=normal

# Optional: Request GPU
# Uncomment for GPU runs on SCG
##SBATCH --gres=gpu:1
##SBATCH --partition=gpu

# ============================================================================
# Fluid Surface Dynamics Simulation
# SLURM Job Script for Stanford SCG
# ============================================================================

echo "=========================================="
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Cores: $SLURM_CPUS_PER_TASK"
echo "=========================================="

# Load modules (adjust for your system)
module purge
module load python/3.9
module load cuda/11.7  # If using GPU

# Activate virtual environment (if using one)
# source ~/venvs/fluid_dynamics/bin/activate

# Or use conda
# module load miniconda3
# conda activate fluid_dynamics

# Set environment variables
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Create output directories
mkdir -p logs
mkdir -p output_${SLURM_JOB_ID}

# Change to working directory
cd $SLURM_SUBMIT_DIR

# Print Python environment info
echo ""
echo "Python environment:"
python --version
pip list | grep -E "(numpy|scipy|cupy|matplotlib)"
echo ""

# ============================================================================
# Run simulation
# ============================================================================

# Choose ONE of the following:

# 1. Quick test run
echo "Running quick test..."
python examples/run_simulation.py \
    --preset fast \
    --output output_${SLURM_JOB_ID}/test \
    --backend auto

# 2. Full simulation with default settings
# echo "Running default simulation..."
# python examples/run_simulation.py \
#     --preset default \
#     --output output_${SLURM_JOB_ID}/default \
#     --backend auto

# 3. High resolution run
# echo "Running high-resolution simulation..."
# python examples/run_simulation.py \
#     --preset highres \
#     --output output_${SLURM_JOB_ID}/highres \
#     --backend auto

# 4. Parameter sweep - viscosity
# echo "Running viscosity sweep..."
# python examples/run_simulation.py \
#     --sweep viscosity \
#     --preset default \
#     --output output_${SLURM_JOB_ID}/sweep_viscosity \
#     --backend auto

# 5. Parameter sweep - Reynolds number
# echo "Running Reynolds number sweep..."
# python examples/run_simulation.py \
#     --sweep reynolds \
#     --preset default \
#     --output output_${SLURM_JOB_ID}/sweep_reynolds \
#     --backend auto

# 6. GPU run (if GPU requested)
# echo "Running on GPU..."
# python examples/run_simulation.py \
#     --preset default \
#     --output output_${SLURM_JOB_ID}/gpu \
#     --backend gpu

# ============================================================================
# Post-processing
# ============================================================================

echo ""
echo "=========================================="
echo "Job completed at: $(date)"
echo "Output saved to: output_${SLURM_JOB_ID}/"
echo "=========================================="

# Optional: Send email notification
# mail -s "Fluid Dynamics Job $SLURM_JOB_ID Complete" your_email@stanford.edu <<< "Job finished successfully"

# Optional: Archive results
# tar -czf results_${SLURM_JOB_ID}.tar.gz output_${SLURM_JOB_ID}/
