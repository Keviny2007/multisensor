#!/bin/bash
#SBATCH --job-name=multipair_linear_probe
#SBATCH --output=logs/linear_probe_%j.out
#SBATCH --error=logs/linear_probe_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=YOUR_EMAIL@brown.edu

# Print job info
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "Working directory: $(pwd)"

# Load modules (adjust based on OSCAR's available modules)
# module load python/3.10.8
# module load cuda/11.8
# module load cudnn/8.6

# Create logs directory if it doesn't exist
mkdir -p logs

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python -m venv .venv
fi

# Activate virtual environment
source .venv/bin/activate

# Install uv (fast package installer)
echo "Installing uv..."
pip install uv

# Upgrade pip using uv
uv pip install --upgrade pip

# Install requirements using uv (skip ray for Python 3.10+)
echo "Installing requirements with uv..."
grep -v "ray==" requirements.txt > requirements_no_ray.txt
uv pip install -r requirements_no_ray.txt

# Verify GPU availability
echo "Checking GPU availability..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}'); print(f'GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

# Set environment variables for better performance
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Run training (using GPU config)
echo "Starting training..."
python loo_cross_validation.py \
    -p params/MultiPair_downstream_experiments/MultiPair_FourSensor_bilateral_linear_probe/config_gpu.yml \
    -d data/processed_test_data

echo "Job finished at: $(date)"
