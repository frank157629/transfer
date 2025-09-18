#!/usr/local_rwth/bin/zsh
### SBATCH directives ###
#SBATCH --gres=gpu:1                   # 1 GPU
#SBATCH --mem=128G                     # Memory limit
#SBATCH --time=12:00:00                # 12 hours
#SBATCH --job-name=BA_Haitian          # Job name
#SBATCH --output=logs/%J.out           # Standard output log
#SBATCH --error=logs/%J.err            # Standard error log
#SBATCH --account=rwth1854
##########################

# 1. Load CUDA module (match your PyTorch build)
module load CUDA/12.6.3

# 2. Activate Conda environment
export CONDA_ROOT=$HOME/miniforge3
source $CONDA_ROOT/etc/profile.d/conda.sh
conda activate pinn_env
export PYTHONPATH=$PYTHONPATH:$HOME/transfer/PowerPINN/src
export WANDB_API_KEY="c26f0418182418f6712b79b4457de4faa81b7524"

# 3. Print node & GPU info
echo "Running on $(hostname)"
nvidia-smi || echo "CPU-only run"

python - <<'PY'
import torch, platform, os
print("Torch :", torch.__version__)
print("CUDA? :", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU   :", torch.cuda.get_device_name())
PY

# 4. Change to project directory and launch script
cd ~/transfer/PowerPINN
python -u test_sweep.py