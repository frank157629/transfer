#!/usr/local_rwth/bin/zsh
### SBATCH directives ###
#SBATCH --gres=gpu:1                   # 1 块 GPU
#SBATCH --mem=128G                      # 内存上限
#SBATCH --time=12:00:00                # 12 小时
#SBATCH --job-name=BA_Haitian           # 作业名称
#SBATCH --output=logs/%J.out           # 标准输出日志
#SBATCH --error=logs/%J.err            # 标准错误日志
#SBATCH --account=rwth1854
##########################

# 1. 加载 CUDA 模块（与 PyTorch 版本匹配）
module load cuda/11.8

# 2. 激活 Conda 环境
export CONDA_ROOT=$HOME/miniforge3
source $CONDA_ROOT/etc/profile.d/conda.sh
conda activate pinn_env

# 3. 打印节点 & GPU 信息
echo "Running on $(hostname)"
nvidia-smi || echo "CPU-only run"

# 4. 进入项目目录并启动脚本
cd ~/transfer/PowerPINN

echo "Python  :" $(which python)
python - <<'PY'
import torch, os, sys
print("Torch   :", torch.__version__)
print("CUDA OK :", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU     :", torch.cuda.get_device_name())
PY
python test_sweep.py