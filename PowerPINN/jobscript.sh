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
module load CUDA/12.6.3

# 2. 激活 Conda 环境
export CONDA_ROOT=$HOME/miniforge3
source $CONDA_ROOT/etc/profile.d/conda.sh
conda activate pinn_env
export PYTHONPATH=$PYTHONPATH:$HOME/transfer/PowerPINN/src
export WANDB_API_KEY="c26f0418182418f6712b79b4457de4faa81b7524"

# 3. 打印节点 & GPU 信息
echo "Running on $(hostname)"
nvidia-smi || echo "CPU-only run"

python - <<'PY'
import torch, platform, os
print("Torch :", torch.__version__)
print("CUDA? :", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU   :", torch.cuda.get_device_name())
PY

# 4. 进入项目目录并启动脚本
cd ~/transfer/PowerPINN
python -u test_sweep.py