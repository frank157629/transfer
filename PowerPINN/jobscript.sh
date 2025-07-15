#!/usr/local_rwth/bin/zsh
#SBATCH --partition=gpuv100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=8
#SBATCH --time=12:00:00
#SBATCH --mem=16G
#SBATCH --job-name=powerpinn
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

# ①（可选）加载与 PyTorch 对应的 CUDA 模块
# 若你保留 torch 2.1.0+cu118 ⇒
module load cuda/11.8
# 若换回 2.5.1+cu12.4 ⇒
# module load cuda/12.4
# 若用 CPU 版 ⇒ 注释掉 module 行

# ② 激活 conda 环境
export CONDA_ROOT=$HOME/miniforge3
source $CONDA_ROOT/etc/profile.d/conda.sh
conda activate pinn_env

# ③ 打印节点 / GPU 信息（调试用）
echo "Running on $(hostname)"
nvidia-smi || echo "CPU-only run"

# ④ 进入项目目录并启动脚本
python test_sweep.py