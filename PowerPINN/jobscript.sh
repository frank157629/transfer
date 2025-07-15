#!/usr/local_rwth/bin/zsh
### SBATCH directives ###
#SBATCH --account=rwth1854     # ← 换成自己的账号；没有就删除此行
#SBATCH --partition=gpuv100            # GPU 队列名；CPU 调试可改 c23ms
#SBATCH --gres=gpu:1                   # 1 块 GPU
#SBATCH --mem=16G                      # 内存上限
#SBATCH --time=12:00:00                # 12 小时
#SBATCH --job-name=powerpinn           # 作业名称
#SBATCH --output=logs/%J.out           # 标准输出日志
#SBATCH --error=logs/%J.err            # 标准错误日志
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
python test_sweep.py