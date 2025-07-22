#!/usr/local_rwth/bin/zsh
### SBATCH Section
#request one gpu per node
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --job-name=MA_fois
#SBATCH --output=ma_fois/%J.out
#SBATCH --time=0-24:00:00
#SBATCH --account=rwth1854

### Program Section
module load cuda/11.6
export CONDA_ROOT=$HOME/miniconda3
. $CONDA_ROOT/etc/profile.d/conda.sh
export PATH="$CONDA_ROOT/bin:$PATH"
conda activate env_fois
cd ma_fois
python3 test_sweep.py