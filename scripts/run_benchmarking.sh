#!/bin/bash -l
#SBATCH -o ./logs/%A.out
#SBATCH -e ./logs/%A.err
#SBATCH --time=24:00:00
#SBATCH --constraint="gpu"
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=80G
#SBATCH --cpus-per-task=18
#SBATCH --mail-type=ALL
#SBATCH --mail-user=akshaykjagadish@gmail.com

# modules load
cd ~/ermi/categorisation/
module purge
module load anaconda/3/2021.11
module load gcc/11 impi/2021.6
module load cuda/11.6
module load pytorch_distributed/gpu-cuda-11.6/1.13.0
pip3 install tabpfn xgboost

python benchmark/eval.py 

