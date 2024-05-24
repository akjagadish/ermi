#!/bin/bash -l
#SBATCH -o ./logs/%A.out
#SBATCH -e ./logs/%A.err
#SBATCH --job-name=OpenML
#SBATCH --mail-type=ALL
#SBATCH --mail-user=akshaykjagadish@gmail.com
#SBATCH --time=72:00:00
#SBATCH --cpus-per-task=18


cd ~/ermi/categorisation/openml

module purge
module load anaconda/3/2023.03
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
pip3 install --user openml

python compute_stats.py