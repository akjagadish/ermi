#!/bin/bash -l
#SBATCH -o ./logs/%A.out
#SBATCH -e ./logs/%A.err
#SBATCH --job-name=SyntheticTaskGeneration  
#SBATCH --mail-type=ALL
#SBATCH --mail-user=akshaykjagadish@gmail.com
#SBATCH --time=72:00:00
#SBATCH --cpus-per-task=18


cd ~/ermi/mi/

module purge
module load anaconda/3/2023.03
module load gcc/13 impi/2021.9


python simulate_data.py --num-tasks 10000 --num-dims 1 --max-steps 25 --paradigm functionlearning