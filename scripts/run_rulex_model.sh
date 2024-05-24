#!/bin/bash -l
#SBATCH -o ./logs/%A.out
#SBATCH -e ./logs/%A.err
#SBATCH --job-name=RULEX
#SBATCH --mail-type=ALL
#SBATCH --mail-user=akshaykjagadish@gmail.com
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=18


cd ~/ermi/categorisation/baselines/
module purge
module load cuda/11.6
module load anaconda/3/2023.03
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# python run_rulex.py --num-iter 1 --loss 'nll' --num-blocks 1 --fit-human-data
# python run_rulex.py --num-iter 1 --loss 'nll' --num-blocks 1 --fit-human-data --exception --task-name badham2017
python run_rulex.py --num-iter 1 --loss 'nll' --num-blocks 1 --fit-human-data --task-name badham2017
