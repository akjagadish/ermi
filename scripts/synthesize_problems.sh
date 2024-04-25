#!/bin/bash -l
#SBATCH -o ./logs/%A.out
#SBATCH -e ./logs/%A.err
#SBATCH --job-name=Claude
#SBATCH --mail-type=ALL
#SBATCH --mail-user=akshaykjagadish@gmail.com
#SBATCH --time=72:00:00
#SBATCH --cpus-per-task=18


cd ~/ermi/categorisation/task_generation

module purge
module load anaconda/3/2021.11
pip3 install --user accelerate openai gym ipdb transformers tensorboard python-dotenv
pip3 install --user fire sentencepiece ipdb accelerate tqdm
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

## generate task labels
python synthesize_problems.py --model NA --task functionlearning --proc-id 0 --num-runs 1 --num-tasks 250 --num-dim 1 --max-length 10000 --run-gpt claude --prompt-version 0 
## pool generated task labels
# python synthesize_problems.py --model NA --proc-id 0 --num-runs 100 --num-tasks 250 --num-dim 3 --max-length 10000 --run-gpt claude --prompt-version 5 --pool --path /u/ajagadish/vanilla-llama/categorisation/data/tasklabels
