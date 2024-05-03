#!/bin/bash -l
#SBATCH -o ./logs/%A.out
#SBATCH -e ./logs/%A.err
#SBATCH --job-name=Claude
#SBATCH --mail-type=ALL
#SBATCH --mail-user=akshaykjagadish@gmail.com
#SBATCH --time=72:00:00
#SBATCH --cpus-per-task=18


cd ~/ermi/taskgeneration

module purge
module load anaconda/3/2021.11
pip3 install --user accelerate openai gym ipdb transformers tensorboard python-dotenv
pip3 install --user fire sentencepiece ipdb accelerate tqdm
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

## generate task labels
python synthesize_problems.py --model NA --task decisionmaking --proc-id 0 --num-runs 1 --num-tasks 2 --num-dim 2 --max-length 10000 --run-gpt claude --prompt-version 0 --path /u/ajagadish/ermi/decisionmaking/data/synthesize_problems

# python synthesize_
## pool generated task labels
# python synthesize_problems.py --model NA --proc-id 0 --num-runs 100 --num-tasks 250 --num-dim 3 --max-length 10000 --run-gpt claude --prompt-version 5 --pool --path /u/ajagadish/vanilla-llama/categorisation/data/tasklabels
# python synthesize_problems.py --model NA --proc-id 0 --num-runs 100 --num-tasks 100 --num-dim 1 --max-length 10000 --run-gpt claude --prompt-version 0 --pool --path /u/ajagadish/ermi/categorisation/data/synthesize_problems
