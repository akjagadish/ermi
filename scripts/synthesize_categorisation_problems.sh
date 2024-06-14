#!/bin/bash -l
#SBATCH -o ./logs/%A.out
#SBATCH -e ./logs/%A.err
#SBATCH --job-name=synthesize_problems_llama
#SBATCH --time=24:00:00
#SBATCH --constraint="gpu"
#SBATCH --gres=gpu:a100:4
#SBATCH --mem=160G
#SBATCH --cpus-per-task=72
#SBATCH --mail-type=ALL
#SBATCH --mail-user=akshaykjagadish@gmail.com


cd ~/ermi/taskgeneration

module purge
module load anaconda/3/2021.11
pip3 install --user accelerate openai gym ipdb transformers tensorboard python-dotenv
pip3 install --user fire sentencepiece ipdb accelerate tqdm
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

## generate task labels
# python synthesize_problems.py --task categorisation --model 70B --proc-id 0 --num-runs 100 --num-tasks 100 --num-dim 2 --max-length 10000 --run-gpt llama-3 --prompt-version 0 --path /u/ajagadish/ermi/categorisation/data/synthesize_problems

# python synthesize_
## pool generated task labels
python synthesize_problems.py --task categorisation --model 70B --proc-id 0 --num-runs 100 --num-tasks 100 --num-dim 2 --max-length 10000 --run-gpt llama-3 --prompt-version 0 --pool --path /u/ajagadish/ermi/categorisation/data/synthesize_problems
