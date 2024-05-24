#!/bin/bash -l
#SBATCH -o ./logs/%A.out
#SBATCH -e ./logs/%A.err
#SBATCH --job-name=Claude
#SBATCH --mail-type=ALL
#SBATCH --mail-user=akshaykjagadish@gmail.com
#SBATCH --time=72:00:00
#SBATCH --cpus-per-task=18


cd ~/ermi/categorisation/taskgeneration

module purge
module load anaconda/3/2021.11
pip3 install --user accelerate openai gym ipdb transformers tensorboard python-dotenv
pip3 install --user fire sentencepiece ipdb accelerate tqdm
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}


## two stages for 4 dims data
python parse_generated_tasks.py  --stage 1 --proc_ids 0 1 2 3 4 5 6 7 8 8 10 11 12 13 14  --gpt claude --dims 4 --num-data 650 --tasks 1000 --runs 0 --prompt_version 5 --use_generated_tasklabels --path /u/ajagadish/vanilla-llama/categorisation/data

## two stages for 6 dims data
# python parse_generated_tasks.py  --proc_ids 3 5 6 7 8 9 --dims 6 --tasks 3000 --stage 2 --num-data 500 --prompt_version 5 --runs 0 --gpt claude --use_generated_tasklabels  --path /u/ajagadish/vanilla-llama/categorisation/data
## after renaming the files
# python parse_generated_tasks.py  --proc_ids 0 1 2 4 3 5 6 7 10 8 9 --dims 6 --tasks 3000 --stage 2 --num-data 500 --prompt_version 5 --runs 0 --gpt claude --use_generated_tasklabels  --path /u/ajagadish/vanilla-llama/categorisation/data

# testing claude 2.1
# python parse_generated_tasks.py  --proc_ids 0 --gpt claude_2.1 --dims 6 --tasks 10 --stage 1 --num-data 500 --prompt_version 5 --runs 0 --use_generated_tasklabels  --path /u/ajagadish/vanilla-llama/categorisation/data

# python parse_generated_tasks.py  --gpt claude --dims 3 --num-data 100 --tasks 1000 --runs 0  --prompt_version 4 --use_generated_tasklabels --proc_ids 1 0 2 3 4 5 6 7 8 --path /u/ajagadish/vanilla-llama/categorisation/data
# python parse_generated_tasks.py  --gpt claude --dims 6 --num-data 500 --tasks 1000 --runs 0  --prompt_version 5 --use_generated_tasklabels --proc_ids 0 --path /u/ajagadish/vanilla-llama/categorisation/data
# python parse_generated_tasks.py  --gpt claude --dims 4 --num-data 500 --tasks 1000 --runs 0  --prompt_version 5 --use_generated_tasklabels --proc_ids 0 1 --path /u/ajagadish/vanilla-llama/categorisation/data
# python parse_generated_tasks.py  --gpt claude --dims 6 --num-data 500 --tasks 100 --runs 0  --prompt_version 5 --use_generated_tasklabels --proc_ids 2 --path /u/ajagadish/vanilla-llama/categorisation/data