#!/bin/bash -l
#SBATCH -o ./logs/%A.out
#SBATCH -e ./logs/%A.err
#SBATCH --job-name=generate_tasks_llama
#SBATCH --time=24:00:00
#SBATCH --constraint="gpu"
#SBATCH --gres=gpu:a100:4
#SBATCH --mem=160G
#SBATCH --cpus-per-task=72
#SBATCH --mail-type=ALL
#SBATCH --mail-user=akshaykjagadish@gmail.com

cd ~/ermi/taskgeneration
## llama-3
#python generate_data.py --model NA --task categorisation --proc-id 0 --num-tasks 100 --start-task-id 0 --num-dim 3 --num-data 100 --max-length 8000 --run-gpt llama-3 --prompt-version 4 --file-name-tasklabels claude_generated_tasklabels_paramsNA_dim3_tasks23421_pversion5 --path-tasklabels /u/ajagadish/ermi/categorisation/data/task_labels
#python generate_data.py --model NA --task categorisation --proc-id 0 --num-tasks 1000 --start-task-id 0 --num-dim 3 --num-data 100 --max-length 8000 --run-gpt llama-3 --prompt-version 4 --file-name-tasklabels claude_generated_tasklabels_paramsNA_dim3_tasks23421_pversion5 --path-tasklabels /u/ajagadish/ermi/categorisation/data/task_labels
# python generate_data.py --model NA --task categorisation --proc-id 0  --num-tasks 1000 --start-task-id 0 --num-dim 6 --num-data 100 --max-length 8000 --run-gpt llama-3 --prompt-version 5 --file-name-tasklabels claude_generated_tasklabels_paramsNA_dim6_tasks13693_pversion5 --path-tasklabels /u/ajagadish/ermi/categorisation/data/task_labels
python generate_data.py --model NA --task categorisation --proc-id 0 --num-tasks 1000 --start-task-id 0 --num-dim 4 --num-data 100 --max-length 8000 --run-gpt llama-3 --prompt-version 5  --file-name-tasklabels claude_generated_tasklabels_paramsNA_dim4_tasks20690_pversion5 --path-tasklabels /u/ajagadish/ermi/categorisation/data/task_labels
