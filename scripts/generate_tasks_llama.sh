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

## categorisation
#python generate_data.py --model NA --task categorisation --proc-id 0 --num-tasks 100  --start-task-id 0 --num-dim 3 --num-data 100 --max-length 8000 --run-gpt llama-3 --prompt-version 4 --file-name-tasklabels claude_generated_tasklabels_paramsNA_dim3_tasks23421_pversion5 --path-tasklabels /u/ajagadish/ermi/categorisation/data/task_labels
#python generate_data.py --model NA --task categorisation --proc-id 0 --num-tasks 1000 --start-task-id 0 --num-dim 3 --num-data 100 --max-length 8000 --run-gpt llama-3 --prompt-version 5 --file-name-tasklabels claude_generated_tasklabels_paramsNA_dim3_tasks23421_pversion5 --path-tasklabels /u/ajagadish/ermi/categorisation/data/task_labels
#python generate_data.py --model NA --task categorisation --proc-id 0 --num-tasks 1000 --start-task-id 0 --num-dim 6 --num-data 100 --max-length 8000 --run-gpt llama-3 --prompt-version 5 --file-name-tasklabels claude_generated_tasklabels_paramsNA_dim6_tasks13693_pversion5 --path-tasklabels /u/ajagadish/ermi/categorisation/data/task_labels
#python generate_data.py --model NA --task categorisation --proc-id 0 --num-tasks 1000 --start-task-id 0 --num-dim 4 --num-data 100 --max-length 8000 --run-gpt llama-3 --prompt-version 5 --file-name-tasklabels claude_generated_tasklabels_paramsNA_dim4_tasks20690_pversion5 --path-tasklabels /u/ajagadish/ermi/categorisation/data/task_labels
# python generate_data.py --model NA --task categorisation --proc-id 1 --num-tasks 1000 --start-task-id 373 --num-dim 3 --num-data 100 --max-length 8000 --run-gpt llama-3 --prompt-version 5 --file-name-tasklabels claude_generated_tasklabels_paramsNA_dim3_tasks23421_pversion5 --path-tasklabels /u/ajagadish/ermi/categorisation/data/task_labels
python generate_data.py --model NA --task categorisation --proc-id 2 --num-tasks 1000 --start-task-id 5000 --num-dim 3 --num-data 100 --max-length 8000 --run-gpt llama-3 --prompt-version 5 --file-name-tasklabels claude_generated_tasklabels_paramsNA_dim3_tasks23421_pversion5 --path-tasklabels /u/ajagadish/ermi/categorisation/data/task_labels

## function learning (proc-id 1 refers to adjusting the sorting order line of the prompt)
# python generate_data.py --model NA --task functionlearning --proc-id 1  --num-tasks 8770 --start-task-id 0 --num-dim 4 --num-data 20 --max-length 100000 --run-gpt llama-3 --prompt-version ranked --file-name-tasklabels claude_synthesized_functionlearning_problems_paramsNA_dim4_tasks8770_pversionranked  --path-tasklabels /u/ajagadish/ermi/decisionmaking/data/synthesize_problems
# python generate_data.py --model NA --task functionlearning --proc-id 1  --num-tasks 8220 --start-task-id 0 --num-dim 4 --num-data 20 --max-length 100000 --run-gpt claude --prompt-version direction --file-name-tasklabels claude_synthesized_functionlearning_problems_paramsNA_dim4_tasks8220_pversiondirection --path-tasklabels /u/ajagadish/ermi/decisionmaking/data/synthesize_problems
# python generate_data.py --model NA --task functionlearning --proc-id 1  --num-tasks 7284 --start-task-id 0 --num-dim 4 --num-data 20 --max-length 100000 --run-gpt claude --prompt-version unknown --file-name-tasklabels claude_synthesized_functionlearning_problems_paramsNA_dim4_tasks7284_pversion0  --path-tasklabels /u/ajagadish/ermi/decisionmaking/data/synthesize_problems
