#!/bin/bash -l
#SBATCH -o ./logs/%A.out
#SBATCH -e ./logs/%A.err
#SBATCH --job-name=simulate_metalearner
#SBATCH --time=24:00:00
#SBATCH --constraint="gpu"
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=40G
#SBATCH --cpus-per-task=18
#SBATCH --mail-type=ALL
#SBATCH --mail-user=akshaykjagadish@gmail.com

cd ~/ermi/
module purge
module load anaconda/3/2023.03
module load gcc/13 impi/2021.9
module load cuda/12.1
module load pytorch/gpu-cuda-12.1/2.2.0
pip3 install --user ipdb torch transformers tensorboard ipdb tqdm nnsight sentencepiece 


## experiment 3
## paired
python mi/ablate_model.py --paradigm decisionmaking --task-name binz2022 --exp-id 3 --policy greedy --layer all_layers --percent-neurons 5 --paired --model-name env=claude_generated_functionlearningtasks_paramsNA_dim2_data20_tasks9254_run0_procid0_pversion2_unknown_model=transformer_num_episodes100000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8_noise0.0_shuffleTrue_pairedTrue_lossnll_run=0
python mi/ablate_model.py --paradigm decisionmaking --task-name binz2022 --exp-id 3 --policy greedy --layer all_layers --percent-neurons 10 --paired --model-name env=claude_generated_functionlearningtasks_paramsNA_dim2_data20_tasks9254_run0_procid0_pversion2_unknown_model=transformer_num_episodes100000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8_noise0.0_shuffleTrue_pairedTrue_lossnll_run=0
python mi/ablate_model.py --paradigm decisionmaking --task-name binz2022 --exp-id 3 --policy greedy --layer all_layers --percent-neurons 15 --paired --model-name env=claude_generated_functionlearningtasks_paramsNA_dim2_data20_tasks9254_run0_procid0_pversion2_unknown_model=transformer_num_episodes100000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8_noise0.0_shuffleTrue_pairedTrue_lossnll_run=0
python mi/ablate_model.py --paradigm decisionmaking --task-name binz2022 --exp-id 3 --policy greedy --layer all_layers --percent-neurons 20 --paired --model-name env=claude_generated_functionlearningtasks_paramsNA_dim2_data20_tasks9254_run0_procid0_pversion2_unknown_model=transformer_num_episodes100000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8_noise0.0_shuffleTrue_pairedTrue_lossnll_run=0
python mi/ablate_model.py --paradigm decisionmaking --task-name binz2022 --exp-id 3 --policy greedy --layer all_layers --percent-neurons 25 --paired --model-name env=claude_generated_functionlearningtasks_paramsNA_dim2_data20_tasks9254_run0_procid0_pversion2_unknown_model=transformer_num_episodes100000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8_noise0.0_shuffleTrue_pairedTrue_lossnll_run=0
python mi/ablate_model.py --paradigm decisionmaking --task-name binz2022 --exp-id 3 --policy greedy --layer all_layers --percent-neurons 30 --paired --model-name env=claude_generated_functionlearningtasks_paramsNA_dim2_data20_tasks9254_run0_procid0_pversion2_unknown_model=transformer_num_episodes100000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8_noise0.0_shuffleTrue_pairedTrue_lossnll_run=0
python mi/ablate_model.py --paradigm decisionmaking --task-name binz2022 --exp-id 3 --policy greedy --layer all_layers --percent-neurons 35 --paired --model-name env=claude_generated_functionlearningtasks_paramsNA_dim2_data20_tasks9254_run0_procid0_pversion2_unknown_model=transformer_num_episodes100000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8_noise0.0_shuffleTrue_pairedTrue_lossnll_run=0
python mi/ablate_model.py --paradigm decisionmaking --task-name binz2022 --exp-id 3 --policy greedy --layer all_layers --percent-neurons 40 --paired --model-name env=claude_generated_functionlearningtasks_paramsNA_dim2_data20_tasks9254_run0_procid0_pversion2_unknown_model=transformer_num_episodes100000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8_noise0.0_shuffleTrue_pairedTrue_lossnll_run=0
python mi/ablate_model.py --paradigm decisionmaking --task-name binz2022 --exp-id 3 --policy greedy --layer all_layers --percent-neurons 45 --paired --model-name env=claude_generated_functionlearningtasks_paramsNA_dim2_data20_tasks9254_run0_procid0_pversion2_unknown_model=transformer_num_episodes100000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8_noise0.0_shuffleTrue_pairedTrue_lossnll_run=0
python mi/ablate_model.py --paradigm decisionmaking --task-name binz2022 --exp-id 3 --policy greedy --layer all_layers --percent-neurons 50 --paired --model-name env=claude_generated_functionlearningtasks_paramsNA_dim2_data20_tasks9254_run0_procid0_pversion2_unknown_model=transformer_num_episodes100000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8_noise0.0_shuffleTrue_pairedTrue_lossnll_run=0
