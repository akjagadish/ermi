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
pip3 install --user ipdb torch transformers tensorboard ipdb tqdm schedulefree


## experiment 1
## paired
# python mi/simulate_model.py --paradigm decisionmaking --task-name binz2022 --exp-id 1 --policy greedy --paired --model-name env=synthetic_dim4_model=transformer_num_episodes100000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8_noise0.0_shuffleTrue_pairedTrue_lossnll_run=0_ranking
# python mi/simulate_model.py --paradigm decisionmaking --task-name binz2022 --exp-id 1 --policy greedy --paired --model-name env=synthetic_dim4_model=transformer_num_episodes100000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8_noise0.0_shuffleTrue_pairedTrue_lossnll_run=0_unknown
# python mi/simulate_model.py --paradigm decisionmaking --task-name binz2022 --exp-id 1 --policy greedy --paired --model-name env=claude_generated_functionlearningtasks_paramsNA_dim4_data20_tasks8770_run0_procid1_pversionranked_model=transformer_num_episodes100000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8_noise0.0_shuffleTrue_pairedTrue_lossnll_run=0
# python mi/simulate_model.py --paradigm decisionmaking --task-name binz2022 --exp-id 1 --policy greedy --paired --model-name env=claude_generated_functionlearningtasks_paramsNA_dim4_data20_tasks7284_run0_procid1_pversionunknown_model=transformer_num_episodes100000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8_noise0.0_shuffleTrue_pairedTrue_lossnll_run=0
## unpaired
# python mi/simulate_model.py --paradigm decisionmaking --task-name binz2022 --exp-id 1 --policy greedy --model-name env=synthetic_dim4_model=transformer_num_episodes100000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8_noise0.0_shuffleTrue_pairedFalse_lossnll_run=0_ranking
# python mi/simulate_model.py --paradigm decisionmaking --task-name binz2022 --exp-id 1 --policy greedy --model-name env=synthetic_dim4_model=transformer_num_episodes100000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8_noise0.0_shuffleTrue_pairedFalse_lossnll_run=0_unknown
# python mi/simulate_model.py --paradigm decisionmaking --task-name binz2022 --exp-id 1 --policy greedy --model-name env=claude_generated_functionlearningtasks_paramsNA_dim4_data20_tasks8770_run0_procid1_pversionranked_model=transformer_num_episodes100000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8_noise0.0_shuffleTrue_pairedFalse_lossnll_run=0
# python mi/simulate_model.py --paradigm decisionmaking --task-name binz2022 --exp-id 1 --policy greedy --model-name env=claude_generated_functionlearningtasks_paramsNA_dim4_data20_tasks7284_run0_procid1_pversionunknown_model=transformer_num_episodes100000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8_noise0.0_shuffleTrue_pairedFalse_lossnll_run=0


## experiment 2
## paired
# python mi/simulate_model.py --paradigm decisionmaking --task-name binz2022 --exp-id 2 --policy greedy --paired --model-name env=synthetic_dim4_model=transformer_num_episodes100000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8_noise0.0_shuffleTrue_pairedTrue_lossnll_run=0_direction
# python mi/simulate_model.py --paradigm decisionmaking --task-name binz2022 --exp-id 2 --policy greedy --paired --model-name env=synthetic_dim4_model=transformer_num_episodes100000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8_noise0.0_shuffleTrue_pairedTrue_lossnll_run=0_unknown
# python mi/simulate_model.py --paradigm decisionmaking --task-name binz2022 --exp-id 2 --policy greedy --paired --model-name env=claude_generated_functionlearningtasks_paramsNA_dim4_data20_tasks8220_run0_procid1_pversiondirection_model=transformer_num_episodes100000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8_noise0.0_shuffleTrue_pairedTrue_lossnll_run=0
# python mi/simulate_model.py --paradigm decisionmaking --task-name binz2022 --exp-id 2 --policy greedy --paired --model-name env=claude_generated_functionlearningtasks_paramsNA_dim4_data20_tasks7284_run0_procid1_pversionunknown_model=transformer_num_episodes100000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8_noise0.0_shuffleTrue_pairedTrue_lossnll_run=0
## unpaired
# python mi/simulate_model.py --paradigm decisionmaking --task-name binz2022 --exp-id 2 --policy greedy --model-name env=synthetic_dim4_model=transformer_num_episodes100000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8_noise0.0_shuffleTrue_pairedFalse_lossnll_run=0_direction
# python mi/simulate_model.py --paradigm decisionmaking --task-name binz2022 --exp-id 2 --policy greedy --model-name env=synthetic_dim4_model=transformer_num_episodes100000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8_noise0.0_shuffleTrue_pairedFalse_lossnll_run=0_unknown
# python mi/simulate_model.py --paradigm decisionmaking --task-name binz2022 --exp-id 2 --policy greedy --model-name env=claude_generated_functionlearningtasks_paramsNA_dim4_data20_tasks8220_run0_procid1_pversiondirection_model=transformer_num_episodes100000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8_noise0.0_shuffleTrue_pairedFalse_lossnll_run=0
# python mi/simulate_model.py --paradigm decisionmaking --task-name binz2022 --exp-id 2 --policy greedy --model-name env=claude_generated_functionlearningtasks_paramsNA_dim4_data20_tasks7284_run0_procid1_pversionunknown_model=transformer_num_episodes100000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8_noise0.0_shuffleTrue_pairedFalse_lossnll_run=0


## experiment 3
## paired
# python mi/simulate_model.py --paradigm decisionmaking --task-name binz2022 --exp-id 3 --policy greedy --paired --model-name env=synthetic_dim2_model=transformer_num_episodes100000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8_noise0.0_shuffleTrue_pairedTrue_lossnll_run=0_unknown
# python mi/simulate_model.py --paradigm decisionmaking --task-name binz2022 --exp-id 3 --policy greedy --paired --model-name env=claude_generated_functionlearningtasks_paramsNA_dim2_data20_tasks9254_run0_procid0_pversion2_unknown_model=transformer_num_episodes100000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8_noise0.0_shuffleTrue_pairedTrue_lossnll_run=0

## BERMI
python mi/simulate_model.py --paradigm decisionmaking --task-name binz2022 --exp-id 3 --policy greedy --paired --ess ${SLURM_ARRAY_TASK_ID} --job-array --offset 30000 --scale 100 --model-name env=claude_dim2_model=transformer_num_episodes100000_num_hidden=256_lr0.1_num_layers=6_d_model=64_num_head=8_noise0.0_shuffleTrue_pairedTrue_lossvariational_essNone_std0.1_run=0
# python mi/simulate_model.py --paradigm decisionmaking --task-name binz2022 --exp-id 3 --policy greedy --paired --ess None --model-name env=claude_dim2_model=transformer_num_episodes100000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8_noise0.0_shuffleTrue_pairedTrue_lossvariational_essNone_std0.1_run=0
# python mi/simulate_model.py --paradigm decisionmaking --task-name binz2022 --exp-id 3 --policy greedy --paired --ess 1000000.0 --model-name env=claude_dim2_model=transformer_num_episodes100000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8_noise0.0_shuffleTrue_pairedTrue_lossvariational_ess1000000.0_std0.1_run=0
# python mi/simulate_model.py --paradigm decisionmaking --task-name binz2022 --exp-id 3 --policy greedy --paired --ess 1.0 --model-name env=claude_dim2_model=transformer_num_episodes100000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8_noise0.0_shuffleTrue_pairedTrue_lossvariational_ess1.0_std0.1_run=0

## unpaired
# python mi/simulate_model.py --paradigm decisionmaking --task-name binz2022 --exp-id 3 --policy greedy --model-name env=synthetic_dim2_model=transformer_num_episodes100000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8_noise0.0_shuffleTrue_pairedFalse_lossnll_run=0_unknown
# python mi/simulate_model.py --paradigm decisionmaking --task-name binz2022 --exp-id 3 --policy greedy --model-name env=claude_generated_functionlearningtasks_paramsNA_dim2_data20_tasks9254_run0_procid0_pversion2_unknown_model=transformer_num_episodes100000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8_noise0.0_shuffleTrue_pairedFalse_lossnll_run=0