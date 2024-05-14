#!/bin/bash -l
#SBATCH -o ./logs/%A.out
#SBATCH -e ./logs/%A.err
#SBATCH --job-name=fit_humans
#SBATCH --mail-type=ALL
#SBATCH --mail-user=akshaykjagadish@gmail.com
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=18


cd ~/ermi/
module purge
module load cuda/11.6
module load anaconda/3/2023.03
pip3 install --user accelerate openai gym ipdb transformers tensorboard anthropic openml wordcloud mycolorpy Pillow pyro-ppl 
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}


# optimize model parameters for meta-learners

##  binz2022
python mi/fit_humans.py --paradigm decisionmaking --task-name binz2022 --optimizer --num-iters 1 --model-name env=claude_generated_functionlearningtasks_paramsNA_dim2_data20_tasks9254_run0_procid0_pversion2_model=transformer_num_episodes500000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8_noise0.0_shuffleTrue_run=0
# python mi/fit_humans.py --paradigm decisionmaking --task-name binz2022 --optimizer --num-iters 1 --model-name env=synthetic_model=transformer_num_episodes100000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8_noise0.0_shuffleTrue_run=0_synthetic
# python mi/fit_humans.py --paradigm decisionmaking --task-name binz2022 --optimizer --num-iters 1 --model-name env=claude_generated_functionlearningtasks_paramsNA_dim2_data20_tasks9254_run0_procid0_pversion2_model=transformer_num_episodes100000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8_noise0.0_shuffleTrue_run=0

# paired model on binz2022
# python mi/fit_humans.py --paradigm decisionmaking --task-name binz2022 --optimizer --paired --num-iters 1 --model-name env=claude_generated_functionlearningtasks_paramsNA_dim2_data20_tasks9254_run0_procid0_pversion2_unknown_model=transformer_num_episodes100000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8_noise0.0_shuffleTrue_pairedTrue_run=0
# python mi/fit_humans.py --paradigm decisionmaking --task-name binz2022 --optimizer --paired --num-iters 1 --model-name env=synthetic_model=transformer_num_episodes100000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8_noise0.0_shuffleTrue_pairedTrue_run=0_synthetic

## badham2017
# python model_comparison/fit_humans.py --model-name env=claude_generated_tasks_paramsNA_dim3_data100_tasks11518_pversion4_model=transformer_num_episodes500000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8_noise0.0_shuffleTrue_run=3 --task-name badham2017 --optimizer
# python model_comparison/fit_humans.py --model-name env=dim3synthetic_model=transformer_num_episodes500000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8_noise0.0_shuffleTrue_run=0_synthetic  --task-name badham2017 --optimizer
# python model_comparison/fit_humans.py --model-name env=dim3synthetic_model=transformer_num_episodes500000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8_noise0.0_shuffleTrue_run=0_syntheticnonlinear  --task-name badham2017 --optimizer
# python model_comparison/fit_humans.py --model-name env=rmc_tasks_dim3_data100_tasks11499_model=transformer_num_episodes500000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8_noise0.0_shuffleTrue_run=1_rmc  --task-name badham2017 --optimizer


## devraj2022
# python model_comparison/fit_humans.py --model-name env=claude_generated_tasks_paramsNA_dim6_data500_tasks12910_pversion5_stage2_model=transformer_num_episodes500000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8_noise0.0_shuffleTrue_run=1 --task-name devraj2022 --optimizer
# python model_comparison/fit_humans.py --model-name env=dim6synthetic_model=transformer_num_episodes500000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8_noise0.0_shuffleTrue_run=2_synthetic  --task-name devraj2022 --optimizer
# python model_comparison/fit_humans.py --model-name env=dim6synthetic_model=transformer_num_episodes500000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8_noise0.0_shuffleTrue_run=2_syntheticnonlinear  --task-name devraj2022 --optimizer
# python model_comparison/fit_humans.py --model-name env=rmc_tasks_dim6_data600_tasks12911_model=transformer_num_episodes500000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8_noise0.0_shuffleTrue_run=0_rmc  --task-name devraj2022 --optimizer


# baseline models

## badham2017
# python baselines/run_gcm.py --num-iter 1 --loss 'nll' --num-blocks 1 --fit-human-data --task-name badham2017
# python baselines/run_pm.py --num-iter 1 --loss 'nll' --num-blocks 1 --fit-human-data --prototypes from_data --task-name badham2017

## devraj2022
# python baselines/run_gcm.py --num-iter 1 --loss 'nll' --num-blocks 1 --fit-human-data --task-name devraj2022
# python baselines/run_pm.py --num-iter 1 --loss 'nll' --num-blocks 1 --fit-human-data --prototypes from_data --task-name devraj2022


# grid search for meta-learners

### beta sweep

## badham2017
# python model_comparison/fit_humans.py --model-name env=claude_generated_tasks_paramsNA_dim3_data100_tasks11518_pversion4_model=transformer_num_episodes500000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8_noise0.0_shuffleTrue_run=0 --task-name badham2017
# python model_comparison/fit_humans.py --model-name env=dim3synthetic_model=transformer_num_episodes500000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8_noise0.0_shuffleTrue_run=0_synthetic  --task-name badham2017
# python model_comparison/fit_humans.py --model-name env=dim3synthetic_model=transformer_num_episodes500000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8_noise0.0_shuffleTrue_run=0_syntheticnonlinear  --task-name badham2017

## devraj2022
# python model_comparison/fit_humans.py --model-name env=claude_generated_tasks_paramsNA_dim6_data500_tasks12910_pversion5_stage2_model=transformer_num_episodes500000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8_noise0.0_shuffleTrue_run=1 --task-name devraj2022
# python model_comparison/fit_humans.py --model-name env=dim6synthetic_model=transformer_num_episodes500000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8_noise0.0_shuffleTrue_run=0_synthetic  --task-name devraj2022
# python model_comparison/fit_humans.py --model-name env=dim6synthetic_model=transformer_num_episodes500000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8_noise0.0_shuffleTrue_run=0_syntheticnonlinear  --task-name devraj2022

### epsilon sweep

# badham2017
# python model_comparison/fit_humans.py --model-name env=claude_generated_tasks_paramsNA_dim3_data100_tasks11518_pversion4_model=transformer_num_episodes500000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8_noise0.0_shuffleTrue_run=0 --task-name badham2017 --method eps_greedy
# python model_comparison/fit_humans.py --model-name env=dim3synthetic_model=transformer_num_episodes500000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8_noise0.0_shuffleTrue_run=0_synthetic  --task-name badham2017 --method eps_greedy
# python model_comparison/fit_humans.py --model-name env=dim3synthetic_model=transformer_num_episodes500000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8_noise0.0_shuffleTrue_run=0_syntheticnonlinear  --task-name badham2017 --method eps_greedy

# devraj2022
# python model_comparison/fit_humans.py --model-name env=claude_generated_tasks_paramsNA_dim6_data500_tasks12910_pversion5_stage2_model=transformer_num_episodes500000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8_noise0.0_shuffleTrue_run=1 --task-name devraj2022 --method eps_greedy
# python model_comparison/fit_humans.py --model-name env=dim6synthetic_model=transformer_num_episodes500000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8_noise0.0_shuffleTrue_run=0_synthetic  --task-name devraj2022 --method eps_greedy
# python model_comparison/fit_humans.py --model-name env=dim6synthetic_model=transformer_num_episodes500000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8_noise0.0_shuffleTrue_run=0_syntheticnonlinear  --task-name devraj2022 --method eps_greedy

