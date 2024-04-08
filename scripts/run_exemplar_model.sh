#!/bin/bash -l
#SBATCH -o ./logs/%A.out
#SBATCH -e ./logs/%A.err
#SBATCH --job-name=exemplar_model
#SBATCH --mail-type=ALL
#SBATCH --mail-user=akshaykjagadish@gmail.com
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=18


cd ~/ermi/categorisation/
module purge
module load cuda/11.6
module load anaconda/3/2023.03
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

## devraj et al 2022 fit to fitted meta-learner simulations
# python baselines/run_gcm.py --num-iter 10 --task-name devraj2022 --loss mse_transfer --num-blocks 11  --model-name env=claude_generated_tasks_paramsNA_dim6_data500_tasks12910_pversion5_stage2_model=transformer_num_episodes500000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8_noise0.0_shuffleTrue_run=1  --method soft_sigmoid
# python baselines/run_gcm.py --num-iter 10 --task-name devraj2022 --loss mse_transfer --num-blocks 11  --model-name env=dim6synthetic_model=transformer_num_episodes500000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8_noise0.0_shuffleTrue_run=2_synthetic  --method soft_sigmoid
# python baselines/run_gcm.py --num-iter 10 --task-name devraj2022 --loss mse_transfer --num-blocks 11  --model-name env=dim6synthetic_model=transformer_num_episodes500000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8_noise0.0_shuffleTrue_run=2_syntheticnonlinear  --method soft_sigmoid

python baselines/run_gcm.py --num-iter 1 --task-name devraj2022 --loss mse_transfer --num-blocks 33  --model-name env=claude_generated_tasks_paramsNA_dim6_data500_tasks12910_pversion5_stage2_model=transformer_num_episodes500000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8_noise0.0_shuffleTrue_run=1  --method soft_sigmoid
# python baselines/run_gcm.py --num-iter 10 --task-name devraj2022 --loss mse_transfer --num-blocks 22  --model-name env=dim6synthetic_model=transformer_num_episodes500000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8_noise0.0_shuffleTrue_run=2_synthetic  --method soft_sigmoid
# python baselines/run_gcm.py --num-iter 10 --task-name devraj2022 --loss mse_transfer --num-blocks 22  --model-name env=dim6synthetic_model=transformer_num_episodes500000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8_noise0.0_shuffleTrue_run=2_syntheticnonlinear  --method soft_sigmoid

# python baselines/run_gcm.py --num-iter 1 --loss mse_transfer --num-blocks 11 --fit-llm --task-name devraj2022


#### deprecated
# python run_gcm.py --beta 0.0 --num-iter 10
# python run_gcm.py --beta 0.1 --num-iter 10
# python run_gcm.py --beta 0.2 --num-iter 10
# python run_gcm.py --beta 0.3 --num-iter 10
# python run_gcm.py --beta 0.4 --num-iter 10
# python run_gcm.py --beta 0.5 --num-iter 10
# python run_gcm.py --beta 0.6 --num-iter 10
# python run_gcm.py --beta 0.7 --num-iter 10
# python run_gcm.py --beta 0.8 --num-iter 10
# python run_gcm.py --beta 0.9 --num-iter 10
# python run_gcm.py --beta 1.0 --num-iter 10
# python run_gcm_parallelised.py

## devraj et al 2022
# python run_gcm.py --num-iter 1 --loss 'nll' --num-blocks 1 --fit-human-data
# python run_gcm.py --num-iter 1 --loss 'nll' --num-blocks 11 --fit-human-data

## badham et al 2017
# python run_gcm.py --num-iter 1 --loss 'nll' --num-blocks 1 --fit-human-data
