#!/bin/bash -l
#SBATCH -o ./logs/%A.out
#SBATCH -e ./logs/%A.err
#SBATCH --job-name=fit_humans
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



# optimizer

## badham2017
# python mi/fit_humans.py --paradigm categorisation --task-name badham2017  --num-iters 1 --method bounded_soft_sigmoid --model-name env=claude_generated_tasks_paramsNA_dim3_data100_tasks11518_pversion4_model=transformer_num_episodes500000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8_noise0.0_shuffleTrue_run=3
# python mi/fit_humans.py --paradigm categorisation --task-name badham2017  --num-iters 1 --method bounded_soft_sigmoid --model-name env=dim3synthetic_model=transformer_num_episodes500000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8_noise0.0_shuffleTrue_run=0_synthetic
# python mi/fit_humans.py --paradigm categorisation --task-name badham2017  --num-iters 1 --method bounded_soft_sigmoid --model-name env=dim3synthetic_model=transformer_num_episodes500000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8_noise0.0_shuffleTrue_run=0_syntheticnonlinear

## devraj2022
# python mi/fit_humans.py --paradigm categorisation --task-name devraj2022  --num-iters 1 --method bounded_soft_sigmoid  --model-name env=claude_generated_tasks_paramsNA_dim6_data500_tasks12910_pversion5_stage2_model=transformer_num_episodes500000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8_noise0.0_shuffleTrue_run=1
# python mi/fit_humans.py --paradigm categorisation --task-name devraj2022  --num-iters 1 --method bounded_soft_sigmoid  --model-name env=dim6synthetic_model=transformer_num_episodes500000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8_noise0.0_shuffleTrue_run=2_synthetic
# python mi/fit_humans.py --paradigm categorisation --task-name devraj2022 --num-iters 1 --method bounded_soft_sigmoid --model-name env=dim6synthetic_model=transformer_num_episodes500000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8_noise0.0_shuffleTrue_run=2_syntheticnonlinear

