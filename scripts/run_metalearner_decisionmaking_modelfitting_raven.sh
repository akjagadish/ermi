#!/bin/bash -l
#SBATCH -o ./logs/%A.out
#SBATCH -e ./logs/%A.err
#SBATCH --job-name=fit_humans
#SBATCH --time=24:00:00
#SBATCH --mem=40G
#SBATCH --cpus-per-task=72
#SBATCH --mail-type=ALL
#SBATCH --mail-user=akshaykjagadish@gmail.com

cd ~/ermi/
module purge
module load anaconda/3/2023.03
module load gcc/13 impi/2021.9
module load cuda/12.1
module load pytorch/gpu-cuda-12.1/2.2.0
pip3 install --user ipdb torch transformers tensorboard ipdb tqdm schedulefree


###  binz2022
## experiment 3
# python mi/fit_humans.py --paradigm decisionmaking --task-name binz2022  --exp-id 3 --num-iters 1 --paired --model-name env=synthetic_dim2_model=transformer_num_episodes100000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8_noise0.0_shuffleTrue_pairedTrue_lossnll_run=0_unknown
# python mi/fit_humans.py --paradigm decisionmaking --task-name binz2022  --exp-id 3 --paired --num-iters 1 --model-name env=claude_generated_functionlearningtasks_paramsNA_dim2_data20_tasks9254_run0_procid0_pversion2_unknown_model=transformer_num_episodes100000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8_noise0.0_shuffleTrue_pairedTrue_lossnll_run=0
# python mi/fit_humans.py --paradigm decisionmaking --task-name binz2022  --exp-id 3 --paired --num-iters 5 --method bounded_soft_sigmoid  --model-name env=claude_generated_functionlearningtasks_paramsNA_dim2_data20_tasks9254_run0_procid0_pversion2_unknown_model=transformer_num_episodes100000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8_noise0.0_shuffleTrue_pairedTrue_lossnll_run=0
python mi/fit_humans.py --paradigm decisionmaking --task-name binz2022  --exp-id 3 --paired --num-iters 1 --method bounded_soft_sigmoid --model-name env=synthetic_dim2_model=transformer_num_episodes100000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8_noise0.0_shuffleTrue_pairedTrue_lossnll_run=0_unknown
# unpaired
# python mi/fit_humans.py --paradigm decisionmaking --task-name binz2022  --exp-id 3 --num-iters 1 --model-name env=synthetic_dim2_model=transformer_num_episodes100000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8_noise0.0_shuffleTrue_pairedFalse_lossnll_run=0_unknown
# python mi/fit_humans.py --paradigm decisionmaking --task-name binz2022  --exp-id 3 --num-iters 1 --model-name env=claude_generated_functionlearningtasks_paramsNA_dim2_data20_tasks9254_run0_procid0_pversion2_unknown_model=transformer_num_episodes100000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8_noise0.0_shuffleTrue_pairedFalse_lossnll_run=0

## experiment 1
## paired
# python mi/fit_humans.py --paradigm decisionmaking --task-name binz2022  --exp-id 1 --num-iters 1 --paired --model-name env=synthetic_dim4_model=transformer_num_episodes100000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8_noise0.0_shuffleTrue_pairedTrue_lossnll_run=0_ranking
# python mi/fit_humans.py --paradigm decisionmaking --task-name binz2022  --exp-id 1 --num-iters 1 --paired --model-name env=synthetic_dim4_model=transformer_num_episodes100000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8_noise0.0_shuffleTrue_pairedTrue_lossnll_run=0_unknown
# python mi/fit_humans.py --paradigm decisionmaking --task-name binz2022  --exp-id 1 --num-iters 1 --paired --model-name env=claude_generated_functionlearningtasks_paramsNA_dim4_data20_tasks8770_run0_procid1_pversionranked_model=transformer_num_episodes100000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8_noise0.0_shuffleTrue_pairedTrue_lossnll_run=0
# python mi/fit_humans.py --paradigm decisionmaking --task-name binz2022  --exp-id 1 --num-iters 1 --paired --model-name env=claude_generated_functionlearningtasks_paramsNA_dim4_data20_tasks7284_run0_procid1_pversionunknown_model=transformer_num_episodes100000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8_noise0.0_shuffleTrue_pairedTrue_lossnll_run=0
# python mi/fit_humans.py --paradigm decisionmaking --task-name binz2022  --exp-id 1 --num-iters 1 --paired --method bounded_soft_sigmoid --model-name env=synthetic_dim4_model=transformer_num_episodes100000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8_noise0.0_shuffleTrue_pairedTrue_lossnll_run=0_unknown
# python mi/fit_humans.py --paradigm decisionmaking --task-name binz2022  --exp-id 1 --num-iters 1 --paired --method bounded_soft_sigmoid --model-name env=claude_generated_functionlearningtasks_paramsNA_dim4_data20_tasks7284_run0_procid1_pversionunknown_model=transformer_num_episodes100000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8_noise0.0_shuffleTrue_pairedTrue_lossnll_run=0
# python mi/fit_humans.py --paradigm decisionmaking --task-name binz2022  --exp-id 1 --num-iters 1 --paired --method bounded_soft_sigmoid --model-name env=synthetic_dim4_model=transformer_num_episodes100000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8_noise0.0_shuffleTrue_pairedTrue_lossnll_run=0_ranking
# python mi/fit_humans.py --paradigm decisionmaking --task-name binz2022  --exp-id 1 --num-iters 1 --paired --method bounded_soft_sigmoid --model-name env=claude_generated_functionlearningtasks_paramsNA_dim4_data20_tasks8770_run0_procid1_pversionranked_model=transformer_num_episodes100000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8_noise0.0_shuffleTrue_pairedTrue_lossnll_run=0

## unpaired
# python mi/fit_humans.py --paradigm decisionmaking --task-name binz2022  --exp-id 1 --num-iters 1 --model-name env=synthetic_dim4_model=transformer_num_episodes100000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8_noise0.0_shuffleTrue_pairedFalse_lossnll_run=0_ranking
# python mi/fit_humans.py --paradigm decisionmaking --task-name binz2022  --exp-id 1 --num-iters 1 --model-name env=synthetic_dim4_model=transformer_num_episodes100000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8_noise0.0_shuffleTrue_pairedFalse_lossnll_run=0_unknown
# python mi/fit_humans.py --paradigm decisionmaking --task-name binz2022  --exp-id 1 --num-iters 1 --model-name env=claude_generated_functionlearningtasks_paramsNA_dim4_data20_tasks8770_run0_procid1_pversionranked_model=transformer_num_episodes100000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8_noise0.0_shuffleTrue_pairedFalse_lossnll_run=0
# python mi/fit_humans.py --paradigm decisionmaking --task-name binz2022  --exp-id 1 --num-iters 1 --model-name env=claude_generated_functionlearningtasks_paramsNA_dim4_data20_tasks7284_run0_procid1_pversionunknown_model=transformer_num_episodes100000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8_noise0.0_shuffleTrue_pairedFalse_lossnll_run=0

## experiment 2
## paired
# python mi/fit_humans.py --paradigm decisionmaking --task-name binz2022  --exp-id 2 --num-iters 1 --paired --model-name env=synthetic_dim4_model=transformer_num_episodes100000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8_noise0.0_shuffleTrue_pairedTrue_lossnll_run=0_direction
# python mi/fit_humans.py --paradigm decisionmaking --task-name binz2022  --exp-id 2 --num-iters 1 --paired --model-name env=synthetic_dim4_model=transformer_num_episodes100000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8_noise0.0_shuffleTrue_pairedTrue_lossnll_run=0_unknown
# python mi/fit_humans.py --paradigm decisionmaking --task-name binz2022  --exp-id 2 --num-iters 1 --paired --model-name env=claude_generated_functionlearningtasks_paramsNA_dim4_data20_tasks8220_run0_procid1_pversiondirection_model=transformer_num_episodes100000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8_noise0.0_shuffleTrue_pairedTrue_lossnll_run=0
# python mi/fit_humans.py --paradigm decisionmaking --task-name binz2022  --exp-id 2 --num-iters 1 --paired --model-name env=claude_generated_functionlearningtasks_paramsNA_dim4_data20_tasks7284_run0_procid1_pversionunknown_model=transformer_num_episodes100000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8_noise0.0_shuffleTrue_pairedTrue_lossnll_run=0
# python mi/fit_humans.py --paradigm decisionmaking --task-name binz2022  --exp-id 2 --num-iters 1 --paired --method bounded_soft_sigmoid --model-name env=synthetic_dim4_model=transformer_num_episodes100000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8_noise0.0_shuffleTrue_pairedTrue_lossnll_run=0_direction
# python mi/fit_humans.py --paradigm decisionmaking --task-name binz2022  --exp-id 2 --num-iters 1 --paired --method bounded_soft_sigmoid --model-name env=claude_generated_functionlearningtasks_paramsNA_dim4_data20_tasks8220_run0_procid1_pversiondirection_model=transformer_num_episodes100000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8_noise0.0_shuffleTrue_pairedTrue_lossnll_run=0
# python mi/fit_humans.py --paradigm decisionmaking --task-name binz2022  --exp-id 2 --num-iters 1 --paired --method bounded_soft_sigmoid --model-name env=synthetic_dim4_model=transformer_num_episodes100000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8_noise0.0_shuffleTrue_pairedTrue_lossnll_run=0_unknown
# python mi/fit_humans.py --paradigm decisionmaking --task-name binz2022  --exp-id 2 --num-iters 1 --paired --method bounded_soft_sigmoid --model-name env=claude_generated_functionlearningtasks_paramsNA_dim4_data20_tasks7284_run0_procid1_pversionunknown_model=transformer_num_episodes100000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8_noise0.0_shuffleTrue_pairedTrue_lossnll_run=0


## unpaired
# python mi/fit_humans.py --paradigm decisionmaking --task-name binz2022  --exp-id 2 --num-iters 1 --model-name env=synthetic_dim4_model=transformer_num_episodes100000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8_noise0.0_shuffleTrue_pairedFalse_lossnll_run=0_direction
# python mi/fit_humans.py --paradigm decisionmaking --task-name binz2022  --exp-id 2 --num-iters 1 --model-name env=synthetic_dim4_model=transformer_num_episodes100000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8_noise0.0_shuffleTrue_pairedFalse_lossnll_run=0_unknown
# python mi/fit_humans.py --paradigm decisionmaking --task-name binz2022  --exp-id 2 --num-iters 1 --model-name env=claude_generated_functionlearningtasks_paramsNA_dim4_data20_tasks8220_run0_procid1_pversiondirection_model=transformer_num_episodes100000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8_noise0.0_shuffleTrue_pairedFalse_lossnll_run=0
# python mi/fit_humans.py --paradigm decisionmaking --task-name binz2022  --exp-id 2 --num-iters 1 --model-name env=claude_generated_functionlearningtasks_paramsNA_dim4_data20_tasks7284_run0_procid1_pversionunknown_model=transformer_num_episodes100000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8_noise0.0_shuffleTrue_pairedFalse_lossnll_run=0


### badham 2017


## devraj 2022