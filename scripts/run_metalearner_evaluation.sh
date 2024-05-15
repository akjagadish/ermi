#!/bin/bash -l
#SBATCH -o ./logs/%A.out
#SBATCH -e ./logs/%A.err
#SBATCH --job-name=evaluate_MI
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

##  binz2022
python mi/evaluate_humanexperiments.py --paradigm decisionmaking --task-name binz2022 --method greedy --model-name env=claude_generated_functionlearningtasks_paramsNA_dim2_data20_tasks9254_run0_procid0_pversion2_unknown_model=transformer_num_episodes500000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8_noise0.0_shuffleTrue_pairedFalse_run=0
python mi/evaluate_humanexperiments.py --paradigm decisionmaking --task-name binz2022 --method greedy --model-name env=synthetic_model=transformer_num_episodes500000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8_noise0.0_shuffleTrue_run=0_synthetic
python mi/evaluate_humanexperiments.py --paradigm decisionmaking --task-name binz2022 --method greedy --paired --model-name env=claude_generated_functionlearningtasks_paramsNA_dim2_data20_tasks9254_run0_procid0_pversion2_unknown_model=transformer_num_episodes500000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8_noise0.0_shuffleTrue_pairedTrue_run=0
python mi/evaluate_humanexperiments.py --paradigm decisionmaking --task-name binz2022 --method greedy --paired --model-name env=synthetic_model=transformer_num_episodes100000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8_noise0.0_shuffleTrue_pairedTrue_run=0_synthetic


python mi/evaluate_humanexperiments.py --paradigm decisionmaking --task-name binz2022 --method binomial --model-name env=claude_generated_functionlearningtasks_paramsNA_dim2_data20_tasks9254_run0_procid0_pversion2_unknown_model=transformer_num_episodes500000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8_noise0.0_shuffleTrue_pairedFalse_run=0
python mi/evaluate_humanexperiments.py --paradigm decisionmaking --task-name binz2022 --method binomial --model-name env=synthetic_model=transformer_num_episodes500000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8_noise0.0_shuffleTrue_run=0_synthetic
python mi/evaluate_humanexperiments.py --paradigm decisionmaking --task-name binz2022 --method binomial --paired --model-name env=claude_generated_functionlearningtasks_paramsNA_dim2_data20_tasks9254_run0_procid0_pversion2_unknown_model=transformer_num_episodes500000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8_noise0.0_shuffleTrue_pairedTrue_run=0
python mi/evaluate_humanexperiments.py --paradigm decisionmaking --task-name binz2022 --method binomial --paired --model-name env=synthetic_model=transformer_num_episodes100000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8_noise0.0_shuffleTrue_pairedTrue_run=0_synthetic