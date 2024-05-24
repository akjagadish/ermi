#!/bin/bash -l
#SBATCH -o ./logs/%A.out
#SBATCH -e ./logs/%A.err
#SBATCH --job-name=FittedSimulations
#SBATCH --mail-type=ALL
#SBATCH --mail-user=akshaykjagadish@gmail.com
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=18


cd ~/ermi/categorisation/
module purge
module load cuda/11.6
module load anaconda/3/2023.03
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

## badham2017
# python mi/fitted_simulations.py --model-name env=claude_generated_tasks_paramsNA_dim3_data100_tasks11518_pversion4_model=transformer_num_episodes500000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8_noise0.0_shuffleTrue_run=0 --task-name badham2017 --optimizer differential_evolution
# python mi/fitted_simulations.py --model-name env=dim3synthetic_model=transformer_num_episodes500000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8_noise0.0_shuffleTrue_run=0_synthetic --task-name badham2017 --optimizer differential_evolution
# python mi/fitted_simulations.py --model-name env=dim3synthetic_model=transformer_num_episodes500000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8_noise0.0_shuffleTrue_run=0_syntheticnonlinear --task-name badham2017 --optimizer differential_evolution

## devraj2022
# python mi/fitted_simulations.py --model-name env=claude_generated_tasks_paramsNA_dim6_data500_tasks12910_pversion5_stage2_model=transformer_num_episodes500000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8_noise0.0_shuffleTrue_run=1 --task-name devraj2022 --optimizer differential_evolution
# python mi/fitted_simulations.py --model-name env=dim6synthetic_model=transformer_num_episodes500000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8_noise0.0_shuffleTrue_run=2_synthetic --task-name devraj2022 --optimizer differential_evolution
# python mi/fitted_simulations.py --model-name env=dim6synthetic_model=transformer_num_episodes500000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8_noise0.0_shuffleTrue_run=2_syntheticnonlinear  --task-name devraj2022 --optimizer differential_evolution


## shepard1961
python mi/simulate_shepard1961.py