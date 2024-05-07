#!/bin/bash -l
#SBATCH -o ./logs/%A.out
#SBATCH -e ./logs/%A.err
#SBATCH --job-name=Plots
#SBATCH --mail-type=ALL
#SBATCH --mail-user=akshaykjagadish@gmail.com
#SBATCH --time=72:00:00
#SBATCH --cpus-per-task=18


# cd ~/ermi/categorisation/

# module purge
# module load anaconda/3/2023.03
# pip install groupBMC==1.0
# export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# python make_plots.py

# cd ~/ermi/functionlearning/

# module purge
# module load anaconda/3/2023.03
# pip install groupBMC==1.0
# pip3 install --user openai ipdb transformers tensorboard anthropic openml wordcloud mycolorpy Pillow
# export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# python make_plots.py


cd ~/ermi/decisionmaking/

module purge
module load anaconda/3/2023.03
pip install groupBMC==1.0
pip3 install --user openai ipdb transformers tensorboard anthropic openml wordcloud mycolorpy Pillow
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

python make_plots.py
