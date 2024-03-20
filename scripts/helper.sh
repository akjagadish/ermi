
#!/bin/bash -l
#SBATCH -o ./logs/%A.out
#SBATCH -e ./logs/%A.err
#SBATCH --job-name=llama
#SBATCH --time=00:15:00
#SBATCH --constraint="gpu"
#SBATCH --gres=gpu:a100:4
#SBATCH --mem=240G
#SBATCH --cpus-per-task=72
#SBATCH --mail-type=ALL
#SBATCH --mail-user=akshaykjagadish@gmail.com

# #export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK} # set it to 20 if you changed it multiples of 32

cd ~/ermi/categorisation/
module purge
module load anaconda/3/2021.11
module load gcc/11 impi/2021.6
module load cuda/11.6
module load pytorch_distributed/gpu-cuda-11.6/1.13.0
pip3 install --user accelerate openai gym ipdb transformers tensorboard anthropic openml
clear
jupyter-lab

cd ~/ermi/categorisation/
module purge
module load anaconda/3/2021.11
module load gcc/11 impi/2021.6
module load cuda/11.6
module load pytorch_distributed/gpu-cuda-11.6/1.13.0
pip3 install --user accelerate openai gym ipdb transformers tensorboard anthropic
clear
tensorboard --logdir=runs/trained_models/ --port=6006

python query.py --llama-path /ptmp/mbinz/new --model 7B
python query.py --llama-path /ptmp/mbinz/new --model 65B

python generate_tasks.py --llama-path /ptmp/mbinz/new --model 7B
python generate_tasks.py --llama-path /ptmp/mbinz/new --model 65B

#srun --time=00:10:00  --cpus-per-task=18 --gres=gpu:a100:1 --mem=50G --pty bash
#srun --time=01:00:00  --cpus-per-task=18 --gres=gpu:a100:1 --mem=80G --pty bash
#srun --time=00:10:00  --cpus-per-task=20 --gres=gpu:a100:4 --mem=240G --pty bash
#srun --time=01:00:00  --cpus-per-task=20 --gres=gpu:a100:4 --mem=240G --pty bash
#srun --time=24:00:00  --cpus-per-task=20 --gres=gpu:a100:4 --mem=240G --pty bash
#raw_response = llama.generate([text], temperature=1., max_length=1); raw_response[0][0][len(text):].replace(' ', '')