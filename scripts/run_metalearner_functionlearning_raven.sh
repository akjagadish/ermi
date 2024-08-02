#!/bin/bash -l
#SBATCH -o ./logs/%A.out
#SBATCH -e ./logs/%A.err
#SBATCH --job-name=train_metalearner_functionlearning
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


python mi/train_functionlearning.py --num-episodes 100000 --max-steps 25 --num-dims 1 --sample-to-match-max-steps --loss nll --env-name claude_generated_functionlearningtasks_paramsNA_dim1_data20_tasks9991_run0_procid0_pversion2 --env-dir /u/ajagadish/ermi/functionlearning/data/generated_tasks --save-dir /u/ajagadish/ermi/functionlearning/trained_models/ --save-every 100 --print-every 10  --first-run-id 0 --noise 0.0 --model-name transformer --num_hidden 256 --num_layers 6 --d_model 64 --num_head 8 --batch_size 64 --shuffle 
# python mi/train_functionlearning.py --num-episodes 100000 --max-steps 20 --num-dims 1 --sample-to-match-max-steps --loss nll --env-name claude_generated_functionlearningtasks_paramsNA_dim1_data20_tasks9991_run0_procid0_pversion2 --env-dir /u/ajagadish/ermi/functionlearning/data/generated_tasks --save-dir /u/ajagadish/ermi/functionlearning/trained_models/ --save-every 100 --print-every 10  --first-run-id 0 --noise 0.0 --model-name transformer --num_hidden 256 --num_layers 6 --d_model 64 --num_head 8 --batch_size 64 --shuffle