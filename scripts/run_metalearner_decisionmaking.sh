#!/bin/bash -l
#SBATCH -o ./logs/%A.out
#SBATCH -e ./logs/%A.err
#SBATCH --job-name=LLM_Priors
#SBATCH --mail-type=ALL
#SBATCH --mail-user=akshaykjagadish@gmail.com
#SBATCH --time=100:00:00
#SBATCH --constraint="gpu"
#SBATCH --cpus-per-task=18
#SBATCH --gres=gpu:1

cd ~/ermi/
module purge 
module load cuda/11.6
module load anaconda/3/2021.11 # need to use old anaconda version so it uses torch that is compatible with cuda 11.6 and allows using GPU
pip3 install --user ipdb transformers tensorboard ipdb tqdm 

# per feature minmax normalization
# python mi/train_decisionmaking.py --num-episodes 100000 --max-steps 20 --num-dims 2  --first-run-id 0 --loss bce --env-name claude_generated_functionlearningtasks_paramsNA_dim2_data20_tasks9254_run0_procid0_pversion2 --env-dir /u/ajagadish/ermi/decisionmaking/data --save-dir /u/ajagadish/ermi/decisionmaking/trained_models/ --save-every 100 --print-every 10  --noise 0.0 --model-name transformer --num_hidden 256 --num_layers 6 --d_model 64 --num_head 8 --batch_size 64 --shuffle 
# python mi/train_decisionmaking.py --num-episodes 500000 --max-steps 20 --num-dims 2  --first-run-id 0 --loss bce --env-name claude_generated_functionlearningtasks_paramsNA_dim2_data20_tasks9254_run0_procid0_pversion2 --env-dir /u/ajagadish/ermi/decisionmaking/data --save-dir /u/ajagadish/ermi/decisionmaking/trained_models/ --save-every 100 --print-every 10  --noise 0.0 --model-name transformer --num_hidden 256 --num_layers 6 --d_model 64 --num_head 8 --batch_size 64 --shuffle 
# python mi/train_decisionmaking.py --num-episodes 500000 --max-steps 20 --num-dims 2  --first-run-id 0 --loss bce --env-name claude_generated_functionlearningtasks_paramsNA_dim2_data20_tasks9254_run0_procid0_pversion2 --env-dir /u/ajagadish/ermi/decisionmaking/data --save-dir /u/ajagadish/ermi/decisionmaking/trained_models/ --save-every 100 --print-every 10  --noise 0.0 --model-name transformer --num_hidden 256 --num_layers ${SLURM_ARRAY_TASK_ID} --d_model 64 --num_head 8 --batch_size 64 --shuffle 

# max steps is set to 10
#python mi/train_decisionmaking.py --synthetic --env-name synthetic --num-episodes 100000 --max-steps 10 --num-dims 2  --first-run-id 0 --loss bce --save-dir /u/ajagadish/ermi/decisionmaking/trained_models/ --save-every 100 --print-every 10  --noise 0.0 --model-name transformer --num_hidden 256 --num_layers 6 --d_model 64 --num_head 8 --batch_size 64 --shuffle 
#python mi/train_decisionmaking.py --synthetic --env-name synthetic --num-episodes 500000 --max-steps 10 --num-dims 2  --first-run-id 0 --loss bce --save-dir /u/ajagadish/ermi/decisionmaking/trained_models/ --save-every 100 --print-every 10  --noise 0.0 --model-name transformer --num_hidden 256 --num_layers 6 --d_model 64 --num_head 8 --batch_size 64 --shuffle 
python mi/train_decisionmaking.py --synthetic --env-name synthetic --num-episodes 500000 --max-steps 10 --num-dims 2  --first-run-id 0 --loss bce --save-dir /u/ajagadish/ermi/decisionmaking/trained_models/ --save-every 100 --print-every 10  --noise 0.0 --model-name transformer --num_hidden 256 --num_layers ${SLURM_ARRAY_TASK_ID} --d_model 64 --num_head 8 --batch_size 64 --shuffle 