#!/bin/bash -l
#SBATCH -o ./logs/%A.out
#SBATCH -e ./logs/%A.err
#SBATCH --job-name=simulate_metalearner
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

# ERMI: 2 dimensions
# python mi/train_decisionmaking.py  --paired --num-episodes 100000 --max-steps 20 --num-dims 2 --first-run-id 0 --loss nll --env-name claude_generated_functionlearningtasks_paramsNA_dim2_data20_tasks9254_run0_procid0_pversion2_unknown --env-dir /u/ajagadish/ermi/decisionmaking/data --save-dir /u/ajagadish/ermi/decisionmaking/trained_models/ --save-every 100 --print-every 10  --noise 0.0 --model-name transformer --num_hidden 256 --num_layers 6 --d_model 64 --num_head 8 --batch_size 64 --shuffle 
# python mi/train_decisionmaking.py --num-episodes 100000 --max-steps 20 --num-dims 2  --first-run-id 0 --loss nll --env-name claude_generated_functionlearningtasks_paramsNA_dim2_data20_tasks9254_run0_procid0_pversion2_unknown --env-dir /u/ajagadish/ermi/decisionmaking/data --save-dir /u/ajagadish/ermi/decisionmaking/trained_models/ --save-every 100 --print-every 10  --noise 0.0 --model-name transformer --num_hidden 256 --num_layers 6 --d_model 64 --num_head 8 --batch_size 64 --shuffle 

# MI: 2 dimensions
# python mi/train_decisionmaking.py --synthetic  --env-name synthetic --num-episodes 100000 --max-steps 10 --num-dims 2 --first-run-id 0 --loss nll --save-dir /u/ajagadish/ermi/decisionmaking/trained_models/ --save-every 100 --print-every 10  --noise 0.0 --model-name transformer --num_hidden 256 --num_layers 6 --d_model 64 --num_head 8 --batch_size 64 --shuffle 
python mi/train_decisionmaking.py --synthetic --paired --env-name synthetic --num-episodes 100000 --max-steps 10 --num-dims 2 --first-run-id 0 --loss nll --save-dir /u/ajagadish/ermi/decisionmaking/trained_models/ --save-every 100 --print-every 10  --noise 0.0 --model-name transformer --num_hidden 256 --num_layers 6 --d_model 64 --num_head 8 --batch_size 64 --shuffle 

# ERMI: 4 dimensions
# python mi/train_decisionmaking.py  --paired --num-episodes 100000 --max-steps 20 --num-dims 4 --first-run-id 0 --loss nll --env-name claude_generated_functionlearningtasks_paramsNA_dim4_data20_tasks7284_run0_procid1_pversionunknown --env-dir /u/ajagadish/ermi/decisionmaking/data --save-dir /u/ajagadish/ermi/decisionmaking/trained_models/ --save-every 100 --print-every 10  --noise 0.0 --model-name transformer --num_hidden 256 --num_layers 6 --d_model 64 --num_head 8 --batch_size 64 --shuffle 
# python mi/train_decisionmaking.py  --paired --num-episodes 100000 --max-steps 20 --num-dims 4 --first-run-id 0 --loss nll --env-name claude_generated_functionlearningtasks_paramsNA_dim4_data20_tasks8220_run0_procid1_pversiondirection --env-dir /u/ajagadish/ermi/decisionmaking/data --save-dir /u/ajagadish/ermi/decisionmaking/trained_models/ --save-every 100 --print-every 10  --noise 0.0 --model-name transformer --num_hidden 256 --num_layers 6 --d_model 64 --num_head 8 --batch_size 64 --shuffle 
# python mi/train_decisionmaking.py  --paired --num-episodes 100000 --max-steps 20 --num-dims 4 --first-run-id 0 --loss nll --env-name claude_generated_functionlearningtasks_paramsNA_dim4_data20_tasks8770_run0_procid1_pversionranked --env-dir /u/ajagadish/ermi/decisionmaking/data --save-dir /u/ajagadish/ermi/decisionmaking/trained_models/ --save-every 100 --print-every 10  --noise 0.0 --model-name transformer --num_hidden 256 --num_layers 6 --d_model 64 --num_head 8 --batch_size 64 --shuffle 
# python mi/train_decisionmaking.py  --num-episodes 100000 --max-steps 20 --num-dims 4 --first-run-id 0 --loss nll --env-name claude_generated_functionlearningtasks_paramsNA_dim4_data20_tasks7284_run0_procid1_pversionunknown --env-dir /u/ajagadish/ermi/decisionmaking/data --save-dir /u/ajagadish/ermi/decisionmaking/trained_models/ --save-every 100 --print-every 10  --noise 0.0 --model-name transformer --num_hidden 256 --num_layers 6 --d_model 64 --num_head 8 --batch_size 64 --shuffle 
# python mi/train_decisionmaking.py  --num-episodes 100000 --max-steps 20 --num-dims 4 --first-run-id 0 --loss nll --env-name claude_generated_functionlearningtasks_paramsNA_dim4_data20_tasks8220_run0_procid1_pversiondirection --env-dir /u/ajagadish/ermi/decisionmaking/data --save-dir /u/ajagadish/ermi/decisionmaking/trained_models/ --save-every 100 --print-every 10  --noise 0.0 --model-name transformer --num_hidden 256 --num_layers 6 --d_model 64 --num_head 8 --batch_size 64 --shuffle 
# python mi/train_decisionmaking.py  --num-episodes 100000 --max-steps 20 --num-dims 4 --first-run-id 0 --loss nll --env-name claude_generated_functionlearningtasks_paramsNA_dim4_data20_tasks8770_run0_procid1_pversionranked --env-dir /u/ajagadish/ermi/decisionmaking/data --save-dir /u/ajagadish/ermi/decisionmaking/trained_models/ --save-every 100 --print-every 10  --noise 0.0 --model-name transformer --num_hidden 256 --num_layers 6 --d_model 64 --num_head 8 --batch_size 64 --shuffle 

# MI: 4 dimensions
# python mi/train_decisionmaking.py --synthetic --paired --env-name synthetic --num-episodes 100000 --max-steps 10 --num-dims 4 --first-run-id 0 --loss nll --save-dir /u/ajagadish/ermi/decisionmaking/trained_models/ --save-every 100 --print-every 10  --noise 0.0 --model-name transformer --num_hidden 256 --num_layers 6 --d_model 64 --num_head 8 --batch_size 64 --shuffle 
# python mi/train_decisionmaking.py --synthetic --paired --ranking --env-name synthetic --num-episodes 100000 --max-steps 10 --num-dims 4 --first-run-id 0 --loss nll --save-dir /u/ajagadish/ermi/decisionmaking/trained_models/ --save-every 100 --print-every 10  --noise 0.0 --model-name transformer --num_hidden 256 --num_layers 6 --d_model 64 --num_head 8 --batch_size 64 --shuffle 
# python mi/train_decisionmaking.py --synthetic --paired --direction --env-name synthetic --num-episodes 100000 --max-steps 10 --num-dims 4 --first-run-id 0 --loss nll --save-dir /u/ajagadish/ermi/decisionmaking/trained_models/ --save-every 100 --print-every 10  --noise 0.0 --model-name transformer --num_hidden 256 --num_layers 6 --d_model 64 --num_head 8 --batch_size 64 --shuffle 
# python mi/train_decisionmaking.py --synthetic  --env-name synthetic --num-episodes 100000 --max-steps 10 --num-dims 4 --first-run-id 0 --loss nll --save-dir /u/ajagadish/ermi/decisionmaking/trained_models/ --save-every 100 --print-every 10  --noise 0.0 --model-name transformer --num_hidden 256 --num_layers 6 --d_model 64 --num_head 8 --batch_size 64 --shuffle 
# python mi/train_decisionmaking.py --synthetic  --ranking --env-name synthetic --num-episodes 100000 --max-steps 10 --num-dims 4 --first-run-id 0 --loss nll --save-dir /u/ajagadish/ermi/decisionmaking/trained_models/ --save-every 100 --print-every 10  --noise 0.0 --model-name transformer --num_hidden 256 --num_layers 6 --d_model 64 --num_head 8 --batch_size 64 --shuffle 
# python mi/train_decisionmaking.py --synthetic   --direction --env-name synthetic --num-episodes 100000 --max-steps 10 --num-dims 4 --first-run-id 0 --loss nll --save-dir /u/ajagadish/ermi/decisionmaking/trained_models/ --save-every 100 --print-every 10  --noise 0.0 --model-name transformer --num_hidden 256 --num_layers 6 --d_model 64 --num_head 8 --batch_size 64 --shuffle 

# test parallel sampling
# python mi/train_decisionmaking.py --synthetic --paired --env-name synthetic --num-episodes 100000 --max-steps 10 --num-dims 4 --first-run-id 99 --loss nll --save-dir /u/ajagadish/ermi/decisionmaking/trained_models/ --save-every 100 --print-every 10  --noise 0.0 --model-name transformer --num_hidden 256 --num_layers 6 --d_model 64 --num_head 8 --batch_size 64 --shuffle 
python mi/train_decisionmaking.py --synthetic --paired --direction --env-name synthetic --num-episodes 100000 --max-steps 10 --num-dims 4 --first-run-id 99 --loss nll --save-dir /u/ajagadish/ermi/decisionmaking/trained_models/ --save-every 100 --print-every 10  --noise 0.0 --model-name transformer --num_hidden 256 --num_layers 6 --d_model 64 --num_head 8 --batch_size 64 --shuffle 
