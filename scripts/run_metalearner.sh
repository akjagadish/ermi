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

cd ~/ermi/categorisation/
module purge 
module load cuda/11.6
module load anaconda/3/2021.11 # need to use old anaconda version so it uses torch that is compatible with cuda 11.6 and allows using GPU
pip3 install --user accelerate openai gym ipdb transformers tensorboard
pip3 install --user fire sentencepiece ipdb accelerate tqdm 


## llm generated data: prompt version 4 for dim 3 data
# python mi/train_transformer.py --num-episodes 500000 --save-every 100 --print-every 100 --max-steps 250 --env-name claude_generated_tasks_paramsNA_dim3_data100_tasks11518_pversion4  --first-run-id 0 --noise 0.0 --model-name transformer --num_hidden 256 --num_layers 6 --d_model 64 --num_head 8 --batch_size 64 --shuffle --env-dir u/ajagadish/vanilla-llama/categorisation/data --first-run-id 2
# shuffled trials: python mi/train_transformer.py --num-episodes 500000 --save-every 100 --print-every 100 --max-steps 250 --env-name claude_generated_tasks_paramsNA_dim3_data100_tasks11518_pversion4  --first-run-id 0 --noise 0.0 --model-name transformer --num_hidden 256 --num_layers 6 --d_model 64 --num_head 8 --batch_size 64 --env-dir u/ajagadish/vanilla-llama/categorisation/data --first-run-id 1
# shuffled features: python mi/train_transformer.py --num-episodes 500000 --save-every 100 --print-every 100 --max-steps 250 --env-name claude_generated_tasks_paramsNA_dim3_data100_tasks11518_pversion4 --noise 0.0 --model-name transformer --num_hidden 256 --num_layers 6 --d_model 64 --num_head 8 --batch_size 64 --shuffle --env-dir u/ajagadish/vanilla-llama/categorisation/data --shuffle-features --first-run-id 2
# shuffled features and with more trials: python mi/train_transformer.py --sample-to-match-max-steps --num-episodes 500000 --save-every 100 --print-every 100 --max-steps 300 --env-name claude_generated_tasks_paramsNA_dim3_data100_tasks11518_pversion4 --noise 0.0 --model-name transformer --num_hidden 256 --num_layers 6 --d_model 64 --num_head 8 --batch_size 64 --shuffle --env-dir u/ajagadish/vanilla-llama/categorisation/data --shuffle-features --first-run-id 3

## llm generated data: prompt version 5 for dim 6 data
# (done) python mi/train_transformer.py --num-episodes 500000 --save-every 100 --print-every 100 --max-steps 300 --num-dims 6 --env-name claude_generated_tasks_paramsNA_dim6_data500_tasks12911_pversion5_stage1  --first-run-id 0 --noise 0.0 --model-name transformer --num_hidden 256 --num_layers 6 --d_model 64 --num_head 8 --batch_size 64 --shuffle --env-dir u/ajagadish/vanilla-llama/categorisation/data --first-run-id 1
# (done) python mi/train_transformer.py --num-episodes 500000 --save-every 100 --print-every 100 --max-steps 650 --num-dims 6 --env-name claude_generated_tasks_paramsNA_dim6_data500_tasks12910_pversion5_stage2  --first-run-id 0 --noise 0.0 --model-name transformer --num_hidden 256 --num_layers 6 --d_model 64 --num_head 8 --batch_size 64 --shuffle --env-dir u/ajagadish/vanilla-llama/categorisation/data --first-run-id 0
# (done) python mi/train_transformer.py --sample-to-match-max-steps --num-episodes 500000 --save-every 100 --print-every 100 --max-steps 650 --num-dims 6 --env-name claude_generated_tasks_paramsNA_dim6_data500_tasks12910_pversion5_stage2  --first-run-id 0 --noise 0.0 --model-name transformer --num_hidden 256 --num_layers 6 --d_model 64 --num_head 8 --batch_size 32 --shuffle --env-dir u/ajagadish/vanilla-llama/categorisation/data --first-run-id 1
# (done) python mi/train_transformer.py --sample-to-match-max-steps --num-episodes 500000 --save-every 100 --print-every 100 --max-steps 650 --num-dims 6 --env-name claude_generated_tasks_paramsNA_dim6_data500_tasks12910_pversion5_stage2 --noise 0.0 --model-name transformer --num_hidden 256 --num_layers 6 --d_model 64 --num_head 8 --batch_size 32 --shuffle --env-dir u/ajagadish/vanilla-llama/categorisation/data --shuffle-features --first-run-id 2

## llm generated data: prompt version 5 for dim 4 data
# (done) python mi/train_transformer.py --first-run-id 0 --sample-to-match-max-steps --num-episodes 500000 --max-steps 300 --num-dims 4 --env-name  claude_generated_tasks_paramsNA_dim4_data650_tasks8950_pversion5_stage1 --save-every 100 --print-every 100  --noise 0.0 --model-name transformer --num_hidden 256 --num_layers 6 --d_model 64 --num_head 8 --batch_size 32 --shuffle --env-dir u/ajagadish/vanilla-llama/categorisation/data 
# (shuffled features; done) python mi/train_transformer.py --first-run-id 1 --sample-to-match-max-steps --num-episodes 500000 --max-steps 300 --num-dims 4 --env-name  claude_generated_tasks_paramsNA_dim4_data650_tasks8950_pversion5_stage1 --save-every 100 --print-every 100  --noise 0.0 --model-name transformer --num_hidden 256 --num_layers 6 --d_model 64 --num_head 8 --batch_size 32 --shuffle --env-dir u/ajagadish/vanilla-llama/categorisation/data --shuffle-features

## rmc generated data for dim 3
# python mi/train_transformer.py  --rmc --num-episodes 500000 --save-every 100 --print-every 10 --max-steps 400 --num-dims 3  --env-name dim3RMC --first-run-id 0 --noise 0.0 --model-name transformer --num_hidden 256 --num_layers 6 --d_model 64 --num_head 8 --batch_size 64 --shuffle 
# (done) python mi/train_transformer.py  --rmc --env-name rmc_tasks_dim3_data100_tasks11499  --num-episodes 500000 --save-every 100 --print-every 10 --max-steps 400 --num-dims 3 --first-run-id 1 --noise 0.0 --model-name transformer --num_hidden 256 --num_layers 6 --d_model 64 --num_head 8 --batch_size 64 --shuffle --env-dir u/ajagadish/vanilla-llama/categorisation/data 
# (done) python mi/train_transformer.py  --rmc --env-name rmc_tasks_dim6_data600_tasks12911  --num-episodes 500000 --save-every 100 --print-every 10 --max-steps 600 --num-dims 6 --first-run-id 0 --noise 0.0 --model-name transformer --num_hidden 256 --num_layers 6 --d_model 64 --num_head 8 --batch_size 32 --shuffle --env-dir u/ajagadish/vanilla-llama/categorisation/data 
# (done) python mi/train_transformer.py  --rmc --env-name rmc_tasks_dim6_data600_tasks12911 --sample-to-match-max-steps --num-episodes 10 --save-every 100 --print-every 10 --max-steps 650 --num-dims 6 --first-run-id 0 --noise 0.0 --model-name transformer --num_hidden 256 --num_layers 6 --d_model 64 --num_head 8 --batch_size 32 --shuffle --env-dir u/ajagadish/vanilla-llama/categorisation/data 
# (done) python mi/train_transformer.py  --rmc --env-name rmc_tasks_dim4_data300_tasks8950  --num-episodes 500000 --save-every 100 --print-every 10 --max-steps 600 --num-dims 4 --first-run-id 0 --noise 0.0 --model-name transformer --num_hidden 256 --num_layers 6 --d_model 64 --num_head 8 --batch_size 64 --shuffle --env-dir u/ajagadish/vanilla-llama/categorisation/data 

# dim 3 synthetic data
# (done) python mi/train_transformer.py  --synthetic --num-episodes 500000 --save-every 100 --print-every 10 --max-steps 400 --num-dims 3  --env-name dim3synthetic --first-run-id 0 --noise 0.0 --model-name transformer --num_hidden 256 --num_layers 6 --d_model 64 --num_head 8 --batch_size 64 --shuffle 
# (done) python mi/train_transformer.py  --synthetic --nonlinear --num-episodes 500000 --save-every 100 --print-every 10 --max-steps 400 --num-dims 3  --env-name dim3synthetic --first-run-id 0 --noise 0.0 --model-name transformer --num_hidden 256 --num_layers 6 --d_model 64 --num_head 8 --batch_size 64 --shuffle 

# dim 6 synthetic data
# (done) python mi/train_transformer.py  --synthetic --nonlinear --num-episodes 500000 --save-every 100 --print-every 100 --max-steps 650 --num-dims 6 --env-name dim6synthetic --first-run-id 2 --noise 0.0 --model-name transformer --num_hidden 256 --num_layers 6 --d_model 64 --num_head 8 --batch_size 32 --shuffle
# (done) python mi/train_transformer.py  --synthetic --num-episodes 500000 --save-every 100 --print-every 100 --max-steps 650 --num-dims 6 --env-name dim6synthetic --first-run-id 2 --noise 0.0 --model-name transformer --num_hidden 256 --num_layers 6 --d_model 64 --num_head 8 --batch_size 32 --shuffle

# dim 4 synthetic data
# (done) python mi/train_transformer.py  --synthetic --nonlinear --num-episodes 500000 --max-steps 300 --num-dims 6 --env-name dim4synthetic --first-run-id 0 --save-every 100 --print-every 100  --noise 0.0 --model-name transformer --num_hidden 256 --num_layers 6 --d_model 64 --num_head 8 --batch_size 64 --shuffle
# (done) python mi/train_transformer.py  --synthetic --num-episodes 500000 --max-steps 300 --num-dims 6 --env-name dim4synthetic --first-run-id 0 --save-every 100 --print-every 100  --noise 0.0 --model-name transformer --num_hidden 256 --num_layers 6 --d_model 64 --num_head 8 --batch_size 64 --shuffle
# (done) python mi/train_transformer.py  --synthetic --nonlinear --num-episodes 500000 --max-steps 300 --num-dims 4 --env-name dim4synthetic --first-run-id 1 --save-every 100 --print-every 100  --noise 0.0 --model-name transformer --num_hidden 256 --num_layers 6 --d_model 64 --num_head 8 --batch_size 64 --shuffle
# (done) python mi/train_transformer.py  --synthetic --num-episodes 500000 --max-steps 300 --num-dims 4 --env-name dim4synthetic --first-run-id 1 --save-every 100 --print-every 100  --noise 0.0 --model-name transformer --num_hidden 256 --num_layers 6 --d_model 64 --num_head 8 --batch_size 64 --shuffle