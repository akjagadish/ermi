import numpy as np
import torch
import pandas as pd
import sys
SYS_PATH = '/u/ajagadish/vanilla-llama' #'/raven/u/ajagadish/vanilla-llama/'
sys.path.append(f'{SYS_PATH}/categorisation/')
sys.path.append(f'{SYS_PATH}/categorisation/data')
sys.path.append(f'{SYS_PATH}/categorisation/rl2')
from evaluate import evaluate_metalearner
import json

def simulate_shepard1961(models=None, tasks=np.arange(1,7), betas=None, num_runs=5, num_trials=96, num_blocks=1, batch_size=64):

    corrects = np.ones((len(models), len(tasks), batch_size, num_trials))
    assert len(models)==len(betas), "Number of models and betas should be the same"
    for m_idx, (model_name, beta) in enumerate(zip(models, betas)):
        if model_name != 'humans':
            for t_idx, task in enumerate(tasks):
                model_path = f"{SYS_PATH}/categorisation/trained_models/{model_name}.pt"
                corrects[m_idx, t_idx] = evaluate_metalearner(task, model_path, 'shepard_categorisation', beta=beta, shuffle_trials=True, num_trials=num_trials, num_runs=num_runs)
            
    # compuate error rates across trials using corrects
    errors = 1. - corrects.mean(2)

    # load json file containing the data
    with open(f'{SYS_PATH}/categorisation/data/human/nosofsky1994.json') as json_file:
        data = json.load(json_file)

    mse_distance = np.zeros((len(models),))
    mi_block_errors = np.zeros((len(models), len(tasks), num_blocks))

    for idx in np.arange(len(models)):
        for t_idx, rule in enumerate(data.keys()):
            block_errors = np.stack(np.split(errors[idx, t_idx], num_blocks)).mean(1)
            human_block_error = data[rule]['y'][:num_blocks]
            # compute mse between human and model error rates for a model summed across tasks
            mse_distance[idx] += np.mean((block_errors-human_block_error)**2)
            mi_block_errors[idx, t_idx] = block_errors
            
    return mse_distance, mi_block_errors

def compute_mse(models):
    num_blocks = 10 # 16
    num_trials_per_block = 16
    num_runs = 50
    min_mse_distance = np.inf
    mse_distances = []
    beta_range = np.arange(0., 1.0, 0.1)
    block_errors = []
    for beta in beta_range:
        betas = [None, beta]
        mse_distance_beta, mi_block_errors = simulate_shepard1961(models=models, betas=betas, num_runs=num_runs, num_blocks=num_blocks, num_trials=num_trials_per_block*num_blocks)
        if mse_distance_beta[1] <= min_mse_distance:
            min_mse_distance = mse_distance_beta[1]
            best_beta = beta
            best_block_errors = mi_block_errors
        mse_distances.append(mse_distance_beta[1])
        block_errors.append(mi_block_errors)

    print(f'best beta: {best_beta}, min_mse_distance: {min_mse_distance}')
    # save results
    model_name = 'ermi' if 'claude' in models[1] else 'rmc' if 'rmc' in models[1] else 'pfn' if 'syntheticnonlinear' in models[1] else 'mi'
    np.save(f'{SYS_PATH}/categorisation/data/fitted_simulation/shepard1961_{model_name}_num_runs={num_runs}_num_blocks={num_blocks}_num_trials_per_block={num_trials_per_block}.npy', [np.array(mse_distances), beta_range])
    np.save(f'{SYS_PATH}/categorisation/data/fitted_simulation/shepard1961_{model_name}_num_runs={num_runs}_num_blocks={num_blocks}_num_trials_per_block={num_trials_per_block}_block_errors.npy', np.stack(block_errors))

if __name__ == '__main__':

      # humans and MI
      models = ['humans',\
            'env=dim3synthetic_model=transformer_num_episodes500000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8_noise0.0_shuffleTrue_run=0_synthetic',\
            ]
      compute_mse(models)

      # humans and ERMI
      models = ['humans',\
            'env=claude_generated_tasks_paramsNA_dim3_data100_tasks11518_pversion4_model=transformer_num_episodes500000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8_noise0.0_shuffleTrue_run=0',
            ]
      compute_mse(models)

      # humans and RMC
      models = ['humans',\
            'env=rmc_tasks_dim3_data100_tasks11499_model=transformer_num_episodes500000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8_noise0.0_shuffleTrue_run=1_rmc',
            ]
      compute_mse(models)

      # humans and PFN
      models = ['humans',\
            'env=dim3synthetic_model=transformer_num_episodes500000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8_noise0.0_shuffleTrue_run=0_syntheticnonlinear',\
            ]
      compute_mse(models)