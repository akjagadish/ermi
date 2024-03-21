import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import sys
SYS_PATH = '/u/ajagadish/ermi'
sys.path.append(f"{SYS_PATH}/categorisation/")
sys.path.append(f"{SYS_PATH}/categorisation/rl2")
sys.path.append(f"{SYS_PATH}/categorisation/data")
# from evaluate import evaluate_metalearner
import json
from groupBMC.groupBMC import GroupBMC
FONTSIZE=20


def posterior_model_frequency(bics, models, horizontal=False, FIGSIZE=(5,5), task_name=None):
    result = {}
    LogEvidence = np.stack(-bics/2)
    result = GroupBMC(LogEvidence).get_result()

    # rename models for plot
    
    if task_name == 'Badham et al. (2017)':
        colors = ['#173b4f', '#4d6a75','#5d7684', '#748995','#4d6a75', '#0d2c3d', '#a2c0a9', '#2f4a5a', '#8b9da7']
    elif task_name == 'Devraj et al. (2022)':
        colors = ['#173b4f', '#8b9da7', '#5d7684', '#2f4a5a', '#0d2c3d', '#4d6a75', '#748995', '#a2c0a9']
    # sort result in descending order
    sort_order = np.argsort(result.frequency_mean)[::-1]
    result.frequency_mean = result.frequency_mean[sort_order]
    result.frequency_var = result.frequency_var[sort_order]
    models = np.array(models)[sort_order]
    colors = np.array(colors)[sort_order]

    f, ax = plt.subplots(1, 1, figsize=FIGSIZE)
    
    if horizontal:
        # composed
        ax.barh(np.arange(len(models)), result.frequency_mean, xerr=np.sqrt(result.frequency_var), align='center', color=colors, height=0.6)#, edgecolor='k')#, hatch='//', label='Compostional Subtask')
        # plt.legend(fontsize=FONTSIZE-4, frameon=False)
        ax.set_ylabel('Models', fontsize=FONTSIZE)
        # ax.set_xlim(0, 0.7)
        ax.set_xlabel('Posterior model frequency', fontsize=FONTSIZE) 
        plt.yticks(ticks=np.arange(len(models)), labels=models, fontsize=FONTSIZE-2)
        ax.set_xticks(np.arange(0, result.frequency_mean.max(), 0.1))
        plt.xticks(fontsize=FONTSIZE-2)
    else:
        bar_positions = np.arange(len(result.frequency_mean))*0.5
        ax.bar(bar_positions, result.frequency_mean, color=colors, width=0.4)
        ax.errorbar(bar_positions, result.frequency_mean, yerr= np.sqrt(result.frequency_var), c='k', lw=3, fmt="o")
        ax.set_xlabel('Models', fontsize=FONTSIZE)
        ax.set_ylabel('Model frequency', fontsize=FONTSIZE)
        ax.set_xticks(bar_positions)  # Set x-tick positions to bar_positions
        ax.set_xticklabels(models, fontsize=FONTSIZE-2)  # Assign category names to x-tick labels
        plt.yticks(fontsize=FONTSIZE-2)
        # start bar plot from 0
        ax.set_ylim([-0.01, .55]) if task_name == 'Badham et al. (2017)' else ax.set_ylim([-0.01, .40])
        # y ticks at 0.1 interval
        ax.set_yticks(np.arange(0.0, .65, 0.1)) if task_name == 'Badham et al. (2017)' else ax.set_yticks(np.arange(0.0, .50, 0.1))

    ax.set_title(f'Model Comparison', fontsize=FONTSIZE)
    # print model names, mean frequencies and std error of mean frequencies
    for i, model in enumerate(models):
        print(f'{model}: {result.frequency_mean[i]} +- {np.sqrt(result.frequency_var[i])}')

    sns.despine()
    f.tight_layout()
    f.savefig(f'{SYS_PATH}/figures/posterior_model_frequency_{task_name}.svg', bbox_inches='tight', dpi=300)
    plt.show()

def exceedance_probability(bics, models, horizontal=False, FIGSIZE=(5,5), task_name=None):
    result = {}
    LogEvidence = np.stack(-bics/2)
    result = GroupBMC(LogEvidence).get_result()

    # rename models for plot
    colors = ['#173b4f', '#8b9da7', '#5d7684', '#2f4a5a', '#0d2c3d', '#4d6a75', '#748995', '#a2c0a9', '#c4d9c2']
    # sort result in descending order
    sort_order = np.argsort(result.exceedance_probability)[::-1]
    result.exceedance_probability = result.exceedance_probability[sort_order]
    models = np.array(models)[sort_order]
    colors = np.array(colors)[sort_order]

    f, ax = plt.subplots(1, 1, figsize=FIGSIZE)
    if horizontal:
        # composed
        ax.barh(np.arange(len(models)), result.exceedance_probability, align='center', color=colors, height=0.6)#, hatch='//', label='Compostional Subtask')
        # plt.legend(fontsize=FONTSIZE-4, frameon=False)
        ax.set_ylabel('Models', fontsize=FONTSIZE)
        # ax.set_xlim(0, 0.7)
        ax.set_xlabel('Exceedance probability', fontsize=FONTSIZE) 
        plt.yticks(ticks=np.arange(len(models)), labels=models, fontsize=FONTSIZE-3.)
        # ax.set_xticks(np.arange(0, result.exceedance_probability.max(), 0.1))
        plt.xticks(fontsize=FONTSIZE-4)
    else:
        # composed
        bar_positions = np.arange(len(result.exceedance_probability))*0.5
        ax.bar(bar_positions, result.exceedance_probability, color=colors, width=0.4)
        # plt.legend(fontsize=FONTSIZE, frameon=False)
        ax.set_xlabel('Models', fontsize=FONTSIZE)
        # ax.set_ylim(0, 0.7)
        ax.set_ylabel('Exceedance probability', fontsize=FONTSIZE) 
        ax.set_xticks(bar_positions)  # Set x-tick positions to bar_positions
        ax.set_xticklabels(models, fontsize=FONTSIZE-2)  # Assign category names to x-tick labels
        plt.yticks(fontsize=FONTSIZE-2)
    
    ax.set_title(f'Model Comparison', fontsize=FONTSIZE)
    sns.despine()
    f.tight_layout()
    f.savefig(f'{SYS_PATH}/figures/exceedance_probability_{task_name}.svg', bbox_inches='tight', dpi=300)
    plt.show()
    
def model_comparison_badham2017(FIGSIZE=(6,5)):
    models = [
              'badham2017_env=claude_generated_tasks_paramsNA_dim3_data100_tasks11518_pversion4_model=transformer_num_episodes500000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8_noise0.0_shuffleTrue_run=0_soft_sigmoid_differential_evolution',\
              'badham2017_env=rmc_tasks_dim3_data100_tasks11499_model=transformer_num_episodes500000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8_noise0.0_shuffleTrue_run=1_rmc_soft_sigmoid_differential_evolution',
              'badham2017_llm_runs=1_iters=1_blocks=1_loss=nll',\
              'badham2017_env=dim3synthetic_model=transformer_num_episodes500000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8_noise0.0_shuffleTrue_run=0_syntheticnonlinear_soft_sigmoid_differential_evolution',\
              'badham2017_env=dim3synthetic_model=transformer_num_episodes500000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8_noise0.0_shuffleTrue_run=0_synthetic_soft_sigmoid_differential_evolution',\
              'badham2017_gcm_runs=1_iters=1_blocks=1_loss=nll',\
              'badham2017_rulex_runs=1_iters=1_blocks=1_loss=nll_exception=True',\
              'badham2017_rulex_runs=1_iters=1_blocks=1_loss=nll_exception=False',\
              'badham2017_pm_runs=1_iters=1_blocks=1_loss=nll',\
                ]
    nlls,fitted_betas, r2s = [], [], []
    model_accs = []
    bics = []
    NUM_TASKS = 4
    NUM_TRIALs = 96
    num_trials = NUM_TRIALs*NUM_TASKS
    # FONTSIZE = 16
    # MODELS = ['ERMI', 'RMC-MI', 'L-MI', 'PFN-MI', 'GCM', 'Rulex', 'Rule',  'PM']
    MODELS = ['ERMI', 'RMC',  'LLM', 'MI', 'PFN', 'GCM', 'Rulex', 'Rule',  'PM']


    for model_name in models:
        fits =  np.load(f'{SYS_PATH}/categorisation/data/model_comparison/{model_name}.npz')
        if 'model=transformer' in model_name:
            betas, pnlls, pr2s = fits['betas'], fits['nlls'], fits['pr2s']
            model_accs.append(fits['accs'].max(0).mean())
            nlls_min_nlls = np.array(pnlls).squeeze()
            pr2s_min_nll = np.array(pr2s).squeeze()
            fitted_betas.append(betas)
            num_parameters = 1 
            bic = np.array(nlls_min_nlls)*2 + np.log(num_trials)*num_parameters
        elif ('gcm' in model_name) or ('pm' in model_name):
            betas, pnlls, pr2s = fits['params'], fits['lls'], fits['r2s']
            # summing the fits for the four conditions separately; hence the total number of parameters is model_parameters*NUM_TASKS
            nlls_min_nlls = np.array(pnlls).squeeze().sum(1)
            pr2s_min_nll = np.array(pr2s).squeeze().mean(1)
            num_parameters = 5*NUM_TASKS if ('gcm' in model_name) else 11*NUM_TASKS
            fitted_betas.append(betas.squeeze()[..., 1].mean(1))
        elif 'llm' in model_name:
            betas, pnlls, pr2s = fits['params'], fits['lls'], fits['r2s']
            # summing the fits for the four conditions separately; hence the total number of parameters is model_parameters*NUM_TASKS
            nlls_min_nlls = np.array(pnlls).squeeze().sum(1)
            pr2s_min_nll = np.array(pr2s).squeeze().mean(1)
            num_parameters = 1*NUM_TASKS 
            fitted_betas.append(betas.squeeze().mean(1))
        elif ('rulex' in model_name):
            betas, pnlls, pr2s = fits['params'], fits['lls'], fits['r2s']
            nlls_min_nlls = np.array(pnlls).squeeze().sum(1)
            pr2s_min_nll = np.array(pr2s).squeeze().mean(1)
            num_parameters = 2*NUM_TASKS
            
        bic = np.array(nlls_min_nlls)*2 + np.log(num_trials)*num_parameters
        nlls.append(nlls_min_nlls)
        r2s.append(pr2s_min_nll)
        bics.append(bic)

    # keep models and choose colors
    num_participants = len(nlls[0])
    MODELS = MODELS[:len(nlls)]
    # set colors depending on number of models in MODELS
    colors = ['#173b4f', '#8b9da7', '#5d7684', '#2f4a5a', '#0d2c3d', '#4d6a75', '#748995', '#a2c0a9', '#c4d9c2'][:len(nlls)]


    # compare mean nlls across models in a bar plot
    f, ax = plt.subplots(1, 1, figsize=FIGSIZE)
    bar_positions = np.arange(len(nlls))*0.5
    ax.bar(bar_positions, np.array(nlls).mean(1), color=colors, width=0.4)
    ax.errorbar(bar_positions, np.array(nlls).mean(1), yerr=np.array(nlls).std(1)/np.sqrt(num_participants-1), c='k', lw=3, fmt="o")
    # add chance level line for 616 trials with binary choices
    ax.axhline(y=-np.log(0.5)*num_trials, color='k', linestyle='--', lw=3)
    ax.set_xlabel('Models', fontsize=FONTSIZE)
    ax.set_ylabel('NLL', fontsize=FONTSIZE)
    ax.set_xticks(bar_positions)  # Set x-tick positions to bar_positions
    ax.set_xticklabels(MODELS, fontsize=FONTSIZE-5)  # Assign category names to x-tick labels
    # ax.set_title(f'Model comparison for Badham et al. (2017)', fontsize=FONTSIZE)
    # plt.xticks(fontsize=FONTSIZE-2)
    plt.yticks(fontsize=FONTSIZE-2)
    sns.despine()
    f.tight_layout()
    plt.show()

    # compare mean BICS across models in a bar plot
    f, ax = plt.subplots(1, 1, figsize=FIGSIZE)
    bar_positions = np.arange(len(bics))*1.5
    ax.bar(bar_positions, np.array(bics).mean(1), color=colors, width=1.)
    ax.errorbar(bar_positions, np.array(bics).mean(1), yerr=np.array(bics).std(1)/np.sqrt(num_participants-1), c='k', lw=3, fmt="o")
    # add chance level line for 616 trials with binary choices
    # ax.axhline(y=-np.log(0.5)*num_trials*2, color='k', linestyle='--', lw=3)
    ax.set_xlabel('Models', fontsize=FONTSIZE)
    ax.set_ylabel('BIC', fontsize=FONTSIZE)
    ax.set_xticks(bar_positions)  # Set x-tick positions to bar_positions
    ax.set_xticklabels(MODELS, fontsize=FONTSIZE-6)  # Assign category names to x-tick labels
    # ax.set_title(f'Model comparison for Badham et al. (2017)', fontsize=FONTSIZE)
    # plt.xticks(fontsize=FONTSIZE-2)
    plt.yticks(fontsize=FONTSIZE-2)
    sns.despine()
    f.tight_layout()
    plt.show()
    f.savefig(f'{SYS_PATH}/figures/bic_badham2017.svg', bbox_inches='tight', dpi=300)

    # compare mean r2s across models in a bar plot
    f, ax = plt.subplots(1, 1, figsize=FIGSIZE)
    bar_positions = np.arange(len(r2s))*0.5
    ax.bar(bar_positions, np.array(r2s).mean(1), color=colors, width=0.4)
    ax.errorbar(bar_positions, np.array(r2s).mean(1), yerr=np.array(r2s).std(1)/np.sqrt(num_participants-1), c='k', lw=3, fmt="o")
    ax.set_xlabel('Models', fontsize=FONTSIZE)
    ax.set_ylabel('R2', fontsize=FONTSIZE)
    ax.set_xticks(bar_positions)  # Set x-tick positions to bar_positions
    ax.set_xticklabels(MODELS, fontsize=FONTSIZE-5)  # Assign category names to x-tick labels

    # ax.set_title(f'Model comparison for  Badham et al. (2017)', fontsize=FONTSIZE)
    plt.yticks(fontsize=FONTSIZE-2)
    sns.despine()
    f.tight_layout()
    plt.show()   

    task_name = 'Badham et al. (2017)'
    posterior_model_frequency(np.array(bics), MODELS, task_name=task_name, FIGSIZE=(7.5,5))
    exceedance_probability(np.array(bics), MODELS, task_name=task_name, FIGSIZE=(7.5,5))

def model_comparison_devraj2022(FIGSIZE=(6,5)):
    models = ['devraj2022_env=claude_generated_tasks_paramsNA_dim6_data500_tasks12910_pversion5_stage2_model=transformer_num_episodes500000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8_noise0.0_shuffleTrue_run=1_soft_sigmoid_differential_evolution', \
              'devraj2022_env=dim6synthetic_model=transformer_num_episodes500000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8_noise0.0_shuffleTrue_run=2_synthetic_soft_sigmoid_differential_evolution', \
              'devraj2022_env=dim6synthetic_model=transformer_num_episodes500000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8_noise0.0_shuffleTrue_run=2_syntheticnonlinear_soft_sigmoid_differential_evolution',\
              'devraj2022_gcm_runs=1_iters=1_blocks=1_loss=nll',\
              'devraj2022_pm_runs=1_iters=1_blocks=1_loss=nll',\
              'devraj2022_rulex_runs=1_iters=1_blocks=1_loss=nll_exception=True', \
              'devraj2022_rulex_runs=1_iters=1_blocks=1_loss=nll_exception=False', 
         ]
    nlls,fitted_betas, r2s = [], [], []
    model_accs = []
    bics = []
    NUM_TASKS = 1
    NUM_TRIALs = 616
    num_trials = NUM_TRIALs*NUM_TASKS
    # FONTSIZE = 16
    MODELS = ['ERMI', 'MI', 'PFN', 'GCM', 'PM', 'Rulex', 'Rule']

    for model_name in models:
        fits =  np.load(f'{SYS_PATH}/categorisation/data/model_comparison/{model_name}.npz')
        if 'model=transformer' in model_name:
            betas, pnlls, pr2s = fits['betas'], fits['nlls'], fits['pr2s']
            model_accs.append(fits['accs'].max(0).mean())
            nlls_min_nlls = np.array(pnlls).squeeze()
            pr2s_min_nll = np.array(pr2s).squeeze()
            fitted_betas.append(betas)
            num_parameters = 1 
            bic = np.array(nlls_min_nlls)*2 + np.log(num_trials)*num_parameters
        elif ('gcm' in model_name) or ('pm' in model_name):
            betas, pnlls, pr2s = fits['params'], fits['lls'], fits['r2s']
            # summing the fits for the four conditions separately; hence the total number of parameters is model_parameters*NUM_TASKS
            nlls_min_nlls = np.array(pnlls).squeeze()
            pr2s_min_nll = np.array(pr2s).squeeze()
            num_parameters = 8*NUM_TASKS
            bic = np.array(nlls_min_nlls)*2 + np.log(num_trials)*num_parameters
            fitted_betas.append(betas.squeeze()[:, 1])
        elif ('rulex' in model_name):
            betas, pnlls, pr2s = fits['params'], fits['lls'], fits['r2s']
            nlls_min_nlls = np.array(pnlls).squeeze()
            pr2s_min_nll = np.array(pr2s).squeeze()
            num_parameters = 2*NUM_TASKS
            bic = np.array(nlls_min_nlls)*2 + np.log(num_trials)*num_parameters

        nlls.append(nlls_min_nlls)
        r2s.append(pr2s_min_nll)
        bics.append(bic)

    # keep models and choose colors
    num_participants = len(nlls[0])
    MODELS = MODELS[:len(nlls)]
    # set colors depending on number of models in MODELS
    colors = ['#173b4f', '#8b9da7', '#5d7684', '#2f4a5a', '#0d2c3d', '#4d6a75', '#748995', '#a2c0a9'][:len(nlls)]


    # compare mean nlls across models in a bar plot
    f, ax = plt.subplots(1, 1, figsize=FIGSIZE)
    bar_positions = np.arange(len(nlls))*0.5
    ax.bar(bar_positions, np.array(nlls).mean(1), color=colors, width=0.4)
    ax.errorbar(bar_positions, np.array(nlls).mean(1), yerr=np.array(nlls).std(1)/np.sqrt(num_participants-1), c='k', lw=3, fmt="o")
    # add chance level line for 616 trials with binary choices
    ax.axhline(y=-np.log(0.5)*num_trials, color='k', linestyle='--', lw=3)
    ax.set_xlabel('Models', fontsize=FONTSIZE)
    ax.set_ylabel('NLL', fontsize=FONTSIZE)
    ax.set_xticks(bar_positions)  # Set x-tick positions to bar_positions
    ax.set_xticklabels(MODELS, fontsize=FONTSIZE-5)  # Assign category names to x-tick labels
    # ax.set_title(f'Model comparison for Badham et al. (2017)', fontsize=FONTSIZE)
    # plt.xticks(fontsize=FONTSIZE-2)
    plt.yticks(fontsize=FONTSIZE-2)
    sns.despine()
    f.tight_layout()
    plt.show()

    # compare mean BICS across models in a bar plot
    f, ax = plt.subplots(1, 1, figsize=FIGSIZE)
    bar_positions = np.arange(len(bics))*1.5
    ax.bar(bar_positions, np.array(bics).mean(1), color=colors, width=1.)
    ax.errorbar(bar_positions, np.array(bics).mean(1), yerr=np.array(bics).std(1)/np.sqrt(num_participants-1), c='k', lw=3, fmt="o")
    # add chance level line for 616 trials with binary choices
    # ax.axhline(y=-np.log(0.5)*num_trials*2, color='k', linestyle='--', lw=3)
    ax.set_xlabel('Models', fontsize=FONTSIZE)
    ax.set_ylabel('BIC', fontsize=FONTSIZE)
    ax.set_xticks(bar_positions)  # Set x-tick positions to bar_positions
    ax.set_xticklabels(MODELS, fontsize=FONTSIZE-6)  # Assign category names to x-tick labels
    # ax.set_title(f'Model comparison for Badham et al. (2017)', fontsize=FONTSIZE)
    # plt.xticks(fontsize=FONTSIZE-2)
    plt.yticks(fontsize=FONTSIZE-2)
    sns.despine()
    f.tight_layout()
    plt.show()
    f.savefig(f'{SYS_PATH}/figures/bic_devraj2022.svg', bbox_inches='tight', dpi=300)

    # compare mean r2s across models in a bar plot
    f, ax = plt.subplots(1, 1, figsize=FIGSIZE)
    bar_positions = np.arange(len(r2s))*0.5
    ax.bar(bar_positions, np.array(r2s).mean(1), color=colors, width=0.4)
    ax.errorbar(bar_positions, np.array(r2s).mean(1), yerr=np.array(r2s).std(1)/np.sqrt(num_participants-1), c='k', lw=3, fmt="o")
    ax.set_xlabel('Models', fontsize=FONTSIZE)
    ax.set_ylabel('R2', fontsize=FONTSIZE)
    ax.set_xticks(bar_positions)  # Set x-tick positions to bar_positions
    ax.set_xticklabels(MODELS, fontsize=FONTSIZE-5)  # Assign category names to x-tick labels

    # ax.set_title(f'Model comparison for  Badham et al. (2017)', fontsize=FONTSIZE)
    plt.yticks(fontsize=FONTSIZE-2)
    sns.despine()
    f.tight_layout()
    plt.show() 

    task_name = 'Devraj et al. (2022)'
    posterior_model_frequency(np.array(bics), MODELS, task_name=task_name)
    exceedance_probability(np.array(bics), MODELS, task_name=task_name)

def model_simulations_smith1998():

    models = ['smith1998', 'ermi', 'synthetic',]# 'human', 'syntheticnonlinear']
    f, ax = plt.subplots(1, len(models), figsize=(5*len(models),5))
    colors = ['#173b4f', '#37761e']
    num_blocks = None
    for idx, model in enumerate(models):
        if model=='smith1998':
       
            with open(f'{SYS_PATH}/categorisation/data/human/{model}.json') as file:
                human_data = json.load(file)

            # human data procesing
            fits_gcm, fits_pm = {}, {}
            mses_gcm = np.array(human_data['exemplar']['y'])
            mses_pm = np.array(human_data['prototype']['y'])
            # std error of mean across participants set to 0.
            stds_gcm = np.zeros_like(mses_gcm)
            stds_pm = np.zeros_like(mses_pm)
            # unsquezze to add a dimension for participants
            mses_gcm = np.expand_dims(mses_gcm, axis=0)
            mses_pm = np.expand_dims(mses_pm, axis=0)
    
        else:

            fits_gcm = np.load(f'{SYS_PATH}/categorisation/data/fitted_simulation/devraj2022_gcm_runs=1_iters=10_blocks=11_loss=mse_transfer_model={model}.npz')
            fits_pm = np.load(f'{SYS_PATH}/categorisation/data/fitted_simulation/devraj2022_pm_runs=1_iters=10_blocks=11_loss=mse_transfer_model={model}.npz')

            # load mses
            mses_gcm = fits_gcm['lls']
            mses_pm = fits_pm['lls']
            # mean mses across participants: mses are of shape (n_runs=1, n_participants, n_conditions=1, n_blocks)
            mses_gcm = np.squeeze(mses_gcm)
            mses_pm = np.squeeze(mses_pm)
            # std error of mean across participants
            stds_gcm = np.std(mses_gcm, axis=0)/np.sqrt(len(mses_gcm)-1)
            stds_pm = np.std(mses_pm, axis=0)/np.sqrt(len(mses_pm)-1)
             
        # keep only the first num_blocks (useful when using smith1998 data)
        num_blocks = 10 if 'smith1998' in models else 11
        mses_gcm = mses_gcm[:, :num_blocks]
        mses_pm = mses_pm[:, :num_blocks]
        stds_gcm = stds_gcm[:num_blocks]
        stds_pm = stds_pm[:num_blocks]

        # plot mean mses across participants for each trial segment for both models
        sns.lineplot(x=np.arange(mses_pm.shape[1])+1, y=np.mean(mses_pm, axis=0), ax=ax[idx], color=colors[0], label='PM')
        sns.lineplot(x=np.arange(mses_pm.shape[1])+1, y=np.mean(mses_gcm, axis=0), ax=ax[idx], color=colors[1], label='GCM')
        # add standard error of mean as error bars
        ax[idx].fill_between(np.arange(mses_pm.shape[1])+1, np.mean(mses_pm, axis=0)-stds_pm, np.mean(mses_pm, axis=0)+stds_pm, alpha=0.2, color=colors[0])
        ax[idx].fill_between(np.arange(mses_pm.shape[1])+1, np.mean(mses_gcm, axis=0)-stds_gcm, np.mean(mses_gcm, axis=0)+stds_gcm, alpha=0.2, color=colors[1])
        ax[idx].set_xlabel('Trial segment', fontsize=FONTSIZE)
        ax[idx].set_ylim([0, 1.])
        ax[idx].set_xticks(np.arange(mses_pm.shape[1])+1)
        # set y ticks font size
        ax[idx].tick_params(axis='y', labelsize=FONTSIZE-2)
        ax[idx].set_xticklabels(np.arange(mses_pm.shape[1])+1,fontsize=FONTSIZE-2)
        if idx==0:
            ax[idx].set_ylabel('Error', fontsize=FONTSIZE)
            # remove bounding box around the legend
            ax[idx].legend(frameon=False, fontsize=FONTSIZE-2)
        else:
            # remove legend
            ax[idx].legend([], frameon=False, fontsize=FONTSIZE-2)
        
    sns.despine()
    f.tight_layout()
    plt.show()
    f.savefig(f'{SYS_PATH}/figures/model_simulations_smith1998.svg', bbox_inches='tight', dpi=300)


def model_simulations_shepard1961(tasks=np.arange(1,7), batch_size=64):

    models = ['humans',\
              'env=claude_generated_tasks_paramsNA_dim3_data100_tasks11518_pversion4_model=transformer_num_episodes500000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8_noise0.0_shuffleTrue_run=0',
              'env=dim3synthetic_model=transformer_num_episodes500000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8_noise0.0_shuffleTrue_run=0_synthetic',\
              ]
    num_blocks = 15 # 16
    num_trials_per_block = 16
    num_trials = num_blocks*num_trials_per_block
    num_runs = 50
    betas = []
    for model in models:
        if model == 'humans':
            betas.append(None)
        else:
            model_name = 'ermi' if 'claude' in models[1] else 'rmc' if 'rmc' in models[1] else 'pfn' if 'syntheticnonlinear' in models[1] else 'mi'
            mse_distances, beta_range = np.load(f'{SYS_PATH}/categorisation/data/fitted_simulation/shepard1961_{model_name}_num_runs={num_runs}_num_blocks={num_blocks}_num_trials_per_block={num_trials_per_block}.npy', allow_pickle=True)
            betas.append(beta_range[np.argmin(mse_distances)])

    #betas = [None, 1, 1, ]
    corrects = np.ones((len(models), len(tasks), batch_size, num_trials))
    assert len(models)==len(betas), "Number of models and betas should be the same"
    for m_idx, (model_name, beta) in enumerate(zip(models, betas)):
        if model_name != 'humans':
            for t_idx, task in enumerate(tasks):
                model_path = f"/u/ajagadish/vanilla-llama/categorisation/trained_models/{model_name}.pt"
                corrects[m_idx, t_idx] = evaluate_metalearner(task, model_path, 'shepard_categorisation', beta=beta, shuffle_trials=True, num_trials=num_trials, num_runs=num_runs)      
    # compuate error rates across trials using corrects
    errors = 1. - corrects.mean(2)

    # load json file containing the data
    with open(f'{SYS_PATH}/categorisation/data/human/nosofsky1994.json') as json_file:
        data = json.load(json_file)

    # compare the error rate over trials between different tasks meaned over noise levels, shuffles and shuffle_evals
    f, axes = plt.subplots(1, len(models), figsize=(6*len(models), 5))
    colors = ['#E0E1DD', '#B6B9B9', '#8C9295', '#616A72','#37434E','#0D1B2A']
    # markers for the six types of rules in the plot: circle, cross, plus, inverted triangle, asterisk, triangle
    markers = ['o', 'x', '+', '*', 'v', '^']

    for idx, ax in enumerate(axes):

        if models[idx]=='humans':
            for i, rule in enumerate(data.keys()):
                ax.plot(np.arange(len(data[rule]['y'][:num_blocks]))+1, data[rule]['y'][:num_blocks], label=f'Type {i+1}', lw=3, color=colors[i], marker=markers[i], markersize=8)
        else:
            for t_idx, task in enumerate(tasks):
                block_errors = np.stack(np.split(errors[idx, t_idx], num_blocks)).mean(1)                
                ax.plot(np.arange(1, num_blocks+1), block_errors, label=f'Type {task}', lw=3, color=colors[t_idx], marker=markers[t_idx], markersize=8)

        ax.set_xticks(np.arange(1, num_blocks+1))
        ax.set_xlabel('Block', fontsize=FONTSIZE)
        if idx==0:
            ax.set_ylabel('Error rate', fontsize=FONTSIZE)
        ax.set_ylim([-0.05, .55])
        locs, labels = ax.get_xticks(), ax.get_xticklabels()
        # Set new x-tick locations and labels
        ax.set_xticks(locs[::2])
        ax.set_xticklabels(np.arange(1, num_blocks+1)[::2], fontsize=FONTSIZE-2)
        ax.tick_params(axis='y', labelsize=FONTSIZE-2)

    # add legend that spans across all subplots, in one row, at the center for the subplots, and place it outside the plot 
    f.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=6, fontsize=FONTSIZE-2, frameon=False, labels=[f'Type {task}' for task in tasks])
    #axes[int(len(models)/2)].legend(fontsize=FONTSIZE-4, frameon=False,  loc="upper center", bbox_to_anchor=(.5, 1.2), ncol=6)  # place legend outside the plot
    sns.despine()
    f.tight_layout()
    plt.show()
    f.savefig(f'{SYS_PATH}/figures/model_simulations_shepard1961.svg', bbox_inches='tight', dpi=300)


def model_comparison_johanssen2002():
    # choose  params for ermi simulations
    ermi_beta = 0.9
    mi_beta = 0.1
    num_runs = 1
    task_block = 32 # chose task block from ERMI to compare with human data

    data = pd.read_csv(f'{SYS_PATH}/categorisation/data/meta_learner/johanssen_categorisation__tasks8950_pversion5_stage1_model=transformer_num_episodes500000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8_noise0.0_shuffleTrue_run=1_beta={ermi_beta}_num_trials=288_num_runs={num_runs}.csv')
    transfer_stimulus_ids = data[data['stimulus_id'].str.contains('T')]['stimulus_id']
    transfer_data = data[data['stimulus_id'].isin(transfer_stimulus_ids)]
    # choose a subset of the transfer_data dataframe where the task_feature is equal to 1
    transfer_data = transfer_data[transfer_data['task_feature'] == task_block]

    mi_data = pd.read_csv(f'{SYS_PATH}/categorisation/data/meta_learner/johanssen_categorisation_500000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8_noise0.0_shuffleTrue_run=1_synthetic_beta={mi_beta}_num_trials=288_num_runs={num_runs}.csv')
    mi_transfer_stimulus_ids = mi_data[mi_data['stimulus_id'].str.contains('T')]['stimulus_id']
    mi_transfer_data = mi_data[mi_data['stimulus_id'].isin(mi_transfer_stimulus_ids)]
    # choose a subset of the transfer_data dataframe where the task_feature is equal to 1
    mi_transfer_data = mi_transfer_data[mi_transfer_data['task_feature'] == task_block]

    import json
    with open(f'{SYS_PATH}/categorisation/data/human/johanssen2002.json') as f:
        human_data = json.load(f)

    # human data procesing
    human_data_dict = {}
    for i, stimulus_id in enumerate(human_data['x']):
        human_data_dict[stimulus_id] = human_data['y'][i]
    human_data_df = pd.DataFrame.from_dict(human_data_dict, orient='index', columns=['human_choice'])
    human_data_df['stimulus_id'] = human_data_df.index
    human_data_df = human_data_df[human_data_df['stimulus_id'].str.contains('T')]

    # get the mean choice for each stimulus_id and sort the values in the order of the magnitude of the mean for both the transfer_data and human_data_df
    human_generalisation = human_data_df['human_choice'].sort_values()
    ermi_meta_learner_generalisation = (1-transfer_data.groupby('stimulus_id')['choice'].mean())
    ermi_meta_learner_generalisation = ermi_meta_learner_generalisation[human_generalisation.index]
    mi_meta_learner_generalisation = (1-mi_transfer_data.groupby('stimulus_id')['choice'].mean())
    mi_meta_learner_generalisation = mi_meta_learner_generalisation[human_generalisation.index] # keep the same order of the stimulus_ids for both human_generalisation and meta_learner_generalisation

    # compare the meta_learner_generalisation with human_generalisation in two subplots side by side
    fig, ax = plt.subplots(1, 3, figsize=(5*3, 5))
    # plot the human_generalisation in the left subplot
    human_generalisation.plot(kind='bar', ax=ax[0], color='#8b9da7', width=0.8)
    # plot the meta_learner_generalisation in the right subplot
    ermi_meta_learner_generalisation.plot(kind='bar', ax=ax[1], color='#173b4f', width=0.8)
    mi_meta_learner_generalisation.plot(kind='bar', ax=ax[2], color='#5d7684', width=0.8)

    # set the x-ticks for both subplots
    ax[0].set_xticks(np.arange(human_generalisation.shape[0]))
    ax[1].set_xticks(np.arange(ermi_meta_learner_generalisation.shape[0]))
    ax[2].set_xticks(np.arange(mi_meta_learner_generalisation.shape[0]))
    # set the x-tick labels for both subplots
    ax[0].set_xticklabels(human_generalisation.index, rotation=0)
    ax[1].set_xticklabels(ermi_meta_learner_generalisation.index, rotation=0)
    ax[2].set_xticklabels(mi_meta_learner_generalisation.index, rotation=0)
    # set the y-ticks for both subplotsand only keep alternating y-tick labels
    y_ticks = np.round(np.arange(0, 1.1, 0.1)[::2],1)
    ax[0].set_yticks(y_ticks)
    ax[1].set_yticks(y_ticks)
    ax[2].set_yticks(y_ticks)
    ax[0].set_yticklabels(y_ticks, fontsize=FONTSIZE-2)
    ax[1].set_yticklabels(y_ticks, fontsize=FONTSIZE-2)
    ax[2].set_yticklabels(y_ticks, fontsize=FONTSIZE-2)
    # set the x-label for both subplots
    ax[0].set_xlabel('Generalization stimulus', fontsize=FONTSIZE)
    # ax[1].set_xlabel('Generalization stimulus', fontsize=FONTSIZE)
    # set the y-label for both subplots
    ax[0].set_ylabel('p(A)', fontsize=FONTSIZE)
    # set the title for both subplots
    # ax[0].set_title('Human', fontsize=FONTSIZE)
    # ax[1].set_title('ERMI', fontsize=FONTSIZE)
    # set the fontsize for both subplots
    ax[0].tick_params(axis='both', which='major', labelsize=FONTSIZE-2)
    ax[1].tick_params(axis='both', which='major', labelsize=FONTSIZE-2)
    ax[2].tick_params(axis='both', which='major', labelsize=FONTSIZE-2)
    # set the ylim for both subplots
    ax[0].set_ylim([0, 1.0])
    ax[1].set_ylim([0, 1.0])
    ax[2].set_ylim([0, 1.0])
    # draw a horizontal line at y=0.5
    ax[0].axhline(y=0.5, linestyle='--', color='black')
    ax[1].axhline(y=0.5, linestyle='--', color='black')
    ax[2].axhline(y=0.5, linestyle='--', color='black')
    fig.tight_layout()
    sns.despine()
    plt.show() 


def plot_dataset_statistics(mode=0):

    from sklearn.preprocessing import PolynomialFeatures
    import statsmodels.api as sm
    
    def gini_compute(x):
        mad = np.abs(np.subtract.outer(x, x)).mean()
        rmad = mad/np.mean(x)
        return 0.5 * rmad

    def return_data_stats(data, poly_degree=2):

        df = data.copy()
        max_tasks = 400
        max_trial = 50
        all_corr, all_bics_linear, all_bics_quadratic, gini_coeff, all_accuraries_linear, all_accuraries_polynomial = [], [], [], [], [], []
        for i in range(0, max_tasks):
            df_task = df[df['task_id'] == i]
            if len(df_task) > 50: # arbitary data size threshold
                y = df_task['target'].to_numpy()
                y = np.unique(y, return_inverse=True)[1]

                X = df_task["input"].to_numpy()
                X = np.stack(X)
                X = (X - X.min())/(X.max() - X.min())

                all_corr.append(np.corrcoef(X[:, 0], X[:, 1])[0, 1])
                all_corr.append(np.corrcoef(X[:, 0], X[:, 2])[0, 1])
                all_corr.append(np.corrcoef(X[:, 1], X[:, 2])[0, 1])


                if (y == 0).all() or (y == 1).all():
                    pass
                else:
                    X_linear = PolynomialFeatures(1).fit_transform(X)
                    log_reg = sm.Logit(y, X_linear).fit(method='bfgs', maxiter=10000, disp=0)

                    gini = gini_compute(np.abs(log_reg.params[1:]))
                    gini_coeff.append(gini)

                    X_poly = PolynomialFeatures(poly_degree).fit_transform(X)
                    log_reg_quadratic = sm.Logit(y, X_poly).fit(method='bfgs', maxiter=10000, disp=0)

                    all_bics_linear.append(log_reg.bic)
                    all_bics_quadratic.append(log_reg_quadratic.bic)

                    if X.shape[0] < max_trial:
                        pass
                    else:
                        task_accuraries_linear = []
                        task_accuraries_polynomial = []
                        for trial in range(max_trial):
                            X_linear_uptotrial = X_linear[:trial]
                            #X_poly_uptotrial = X_poly[:trial]
                            y_uptotrial = y[:trial]

                            if (y_uptotrial == 0).all() or (y_uptotrial == 1).all() or trial == 0:
                                task_accuraries_linear.append(0.5)
                                #task_accuraries_polynomial.append(0.5)
                            else:
                                log_reg = sm.Logit(y_uptotrial, X_linear_uptotrial).fit(method='bfgs', maxiter=10000, disp=0)
                                #log_reg_quadratic = sm.Logit(y_uptotrial, X_poly_uptotrial).fit(method='bfgs', maxiter=10000, disp=0)

                                y_linear_trial = log_reg.predict(X_linear[trial])
                                #y_poly_trial = log_reg_quadratic.predict(X_poly[trial])

                                task_accuraries_linear.append(float((y_linear_trial.round() == y[trial]).item()))
                                #task_accuraries_polynomial.append(float((y_poly_trial.round() == y[trial]).item()))

                    all_accuraries_linear.append(task_accuraries_linear)
                    #all_accuraries_polynomial.append(task_accuraries_polynomial)
        all_accuraries_linear = np.array(all_accuraries_linear).mean(0)
        #all_accuraries_polynomial = np.array(all_accuraries_polynomial).mean(0)

        logprobs = torch.from_numpy(-0.5 * np.stack((all_bics_linear, all_bics_quadratic), -1))
        joint_logprob = logprobs + torch.log(torch.ones([]) /logprobs.shape[1])
        marginal_logprob = torch.logsumexp(joint_logprob, dim=1, keepdim=True)
        posterior_logprob = joint_logprob - marginal_logprob

        return all_corr, gini_coeff, posterior_logprob, all_accuraries_linear, all_accuraries_polynomial

    # set env_name and color_stats based on mode
    if mode == 0:
        env_name = f'{SYS_PATH}/categorisation/data/claude_generated_tasks_paramsNA_dim4_data650_tasks8950_pversion5_stage1'
        color_stats = '#405A63' #'#2F4A5A'# '#173b4f'
    elif mode == 1:#last plot
        env_name = f'{SYS_PATH}/categorisation/data/linear_data'
        color_stats = '#66828F' #5d7684'# '#5d7684'
    elif mode == 2:#first plot
        env_name = f'{SYS_PATH}/categorisation/data/real_data'
        color_stats = '#173b4f'#'#0D2C3D' #'#8b9da7'

    # load data
    data = pd.read_csv(f'{env_name}.csv')
    data = data.groupby(['task_id']).filter(lambda x: len(x['target'].unique()) == 2) # check if data has only two values for target in each task
    data.input = data['input'].apply(lambda x: np.array(eval(x)))

    if os.path.exists(f'{SYS_PATH}/categorisation/data/stats/stats_{str(mode)}.npz'):
        stats = np.load(f'{SYS_PATH}/categorisation/data/stats/stats_{str(mode)}.npz', allow_pickle=True)
        all_corr, gini_coeff, posterior_logprob, all_accuraries_linear = stats['all_corr'], stats['gini_coeff'], stats['posterior_logprob'], stats['all_accuraries_linear']
    else:
        all_corr, gini_coeff, posterior_logprob, all_accuraries_linear, all_accuraries_polynomial = return_data_stats(data)
        gini_coeff = np.array(gini_coeff)
        gini_coeff = gini_coeff[~np.isnan(gini_coeff)]
        posterior_logprob = posterior_logprob[:, 0].exp().detach().numpy()

    FONTSIZE=22 #8
    bin_max = np.max(gini_coeff)
    fig, axs = plt.subplots(1, 4,  figsize = (6*4,4))#figsize=(6.75, 1.5))
    axs[0].plot(all_accuraries_linear, color=color_stats, alpha=1., lw=3)
    #axs[0].plot(all_accuraries_polynomial, alpha=0.7)
    sns.histplot(np.array(all_corr), ax=axs[1], bins=11, binrange=(-1., 1.), stat='probability', edgecolor='w', linewidth=1, color=color_stats, alpha=1.)
    sns.histplot(gini_coeff, ax=axs[2], bins=11, binrange=(0, bin_max), stat='probability', edgecolor='w', linewidth=1, color=color_stats, alpha=1.)
    sns.histplot(posterior_logprob, ax=axs[3], bins=5, binrange=(0.0, 1.), stat='probability', edgecolor='w', linewidth=1, color=color_stats, alpha=1.)
    axs[1].set_xlim(-1, 1)

    axs[0].set_ylim(0.45, 1.05)
    axs[1].set_ylim(0, 0.4)
    axs[2].set_xlim(0., 0.76)
    axs[3].set_xlim(0., 1.05)

    axs[0].set_yticks(np.arange(0.5, 1.05, 0.25))
    axs[1].set_yticks(np.arange(0, 0.45, 0.2))
    axs[2].set_yticks(np.arange(0, 0.4, 0.15))
    axs[3].set_yticks(np.arange(0, 1.05, 0.5))

    # set tick size
    axs[0].tick_params(axis='both', which='major', labelsize=FONTSIZE-2)
    axs[1].tick_params(axis='both', which='major', labelsize=FONTSIZE-2)
    axs[2].tick_params(axis='both', which='major', labelsize=FONTSIZE-2)
    axs[3].tick_params(axis='both', which='major', labelsize=FONTSIZE-2)

    axs[0].set_ylabel('Accuracy', fontsize=FONTSIZE)
    axs[1].set_ylabel('Proportion', fontsize=FONTSIZE)
    axs[2].set_ylabel('')
    axs[3].set_ylabel('')

    if mode == 1:
        axs[0].set_xlabel('Trials', fontsize=FONTSIZE)
        axs[1].set_xlabel('Pearson\'s r', fontsize=FONTSIZE)
        axs[2].set_xlabel('Gini Coefficient', fontsize=FONTSIZE)
        axs[3].set_xlabel('Posterior probability ', fontsize=FONTSIZE)

    #set title
    if mode == 2:
        axs[0].set_title('Performance', fontsize=FONTSIZE)
        axs[1].set_title('Input correlation', fontsize=FONTSIZE)
        axs[2].set_title('Sparsity', fontsize=FONTSIZE)
        axs[3].set_title('Linearity', fontsize=FONTSIZE)

    plt.tight_layout()
    sns.despine()
    plt.savefig(f'{SYS_PATH}/figures/stats_' + str(mode) + '.svg', bbox_inches='tight')
    plt.show()

    # save corr, gini, posterior_logprob, and all_accuraries_linear for each mode in one .npz file
    if not os.path.exists(f'{SYS_PATH}/categorisation/data/stats/stats_{str(mode)}.npz'):
        np.savez(f'{SYS_PATH}/categorisation/data/stats/stats_{str(mode)}.npz', all_corr=all_corr, gini_coeff=gini_coeff, posterior_logprob=posterior_logprob, all_accuraries_linear=all_accuraries_linear)