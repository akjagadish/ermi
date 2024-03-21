import numpy as np
import torch
import pandas as pd
import torch.nn.functional as F
from scipy.optimize import differential_evolution, minimize
import numpy as np
import torch
import statsmodels.discrete.discrete_model as sm
import ipdb
from scipy.optimize import LinearConstraint, Bounds

   
class LLMCategoryLearning():
    """ LLM based category learning model"""
    
    def __init__(self, num_features=4, num_categories=2, num_iterations=1, opt_method='minimize', loss='nll'):
        
        self.num_features = num_features
        self.num_categories = num_categories
        self.num_iterations = num_iterations
        self.opt_method = opt_method
        self.loss = loss
        self.loss_fn = self.compute_nll if loss == 'nll' else NotImplementedError

    
    def fit_participants(self, df, num_blocks=1, reduce='sum'):
        """ fit llm to individual participants and compute negative log likelihood 
        
        args:
        df: dataframe containing the data
        
        returns:
        nll: negative log likelihood for each participant"""

        num_conditions = len(df['condition'].unique())
        num_participants = len(df['participant'].unique())
        fit_measure, r2 = np.zeros((num_participants, num_conditions, num_blocks)), np.zeros((num_participants, num_conditions, num_blocks))
        self.bounds = [(0., 1.)] # epsilon greedy model
        store_params = np.zeros((num_participants, num_conditions, num_blocks, 1)) # store params for each task feature and block

        for p_idx, participant_id in enumerate(df['participant'].unique()[:num_participants]):
            df_participant = df[(df['participant'] == participant_id)]
            for c_idx, condition_id in enumerate(df['condition'].unique()):
                df_condition = df_participant[(df_participant['condition'] == condition_id)]
                num_trials_per_block = int(len(df_condition)/num_blocks)
                for b_idx, block in enumerate(range(num_blocks)):
                    offset_trial = df_condition.trial.min()
                    df_condition_block = df_condition[(df_condition['trial'] < ((block+1)*num_trials_per_block + offset_trial)) & (df_condition['trial'] >= (block*num_trials_per_block + offset_trial))]
                    best_params = self.fit_parameters(df_condition_block, reduce)
                    fit_measure[p_idx, c_idx, b_idx] = self.loss_fn(best_params, df_condition_block, reduce)
                    if self.loss == 'nll':
                        num_trials = len(df_condition_block)*(df_condition_block.task.max()+1)
                        r2[p_idx, c_idx, b_idx] = 1 - (fit_measure[p_idx, c_idx, b_idx]/(-num_trials*np.log(1/2)))
                    store_params[p_idx, c_idx, b_idx] = best_params
                
        return fit_measure, r2, store_params
    

    def fit_parameters(self, df, reduce='sum'):
        """ fit parameters using scipy optimiser 
        
        args:
        df: dataframe containing the data
        
        returns:
        best_params: best parameters found by the optimiser
        """
        
        best_fun = np.inf
        minimize_loss_function = self.loss_fn
        for _ in range(self.num_iterations):
    
            if self.opt_method == 'minimize':
                init = [np.random.uniform(x, y) for x, y in self.bounds]
                result = minimize(
                    fun=minimize_loss_function,
                    x0=init,
                    args=(df, reduce),
                    bounds=self.bounds
                )
            elif self.opt_method == 'differential_evolution':
                result = differential_evolution(func=minimize_loss_function, 
                                    bounds=self.bounds, 
                                    args=(df, reduce),
                                    maxiter=1)
            else:
                raise NotImplementedError
            best_params = result.x

            if result.fun < best_fun:
                best_fun = result.fun
                best_res = result

        if best_res.success:
            print("The optimiser converged successfully.")
        else:
            Warning("The optimiser did not converge.")
        
        return best_params

    
    def compute_nll(self, params, df, reduce):
        """ compute negative log likelihood of the data given the parameters 
        
        args:
        params: parameters of the model
        df: dataframe containing the data
        
        returns:
        negative log likelihood of the data given the parameters
        """
   
        ll = 0.
        num_tasks = df['task'].max() + 1
        epsilon = 1e-10
        category_labels = df.true_category.unique()
        assert len(category_labels) == self.num_categories, "Number of categories in the data does not match the number of categories in the model"
        human_mapping = {df.correct_choice[df.category==0].values[0]:0, df.correct_choice[df.category==1].values[0]:1}
        llm_mapping = {df.true_category[df.category==0].values[0]:0, df.true_category[df.category==1].values[0]:1}
        for task_id in range(num_tasks):
            df_task = df[(df['task'] == task_id)]
            for trial_id in df_task['trial'].values:
                df_trial = df_task[(df_task['trial'] == trial_id)]
                choice = human_mapping[df_trial['choice'].values[0]]
                llm_choice = llm_mapping[df_trial['llm_category'].values[0]]
                category_probabilities = 1 if llm_choice == choice else 0
                p_choice = category_probabilities*(1-params[0]) + params[0]*(1/self.num_categories)
                ll += np.log(p_choice + epsilon)

        return -ll


    