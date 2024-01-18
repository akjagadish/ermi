import numpy as np
import torch
import pandas as pd
import torch.nn.functional as F
from scipy.optimize import differential_evolution, minimize
import numpy as np
import torch
import statsmodels.discrete.discrete_model as sm
import ipdb
import itertools

class RulExModel():
    """ RuleEX Model """
    
    def __init__(self, num_features=4, num_categories=2, num_iterations=1, exception=False, exceptions=None, opt_method='minimize', loss='nll'):
         
        self.num_iterations = num_iterations
        self.num_features = num_features
        self.num_categories = num_categories
        self.define_loss_fn(loss)
        self.loss = loss
        self.exception = exception
        self.exceptions = exceptions if exceptions is not None else []

        self.bounds = [(0, 1), # guessing term
                       ] 
        self.weight_bounds = [(0, 1)]

        self.opt_method = opt_method

    
    def define_loss_fn(self, loss):
        if loss == 'nll':
            self.loss_fn = self.compute_nll
        elif loss == 'nll_transfer':
            self.loss_fun = self.compute_nll_transfer
        else:
            raise NotImplementedError

    def fit_participants(self, df, num_blocks=1, reduce='sum'):
        """ fit gcm to individual participants and compute negative log likelihood 
        
        args:
        df: dataframe containing the data
        
        returns:
        nll: negative log likelihood for each participant"""

        num_conditions = len(df['condition'].unique())
        num_participants = len(df['participant'].unique())
        fit_measure, r2 = np.zeros((num_participants, num_conditions, num_blocks)), np.zeros((num_participants, num_conditions, num_blocks))
        store_params = np.zeros((num_participants, num_conditions, num_blocks, len(self.bounds)+3)) # store params for each task feature and block

        for p_idx, participant_id in enumerate(df['participant'].unique()[:num_participants]):
            df_participant = df[(df['participant'] == participant_id)]
            for c_idx, condition_id in enumerate(df['condition'].unique()):
                df_condition = df_participant[(df_participant['condition'] == condition_id)]
                num_trials_per_block = int((df_condition.trial.max()+1)/num_blocks)
                for b_idx, block in enumerate(range(num_blocks)):
                    df_condition_block = df_condition[(df_condition['trial'] < (block+1)*num_trials_per_block) & (df_condition['trial'] >= block*num_trials_per_block)]
                    best_params = self.fit_parameters(df_condition_block)
                    fit_measure[p_idx, c_idx, b_idx] = self.loss_fn(best_params[[0]], df_condition_block, best_params[2:].astype('int'), int(best_params[1]))
                    if self.loss == 'nll':
                        num_trials = len(df_condition_block)*(df_condition_block.task.max()+1)
                        r2[p_idx, c_idx, b_idx] = 1 - (fit_measure[p_idx, c_idx, b_idx]/(-num_trials*np.log(1/2)))

                    if condition_id == 2:
                        store_params[p_idx, c_idx, b_idx] = best_params
                    else:
                        store_params[p_idx, c_idx, b_idx] = np.hstack([best_params[:2], np.nan, best_params[-1]])

        return fit_measure, r2, store_params
    

    def fit_parameters(self, df):
        """ fit parameters using scipy optimiser 
        
        args:
        df: dataframe containing the data
        
        returns:
        best_params: best parameters found by the optimiser
        """
        
        best_fun = np.inf
        minimize_loss_function = self.loss_fn
        
        # sample order invariant pairs of features to use for the conjunctive rule
        if df.condition.unique()[0] == 2:
            feature_pairs = np.array(list(itertools.combinations(range(self.num_features), 2)))
        else:
            # pass a list of 1-dim features
            feature_pairs = np.arange(self.num_features).reshape(-1, 1)

        for feature_dim in feature_pairs:

            for category_code in range(self.num_categories):

                for _ in range(self.num_iterations):

                    if self.opt_method == 'minimize':
                        init = [np.random.uniform(x, y) for x, y in self.bounds]
                        result = minimize(
                            fun=minimize_loss_function,
                            x0=init,
                            args=(df, feature_dim, category_code),
                            bounds=self.bounds,
                        )
                    best_params = result.x

                    if result.fun < best_fun:
                        best_fun = result.fun
                        best_res = result
                        best_feature_dim = feature_dim
                        best_category_code = category_code

        if best_res.success:
            print("The optimiser converged successfully.")
        else:
            Warning("The optimiser did not converge.")

        best_params = np.hstack([best_params, best_category_code, best_feature_dim])
        return best_params
    
    def compute_nll(self, params, df, feature_dim, category_code):
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
        categories = {'j': 0, 'f': 1}

        for task_id in range(num_tasks):
            df_task = df[(df['task'] == task_id)]
            num_trials = df_task['trial'].max() + 1
            for trial_id in df_task['trial'].values:
                df_trial = df_task[(df_task['trial'] == trial_id)]
                choice = categories[df_trial.choice.item()] if df_trial.choice.item() in categories else df_trial.choice.item()
                true_choice = categories[df_trial.correct_choice.item()] if df_trial.correct_choice.item() in categories else df_trial.correct_choice.item()
  
                # load num features of the current stimuli
                current_stimuli = df_trial[['feature{}'.format(i+1) for i in range(self.num_features)]].values.squeeze()
                
                # compute probability of choice given the stimuli
                category_probabilities = self.rule_ex(feature_dim, current_stimuli)
                category_probabilities = category_code - category_probabilities if category_code == 1 else category_probabilities
                p_choice = category_probabilities[choice]*(1-params[0]) + params[0]*(1/self.num_categories)
                ll += np.log(p_choice + epsilon)
            
        return -ll

    def compute_nll_transfer(self, params, df_train, df_transfer, feature_dim):
        """ compute negative log likelihood of the data given the parameters

        args:
        params: parameters of the model
        df_train: dataframe containing the training data
        df_transfer: dataframe containing the transfer data

        returns:
        negative log likelihood of the data given the parameters
        """
       
        raise NotImplementedError
                    

    def rule_ex(self, feature_dim, current_stimuli):
        """ return probability of choices given the stimuli based on a rule
        args:
        feature_dim: feature_dim of the stimuli to use for the rule
        current_stimuli: features of the current stimuli
        category_code: category code for which the rule should be applied
        
        returns:
        category_probabilities: probability of choices given the stimuli based on a rule
        """
        
        assert self.num_categories == 2, "only 2 categories are supported"
        if len(feature_dim)==1:
            assert feature_dim < self.num_features, "feature_dim must be less than num_features"
            ## 1-dim rule with exception
            # if current stimuli is one of the exceptions retun the category for the exception
            if self.exception and (current_stimuli.tolist() in self.exceptions):
                # reverse the category assignment for the exception stimuli based on the value taken on the feature_dim
                category_probabilities = np.array([1., .0]) if (current_stimuli[feature_dim] == 1) else np.array([.0, 1.])
            elif current_stimuli[feature_dim] == 0:
                category_probabilities = np.array([1., .0])
            else:
                category_probabilities = np.array([.0, 1.])

        ## conjunctive rule 
        elif len(feature_dim)>1:
            feature_dim1, feature_dim2 = feature_dim
            if current_stimuli[feature_dim1]==current_stimuli[feature_dim2]:
                category_probabilities = np.array([1., .0])
            else:
                category_probabilities = np.array([.0, 1.])

        else:
            raise NotImplementedError
    
        return category_probabilities
    