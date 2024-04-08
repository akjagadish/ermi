import numpy as np
import torch
import pandas as pd
import torch.nn.functional as F
from scipy.optimize import differential_evolution, minimize
import numpy as np
import torch
import statsmodels.discrete.discrete_model as sm
import ipdb

#prototype stimuli are set by the experimenter depending on the type of rule used for categorisation
    
class PrototypeModel():
    """ Prototype model for categorisation task """
    
    def __init__(self, prototypes=None, num_features=3, num_categories=2, distance_measure=1, num_iterations=1, burn_in=False, learn_prototypes=False, loss='nll'):
        
        self.bounds = [(0, 20.), # sensitivity
                       (0, 1), # bias
                       ]     
        self.weight_bound = [(0, 1)] # weights
        self.distance_measure = distance_measure  
        self.num_iterations = num_iterations
        self.num_features = num_features
        self.num_categories = num_categories
        self.burn_in = burn_in
        self.learn_prototypes = learn_prototypes
        if self.learn_prototypes:
            self.bounds.extend(self.weight_bound * self.num_features * self.num_categories)
        self.prototypes = [np.ones(num_features) * 0.5, np.ones(num_features) * 0.5] if prototypes is None else prototypes
        self.define_loss_fn(loss)
        self.loss = loss
    
    def define_loss_fn(self, loss):
        if loss == 'nll':
            self.loss_fn = self.compute_nll
        elif loss == 'nll_transfer':
            self.loss_fun = self.compute_nll_transfer
        elif loss == 'mse':
            raise NotImplementedError
        elif loss == 'mse_transfer':
            self.loss_fn = self.compute_mse_transfer
        else:
            raise NotImplementedError

    def loo_nll(self, df):
        """ compute negative log likelihood for left out participants
        
        args:
        df: dataframe containing the data
        
        returns:
        nll: negative log likelihood for left out participants"""

        df_train = df.loc[df['is_training']]
        df_test = df.loc[~df['is_training']]
        
        self.bounds.extend(self.weight_bound * self.num_features)
        
        best_params = self.fit_parameters(df_train)
        nll = self.compute_nll(best_params, df_test)
        
        return nll

    def fit_participants(self, df, num_blocks=1):
        """ fit pm to individual participants and compute negative log likelihood 
        
        args:
        df: dataframe containing the data
        
        returns:
        nll: negative log likelihood for each participant"""

        num_conditions = len(df['condition'].unique())
        num_participants = len(df['participant'].unique())
        fit_measure, r2 = np.zeros((num_participants, num_conditions, num_blocks)), np.zeros((num_participants, num_conditions, num_blocks))
        self.bounds.extend(self.weight_bound * self.num_features)
        store_params = np.zeros((num_participants, num_conditions, num_blocks, len(self.bounds))) # store params for each task feature and block

        for p_idx, participant_id in enumerate(df['participant'].unique()[:num_participants]):
            df_participant = df[(df['participant'] == participant_id)]
            for c_idx, condition_id in enumerate(df['condition'].unique()):
                df_condition = df_participant[(df_participant['condition'] == condition_id)]
                num_trials_per_block = int(len(df_condition)/num_blocks)
                for b_idx, block in enumerate(range(num_blocks)):
                    offset_trial = df_condition.trial.min()
                    df_condition_block = df_condition[(df_condition['trial'] < ((block+1)*num_trials_per_block + offset_trial)) & (df_condition['trial'] >= (block*num_trials_per_block + offset_trial))]
                    best_params = self.fit_parameters(df_condition_block)
                    fit_measure[p_idx, c_idx, b_idx] = self.loss_fn(best_params, df_condition_block)
                    if self.loss == 'nll':
                        num_trials = len(df_condition_block)*(df_condition_block.task.max()+1)
                        num_trials = num_trials*0.5 if self.burn_in else num_trials
                        r2[p_idx, c_idx, b_idx] = 1 - (fit_measure[p_idx, c_idx, b_idx]/(-num_trials*np.log(1/2)))
                    store_params[p_idx, c_idx, b_idx] = best_params
                
        return fit_measure, r2, store_params
    
    def fit_metalearner(self, df, num_blocks=1):
        """ fit pm to individual meta-learning model runs and compute negative log likelihood 
        
        args:
        df: dataframe containing the data
        
        returns:
        nll: negative log likelihood for each block of a given task_feature"""

        num_task_features = len(df['task_feature'].unique())
        num_runs = len(df['run'].unique())
        fit_measure, r2 = np.zeros((num_runs, num_task_features, num_blocks)), np.zeros((num_runs, num_task_features, num_blocks))
        self.bounds.extend(self.weight_bound * self.num_features)
        store_params = np.zeros((num_runs, num_task_features, num_blocks, len(self.bounds)))

        for r_idx, run_id in enumerate(df['run'].unique()[:num_runs]):
            df_run = df[(df['run'] == run_id)]
            for idx, task_feature_id in enumerate(df_run['task_feature'].unique()):
                df_task_feature = df_run[(df_run['task_feature'] == task_feature_id)]
                num_trials_per_block = int((df_task_feature.trial.max()+1)/num_blocks)
                for b_idx, block in enumerate(range(num_blocks)):
                    df_task_feature_block = df_task_feature[(df_task_feature['trial'] < (block+1)*num_trials_per_block) & (df_task_feature['trial'] >= block*num_trials_per_block)]
                    best_params = self.fit_parameters(df_task_feature_block)
                    fit_measure[r_idx, idx, b_idx] = self.loss_fn(best_params, df_task_feature_block)
                    # fit_measure[idx, b_idx] = -self.compute_nll(best_params, df_task_feature_block, reduce)
                    if self.loss == 'nll':
                        num_trials = len(df_task_feature_block)*(df_task_feature_block.task.max()+1)
                        num_trials = num_trials*0.5 if self.burn_in else num_trials
                        r2[r_idx, idx, b_idx] = 1 - (fit_measure[r_idx, idx, b_idx]/(-num_trials*np.log(1/2)))
                    store_params[r_idx, idx, b_idx] = best_params
        
        return fit_measure, r2, store_params

    def fit_llm(self, df, num_blocks=1, reduce='sum'):
        """ fit pm to llm choices and compute negative log likelihood 
        
        args:
        df: dataframe containing the data
        
        returns:
        nll: negative log likelihood for each participant"""

        num_conditions = len(df['condition'].unique())
        num_participants = len(df['participant'].unique())
        fit_measure, r2 = np.zeros((num_participants, num_conditions, num_blocks)), np.zeros((num_participants, num_conditions, num_blocks))
        self.bounds.extend(self.weight_bound * self.num_features)
        store_params = np.zeros((num_participants, num_conditions, num_blocks, len(self.bounds))) # store params for each task feature and block

        for p_idx, participant_id in enumerate(df['participant'].unique()[:num_participants]):
            df_participant = df[(df['participant'] == participant_id)]
            unique_values = df_participant['llm_category'].unique()
            mapping = {k: v for v, k in enumerate(unique_values)}
            df_participant['choice'] =  df_participant['llm_category'].map(mapping)
            df_participant['correct_choice'] = df_participant['true_category'].map(mapping)
            df_participant['category'] = df_participant['true_category'].map(mapping)
            for c_idx, condition_id in enumerate(df['condition'].unique()):
                df_condition = df_participant[(df_participant['condition'] == condition_id)]
                num_trials_per_block = int(len(df_condition)/num_blocks)
                for b_idx, block in enumerate(range(num_blocks)):
                    offset_trial = df_condition.trial.min()
                    df_condition_block = df_condition[(df_condition['trial'] < ((block+1)*num_trials_per_block + offset_trial)) & (df_condition['trial'] >= (block*num_trials_per_block + offset_trial))]
                    best_params = self.fit_parameters(df_condition_block)
                    fit_measure[p_idx, c_idx, b_idx] = self.loss_fn(best_params, df_condition_block)
                    if self.loss == 'nll':
                        num_trials = len(df_condition_block)*(df_condition_block.task.max()+1)
                        num_trials = num_trials*0.5 if self.burn_in else num_trials
                        r2[p_idx, c_idx, b_idx] = 1 - (fit_measure[p_idx, c_idx, b_idx]/(-num_trials*np.log(1/2)))
                    store_params[p_idx, c_idx, b_idx] = best_params
                
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
        # define the constraint that the weights must sum to 1 as an obj
        constraint_obj = {'type': 'eq', 'fun': self.constraint}
        for _ in range(self.num_iterations):
    
            init = [np.random.uniform(x, y) for x, y in self.bounds]
            result = minimize(
                fun=minimize_loss_function,
                x0=init,
                args=(df),
                bounds=self.bounds,
                constraints=constraint_obj,
                method='SLSQP'if self.loss == 'mse_transfer' else None
            )
            
            # result = differential_evolution(self.compute_nll, 
            #                     bounds=self.bounds, 
            #                     args=(df),
            #                     maxiter=100)
            
            best_params = result.x

            if result.fun < best_fun:
                best_fun = result.fun
                best_res = result

        if best_res.success:
            print("The optimiser converged successfully.")
        else:
            Warning("The optimiser did not converge.")
        
        return best_params

    def constraint(self, params):
        """ define the constraint that the weights must sum to 1 """
        
        return np.sum(params[2+self.num_features*self.num_categories:]) - 1 if self.learn_prototypes else  np.sum(params[2:]) - 1

    def compute_nll(self, params, df):
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
            stimuli_seen = [[] for i in range(self.num_categories)] # list of lists to store objects seen so far within each category
            self.prototypes = [np.array([params[2+i*self.num_features+j] for j in range(self.num_features)]) for i in range(self.num_categories)] if self.learn_prototypes else self.prototypes
            self.prototypes = [df_task[df_task.correct_choice==category].iloc[0][['prototype_feature{}'.format(i+1) for i in range(self.num_features)]].values for category in range(self.num_categories)] if self.prototypes == 'from_data' else self.prototypes
            for trial_id in df_task['trial'].values:
                df_trial = df_task[(df_task['trial'] == trial_id)]
                choice = categories[df_trial.choice.item()] if df_trial.choice.item() in categories else df_trial.choice.item()
                true_choice = categories[df_trial.correct_choice.item()] if df_trial.correct_choice.item() in categories else df_trial.correct_choice.item()
  
                # load num features of the current stimuli
                current_stimuli = df_trial[['feature{}'.format(i+1) for i in range(self.num_features)]].values
                
                # given stimuli and list of objects seen so far within cateogry return probablity the object belongs to each category 
                category_probabilities = self.prototype_model(params, current_stimuli, stimuli_seen, choice)
                p_choice = category_probabilities[choice]*(1-params[1]) + params[1]*(1/self.num_categories)
                if self.burn_in:
                    ll += 0 if (trial_id<int(num_trials/2)) else np.log(p_choice + epsilon)
                else:
                    ll += np.log(p_choice + epsilon)
                # ll += 0 if (self.burn_in and (trial_id<int(num_trials/2))) else self.prototype_model(params, current_stimuli, stimuli_seen, choice)
                # update stimuli seen
                stimuli_seen[true_choice].append(current_stimuli)
    
        return -ll

    def compute_nll_transfer(self, params, df_train, df_transfer):
        """ compute negative log likelihood of the data given the parameters

        args:
        params: parameters of the model
        df_train: dataframe containing the training data
        df_transfer: dataframe containing the transfer data

        returns:
        negative log likelihood of the data given the parameters
        """
       
        ll = 0.
        stimuli_seen = [df_train[df_train['category'] == i][['x{}'.format(i+1) for i in range(self.num_features)]].values for i in range(self.num_categories)]
        stimuli_seen = [np.expand_dims(stimuli_seen[i], axis=1) for i in range(self.num_categories)]

        for trial_id in df_transfer.trial_id.values:
            df_trial = df_transfer[(df_transfer['trial_id'] == trial_id)]
            choice = df_trial['category'].item()
            current_stimuli = df_trial[['x{}'.format(i+1) for i in range(self.num_features)]].values
            ll += self.prototype_model(params, current_stimuli, stimuli_seen, choice)

        return -2*ll

    def compute_mse_transfer(self, params, df_train):
        """ compute mse of the data given the parameters 
        
        args:
        params: parameters of the model
        df: dataframe containing the data
        
        returns:
        mse of the data given the parameters
        """

 
        stimuli_seen = [df_train[df_train['category'] == category][['feature{}'.format(i+1) for i in range(self.num_features)]].values for category in np.sort(df_train['category'].unique())]
        stimuli_seen = [np.expand_dims(stimuli_seen[i], axis=1) for i in range(self.num_categories)]
        self.prototypes = [np.array([params[2+i*self.num_features+j] for j in range(self.num_features)]) for i in range(self.num_categories)] if self.learn_prototypes else self.prototypes
        self.prototypes = [df_train[df_train.category==category].iloc[0][['prototype_feature{}'.format(i+1) for i in range(self.num_features)]].values for category in np.sort(df_train['category'].unique())] if self.prototypes == 'from_data' else self.prototypes
        REF_CATEGORY = 1
        # keep one instance of each stimulus in order of stimulus_id
        df_transfer = df_train.drop_duplicates(subset=['stimulus_id'], keep='first').sort_values(by=['stimulus_id'])
        # compute the proportion of trials (out of those in which stimulus i was seen) in which the participant actually categorized stimulus i in category 1.
        proportion_category_1 = np.array([np.mean(df_train[df_train['stimulus_id'] == i]['choice'].values==REF_CATEGORY) for i in df_transfer['stimulus_id'].values])
        probability_category_1 = np.zeros(len(df_transfer['stimulus_id']))
        for idx, stimulus_id in enumerate(df_transfer.stimulus_id.values):
            df_trial = df_transfer[(df_transfer['stimulus_id'] == stimulus_id)]
            current_stimuli = df_trial[['feature{}'.format(i+1) for i in range(self.num_features)]].values
            category_probabilities = self.prototype_model(params, current_stimuli, stimuli_seen, None)
            probability_category_1[idx] = category_probabilities[REF_CATEGORY]*(1-params[1]) + params[1]*(1/self.num_categories)
        
        mse = np.sum((probability_category_1 - proportion_category_1)**2)

        return mse

    def benchmark(self, df_train, df_transfer):
        """ fit pm to training data and transfer to new data 
        
        args:
        df_train: dataframe containing the training data
        df_transfer: dataframe containing the transfer data
        
        returns:
        nll: negative log likelihood for each participant"""

        self.bounds.extend(self.weight_bound * self.num_features)
        constraint_obj = {'type': 'eq', 'fun': self.constraint}
        result = minimize(
                fun=self.compute_nll_transfer,
                x0=[np.random.uniform(x, y) for x, y in self.bounds],
                args=(df_train, df_transfer),
                bounds=self.bounds,
                constraints=constraint_obj
            )
        if result.success:
            print("The optimiser converged successfully.")
        else:
            Warning("The optimiser did not converge.")

        log_likelihood = -self.compute_nll_transfer(result.x, df_train, df_transfer)/2
        r2 = 1 - (log_likelihood/(len(df_transfer)*np.log(1/2)))
        print(f'fitted log-likelihood: {log_likelihood}')
        print(f'fitted pseudo-r2: {r2} \n')

        return result.x

    def prototype_model(self, params, current_stimuli, stimuli_seen, choice):
        """ return log likelihood of the choice given the stimuli and stimuli seen so far
         
        args:
        params: parameters of the model
        current_stimuli: features of the current stimuli
        stimuli_seen: list of lists of features of stimuli seen so far within each category
        choice: choice made by the participant
        
        returns:
        log likelihood of the choice given the stimuli and stimuli seen so far
        """
    
        sensitivity, bias = params[:2]
        weights = params[2+self.num_features*self.num_categories:] if self.learn_prototypes else params[2:]
        category_similarity = np.zeros(self.num_categories)
       
        for for_category in range(self.num_categories):
            if len(stimuli_seen[for_category]) == 0:
                # if no stimuli seen yet within category, similarity is set to unseen similarity
                category_similarity[for_category] = bias
            else:
                # compute attention weighted similarity measure
                category_similarity[for_category] = self.compute_attention_weighted_similarity(current_stimuli, np.stack(self.prototypes[for_category]), (weights, sensitivity))

        # compute category probabilities
        category_probabilities = self.compute_category_probabilities(category_similarity, bias)
        
        return category_probabilities
    
    def compute_attention_weighted_similarity(self, x, y, params):
        """ compute attention weighted similarity between current stimuli and stimuli seen so far 
        args:
        x: current stimuli
        y: stimuli seen so far
        params: attention weights and sensitivity
        
        returns:
        s: similarity between current stimuli and stimuli seen so far 
        """
        weights, sensitivity = params
       
        # compute distance between stimuli vectors with features weighted by attention weights with broadcasting
        d = np.mean(weights.reshape((1,-1)) @ (np.abs(y-x) ** self.distance_measure).T, axis=1)
        # take root of the distance measure
        d = d ** (1 / self.distance_measure)
        # compute similarity
        s = np.exp(-sensitivity * d)

        return s
    
    def compute_category_probabilities(self, s, b):
        """ compute probabilities for categories given the similarity and bias 

        args:
        s: similarity
        b: bias
        
        return:
        p: probability of each category
        """
        #TODO: bias term b for weighting similarity of categories differently
        
        assert len(s) == 2, "number of categories must be 2"
        weighted_similarities = np.array([1, 1]) * s #np.array([b, 1-b]) * s
        epsilon = 1e-10
        sum_weighted_similarities = np.sum(weighted_similarities)
        p = weighted_similarities / (sum_weighted_similarities + epsilon)
        
        return p