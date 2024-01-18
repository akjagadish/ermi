import pandas as pd
import numpy as np
import torch
import sys
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
import statsmodels.api as sm
SYS_PATH = '/u/ajagadish/ermi/categorisation'
sys.path.append(f"{SYS_PATH}/categorisation/")
sys.path.append(f"{SYS_PATH}/categorisation/rl2")
sys.path.append(f"{SYS_PATH}/categorisation/data")
from baseline_classifiers import benchmark_baseline_models_regex_parsed_random_points, benchmark_baseline_models_regex_parsed
from baseline_classifiers import LogisticRegressionModel, SVMModel

def return_baseline_performance(data, random=False):

    num_tasks = data.task_id.max()+1
    llm_performance = benchmark_baseline_models_regex_parsed(num_tasks, data)
    if random:
        uniform_performance = benchmark_baseline_models_regex_parsed_random_points(data)
        performance = np.concatenate((llm_performance, uniform_performance), axis=1)
    else:
        performance = llm_performance

    means = performance.mean(0)
    std_errors = performance.std(0)/np.sqrt(num_tasks-1)

    return means, std_errors, performance

# computing the distance between consequetive datapoints over trials
def l2_distance_trials(data, within_targets=False, within_consecutive_targets=False):
    '''
    Spatial distance between datapoints for each task
    Args:
        data: pandas dataframe with columns ['task_id', 'trial_id', 'input', 'target']
    Returns:
        None
    ''' 
    tasks = data.task_id.unique()#[:100]
    # extract the spatial distance for each task

    for target in data.target.unique():

        for task in tasks:
            # get the inputs for this task which is numpy array of dim (num_trials, 3)
            inputs = np.stack([eval(val) for val in data[data.task_id==task].input.values])
            # get the targets for this task which is numpy array of dim (num_trials, 1)
            targets = np.stack([val for val in data[data.task_id==task].target.values])

        
            if within_targets:
                inputs = inputs[targets==target]    

            # get the spatial distance between datapoints over trials for only points with the same target
            distance = np.array([np.linalg.norm(inputs[ii,:]-inputs[ii+1,:]) for ii in range(inputs.shape[0]-1)])
            
            if within_consecutive_targets:
                # consequetive datapoints with the same target
                distance = np.array([np.linalg.norm(inputs[ii]-inputs[ii+1]) for ii in range(inputs.shape[0]-1) if targets[ii]==targets[ii+1]])
            
            # pad with Nan's if distances are of unequal length and stack them vertically over tasks
            distance = np.pad(distance, (0, int(data.trial_id.max()*0.6)-distance.shape[0] if within_targets else data.trial_id.max()-distance.shape[0]), mode='constant', constant_values=np.nan)
            if task==0:
                distances = distance
            else:
                distances = np.vstack((distances, distance))

        # # plot the spatial distances
        # f, ax = plt.subplots(1, 1, figsize=(7,7))   
        # sns.heatmap(distances, annot=False, ax=ax, cmap='hot_r', vmin=0, vmax=1)
        # ax.set_title(f'Distance between datapoints')
        # ax.set_xlabel('Trial')
        # ax.set_ylabel('Task')
        # plt.show()
    
    return distances

def l2_distance_trials_all(data, target='A', shift=1, within_targets=False, llama=False, random=False):
    '''
    Compute distance of a datapoint with every other datapoint with shifts over trials
    Args:
        data: pandas dataframe with columns ['task_id', 'trial_id', 'input', 'target']
    Returns:
        None
    ''' 
    tasks = data.task_id.unique()#[:1000]

    # extract the distances for each task
    for task in tasks:
        
        # get the inputs for this task which is numpy array of dim (num_trials, 3)
        inputs = np.stack([eval(val) for val in data[data.task_id==task].input.values])
        # get the targets for this task which is numpy array of dim (num_trials, 1)
        targets = np.stack([val for val in data[data.task_id==task].target.values]) 
        
        if random:
            num_points, dim = 87, 3
            inputs =  np.random.rand(num_points, dim) 
            targets = np.random.choice(['A', 'B'], size=num_points) 

        if within_targets:
            inputs = inputs[targets==target]    

        # get the spatial distance between datapoints over trials 
        distance = np.array([np.linalg.norm(inputs[ii,:]-inputs[ii+shift,:]) for ii in range(inputs.shape[0]-shift)])
        # pad with Nan's if distances are of unequal length and stack them vertically over tasks
        if llama:
            distance = np.pad(distance, (0, int(8*0.6)-distance.shape[0] if within_targets else 8-distance.shape[0]), mode='constant', constant_values=np.nan)
        else:
            distance = np.pad(distance, (0, int(data.trial_id.max()*0.9)-distance.shape[0] if within_targets else data.trial_id.max()+1-distance.shape[0]), mode='constant', constant_values=np.nan)

        if task==0:
            distances = distance
        else:
            distances = np.vstack((distances, distance))
        
    return distances

def probability_same_target_vs_distance(data, target='A', llama=False, random=False):

    #TODO:
    # 1. set max datapoints based on max number of trials in the dataset
    # 2. set values for random more appropriately
    ref_target = 0 if target=='A' else 1
    tasks = data.task_id.unique()#[:1000]
    MAX_SIZE = (data.trial_id.max()+1)**2
    # load data for each task
    for task in tasks:
        
        # get the inputs for this task which is numpy array of dim (num_trials, 3)
        inputs = np.stack([eval(val) for val in data[data.task_id==task].input.values])
        # get the targets for this task which is numpy array of dim (num_trials, 1)
        targets = np.stack([val for val in data[data.task_id==task].target.values]) 
        
        if random:
            num_points, dim = data.trial_id.max(), 3
            inputs =  np.random.rand(num_points, dim) 
            targets = np.random.choice(['A', 'B'], size=num_points) 
   

        # get the spatial distance between datapoints over trials 
        #feature_distance = np.array([np.linalg.norm(inputs[ii,:]-inputs[ii+1,:]) for ii in range(inputs.shape[0]-1)])
        # distance between every pair of datapoints
        feature_distance = np.array([np.linalg.norm(inputs[ii,:]-inputs[jj,:]) for ii in range(inputs.shape[0]) for jj in range(inputs.shape[0]) if ii!=jj])

        # compute difference in probability of target for each pair of datapoints
        svm = SVMModel(inputs, targets)
        probability = svm.predict_proba(inputs)
        #probability_distance = np.array([np.linalg.norm(probability[ii, 0]-probability[ii+1, 0]) for ii in range(probability.shape[0]-1)])
        probability_distance = np.array([np.abs(probability[ii,ref_target]-probability[jj,ref_target]) for ii in range(probability.shape[0]) for jj in range(probability.shape[0]) if ii!=jj])
        
        # pad with Nan's if distances are of unequal length and stack them vertically over tasks
        # 100*100 = 10000 is the maximum number of pairs of datapoints as number of datapoints is 100
        probability_distance = np.pad(probability_distance, (0, MAX_SIZE-feature_distance.shape[0]), mode='constant', constant_values=np.nan)
        feature_distance = np.pad(feature_distance, (0, MAX_SIZE-feature_distance.shape[0]), mode='constant', constant_values=np.nan)
        
        #print(probability_distance.shape)
        if task==0:
            distances = feature_distance
            probabilities = probability_distance
        else:
            distances = np.vstack((distances, feature_distance))
            probabilities = np.vstack((probabilities, probability_distance))

    # # plot probability vs distance
    # f, ax = plt.subplots(1, 1, figsize=(7,7))
    # sns.regplot(distances.flatten(), probabilities.flatten(), ax=ax)
    # ax.set_title(f'Probability of same target vs distance between datapoints')
    # ax.set_xlabel('Distance between datapoints')
    # ax.set_ylabel('Probability of same target')
    # plt.show()

    return distances, probabilities

def evaluate_data_against_baselines(data, upto_trial=15, num_trials=None):

    tasks = data.task_id.unique()#[:1000] 
    accuracy_lm = []
    accuracy_svm = []
    scores = []
    # loop over dataset making predictions for next trial using model trained on all previous trials
    for task in tasks:
        baseline_model_choices, true_choices, baseline_model_scores = [], [], []   
        # get the inputs for this task which is numpy array of dim (num_trials, 3)
        inputs = np.stack(data[data.task_id==task].input.values)
        # normalise data for each task to be between 0 and 1
        inputs = (inputs - inputs.min())/(inputs.max() - inputs.min())
        # get the targets for this task which is numpy array of dim (num_trials, 1)
        # targets = torch.stack([torch.tensor(0) if val=='A' else torch.tensor(1) for val in data[data.task_id==task].target.values])
        targets = data[data.task_id==task].target.to_numpy()
        targets = torch.from_numpy(np.unique(targets, return_inverse=True)[1])
        num_trials = data[data.task_id==task].trial_id.max() if num_trials is None else num_trials

        trial = upto_trial # fit datapoints upto upto_trial; sort of burn-in trials
        # loop over trials
        while trial < num_trials:
            trial_inputs = inputs[:trial]
            trial_targets = targets[:trial]
            # if all targets until then are same, skip this trial
            if (trial_targets == 0).all() or (trial_targets == 1).all() or trial<=5:
                
                # sample probability from uniform distribution
                p = torch.distributions.uniform.Uniform(0, 1).sample()
                lr_model_choice = torch.tensor([[1-p, p]])
                p = torch.distributions.uniform.Uniform(0, 1).sample()
                svm_model_choice = torch.tensor([[p, 1-p]])
                baseline_model_choices.append(torch.stack([lr_model_choice, svm_model_choice]))
                true_choices.append(targets[[trial]])
                baseline_model_scores.append(torch.tensor([p, 1-p]))
            
            else:

                lr_model = LogisticRegressionModel(trial_inputs, trial_targets)
                svm_model = SVMModel(trial_inputs, trial_targets)
                lr_score = lr_model.score(inputs[[trial]], targets[[trial]])
                svm_score = svm_model.score(inputs[[trial]], targets[[trial]])
                lr_model_choice = lr_model.predict_proba(inputs[[trial]])
                svm_model_choice = svm_model.predict_proba(inputs[[trial]])#
                true_choice = targets[[trial]] #trial:trial+1]
                baseline_model_choices.append(torch.tensor(np.array([lr_model_choice, svm_model_choice])))
                true_choices.append(true_choice)
                baseline_model_scores.append(torch.tensor(np.array([lr_score, svm_score])))
            trial += 1
    
        # calculate accuracy
        baseline_model_choices_stacked, true_choices_stacked = torch.stack(baseline_model_choices).squeeze().argmax(2), torch.stack(true_choices).squeeze()
        accuracy_per_task_lm = (baseline_model_choices_stacked[:, 0] == true_choices_stacked) #for model_id in range(1)]
        accuracy_per_task_svm = (baseline_model_choices_stacked[:, 1] == true_choices_stacked) #for model_id in range(1)]
        
        baseline_model_scores_stacked = torch.stack(baseline_model_scores).squeeze()
        scores.append(baseline_model_scores_stacked.squeeze())
        accuracy_lm.append(accuracy_per_task_lm)
        accuracy_svm.append(accuracy_per_task_svm)
        
    return accuracy_lm, accuracy_svm, scores

def find_counts(inputs, dim, xx_min, xx_max):
    return (inputs[:, dim]<xx_max)*(inputs[:, dim]>xx_min)

def data_in_range(inputs, targets, min_value=0, max_value=1):
    inputs_in_range = [(inputs[ii]>min_value).all() * (inputs[ii]<max_value).all() for ii in range(len(inputs))]
    inputs = inputs[inputs_in_range]
    targets = targets[inputs_in_range]
    return inputs, targets

def bin_data_points(num_bins, data, min_value=0, max_value=1):
    inputs = np.stack([eval(val) for val in data.input.values])
    targets = np.stack([val for val in data.target.values])
    inputs, targets = data_in_range(inputs, targets, min_value, max_value)
    bins = np.linspace(0, 1, num_bins+1)[:-1]
    bin_counts, target_counts = [], [] #np.zeros((len(bins)*3))
    for ii in bins:
        x_min = ii 
        x_max = ii + 1/num_bins
        for jj in bins:
            y_min = jj
            y_max = jj + 1/num_bins
            for kk in bins:
                z_min = kk
                z_max = kk + 1/num_bins
                num_points = (find_counts(inputs, 0, x_min, x_max)*find_counts(inputs, 1, y_min, y_max)*find_counts(inputs, 2, z_min, z_max))
                bin_counts.append(num_points.sum())
                target_counts.append((targets[num_points]=='A').sum())

    bin_counts = np.array(bin_counts)
    target_counts = np.array(target_counts)
    return bin_counts, target_counts

def gini_compute(x):
                    mad = np.abs(np.subtract.outer(x, x)).mean()
                    rmad = mad/np.mean(x)
                    return 0.5 * rmad

def return_data_stats(data, poly_degree=2):

    df = data.copy()
    max_tasks = int(df['task_id'].max() + 1)
    all_corr, all_coef, all_bics_linear, all_bics_quadratic  = [], [], [], []
    f1_ceof, f2_coef, f3_coef = [], [], []
    f1_corr, f2_corr, f3_corr = [], [], []
    gini_coeff, advantage = [], []
    for i in range(0, max_tasks):
        df_task = df[df['task_id'] == i]
        if len(df_task) > 50: # arbitary data size threshold
            y = df_task['target'].to_numpy()
            y = np.unique(y, return_inverse=True)[1]

            # df_task['input'] = df_task['input'].apply(eval).apply(np.array)
            X = df_task["input"].to_numpy()
            X = np.stack(X)
            X = (X - X.min())/(X.max() - X.min())  # normalize data
            
            # correlations
            all_corr.append(np.corrcoef(X[:, 0], X[:, 1])[0, 1])
            all_corr.append(np.corrcoef(X[:, 0], X[:, 2])[0, 1])
            all_corr.append(np.corrcoef(X[:, 1], X[:, 2])[0, 1])

            # per feature correlations
            f1_corr.append(np.corrcoef(X[:, 0], X[:, 1])[0, 1])
            f2_corr.append(np.corrcoef(X[:, 0], X[:, 2])[0, 1])
            f3_corr.append(np.corrcoef(X[:, 1], X[:, 2])[0, 1])

            if (y == 0).all() or (y == 1).all():
                pass
            else:
                X_linear = PolynomialFeatures(1).fit_transform(X)
                log_reg = sm.Logit(y, X_linear).fit(method='bfgs', maxiter=10000)

                # weights
                all_coef.append(log_reg.params[1])
                all_coef.append(log_reg.params[2])
                all_coef.append(log_reg.params[3])
                
                # store feature specific weights separately
                f1_ceof.append(log_reg.params[1])
                f2_coef.append(log_reg.params[2])
                f3_coef.append(log_reg.params[3])


                X_poly = PolynomialFeatures(poly_degree).fit_transform(X)
                log_reg_quadratic = sm.Logit(y, X_poly).fit(method='bfgs', maxiter=10000)
                
                # svm using sklearn
                # svm = SVMModel(X, y)
                # score_svm = svm.score(X, y)
                # bic_svm, ll_svm = svm.calculate_bic(X, y)
                
                # logisitic regression with polynomial features using sklearn
                svm = LogisticRegressionModel(X_poly, y)
                score_svm = svm.score(X_poly, y)
                bic_svm, ll_svm = svm.calculate_bic(X_poly, y)
                
                # logisitic regression with linear features using sklearn
                lr = LogisticRegressionModel(X_linear, y)
                score_lr = lr.score(X_linear, y)
                bic_lr, ll_lr = lr.calculate_bic(X_linear, y)
                # print(bic_lr, bic_svm)
               
                advantage.append(0. if score_svm > score_lr else 1.) #0 if score_svm > score_lr else 1) #(score_svm - score_lr) #(ll_svm-ll_lr)

                # bics
                all_bics_linear.append(bic_lr) #(-2*ll_lr) #(log_reg.bic)
                all_bics_quadratic.append(bic_svm) #(-2*ll_svm) #(log_reg_quadratic.bic)

                gini = gini_compute(np.abs(log_reg.params[1:])) 
                gini_coeff.append(gini)
                

    # compute posterior probabilities
    logprobs = torch.from_numpy(-0.5 * np.stack((all_bics_linear, all_bics_quadratic), -1))
    joint_logprob = logprobs + torch.log(torch.ones([]) /logprobs.shape[1])
    marginal_logprob = torch.logsumexp(joint_logprob, dim=1, keepdim=True)
    posterior_logprob = joint_logprob - marginal_logprob

    # store feature specific weights separately
    f1_ceof = np.array(f1_ceof)
    f2_coef = np.array(f2_coef)
    f3_coef = np.array(f3_coef)
    feature_coef = np.stack((f1_ceof, f2_coef, f3_coef), -1)
    # horizontal task the feature coefficients
    # hfeature_coef = feature_coef.reshape(-1, 3)

    # horizontal stack per feature correlations
    features_corrs = np.stack((f1_corr, f2_corr, f3_corr), -1)


    return all_corr, all_coef, posterior_logprob, feature_coef, features_corrs, gini_coeff, advantage