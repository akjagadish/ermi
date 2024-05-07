import numpy as np
import torch
from envs import FunctionlearningTask, DecisionmakingTask
import torch.nn as nn
# import argparse
# from baseline_classifiers import LogisticRegressionModel, SVMModel


def evaluate_regression(env_name=None, model_path=None, experiment='llm_generated', env=None, model=None, mode='val', shuffle_trials=False, loss='mse', beta=1., max_steps=70, nonlinear=False, num_dims=3, device='cpu', return_all=False):

    if env is None:
        # load environment
        if experiment == 'synthetic':
            raise NotImplementedError
        elif experiment == 'llm_generated':
            env = FunctionlearningTask(data=env_name, num_dims=num_dims,
                                       mode=mode, max_steps=max_steps, shuffle_trials=shuffle_trials)

    if model is None:
        # load model
        model = torch.load(model_path, map_location=torch.device('cpu'))[
            1].to(device)

    with torch.no_grad():
        model.eval()
        packed_inputs, sequence_lengths, targets = env.sample_batch()
        model.device = device

        # model_choices = model(
        #     packed_inputs.float().to(device), sequence_lengths)
        # # TODO: this reshaping limits the future possilibites chagne it
        # model_choices = torch.concat([model_choices[i, :seq_len] for i, seq_len in enumerate(
        #     sequence_lengths)], axis=0).squeeze().float()
        # true_choices = targets.reshape(-1).float().to(device) if (
        #     experiment == 'synthetic') else torch.concat(targets, axis=0).float().to(device)
        # target = None
        # accuracy = model.criterion(model_choices, true_choices)

        # predictive_posterior = model(
        #     packed_inputs.float().to(device), sequence_lengths)
        # accuracy = predictive_posterior.log_prob(
        #     torch.stack(targets).unsqueeze(2).float().to(device)).mean()
        # model_choices, true_choices = predictive_posterior.mean, torch.stack(
        #     targets).float().to(device)
        if loss == 'mse':
            criterion = nn.MSELoss()
            predictive_posterior = model(
                packed_inputs, sequence_lengths)
            model_preds = predictive_posterior.mean
            model_preds = torch.concat([model_preds[i, :seq_len] for i, seq_len in enumerate(
                sequence_lengths)], axis=0).squeeze().float()
            true_targets = torch.concat(
                targets, axis=0).float().to(device)
            accuracy = criterion(model_preds, true_targets)
        elif loss == 'nll':
            predictive_posterior = model(
                packed_inputs, sequence_lengths)
            accuracy = - \
                predictive_posterior.log_prob(
                    torch.stack(targets).unsqueeze(2).float().to(device)).mean()

    if return_all:
        return accuracy, None, None, sequence_lengths, targets  # model_preds, true_targets
    else:
        return accuracy


def evaluate_classification(env_name=None, model_path=None, experiment='llm_generated', env=None, model=None, mode='val', shuffle_trials=False, policy='greedy', beta=1., max_steps=70, nonlinear=False, num_dims=3, device='cpu', return_all=False):

    if env is None:
        # load environment
        if experiment == 'synthetic':
            raise NotImplementedError
        elif experiment == 'llm_generated':
            env = DecisionmakingTask(data=env_name, num_dims=num_dims,
                                     mode=mode, max_steps=max_steps, shuffle_trials=shuffle_trials)

    if model is None:
        # load model
        model = torch.load(model_path, map_location=torch.device('cpu'))[
            1].to(device)

    with torch.no_grad():
        model.eval()
        packed_inputs, sequence_lengths, targets = env.sample_batch()
        model.device = device
        model.beta = beta  # model beta is adjustable at test time

        model_choices = model(packed_inputs, sequence_lengths)

        # sample from model choices probs using binomial distribution
        if policy == 'binomial':
            model_choices = torch.distributions.Binomial(
                probs=model_choices).sample()
        elif policy == 'greedy':
            model_choices = model_choices.round()

        model_choices = torch.concat([model_choices[i, :seq_len] for i, seq_len in enumerate(
            sequence_lengths)], axis=0).squeeze().float()
        true_choices = targets.reshape(-1, 1).float().to(device).squeeze()
        accuracy = (model_choices == true_choices).sum() / \
            (model_choices.shape[0])

    if return_all:
        return accuracy, None, None, sequence_lengths, targets  # model_preds, true_targets
    else:
        return accuracy
