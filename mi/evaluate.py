import numpy as np
import torch
from envs import FunctionlearningTask
# import argparse
# from baseline_classifiers import LogisticRegressionModel, SVMModel


def evaluate_regression(env_name=None, model_path=None, experiment='categorisation', env=None, model=None, mode='val', shuffle_trials=False, policy='binomial', beta=1., max_steps=70, nonlinear=False, num_dims=3, device='cpu', return_all=False):

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
        model.beta = beta  # model beta is adjustable at test time
        model.device = device
        model_choices = model(
            packed_inputs.float().to(device), sequence_lengths)

        # TODO: this reshaping limits the future possilibites chagne it
        model_choices = torch.concat([model_choices[i, :seq_len] for i, seq_len in enumerate(
            sequence_lengths)], axis=0).squeeze().float()
        true_choices = targets.reshape(-1).float().to(device) if (
            experiment == 'synthetic') else torch.concat(targets, axis=0).float().to(device)
        targets = None
        accuracy = model.criterion(model_choices, true_choices)

    if return_all:
        return accuracy, model_choices, true_choices, sequence_lengths, targets
    else:
        return accuracy
