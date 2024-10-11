import numpy as np
import torch
from envs import FunctionlearningTask, DecisionmakingTask, SyntheticDecisionmakingTask, SyntheticFunctionlearningTask
import torch.nn as nn
from model import TransformerDecoderClassification, TransformerDecoderLinearWeights
from model_utils import parse_model_path
from torch.distributions import Categorical, Normal, Bernoulli


def evaluate_regression(env_name=None, model_path=None, experiment='llm_generated', env=None, model=None, mode='val', shuffle_trials=False, loss='mse', beta=1., max_steps=70, nonlinear=False, num_dims=3, device='cpu', return_all=False):

    if env is None:
        # load environment
        if experiment == 'synthetic':
            env = SyntheticFunctionlearningTask(num_dims=num_dims, mode=mode, max_steps=max_steps, device=device)
        elif experiment == 'llm_generated':
            env = FunctionlearningTask(data=env_name, num_dims=num_dims,
                                       mode=mode, max_steps=max_steps, shuffle_trials=shuffle_trials, device=device)

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
                packed_inputs.to(device), sequence_lengths)
            model_preds = predictive_posterior.mean
            model_preds = torch.concat([model_preds[i, :seq_len] for i, seq_len in enumerate(
                sequence_lengths)], axis=0).squeeze().float().to(device)
            true_targets = torch.concat(
                targets, axis=0).float().to(device) if isinstance(targets, list) else targets.reshape(-1).float().to(device)
            accuracy = criterion(model_preds, true_targets)
        elif loss == 'nll':
            predictive_posterior = model(
                packed_inputs, sequence_lengths)
            true_targets = torch.stack(targets).float().to(device) if isinstance(targets, list) else targets.float().to(device)
            accuracy = - \
                predictive_posterior.log_prob(true_targets).mean()

    if return_all:
        return accuracy, None, None, sequence_lengths, targets  # model_preds, true_targets
    else:
        return accuracy


def evaluate_classification(env_name=None, model_path=None, experiment='llm_generated', paired=False, env=None, model=None, mode='val', shuffle_trials=False, loss='nll', policy='greedy', beta=1., max_steps=70, nonlinear=False, num_dims=3, device='cpu', optimizer=None, optim='adamw', return_all=False):

    if env is None:
        # load environment
        if experiment == 'synthetic':
            env = SyntheticDecisionmakingTask(num_dims=num_dims,  mode=mode, max_steps=max_steps,
                                              device=device).to(device)
        elif experiment == 'llm_generated':
            env = DecisionmakingTask(data=env_name, num_dims=num_dims,
                                     mode=mode, max_steps=max_steps, shuffle_trials=shuffle_trials,
                                      device=device).to(device)
        
    if model is None:
        # load model
        num_hidden, num_layers, d_model, num_head, loss_fn, model_max_steps = parse_model_path(model_path, {'model_max_steps': max_steps})

        # initialise model
        if paired:
            model = TransformerDecoderLinearWeights(num_input=env.num_dims, num_output=env.num_choices, num_hidden=num_hidden,
                                                    num_layers=num_layers, d_model=d_model, num_head=num_head, max_steps=model_max_steps, loss=loss_fn, device=device).to(device)

        else:
            model = TransformerDecoderClassification(num_input=env.num_dims, num_output=env.num_choices, num_hidden=num_hidden,
                                                    num_layers=num_layers, d_model=d_model, num_head=num_head, max_steps=model_max_steps, loss=loss_fn, device=device).to(device)
        
        # load model weights
        state_dict = torch.load(
            model_path, map_location=device)[1]
        model.load_state_dict(state_dict)
        model.to(device)

    model.eval()
    if optimizer is not None and optim == 'schedulefree':
        optimizer.eval()

    with torch.no_grad():
        # sample batch
        packed_inputs, sequence_lengths, targets = env.sample_batch(
            paired=paired)
        model.device = device
        model.beta = beta  # model beta is adjustable at test time

        # model choices
        model_choice_probs = model(packed_inputs, sequence_lengths)

        # sample from model choices probs using bernoulli distribution or use greedy policy
        model_choices = Bernoulli(probs=model_choice_probs).sample() if policy == 'bernoulli' else model_choice_probs.round()
        model_choices = torch.concat([model_choices[i, :seq_len] for i, seq_len in enumerate(
            sequence_lengths)], axis=0).squeeze().float()
        true_choices = targets.reshape(-1, 1).float().to(device).squeeze()
        accuracy = (model_choices == true_choices).sum() / \
            (model_choices.shape[0])

    if return_all:
        return accuracy, None, None, sequence_lengths, targets  # model_preds, true_targets
    else:
        return accuracy
