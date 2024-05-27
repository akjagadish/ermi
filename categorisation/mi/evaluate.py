import numpy as np
import torch
from .envs import CategorisationTask, ShepardsTask, NosofskysTask, SyntheticCategorisationTask, SmithsTask, RMCTask
import argparse
from .baseline_classifiers import LogisticRegressionModel, SVMModel

# evaluate a model


def evaluate_1d(env_name=None, model_path=None, experiment='categorisation', env=None, model=None, mode='val', shuffle_trials=False, policy='binomial', beta=1., max_steps=70, nonlinear=False, num_dims=3, device='cpu', return_all=False):

    if env is None:
        # load environment
        if experiment == 'synthetic':
            env = SyntheticCategorisationTask(
                nonlinear=nonlinear, num_dims=num_dims, max_steps=max_steps, shuffle_trials=shuffle_trials)
        elif experiment == 'rmc':
            env = RMCTask(data=env_name, num_dims=num_dims,
                          max_steps=max_steps, shuffle_trials=shuffle_trials)
        if experiment == 'categorisation':
            env = CategorisationTask(data=env_name, num_dims=num_dims,
                                     mode=mode, max_steps=max_steps, shuffle_trials=shuffle_trials)
        elif experiment == 'shepard_categorisation':
            env = ShepardsTask(task=env_name, max_steps=max_steps,
                               shuffle_trials=shuffle_trials)
        elif experiment == 'nosofsky_categorisation':
            env = NosofskysTask(task=env_name)
        elif experiment == 'smith_categorisation':
            env = SmithsTask(rule=env_name, max_steps=max_steps)

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

        # sample from model choices probs using binomial distribution
        if policy == 'binomial':
            model_choices = torch.distributions.Binomial(
                probs=model_choices).sample()
        elif policy == 'greedy':
            model_choices = model_choices.round()

        # TODO: this reshaping limits the future possilibites chagne it
        model_choices = torch.concat([model_choices[i, :seq_len] for i, seq_len in enumerate(
            sequence_lengths)], axis=0).squeeze().float()
        true_choices = targets.reshape(-1).float().to(device) if (
            experiment == 'synthetic' or experiment == 'rmc') else torch.concat(targets, axis=0).float().to(device)
        category_labels = torch.concat(env.stacked_labels, axis=0).float(
        ) if experiment == 'nosofsky_categorisation' else None
        accuracy = (model_choices == true_choices).sum() / \
            (model_choices.shape[0])

    if return_all:
        return accuracy, model_choices, true_choices, sequence_lengths, category_labels
    else:
        return accuracy

# evaluate a model


def evaluate(env_name=None, model_path=None, env=None, model=None, mode='val', policy='greedy', return_all=False):

    if env is None:
        # load environment
        env = CategorisationTask(data=env_name, mode=mode)
    if model is None:
        # load model
        model = torch.load(model_path)[1]

    with torch.no_grad():
        model.eval()
        inputs, targets, prev_targets, done, info = env.reset()
        hx, cx = model.initial_states(env.batch_size)
        model_choices = []
        true_choices = []

        while not done:
            inputs = model.make_inputs(inputs, prev_targets)
            model_choice, hx, cx = model(inputs.float(), hx, cx)
            true_choice = targets.detach().clone()
            model_choices.append(model_choice)
            true_choices.append(true_choice)
            inputs, targets, prev_targets, done, info = env.step()

        model_choices = torch.stack(model_choices).squeeze()
        true_choices = torch.stack(true_choices)

        predictions = model_choices.argmax(2).reshape(-1) if policy == 'greedy' else \
            model_choices.view(
                model_choices.shape[0]*model_choices.shape[1], model_choices.shape[2]).multinomial(1).reshape(-1)
        accuracy = (true_choices.reshape(-1) ==
                    predictions).sum()/(predictions.shape[0])

    if return_all:
        return accuracy, model_choices, true_choices
    else:
        return accuracy


def evaluate_against_baselines(env_name, model_path, mode='val', return_all=False):

    # load environment
    env = CategorisationTask(data=env_name, mode=mode)

    # load models
    _, _, _, done, info = env.reset()
    inputs, targets = info['inputs'], info['targets']
    baseline_model_choices, true_choices, tasks = [], [], []
    num_tasks = targets.shape[0]
    num_trials = env.max_steps

    # loop over dataset making predictions for next trial using model trained on all previous trials
    for task in range(num_tasks):
        # only evaluate on last trial; not possible to evaluate on first trial as it will only have one class
        trial = env.max_steps-1
        # loop over trials
        while trial < num_trials:
            trial_inputs = inputs[task, :trial]
            trial_targets = targets[task, :trial]
            try:
                lr_model = LogisticRegressionModel(trial_inputs, trial_targets)
                svm_model = SVMModel(trial_inputs, trial_targets)
                lr_model_choice = lr_model.predict_proba(
                    inputs[task, trial:trial+1])
                svm_model_choice = svm_model.predict_proba(
                    inputs[task, trial:trial+1])
                true_choice = targets[task, trial:trial+1]
                baseline_model_choices.append(torch.tensor(
                    [lr_model_choice, svm_model_choice]))
                true_choices.append(true_choice)
                tasks.append(task)
            except:
                print('error')
            trial += 1

    # meta-learned model predictions
    _, metal_choice, metal_true_choices = evaluate(
        env_name=env_name, model_path=model_path, mode=mode, policy='greedy', return_all=True)

    # calculate accuracy
    baseline_model_choices, true_choices = torch.stack(
        baseline_model_choices).squeeze().argmax(2), torch.stack(true_choices).squeeze()
    ml2 = (metal_choice.argmax(2)[-1] == metal_true_choices[-1]
           )[tasks].sum()/len(tasks)  # metal_choice.shape[1]
    accuracy = [(baseline_model_choices[:, model_id] ==
                 true_choices).sum()/len(tasks) for model_id in range(2)]
    accuracy.append(ml2)

    # concatenate all model choices and true choices
    all_model_choices = torch.cat(
        [baseline_model_choices, metal_choice.argmax(2)[-1][tasks].unsqueeze(1)], dim=1)
    all_true_choices = true_choices

    # predictions = model_choices.argmax(2).reshape(-1) if policy=='greedy' else \
    #     model_choices.view(model_choices.shape[0]*model_choices.shape[1], model_choices.shape[2]).multinomial(1).reshape(-1)

    if return_all:
        return accuracy, all_model_choices, all_true_choices
    else:
        return accuracy


def evaluate_metalearner(env_name, model_path, experiment='categorisation', mode='test', shuffle_trials=False, beta=1., num_trials=96, num_runs=5, return_choices=False):

    for run_idx in range(num_runs):

        _, model_choices, true_choices, sequences, category_labels = evaluate_1d(env_name=env_name,
                                                                                 model_path=model_path, experiment=experiment, mode=mode, shuffle_trials=shuffle_trials,
                                                                                 beta=beta, return_all=True, max_steps=num_trials)

        cum_sum = np.array(sequences).cumsum()
        model_choices = model_choices.round()
        if run_idx == 0:
            correct = np.ones((num_runs, len(cum_sum), np.diff(
                cum_sum).max()))
            model_choices_unpacked = np.ones((num_runs, len(cum_sum), np.diff(
                cum_sum).max()))
            true_choices_unpacked = np.ones((num_runs, len(cum_sum), np.diff(
                cum_sum).max()))
            labels_unpacked = np.ones((num_runs, len(cum_sum), np.diff(
                cum_sum).max()))

        for task_idx, _ in enumerate(cum_sum[:-1]):
            # task corrects
            task_correct = (model_choices == true_choices.squeeze())[
                cum_sum[task_idx]:cum_sum[task_idx+1]]
            model_choices_unpacked[run_idx, task_idx, :(
                cum_sum[task_idx+1]-cum_sum[task_idx])] = model_choices[cum_sum[task_idx]:cum_sum[task_idx+1]]
            true_choices_unpacked[run_idx, task_idx, :(
                cum_sum[task_idx+1]-cum_sum[task_idx])] = true_choices.squeeze()[cum_sum[task_idx]:cum_sum[task_idx+1]]
            if experiment == 'nosofsky_categorisation':
                labels_unpacked[run_idx, task_idx, :(
                    cum_sum[task_idx+1]-cum_sum[task_idx])] = category_labels.squeeze()[cum_sum[task_idx]:cum_sum[task_idx+1]]
            correct[run_idx, task_idx, :(
                cum_sum[task_idx+1]-cum_sum[task_idx])] = task_correct.numpy()

    # return mean over runs
    correct = correct[..., :num_trials].mean(
        0) if num_trials is not None else correct.mean(0)

    if return_choices:
        return correct, model_choices_unpacked, true_choices_unpacked, labels_unpacked
    else:
        return correct
