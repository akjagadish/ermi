import numpy as np
import torch
from envs import CategorisationTask, ShepardsTask, NosofskysTask, SyntheticCategorisationTask, SmithsTask, JohanssensTask
import pandas as pd
import argparse
SYS_PATH = '/u/ajagadish/ermi'


def simulate(task_feature=None, model_path=None, experiment='categorisation', env=None, model=None, mode='val', shuffle_trials=True, policy='binomial', beta=1., max_steps=100, batch_size=1, device='cpu'):

    if env is None:
        # load environment
        if experiment == 'synthetic':
            env = SyntheticCategorisationTask(
                max_steps=max_steps, shuffle_trials=shuffle_trials)
        if experiment == 'categorisation':
            env = CategorisationTask(
                data=task_feature, mode=mode, max_steps=max_steps, shuffle_trials=shuffle_trials)
        elif experiment == 'shepard_categorisation':
            env = ShepardsTask(task=task_feature, return_prototype=True,
                               batch_size=batch_size, max_steps=max_steps, shuffle_trials=shuffle_trials)
        elif experiment == 'nosofsky_categorisation':
            env = NosofskysTask(task=task_feature)
        elif experiment == 'smith_categorisation':
            env = SmithsTask(rule=task_feature, return_prototype=True, batch_size=batch_size,
                             max_steps=max_steps, shuffle_trials=shuffle_trials, use_existing_stimuli=True)
        elif experiment == 'johanssen_categorisation':
            env = JohanssensTask(block=task_feature, transfer=True, return_prototype=True, batch_size=batch_size,
                                 max_steps=max_steps, shuffle_trials=shuffle_trials, use_existing_stimuli=True)

    if model is None:  # load model
        model = torch.load(model_path)[1].to(device) if device == 'cuda' else torch.load(
            model_path, map_location=torch.device('cpu'))[1].to(device)

    with torch.no_grad():
        model.eval()
        outputs = env.sample_batch()
        if (env.return_prototype is True) and hasattr(env, 'return_prototype'):
            packed_inputs, sequence_lengths, targets, stacked_prototypes = outputs
        else:
            packed_inputs, sequence_lengths, targets = outputs

        # if experiment == 'joahanssen_categorisation':
        #     batch_stimulus_ids, batch_stimulus_names = env.batch_stimulus_ids, env.batch_stimulus_names

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

    model_choices = np.stack([model_choices[i, :seq_len] for i, seq_len in enumerate(sequence_lengths)]).squeeze(
    ) if batch_size > 1 else np.stack([model_choices[i, :seq_len] for i, seq_len in enumerate(sequence_lengths)])
    true_choices = np.stack(targets).squeeze(
    ) if batch_size > 1 else np.stack(targets)
    prototypes = np.stack(stacked_prototypes) if env.return_prototype else None
    input_features = packed_inputs[..., :-1]
    stimulus_dict = env.stimulus_dict if (experiment == 'smith_categorisation') or (
        experiment == 'johanssen_categorisation') else None

    return model_choices, true_choices, sequence_lengths, stimulus_dict, prototypes, input_features


def simulate_metalearners_choices(task_feature, model_path, experiment='categorisation', mode='test', shuffle_trials=False, beta=1., num_trials=100, num_runs=1, batch_size=1, device='cpu'):

    for run_idx in range(num_runs):

        model_choices, true_choices, sequences, stimulus_dict,  prototypes, input_features = simulate(task_feature=task_feature,
                                                                                                      model_path=model_path, experiment=experiment, mode=mode, shuffle_trials=shuffle_trials,
                                                                                                      beta=beta, batch_size=batch_size, max_steps=num_trials, device=device)
        last_task_trial_idx = 0

        # loop over batches, indexing them as tasks in the pd data frame
        for task_idx, (model_choices_task, true_choices_task, sequence_lengths_task, prototypes_task, input_features_task) in enumerate(zip(model_choices, true_choices, sequences, prototypes, input_features)):
            # loop over trials in each batch
            for trial_idx, (model_choice, true_choice, input_feature) in enumerate(zip(model_choices_task, true_choices_task, input_features_task)):
                stimulus_id = [k for k, v in stimulus_dict.items() if v == list(input_feature.numpy())][0] if (
                    experiment == 'smith_categorisation') or (experiment == 'johanssen_categorisation') else None
                data = {'task_feature': task_feature, 'run': run_idx, 'task': task_idx, 'trial': trial_idx + last_task_trial_idx,  'choice': int(model_choice), 'correct_choice': int(true_choice),
                        'category': int(true_choice)+1, 'all_features': str(input_feature.numpy()),
                        'stimulus_id': np.nan if stimulus_id is None else stimulus_id,
                        **{f'feature{i+1}': input_feature[i].numpy() for i in range(len(input_feature))},
                        **{f'prototype_feature{i+1}': prototypes_task[int(true_choice)][i] for i in range(len(prototypes_task[0]))}}

                # make a pandas data frame
                if run_idx == 0 and task_idx == 0 and trial_idx == 0:
                    df = pd.DataFrame(data, index=[0])
                else:
                    df = pd.concat([df, pd.DataFrame(data, index=[0])])
            # (trial_idx + 1) (uncomment: if you want to pool trials across batches into one big task with indexing continuing from previous batch)
            last_task_trial_idx = 0

    return df


def simulate_task(model_name=None, experiment=None, tasks=[None], beta=1., num_runs=1, num_trials=100, batch_size=1, device='cpu'):

    model_path = f"{SYS_PATH}/categorisation/trained_models/{model_name}.pt"
    for task_feature in tasks:

        df = simulate_metalearners_choices(task_feature, model_path, experiment,
                                           beta=beta, shuffle_trials=True, num_runs=num_runs,
                                           batch_size=batch_size, num_trials=num_trials, device=device)

        # concate into one csv
        if task_feature == tasks[0]:
            df_all = df
        else:
            df_all = pd.concat([df_all, df])

    # save to csv
    df_all.to_csv(
        f'{SYS_PATH}/categorisation/data/meta_learner/{experiment}_{model_name[48:]}_beta={beta}_num_trials={num_trials}_num_runs={num_runs}.csv', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='save meta-learner choices on different categorisation tasks')
    parser.add_argument('--no-cuda', action='store_true',
                        default=False, help='disables CUDA training')
    parser.add_argument('--experiment', type=str,
                        required=True, help='task name')
    parser.add_argument('--model-name', type=str,
                        required=True, help='model name')
    parser.add_argument('--beta', type=float, default=1.,
                        help='beta value for softmax')
    parser.add_argument('--job-id', type=int, default=None, help='job id')
    parser.add_argument('--num-runs', type=int,
                        default=1, help='number of runs')

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    beta = args.beta

    if args.experiment == 'shepard_categorisation':
        simulate_task(model_name=args.model_name, experiment=args.experiment, tasks=np.arange(
            1, 7), beta=beta, num_runs=args.num_runs, batch_size=1, num_trials=100, device=device)  # 1000
    elif args.experiment == 'smith_categorisation':
        simulate_task(model_name=args.model_name,  experiment=args.experiment, tasks=[
                      'nonlinear'], beta=beta, num_runs=args.num_runs, batch_size=1, num_trials=616, device=device)  # 300, rule='linear',
        use_existing_stimuli = True
        # TODO: pass in use_existing_stimuli to simulate_task
    elif args.experiment == 'johanssen_categorisation':
        if args.job_id is not None:
            beta = args.job_id/10
        simulate_task(model_name=args.model_name,  experiment=args.experiment, tasks=[
                      1, 2, 3, 4, 5, 6, 8, 16, 24, 32], beta=beta, num_runs=args.num_runs, batch_size=68*8, num_trials=288, device=device)
        # batch_size = 68*10 [num_participants * num_times_transfer_stimuli_presented (set to 10>>7 transfer stimuli as we are sampling transfer stimuli in each block)]
        # beta = 0.18564321475504575 (badham et al. 2017), 0.09495665880995116 (devraj et al. 2002)
        # task_blocks from the paper [2, 4, 8, 16, 24, 32]
