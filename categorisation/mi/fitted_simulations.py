import numpy as np
import torch
from human_envs import Badham2017, Devraj2022
import argparse
from tqdm import tqdm
from scipy.optimize import differential_evolution, minimize
import sys
import pandas as pd
SYS_PATH = '/u/ajagadish/ermi/'
sys.path.insert(0, f"{SYS_PATH}/categorisation/mi")


def simulate_metalearner_with_fitted_coefficients(env=None, model_path=None, participant=0, beta=1., epsilon=0., method='soft_sigmoid', device='cpu', **kwargs):

    # load model
    model = torch.load(model_path)[1].to(device) if device == 'cuda' else torch.load(
        model_path, map_location=torch.device('cpu'))[1].to(device)

    with torch.no_grad():

        # model setup: eval mode and set beta
        model.eval()
        model.beta = torch.tensor(beta).to(device)
        model.device = device

        # env setup: sample batch from environment and unpack
        outputs = env.sample_batch(participant)

        if (env.return_prototype is True) and hasattr(env, 'return_prototype'):
            packed_inputs, sequence_lengths, correct_choices, human_choices, stacked_prototypes, stacked_stimulus_ids = outputs
        else:
            packed_inputs, sequence_lengths, correct_choices, human_choices, stacked_stimulus_ids = outputs

        # set human choices to correct choices
        # human_choices = correct_choices
        # randomise human choices
        # human_choices = torch.randint(0, 2, human_choices.shape).to(device)

        # get model choices
        model_choice_probs = model(
            packed_inputs.float().to(device), sequence_lengths)

        if method == 'eps_greedy' or method == 'both':
            assert beta == 1. if method == 'eps_greedy' else True, "beta must be 1 for eps_greedy"
            model_choice_probs = model_choice_probs * \
                (1 - epsilon) + epsilon * 0.5

        # task measures
        model_choices = torch.distributions.Binomial(
            probs=model_choice_probs).sample()
        # remove the last column which is the preivous choice
        input_features = packed_inputs[..., :-1]
        prototypes = np.stack(
            stacked_prototypes) if env.return_prototype else torch.empty_like(input_features)
        stimulus_ids = stacked_stimulus_ids if stacked_stimulus_ids is not None else torch.empty_like(
            correct_choices)

        last_task_trial_idx = 0
        # loop over batches, indexing them as tasks in the pd data frame
        for task_idx, (model_choices_task, correct_choices_task, sequence_lengths_task, prototypes_task, stimulus_ids_task, input_features_task) in enumerate(zip(model_choices, correct_choices, sequence_lengths, prototypes, stimulus_ids, input_features)):
            # loop over trials in each batch
            for trial_idx, (model_choice, correct_choice, stimulus_id, input_feature) in enumerate(zip(model_choices_task, correct_choices_task, stimulus_ids_task, input_features_task)):
                data = {'task': 0, 'trial': trial_idx + last_task_trial_idx, 'condition': task_idx+1, 'choice': int(model_choice), 'correct_choice': int(correct_choice),
                        'category': int(correct_choice)+1, 'all_features': str(input_feature.numpy()),
                        'stimulus_id': int(stimulus_id) if stacked_stimulus_ids is not None else np.nan,
                        **{f'feature{i+1}': input_feature[i].numpy() for i in range(len(input_feature))},
                        }

                # append prototpyes if they exist
                data = {**data,  **{f'prototype_feature{i+1}': prototypes_task[int(
                    correct_choice)][i] for i in range(len(prototypes_task[0]))}} if env.return_prototype else data

                # make a pandas data frame
                df = pd.DataFrame(data, index=[0]) if task_idx == 0 and trial_idx == 0 else pd.concat(
                    [df, pd.DataFrame(data, index=[0])])

            # last_task_trial_idx = (trial_idx + 1) (uncomment: if you want to pool trials across batches into one big task with indexing continuing from previous batch)

    model_accuracy = (df.choice.values == df.correct_choice.values).mean()
    print(
        f"model accuracy, beta, epsilon, method: {model_accuracy}, {beta}, {epsilon}, {method}")

    return df


def simulate_metalearner(args):

    def objective(env=None, model_name=None, method='soft_sigmoid', num_runs=1, **task_features):
        '''  pool data from all participants
        '''

        model_path = f"{SYS_PATH}/categorisation/trained_models/{model_name}.pt"
        parameters_path = f"{SYS_PATH}/categorisation/data/model_comparison/{args.task_name}_{args.model_name}_{args.method}_{args.optimizer}.npz"
        parameters = np.load(parameters_path)['betas']
        participants = env.data.participant.unique()
        assert len(participants) == len(
            parameters), "number of participants and parameters must be equal"

        df = None
        for idx, participant in enumerate(participants):
            epsilon = parameters[idx] if args.method == 'eps_greedy' else 0.
            beta = parameters[idx] if args.method == 'soft_sigmoid' else 1.
            df = simulate_metalearner_with_fitted_coefficients(env=env, model_path=model_path, participant=participant, shuffle_trials=True,
                                                               beta=beta, epsilon=epsilon, method=method, **task_features)
            df['participant'] = participant
            # concate into one csv
            if participant == participants[0]:
                df_all = df
            else:
                df_all = pd.concat([df_all, df])

        return df_all

    if args.task_name == 'badham2017':
        env = Badham2017()
        task_features = {}
        df = objective(env=env, model_name=args.model_name,
                       method=args.method, num_runs=1, **task_features)
    elif args.task_name == 'devraj2022':
        env = Devraj2022(return_prototype=True)
        df = objective(env=env, model_name=args.model_name,
                       method=args.method, num_runs=1)
    else:
        raise NotImplementedError

    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='save meta-learner choices on different categorisation tasks')
    parser.add_argument('--no-cuda', action='store_true',
                        default=False, help='disables CUDA training')
    parser.add_argument('--task-name', type=str,
                        required=True, help='task name')
    parser.add_argument('--model-name', type=str,
                        required=True, help='model name')
    parser.add_argument('--method', type=str, default='soft_sigmoid',
                        help='method for computing model choice probabilities')
    parser.add_argument('--optimizer', choices=['differential_evolution', 'grid_search'],
                        default='differential_evolution', help='method for computing model choice probabilities')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    df = simulate_metalearner(args)
    save_path = f"{SYS_PATH}/categorisation/data/fitted_simulation/{args.task_name}_{args.model_name}_{args.method}.csv"
    df.to_csv(save_path, index=False)
