import numpy as np
import torch
from envs import Binz2022, Badham2017, Devraj2022
import argparse
from tqdm import tqdm
from scipy.optimize import differential_evolution, minimize
from model import TransformerDecoderClassification, TransformerDecoderLinearWeights
import sys
import re
sys.path.insert(0, '/u/ajagadish/ermi/mi')
SYS_PATH = '/u/ajagadish/ermi'


def compute_loglikelihood_human_choices_under_model(env=None, model_path=None, participant=0, beta=1., epsilon=0., method='soft_sigmoid', device='cpu', paired=False, **kwargs):

    # parse num_hidden, num_layers, d_model, num_head, noise, shuffle, paired, loss from model_path
    patterns = {
        "num_hidden": r"num_hidden=(\d+)",
        "num_layers": r"num_layers=(\d+)",
        "d_model": r"d_model=(\d+)",
        "num_head": r"num_head=(\d+)",
        "noise": r"noise=(\d+\.\d+)",
        "shuffle": r"shuffle=(True|False)",
        "paired": r"paired=(True|False)",
        "loss": r"loss(\w+)"
    }

    # Initialize a dictionary to store the parsed parameters
    parameters = {}

    # Parse each parameter from the model_path string
    for param, pattern in patterns.items():
        match = re.search(pattern, model_path)
        if match:
            parameters[param] = match.group(1)

    num_hidden = int(parameters.get('num_hidden', 0))
    num_layers = int(parameters.get('num_layers', 0))
    d_model = int(parameters.get('d_model', 0))
    num_head = int(parameters.get('num_head', 0))
    loss_fn = 'nll'  # parameters.get('loss', 'nll')
    # TODO: does not work other than for binz2022
    model_max_steps = kwargs.get('model_max_steps', 10)
    # load model
    if paired:
        model = TransformerDecoderLinearWeights(num_input=env.num_dims, num_output=env.num_choices, num_hidden=num_hidden,
                                                num_layers=num_layers, d_model=d_model, num_head=num_head, max_steps=model_max_steps, loss=loss_fn, device=device).to(device)

    else:
        model = TransformerDecoderClassification(num_input=env.num_dims, num_output=env.num_choices, num_hidden=num_hidden,
                                                 num_layers=num_layers, d_model=d_model, num_head=num_head, max_steps=model_max_steps, loss=loss_fn, device=device).to(device)
    state_dict = torch.load(
        model_path, map_location=torch.device('cpu'))[1]
    model.load_state_dict(state_dict)
    model.to(device)

    with torch.no_grad():

        # model setup: eval mode and set beta
        model.eval()
        model.beta = beta
        model.device = device

        # env setup: sample batch from environment and unpack
        outputs = env.sample_batch(participant, paired=paired)

        if not hasattr(env, 'return_prototype'):
            packed_inputs, sequence_lengths, correct_choices, human_choices, _ = outputs
        elif hasattr(env, 'return_prototype') and (env.return_prototype is True):
            packed_inputs, sequence_lengths, correct_choices, human_choices, stacked_prototypes, _ = outputs

        # get model choices
        model_choice_probs = model(
            packed_inputs.float().to(device), sequence_lengths)

        # task performance
        if model.loss == 'bce':
            model_choices = model_choice_probs > 0.5 if method == 'greedy' else torch.distributions.Binomial(
                probs=model_choice_probs).sample()

        elif model.loss == 'nll':
            model_choices = model_choice_probs.mean > 0.5 if method == 'greedy' else model_choice_probs.sample()

        per_trial_model_accuracy = (model_choices == correct_choices)
        per_trial_human_accuracy = (human_choices == correct_choices)
        model_choices = torch.concat([model_choices[i, :seq_len] for i, seq_len in enumerate(
            sequence_lengths)], axis=0).squeeze().float()
        correct_choices = torch.concat([correct_choices[i, :seq_len] for i, seq_len in enumerate(
            sequence_lengths)], axis=0).squeeze().float()
        correct_choices = correct_choices.reshape(-1).float().to(device)
        model_accuracy = (model_choices == correct_choices).sum() / \
            correct_choices.numel()
        human_accuracy = (human_choices.reshape(-1) ==
                          correct_choices).sum() / correct_choices.numel()

    return model_accuracy, per_trial_model_accuracy, per_trial_human_accuracy


def optimize(args):

    model_path = f"{SYS_PATH}/{args.paradigm}/trained_models/{args.model_name}.pt"
    if args.task_name == 'badham2017':
        env = Badham2017()
        task_features = {}
    elif args.task_name == 'devraj2022':
        env = Devraj2022()
        task_features = {}
    elif args.task_name == 'binz2022':
        env = Binz2022()
        task_features = {'model_max_steps': 10}
    else:
        raise NotImplementedError

    per_trial_accs, human_accs, accs = [], [], []
    participants = env.data.participant.unique()
    for participant in participants:
        beta, epsilon = 1., 0.
        model_accuracy, per_trial_model_accuracy, human_accuracy = compute_loglikelihood_human_choices_under_model(env=env, model_path=model_path, participant=participant, shuffle_trials=True,
                                                                                                                   beta=beta, epsilon=epsilon, method=args.method, paired=args.paired, **task_features)
        human_accs.append(human_accuracy)
        per_trial_accs.append(per_trial_model_accuracy)
        accs.append(model_accuracy)

    return np.array(accs), torch.stack(per_trial_accs).squeeze().sum(1), np.stack(human_accs).squeeze().sum(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='save meta-learners choices for a given task within a paradigm')
    parser.add_argument('--no-cuda', action='store_true',
                        default=False, help='disables CUDA training')
    parser.add_argument('--paradigm', type=str, default='categorisation')
    parser.add_argument('--task-name', type=str,
                        required=True, help='task name')
    parser.add_argument('--model-name', type=str,
                        required=True, help='model name')
    parser.add_argument('--paired', action='store_true',
                        default=False, help='paired')
    parser.add_argument('--method', type=str, default='greedy',
                        help='method to use for computing model choices')

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    model_accuracy, per_trial_model_accuracy, human_accuracy = optimize(args)

    # save list of results
    save_path = f"{SYS_PATH}/{args.paradigm}/data/evaluation/evaluate_{args.task_name}_{args.model_name[:200]}_paired{args.paired}_method{args.method}.npz"
    np.savez(save_path, model_accuracy=model_accuracy,
             per_trial_model_accuracy=per_trial_model_accuracy, human_accuracy=human_accuracy)
