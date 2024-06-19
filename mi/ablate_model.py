import numpy as np
import torch
from envs import Binz2022, Badham2017, Devraj2022
import argparse
from tqdm import tqdm
from scipy.optimize import differential_evolution, minimize
from model import TransformerDecoderClassification, TransformerDecoderLinearWeights
import sys
import re
from model_utils import parse_model_path
from torch.distributions import Bernoulli
from nnsight import NNsight
sys.path.insert(0, '/u/ajagadish/ermi/mi')
SYS_PATH = '/u/ajagadish/ermi'


def compute_loglikelihood_human_choices_under_model(env=None, model_path=None, participant=0, beta=1., epsilon=0., policy='greedy', device='cpu', paired=False, percent_neurons=10, layer='linear', **kwargs):

    # parse model parameters
    num_hidden, num_layers, d_model, num_head, loss_fn, model_max_steps = parse_model_path(model_path, kwargs)

    # initialise model
    if paired:
        net = TransformerDecoderLinearWeights(num_input=env.num_dims, num_output=env.num_choices, num_hidden=num_hidden,
                                                num_layers=num_layers, d_model=d_model, num_head=num_head, max_steps=model_max_steps, loss=loss_fn, device=device).to(device)

    else:
        net = TransformerDecoderClassification(num_input=env.num_dims, num_output=env.num_choices, num_hidden=num_hidden,
                                                 num_layers=num_layers, d_model=d_model, num_head=num_head, max_steps=model_max_steps, loss=loss_fn, device=device).to(device)
    
    # load model weights
    state_dict = torch.load(
        model_path, map_location=device)[1]
    
    # set the values of percent_neurons of all values within the state_dict to 0
    for key in state_dict.keys():
        state_dict[key][..., [np.random.choice(state_dict[key].shape[-1], int(state_dict[key].shape[-1] * percent_neurons / 100), replace=False)]] = 0

    net.load_state_dict(state_dict)
    net.to(device)

    with torch.no_grad():

        # model setup: eval mode and set beta
        net.eval()
        net.beta = beta
        net.device = device  

    
        # env setup: sample batch from environment and unpack
        outputs = env.sample_batch(participant, paired=paired)

        if not hasattr(env, 'return_prototype'):
            packed_inputs, sequence_lengths, correct_choices, human_choices, _ = outputs
        elif hasattr(env, 'return_prototype') and (env.return_prototype is True):
            packed_inputs, sequence_lengths, correct_choices, human_choices, _, _ = outputs

        # get model choiceså
        input = (packed_inputs.float().to(device), sequence_lengths)
        model_choice_probs = net(*input)
        # compute model choices
        model_choices = model_choice_probs.round() if policy == 'greedy' else Bernoulli(
                    probs=model_choice_probs).sample()

        # compute metrics
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

    return model_accuracy, per_trial_model_accuracy, human_accuracy, per_trial_human_accuracy


def optimize(args):

    model_path = f"{SYS_PATH}/{args.paradigm}/trained_models/{args.model_name}.pt"
    if args.task_name == 'badham2017':
        env = Badham2017()
        task_features = {'model_max_steps': 96}
    elif args.task_name == 'devraj2022':
        env = Devraj2022()
        task_features = {'model_max_steps': 616}
    elif args.task_name == 'binz2022':
        env = Binz2022(experiment_id=args.exp_id)
        task_features = {'model_max_steps': 10}
    else:
        raise NotImplementedError

    per_trial_accs, per_trial_human_accs, human_accs, accs = [], [], [], []
    participants = env.data.participant.unique()
    for participant in participants:
        beta, epsilon = 1., 0.
        model_accuracy, per_trial_model_accuracy, human_accuracy, per_trial_human_accuracy = compute_loglikelihood_human_choices_under_model(env=env, model_path=model_path, participant=participant, shuffle_trials=True,
                                                                                                                   beta=beta, epsilon=epsilon, policy=args.policy, paired=args.paired, percent_neurons=args.percent_neurons, 
                                                                                                                    layer=args.layer, **task_features)
        human_accs.append(human_accuracy)
        per_trial_accs.append(per_trial_model_accuracy)
        per_trial_human_accs.append(per_trial_human_accuracy)
        accs.append(model_accuracy)

    return np.array(accs), torch.stack(per_trial_accs).squeeze().sum(1), np.array(human_accs), np.stack(per_trial_human_accs).squeeze().sum(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='simulate ablated meta-learners on given task within a paradigm')
    parser.add_argument('--no-cuda', action='store_true',
                        default=False, help='disables CUDA training')
    parser.add_argument('--paradigm', type=str, default='categorisation')
    parser.add_argument('--task-name', type=str,
                        required=True, help='task name')
    parser.add_argument('--exp-id', type=int, default=1, help='experiment id')
    parser.add_argument('--model-name', type=str,
                        required=True, help='model name')
    parser.add_argument('--paired', action='store_true',
                        default=False, help='paired')
    parser.add_argument('--policy', type=str, default='greedy',
                        help='method to use for computing model choices')
    parser.add_argument('--percent-neurons', type=int, default=10, choices=range(1, 101),
                        help='percentage of hidden units to ablate')
    parser.add_argument('--layer', type=str, default='last_linear',
                        help='from which layer(s) to ablate hidden units')

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    args.device = torch.device("cuda" if use_cuda else "cpu")

    model_accuracy, per_trial_model_accuracy, human_accuracy, per_trial_human_accuracy = optimize(args)

    num_hidden, num_layers, d_model, num_head, loss_fn, _, source, condition = parse_model_path(args.model_name, {}, return_data_info=True)
    
    # save list of results
    save_path = f"{args.paradigm}/data/model_ablation/task={args.task_name}_experiment={args.exp_id}_source={source}_condition={condition}_loss={loss_fn}_paired={args.paired}_policy={args.policy}_percentneurons={args.percent_neurons}_layer={args.layer}.npz"
    np.savez(save_path, model_accuracy=model_accuracy,
             per_trial_model_accuracy=per_trial_model_accuracy, human_accuracy=human_accuracy, per_trial_human_accuracy=per_trial_human_accuracy)
