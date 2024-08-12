import numpy as np
import torch
from envs import Binz2022, Badham2017, Devraj2022, Little2022
import argparse
from tqdm import tqdm
from scipy.optimize import differential_evolution, minimize
from model import TransformerDecoderClassification, TransformerDecoderLinearWeights, TransformerDecoderRegression, TransformerDecoderRegressionLinearWeights
import sys
import re
from model_utils import parse_model_path
from torch.distributions import Bernoulli
sys.path.insert(0, '/u/ajagadish/ermi/mi')
SYS_PATH = '/u/ajagadish/ermi'

def compute_mse(x, y, axis=1):
    return ((x - y) ** 2).mean(axis=axis)

def compute_loglikelihood_human_choices_under_model(env=None, model_path=None, participant=0, beta=1., epsilon=0., policy='greedy', device='cpu', paired=False, **kwargs):

    # parse model parameters
    num_hidden, num_layers, d_model, num_head, loss_fn, model_max_steps = parse_model_path(model_path, kwargs)

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
            packed_inputs, sequence_lengths, correct_choices, human_choices, _, _ = outputs

        # get model choices
        model_choice_probs = model(
            packed_inputs.float().to(device), sequence_lengths)
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
        model_coefficients = model.w.detach().numpy() if paired else None

    return model_accuracy, per_trial_model_accuracy, human_accuracy, per_trial_human_accuracy, model_coefficients

def compute_mses_human_predictions_under_model(env=None, model_path=None, participant=0, device='cpu', paired=False, policy='greedy', **kwargs):

    # parse model parameters
    num_hidden, num_layers, d_model, num_head, loss_fn, model_max_steps = parse_model_path(model_path, kwargs)

    # initialise model
    if paired:
        model = TransformerDecoderRegressionLinearWeights(num_input=env.num_dims, num_output=env.num_choices, num_hidden=num_hidden,
                                                    num_layers=num_layers, d_model=d_model, num_head=num_head, max_steps=model_max_steps, loss=loss_fn, device=device).to(device)

    else:
        model = TransformerDecoderRegression(num_input=env.num_dims, num_output=env.num_choices, num_hidden=num_hidden,
                                                 num_layers=num_layers, d_model=d_model, num_head=num_head, max_steps=model_max_steps, loss=loss_fn, device=device).to(device)
    
    #load model weights
    state_dict = torch.load(
        model_path, map_location=torch.device('cpu'))[1]
    model.load_state_dict(state_dict)
    model.to(device)

    with torch.no_grad():

        # model setup: eval mode and set beta
        model.eval()
        model.device = device

        # env setup: sample batch from environment and unpack
        outputs = env.sample_batch(participant, paired=paired)
        packed_inputs, sequence_lengths, targets, human_preds, ground_truth_functions = outputs
        # packed_inputs: (40*24)X25X1
        
        # get model preds
        model_preds = model(
            packed_inputs.float().to(device), sequence_lengths)
        model_preds = model_preds.mean if policy == 'greedy' else model_preds.sample()

        # compute metrics
        model_error = compute_mse(model_preds[:, -1], targets[:, -1])
        model_preds = torch.concat([model_preds[i, [-1]] for i, _ in enumerate(
            sequence_lengths)], axis=0).squeeze().float()
        targets = torch.concat([targets[i, [-1]] for i, _ in enumerate(
            sequence_lengths)], axis=0).squeeze().float()
        targets = targets.reshape(-1).float().to(device)

    return model_preds, model_error, targets, human_preds, ground_truth_functions


def sample_model(args):

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
    elif args.task_name == 'little2022':
        env = Little2022()
        task_features = {'model_max_steps': 25}
    else:
        raise NotImplementedError
   
    participants = env.data.participant.unique()
    
    if args.task_name in ['little2022']:

        model_errors, model_preds, targets, human_preds, ground_truth_functions = [], [], [], [], []
        for participant in participants:
            model_pred, model_error, target, human_pred, ground_truth_function = compute_mses_human_predictions_under_model(env=env, model_path=model_path, participant=participant, shuffle_trials=True,
                                                                                                                    paired=args.paired, **task_features)
            model_preds.append(model_pred)
            model_errors.append(model_error)
            targets.append(target)
            human_preds.append(human_pred)
            ground_truth_functions.append(ground_truth_function)

        return np.stack(model_preds), np.stack(model_errors), np.stack(targets), np.stack(human_preds), np.stack(ground_truth_functions)
        
    else:

        per_trial_accs, per_trial_human_accs, human_accs, accs, coeffs = [], [], [], [], []
        for participant in participants:  
            beta, epsilon = 1., 0.
            model_accuracy, per_trial_model_accuracy, human_accuracy, per_trial_human_accuracy, model_coeffs = compute_loglikelihood_human_choices_under_model(env=env, model_path=model_path, participant=participant, shuffle_trials=True,
                                                                                                                    beta=beta, epsilon=epsilon, policy=args.policy, paired=args.paired, **task_features)
            human_accs.append(human_accuracy)
            per_trial_accs.append(per_trial_model_accuracy)
            per_trial_human_accs.append(per_trial_human_accuracy)
            accs.append(model_accuracy)
            coeffs.append(model_coeffs)
        
        return np.array(accs), torch.stack(per_trial_accs).squeeze().sum(1), np.array(human_accs), np.stack(per_trial_human_accs).squeeze().sum(1), np.stack(coeffs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='save meta-learners choices for a given task within a paradigm')
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
    parser.add_argument('--ess', type=float, default=None,
                         help='weight for the nll loss term in the ELBO')
    parser.add_argument('--job-array', action='store_true',
                        default=False, help='passed as a job array')
    parser.add_argument('--scale', type=int, default=10000,
                        help='scale for the job array')
    parser.add_argument('--offset', type=int, default=0,
                        help='offset for the job array')

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    args.device = torch.device("cuda" if use_cuda else "cpu")
    
    num_hidden, num_layers, d_model, num_head, loss_fn, _, source, condition = parse_model_path(args.model_name, {}, return_data_info=True)
    save_path = f"{args.paradigm}/data/model_simulation/task={args.task_name}_experiment={args.exp_id}_source={source}_condition={condition}_loss={loss_fn}_paired={args.paired}_policy={args.policy}.npz"
    args.ess = args.ess * args.scale + args.offset if args.job_array else args.ess
    save_path = save_path.replace('.npz', f"_ess={str(int(args.ess))}.npz")
    args.model_name = args.model_name.replace('essNone', f'ess{str(int(args.ess))}')
    
    if args.paradigm == 'functionlearning':
        model_preds, model_errors, targets, human_preds, ground_truth_functions = sample_model(args)
        # save list of results
        np.savez(save_path, model_preds=model_preds, model_errors=model_errors, targets=targets, 
                 human_preds=human_preds, ground_truth_functions=ground_truth_functions)
    else:
        model_accuracy, per_trial_model_accuracy, human_accuracy, per_trial_human_accuracy, model_coefficients = sample_model(args)
        print('saving')
        # save list of results
        np.savez(save_path, model_accuracy=model_accuracy, per_trial_model_accuracy=per_trial_model_accuracy, 
                 human_accuracy=human_accuracy, per_trial_human_accuracy=per_trial_human_accuracy, model_coefficients=model_coefficients)
