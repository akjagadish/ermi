import numpy as np
import torch
from envs import Binz2022, Badham2017, Devraj2022
import argparse
from tqdm import tqdm
from scipy.optimize import differential_evolution, minimize
import sys
import re
from model import TransformerDecoderClassification, TransformerDecoderLinearWeights
from model_utils import parse_model_path
from torch.distributions import Bernoulli
sys.path.insert(0, '/u/ajagadish/ermi/mi')
SYS_PATH = '/u/ajagadish/ermi'


def compute_loglikelihood_human_choices_under_model(env, model, participant=0, beta=1., epsilon=0., method='soft_sigmoid', policy='greedy', device='cpu', paired=False, **kwargs):

    with torch.no_grad():

        # model setup: eval mode and set beta
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

        if method == 'eps_greedy' or method == 'both':
            # make a new tensor containing model_choice_probs for each trial for option 1 and 1-model_choice_probs for option 2
            probs = torch.cat(
                [1-model_choice_probs, model_choice_probs], axis=2)
            # keep only the probabilities for the chosen option from human_choices
            probs = torch.vstack([probs[batch, i, human_choices[batch, i, 0].long(
            )] for batch in range(probs.shape[0]) for i in range(sequence_lengths[batch])])
            probs_with_guessing = probs * (1 - epsilon) + epsilon * 0.5
            summed_loglikelihoods = torch.log(probs_with_guessing).sum()

        elif method == 'soft_sigmoid':

            assert epsilon == 0., "epsilon must be 0 for soft_sigmoid"
            # compute log likelihoods of human choices under model choice probs (binomial distribution)
            loglikehoods = Bernoulli(
                    probs=model_choice_probs).log_prob(human_choices.float())
            summed_loglikelihoods = torch.vstack(
                [loglikehoods[idx, :sequence_lengths[idx]].sum() for idx in range(len(loglikehoods))]).sum()
       
        # sum log likelihoods only for unpadded trials per condition and compute chance log likelihood
        chance_loglikelihood = sum(sequence_lengths) * np.log(0.5)

        # task performance
        model_choices = torch.concat([model_choices[i, :seq_len] for i, seq_len in enumerate(
            sequence_lengths)], axis=0).squeeze().float()
        correct_choices = torch.concat([correct_choices[i, :seq_len] for i, seq_len in enumerate(
            sequence_lengths)], axis=0).squeeze().float()
        correct_choices = correct_choices.reshape(-1).float().to(device)
        model_accuracy = (model_choices == correct_choices).sum() / \
            correct_choices.numel()
    
    return summed_loglikelihoods, chance_loglikelihood, model_accuracy

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
    
    # parse model parameters
    num_hidden, num_layers, d_model, num_head, loss_fn, model_max_steps = parse_model_path(model_path, task_features)
    
    # initialise model
    if args.paired:
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

    def objective(x, participant):
        epsilon = x[0] if args.method == 'eps_greedy' else 0.
        beta = x[0] if args.method == 'soft_sigmoid' else 1.
        if args.method == 'both':
            epsilon = x[0]
            beta = x[1]
        ll, _, _ = compute_loglikelihood_human_choices_under_model(env=env, model=model, participant=participant, shuffle_trials=True,
                                                                   beta=beta, epsilon=epsilon, method=args.method, paired=args.paired, ** task_features)
        return -ll.numpy()

    if args.method == 'soft_sigmoid':
        bounds = [(0., 1.)]
    elif args.method == 'eps_greedy':
        bounds = [(0., 1.)]
    elif args.method == 'both':
        bounds = [(0., 1.), (0., 1.)]
    else:
        raise NotImplementedError

    pr2s, nlls, accs, parameters = [], [], [], []
    participants = env.data.participant.unique()
    for participant in participants:
        res_fun = np.inf
        for _ in tqdm(range(args.num_iters)):

            # x0 = [np.random.uniform(x, y) for x, y in bounds]
            # result = minimize(objective, x0, args=(participant), bounds=bounds, method='SLSQP')
            result = differential_evolution(
                func=objective, args=(participant,), bounds=bounds)

            if result.fun < res_fun:
                res_fun = result.fun
                res = result
                print(f"min nll and parameter: {res_fun, res.x}")

        epsilon = res.x[0] if args.method == 'eps_greedy' else 0.
        beta = res.x[0] if args.method == 'soft_sigmoid' else 1.
        if args.method == 'both':
            epsilon = res.x[0]
            beta = res.x[1]

        ll, chance_ll, acc = compute_loglikelihood_human_choices_under_model(env=env, model=model, participant=participant, shuffle_trials=True,
                                                                             beta=beta, epsilon=epsilon, method=args.method, paired=args.paired, **task_features)
        nlls.append(-ll)
        pr2s.append(1 - (ll/chance_ll))
        accs.append(acc)
        parameters.append(res.x)

    return np.array(pr2s), np.array(nlls), accs, parameters


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
    parser.add_argument('--method', type=str, default='soft_sigmoid',
                        help='method for computing model choice probabilities')
    parser.add_argument('--num-iters', type=int, default=5)
    parser.add_argument('--paired', action='store_true',
                        default=False, help='paired')

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")


    pr2s, nlls, accs, parameters = optimize(args)
    optimizer = 'de'
    
    # save list of results
    num_hidden, num_layers, d_model, num_head, loss_fn, _, source, condition = parse_model_path(args.model_name, {}, return_data_info=True)
    save_path = f"{args.paradigm}/data/model_comparison/task={args.task_name}_experiment={args.exp_id}_source={source}_condition={condition}_loss={loss_fn}_paired={args.paired}_method={args.method}_optimizer={optimizer}.npz"
    np.savez(save_path, betas=parameters, nlls=nlls, pr2s=pr2s, accs=accs)
