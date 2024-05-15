import numpy as np
import torch
from envs import Binz2022, Badham2017, Devraj2022
import argparse
from tqdm import tqdm
from scipy.optimize import differential_evolution, minimize
import sys
sys.path.insert(0, '/u/ajagadish/ermi/mi')
SYS_PATH = '/u/ajagadish/ermi'


def compute_loglikelihood_human_choices_under_model(env=None, model_path=None, participant=0, beta=1., epsilon=0., method='soft_sigmoid', device='cpu', paired=False, **kwargs):

    # load model
    model = torch.load(model_path)[1].to(device) if device == 'cuda' else torch.load(
        model_path, map_location=torch.device('cpu'))[1].to(device)

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

        # set human choices to correct choices
        # human_choices = correct_choices
        # randomise human choices
        # human_choices = torch.randint(0, 2, human_choices.shape).to(device)

        # get model choices
        model_choice_probs = model(
            packed_inputs.float().to(device), sequence_lengths)

        if method == 'eps_greedy' or method == 'both':

            # make a new tensor containing model_choice_probs for each trial for option 1 and 1-model_choice_probs for option 2
            probs = torch.cat(
                [1-model_choice_probs, model_choice_probs], axis=2)
            # keep only the probabilities for the chosen option from human_choices
            probs = torch.vstack([probs[batch, i, human_choices[batch, i, 0].long(
            )] for batch in range(probs.shape[0]) for i in range(sequence_lengths[batch])])
            probs_with_guessing = probs * (1 - epsilon) + epsilon * 0.5
            loglikehoods = torch.log(probs_with_guessing)
            summed_loglikelihoods = loglikehoods.sum()

        elif method == 'soft_sigmoid':

            assert epsilon == 0., "epsilon must be 0 for soft_sigmoid"
            # compute log likelihoods of human choices under model choice probs (binomial distribution)
            loglikehoods = torch.distributions.Binomial(
                probs=model_choice_probs).log_prob(human_choices)
            summed_loglikelihoods = torch.vstack(
                [loglikehoods[idx, :sequence_lengths[idx]].sum() for idx in range(len(loglikehoods))]).sum()

        # sum log likelihoods only for unpadded trials per condition and compute chance log likelihood
        chance_loglikelihood = sum(sequence_lengths) * np.log(0.5)

        # task performance
        model_choices = torch.distributions.Binomial(
            probs=model_choice_probs).sample()
        model_choices = torch.concat([model_choices[i, :seq_len] for i, seq_len in enumerate(
            sequence_lengths)], axis=0).squeeze().float()
        correct_choices = torch.concat([correct_choices[i, :seq_len] for i, seq_len in enumerate(
            sequence_lengths)], axis=0).squeeze().float()
        correct_choices = correct_choices.reshape(-1).float().to(device)
        model_accuracy = (model_choices == correct_choices).sum() / \
            correct_choices.numel()
        # TODO: save human accs per participant
        # human_accuracy = (human_choices.reshape(-1) ==
        #                   correct_choices).sum() / correct_choices.numel()

    return summed_loglikelihoods, chance_loglikelihood, model_accuracy


def grid_search(args):

    if args.method == 'soft_sigmoid':
        # beta sweep
        betas = np.arange(0., 10., 0.05)
        parameters = betas

    elif args.method == 'eps_greedy':
        # epsilon sweep
        epsilons = np.arange(0., 1., 0.05)
        parameters = epsilons

    else:
        raise NotImplementedError

    def objective(env=None, model_name=None, beta=1., epsilon=0., method='soft_sigmoid', num_runs=1, paradigm='categorisation', **task_features):
        '''  compute log likelihoods of human choices under model choice probs based on binomial distribution
        '''

        model_path = f"{SYS_PATH}/{paradigm}/trained_models/{model_name}.pt"
        participants = env.data.participant.unique()
        loglikelihoods, p_r2, model_acc = [], [], []
        for participant in participants:
            ll, chance_ll, acc = compute_loglikelihood_human_choices_under_model(env=env, model_path=model_path, participant=participant, shuffle_trials=True,
                                                                                 beta=beta, epsilon=epsilon, method=method, **task_features)
            loglikelihoods.append(ll)
            p_r2.append(1 - (ll/chance_ll))
            model_acc.append(acc)

        loglikelihoods = np.array(loglikelihoods)
        p_r2 = np.array(p_r2)
        model_acc = np.array(model_acc)

        return -loglikelihoods, p_r2, model_acc

    nlls, pr2s, accs = [], [], []
    for idx, param in enumerate(parameters):
        epsilon = param if args.method == 'eps_greedy' else 0.
        beta = param if args.method == 'soft_sigmoid' else 1.
        if args.task_name == 'badham2017':
            env = Badham2017()
            # TODO: for task in tasks:
            task_features = {}
            nll_per_beta, pr2_per_beta, model_acc_per_beta = objective(
                env=env, model_name=args.model_name, epsilon=epsilon, beta=beta, method=args.method, num_runs=1, **task_features)
        elif args.task_name == 'devraj2022':
            env = Devraj2022()
            nll_per_beta, pr2_per_beta, model_acc_per_beta = objective(
                env=env, model_name=args.model_name, epsilon=epsilon, beta=beta, method=args.method, num_runs=1)
        elif args.task_name == 'binz2022':
            env = Binz2022()
            nll_per_beta, pr2_per_beta, model_acc_per_beta = objective(
                env=env, model_name=args.model_name, epsilon=epsilon, beta=beta, method=args.method, num_runs=1)
        else:
            raise NotImplementedError
        nlls.append(nll_per_beta)
        pr2s.append(pr2_per_beta)
        accs.append(model_acc_per_beta)

    return np.array(pr2s), np.array(nlls), accs, parameters


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
        task_features = {}
    else:
        raise NotImplementedError

    def objective(x, participant):
        epsilon = x[0] if args.method == 'eps_greedy' else 0.
        beta = x[0] if args.method == 'soft_sigmoid' else 1.
        if args.method == 'both':
            epsilon = x[0]
            beta = x[1]
        ll, _, _ = compute_loglikelihood_human_choices_under_model(env=env, model_path=model_path, participant=participant, shuffle_trials=True,
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

        ll, chance_ll, acc = compute_loglikelihood_human_choices_under_model(env=env, model_path=model_path, participant=participant, shuffle_trials=True,
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
    parser.add_argument('--model-name', type=str,
                        required=True, help='model name')
    parser.add_argument('--method', type=str, default='soft_sigmoid',
                        help='method for computing model choice probabilities')
    parser.add_argument('--optimizer', action='store_true',
                        default=False, help='find optimal beta using optimizer')
    parser.add_argument('--num-iters', type=int, default=5)
    parser.add_argument('--paired', action='store_true',
                        default=False, help='paired')

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if args.optimizer:
        pr2s, nlls, accs, parameters = optimize(args)
        optimizer = 'differential_evolution'
    else:
        pr2s, nlls, accs, parameters = grid_search(args)
        optimizer = 'grid_search'

    # save list of results
    save_path = f"{SYS_PATH}/{args.paradigm}/data/model_comparison/{args.task_name}_{args.model_name[:200]}_paired{args.paired}_{args.method}_{optimizer}.npz"
    np.savez(save_path, betas=parameters, nlls=nlls, pr2s=pr2s, accs=accs)
