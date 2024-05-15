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

        # get model choices
        model_choice_probs = model(
            packed_inputs.float().to(device), sequence_lengths)

        # task performance
        model_choices = model_choice_probs > 0.5 if method == 'greedy' else torch.distributions.Binomial(
            probs=model_choice_probs).sample()
        per_trial_model_accuracy = (model_choices == correct_choices)
        model_choices = torch.concat([model_choices[i, :seq_len] for i, seq_len in enumerate(
            sequence_lengths)], axis=0).squeeze().float()
        correct_choices = torch.concat([correct_choices[i, :seq_len] for i, seq_len in enumerate(
            sequence_lengths)], axis=0).squeeze().float()
        correct_choices = correct_choices.reshape(-1).float().to(device)
        model_accuracy = (model_choices == correct_choices).sum() / \
            correct_choices.numel()
        human_accuracy = (human_choices.reshape(-1) ==
                          correct_choices).sum() / correct_choices.numel()

    return model_accuracy, per_trial_model_accuracy, human_accuracy


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

    per_trial_accs, human_accs, accs = [], [], []
    participants = env.data.participant.unique()
    for participant in participants:
        beta, epsilon = 1., 0.
        model_accuracy, per_trial_model_accuracy, human_accuracy = compute_loglikelihood_human_choices_under_model(env=env, model_path=model_path, participant=participant, shuffle_trials=True,
                                                                                                                   beta=beta, epsilon=epsilon, method=args.method, paired=args.paired, **task_features)
        human_accs.append(human_accuracy)
        per_trial_accs.append(per_trial_model_accuracy)
        accs.append(model_accuracy)

    return np.array(accs), torch.stack(per_trial_accs).squeeze().sum(1), np.array(human_accs)


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
