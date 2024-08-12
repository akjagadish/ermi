import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import os
from torch.utils.tensorboard import SummaryWriter
from envs import DecisionmakingTask, SyntheticDecisionmakingTask
from model import TransformerDecoderClassification, TransformerDecoderLinearWeights
import argparse
from tqdm import tqdm
from evaluate import evaluate_classification
import schedulefree
import ivon
from model_utils import get_wd_from_std, compute_elbo

def run(env_name, paired, restart_training, restart_episode_id, num_episodes, train_samples, ess, std, synthetic, ranking, direction, num_dims, max_steps, sample_to_match_max_steps, noise, shuffle, shuffle_features, print_every, save_every, num_hidden, num_layers, d_model, num_head, loss_fn, save_dir, device, lr, batch_size=64):

    writer = SummaryWriter('runs/' + save_dir)
    if synthetic:
        env = SyntheticDecisionmakingTask(num_dims=num_dims, max_steps=max_steps, batch_size=batch_size, noise=noise, ranking=ranking, direction=direction,
                                          device=device).to(device)
        model_max_steps = env.max_steps
    else:
        env = DecisionmakingTask(data=env_name, num_dims=num_dims, max_steps=max_steps, sample_to_match_max_steps=sample_to_match_max_steps,
                                 batch_size=batch_size, noise=noise, shuffle_trials=shuffle, shuffle_features=shuffle_features, device=device).to(device)
        model_max_steps = env.max_steps//2

    # setup model
    if restart_training and os.path.exists(save_dir):
        t, model = torch.load(save_dir)
        model = model.to(device)
        print(f'Loaded model from {save_dir}')
        start_id = restart_episode_id
    else:
        if paired:
            model = TransformerDecoderLinearWeights(num_input=env.num_dims, num_output=env.num_choices, num_hidden=num_hidden,
                                                    num_layers=num_layers, d_model=d_model, num_head=num_head, max_steps=model_max_steps, loss=loss_fn, device=device).to(device)

        else:
            model = TransformerDecoderClassification(num_input=env.num_dims, num_output=env.num_choices, num_hidden=num_hidden,
                                                     num_layers=num_layers, d_model=d_model, num_head=num_head, max_steps=model_max_steps, loss=loss_fn, device=device).to(device)
        start_id = 0

    # setup optimizer
    # optimizer = optim.Adam(model.parameters(), lr=lr)
    #optimizer = schedulefree.AdamWScheduleFree(model.parameters(), lr=args.lr)
    ess = len(env.data) if ess is None else ess
    wd = get_wd_from_std(std, ess)
    optimizer = ivon.IVON(model.parameters(), lr=lr, ess=ess, weight_decay=wd)
    losses = []  # keep track of losses
    accuracy = []  # keep track of accuracies

    # train for num_episodes
    for t in tqdm(range(start_id, int(num_episodes))):
        # optimizer.train()
        # model.train()
        # packed_inputs, sequence_lengths, targets = env.sample_batch(
        #     paired=paired)
        # optimizer.zero_grad()
        # loss = model.compute_loss(packed_inputs, targets, sequence_lengths)
        
        model.train()
        packed_inputs, sequence_lengths, targets = env.sample_batch(paired=paired)
        for _ in range(train_samples):
            with optimizer.sampled_params(train=True):
                optimizer.zero_grad()
                loss = model.compute_loss(packed_inputs, targets, sequence_lengths)
                loss.backward()

        # backprop
        # loss.backward()
        optimizer.step()

        # logging
        losses.append(loss.item())
        # elbo = compute_elbo(optimizer, model, std, packed_inputs, targets, sequence_lengths)

        if (not t % print_every):
            writer.add_scalar('Loss', loss, t)
            # writer.add_scalar('ELBO', elbo, t)

        if (not t % save_every):
            torch.save([t, model.state_dict(), optimizer.state_dict(), std, ess], save_dir)
            experiment = 'synthetic' if synthetic else 'llm_generated'
            acc = evaluate_classification(env_name=env_name, experiment=experiment, paired=paired,
                                          env=env, model=model, mode='val', shuffle_trials=shuffle, loss=loss_fn, max_steps=max_steps, num_dims=num_dims, optimizer=optimizer, device=device)
            accuracy.append(acc)
            writer.add_scalar('Val. Acc.', acc, t)

    return losses, accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='meta-learning for decisionmaking')
    parser.add_argument('--num-episodes', type=int, default=1e6,
                        help='number of trajectories for training')
    parser.add_argument('--train-samples', type=int, default=1,
                        help='number of samples for training')
    parser.add_argument('--job-array', action='store_true',
                        default=False, help='job array')
    parser.add_argument('--ess', type=float, default=None,
                         help='weight for the nll loss term in the ELBO')
    parser.add_argument('--prior-std', type=float, default=0.1,
                         help='std for the prior')
    parser.add_argument('--num-dims', type=int, default=3,
                        help='number of dimensions')
    parser.add_argument('--max-steps', type=int, default=8,
                        help='number of data points per task')
    parser.add_argument('--batch_size', type=int,
                        default=64, help='batch size')
    parser.add_argument('--print-every', type=int,
                        default=100, help='how often to print')
    parser.add_argument('--save-every', type=int,
                        default=100, help='how often to save')
    parser.add_argument('--runs', type=int, default=1,
                        help='total number of runs')
    parser.add_argument('--first-run-id', type=int,
                        default=0, help='id of the first run')
    parser.add_argument('--num_hidden', type=int,
                        default=128, help='number of hidden units')
    parser.add_argument('--lr', type=float, default=3e-4, help='learning rate')
    parser.add_argument('--num_layers', type=int,
                        default=1, help='number of layers')
    parser.add_argument('--d_model', type=int, default=256,
                        help='dimension of the model')
    parser.add_argument('--num_head', type=int,
                        default=4, help='number of heads')
    parser.add_argument('--loss', default='nll', help='loss function')
    parser.add_argument('--no-cuda', action='store_true',
                        default=False, help='disables CUDA training')
    parser.add_argument('--paired', action='store_true',
                        default=False, required=False, help='paired')
    parser.add_argument('--env-type', default=None, 
                        help='name of the environment when name of the dataset does not explain the model fully')
    parser.add_argument('--env-name', required=False, 
                        help='name of the environment')
    parser.add_argument('--env-dir', help='name of the environment', 
                        required=False)
    parser.add_argument('--save-dir', help='directory to save models', 
                        required=True)
    parser.add_argument('--test', action='store_true',
                        default=False, help='test runs')
    parser.add_argument('--synthetic', action='store_true',
                        default=False, help='train models on synthetic data')
    parser.add_argument('--ranking', action='store_true',
                        default=False, help='train models on ranked synthetic data')
    parser.add_argument('--direction', action='store_true',
                        default=False, help='train models on directional synthetic data')
    parser.add_argument('--noise', type=float, default=0., help='noise level')
    parser.add_argument('--shuffle', action='store_true',
                        default=False, help='shuffle trials')
    parser.add_argument('--shuffle-features', action='store_true',
                        default=False, help='shuffle features')
    parser.add_argument('--model-name', default='transformer',
                        help='name of the model')
    parser.add_argument('--sample-to-match-max-steps', action='store_true',
                        default=False, help='sample to match max steps')
    parser.add_argument('--restart-training', action='store_true',
                        default=False, help='restart training')
    parser.add_argument('--restart-episode-id', type=int,
                        default=0, help='restart episode id')
    parser.add_argument('--scale', type=int, default=10000,
                        help='scale for the job array')
    parser.add_argument('--offset', type=int, default=0,
                        help='offset for the job array')
    # parser.add_argument('--eval', default='categorisation', help='what to eval your meta-learner on')

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    env = f'{args.env_name}_dim{args.num_dims}' if args.synthetic else args.env_name if args.env_type is None else args.env_type
    args.ess = args.ess * args.scale + args.offset if args.job_array else args.ess

    for i in range(args.runs):

        save_dir = f'{args.save_dir}env={env}_model={args.model_name}_num_episodes{str(args.num_episodes)}_num_hidden={str(args.num_hidden)}_lr{str(args.lr)}_num_layers={str(args.num_layers)}_d_model={str(args.d_model)}_num_head={str(args.num_head)}_noise{str(args.noise)}_shuffle{str(args.shuffle)}_paired{str(args.paired)}_loss{str(args.loss)}_ess{str(int(args.ess))}_std{str(args.prior_std)}_run={str(args.first_run_id + i)}.pt'
        save_dir = save_dir.replace(
                '.pt', f'_{"ranking" if args.ranking else "direction" if args.direction else "unknown"}.pt') if args.synthetic else save_dir
        save_dir = save_dir.replace(
            '.pt', '_test.pt') if args.test else save_dir
        env_name = f'/{args.env_dir}/{args.env_name}.csv' if not args.synthetic else None

        run(env_name, args.paired, args.restart_training, args.restart_episode_id, args.num_episodes, args.train_samples, args.ess, args.prior_std, args.synthetic, args.ranking, args.direction, args.num_dims, args.max_steps, args.sample_to_match_max_steps,
            args.noise, args.shuffle, args.shuffle_features, args.print_every, args.save_every, args.num_hidden, args.num_layers, args.d_model, args.num_head, args.loss, save_dir, device, args.lr, args.batch_size)
