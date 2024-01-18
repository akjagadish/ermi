import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import os
from torch.utils.tensorboard import SummaryWriter
from envs import CategorisationTask, SyntheticCategorisationTask, RMCTask
from model import Transformer, TransformerDecoder
import argparse
from tqdm import tqdm
from evaluate import evaluate, evaluate_1d


def run(env_name, restart_training, restart_episode_id, num_episodes, synthetic, nonlinear, rmc, num_dims, max_steps, sample_to_match_max_steps, noise, shuffle, shuffle_features, print_every, save_every, num_hidden, num_layers, d_model, num_head, save_dir, device, lr, batch_size=64):

    writer = SummaryWriter('runs/' + save_dir)
    if synthetic:
        env = SyntheticCategorisationTask(nonlinear=nonlinear, num_dims=num_dims, max_steps=max_steps, batch_size=batch_size, noise=noise, shuffle_trials=shuffle, device=device).to(device)
    elif rmc:
        assert num_dims == 3, 'RMC only supports 3 dimensions'
        env = RMCTask(data=env_name, num_dims=num_dims, max_steps=max_steps, batch_size=batch_size, noise=noise, shuffle_trials=shuffle, device=device).to(device)
    else:
        env = CategorisationTask(data=env_name, num_dims=num_dims, max_steps=max_steps, sample_to_match_max_steps=sample_to_match_max_steps, batch_size=batch_size, noise=noise, shuffle_trials=shuffle, shuffle_features=shuffle_features, device=device).to(device)
    
    # setup model
    if restart_training and os.path.exists(save_dir):
        t, model = torch.load(save_dir)
        model = model.to(device)
        print(f'Loaded model from {save_dir}')
        start_id = restart_episode_id
    else:
        model = TransformerDecoder(num_input=env.num_dims, num_output=env.num_choices, num_hidden=num_hidden, num_layers=num_layers, d_model=d_model, num_head=num_head, max_steps=max_steps, device=device).to(device)
        start_id = 0
    # setup optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)
    losses = [] # keep track of losses
    accuracy = [] # keep track of accuracies
    
    # mask = torch.tril(torch.ones(max_steps, max_steps))
    for t in tqdm(range(start_id, int(num_episodes))):

        packed_inputs, sequence_lengths, targets = env.sample_batch()
        model_choices = model(packed_inputs, sequence_lengths)
        model_choices = torch.concat([model_choices[i, :seq_len] for i, seq_len in enumerate(sequence_lengths)], axis=0).squeeze().float()
        true_choices = targets.reshape(-1).float() if synthetic or rmc else torch.concat(targets, axis=0).float().to(device)

        # gradient step
        loss = model.compute_loss(model_choices, true_choices)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # logging
        losses.append(loss.item())
        
        if (not t % print_every):
            writer.add_scalar('Loss', loss, t)

        if (not t % save_every):
            torch.save([t, model], save_dir)
            experiment = 'synthetic' if synthetic else 'rmc' if rmc else 'categorisation'
            acc = evaluate_1d(env_name=env_name, model_path=save_dir, experiment=experiment, mode='val', max_steps=max_steps, nonlinear=nonlinear, num_dims=num_dims, device=device)
            accuracy.append(acc)
            writer.add_scalar('Val. Acc.', acc, t)
        
    return losses, accuracy

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='meta-learning for categorisation')
    parser.add_argument('--num-episodes', type=int, default=1e6, help='number of trajectories for training')
    parser.add_argument('--num-dims', type=int, default=3, help='number of dimensions')
    parser.add_argument('--max-steps', type=int, default=8, help='number of data points per task')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--print-every', type=int, default=100, help='how often to print')
    parser.add_argument('--save-every', type=int, default=100, help='how often to save')
    parser.add_argument('--runs', type=int, default=1, help='total number of runs')
    parser.add_argument('--first-run-id', type=int, default=0, help='id of the first run')
    parser.add_argument('--num_hidden', type=int, default=128, help='number of hidden units')
    parser.add_argument('--lr', type=float, default=3e-4, help='learning rate')
    parser.add_argument('--num_layers', type=int, default=1, help='number of layers')
    parser.add_argument('--d_model', type=int, default=256, help='dimension of the model')
    parser.add_argument('--num_head', type=int, default=4, help='number of heads')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--env-name', default='llama_generated_tasks_params65B_dim3_data8_tasks14500', help='name of the environment')
    parser.add_argument('--env-dir', default='raven/u/ajagadish/vanilla-llama/categorisation/data', help='name of the environment')
    parser.add_argument('--save-dir', default='trained_models/', help='directory to save models')
    parser.add_argument('--test', action='store_true', default=False, help='test runs')
    parser.add_argument('--synthetic', action='store_true', default=False, help='train models on synthetic data')
    parser.add_argument('--nonlinear', action='store_true', default=False, help='train models on nonlinear synthetic data')
    parser.add_argument('--rmc', action='store_true', default=False, help='train models on rmc data')
    parser.add_argument('--noise', type=float, default=0., help='noise level')
    parser.add_argument('--shuffle', action='store_true', default=False, help='shuffle trials')
    parser.add_argument('--shuffle-features', action='store_true', default=False, help='shuffle features')
    parser.add_argument('--model-name', default='transformer', help='name of the model')
    parser.add_argument('--sample-to-match-max-steps', action='store_true', default=False, help='sample to match max steps')
    parser.add_argument('--restart-training', action='store_true', default=False, help='restart training')
    parser.add_argument('--restart-episode-id', type=int, default=0, help='restart episode id')
    # parser.add_argument('--eval', default='categorisation', help='what to eval your meta-learner on')

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu") #"cpu" #
    env_name = f'/{args.env_dir}/{args.env_name}.csv'# if args.env_name is None else args.env_name

    for i in range(args.runs):

        save_dir = f'{args.save_dir}env={args.env_name}_model={args.model_name}_num_episodes{str(args.num_episodes)}_num_hidden={str(args.num_hidden)}_lr{str(args.lr)}_num_layers={str(args.num_layers)}_d_model={str(args.d_model)}_num_head={str(args.num_head)}_noise{str(args.noise)}_shuffle{str(args.shuffle)}_run={str(args.first_run_id + i)}.pt'
        
        if args.synthetic:
            save_dir = save_dir.replace('.pt', f'_synthetic{"nonlinear" if args.nonlinear else ""}.pt')
        elif args.rmc:
            save_dir = save_dir.replace('.pt', f'_rmc.pt')
        save_dir = save_dir.replace('.pt', '_test.pt') if args.test else save_dir
        
        run(env_name, args.restart_training, args.restart_episode_id, args.num_episodes, args.synthetic, args.nonlinear, args.rmc, args.num_dims, args.max_steps, args.sample_to_match_max_steps, args.noise, args.shuffle, args.shuffle_features, args.print_every, args.save_every, args.num_hidden, args.num_layers, args.d_model, args.num_head, save_dir, device, args.lr, args.batch_size)
