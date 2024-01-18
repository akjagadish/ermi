import gym
import math
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from envs import CategorisationTask, SyntheticCategorisationTask
from model import MetaLearner, NoisyMetaLearner
import argparse
from tqdm import tqdm
from evaluate import evaluate, evaluate_1d
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


def run(env_name, num_episodes, synthetic, max_steps, noise, shuffle, print_every, save_every, num_hidden, save_dir, device, lr, batch_size=64):

    writer = SummaryWriter('runs/' + save_dir)
    if synthetic:
        env = SyntheticCategorisationTask(max_steps=max_steps, batch_size=batch_size, noise=noise, shuffle_trials=shuffle, device=device).to(device)
    else:
        env = CategorisationTask(data=env_name, max_steps=max_steps, batch_size=batch_size, noise=noise, shuffle_trials=shuffle, device=device).to(device)
    
    # setup model
    #model = MetaLearner(num_input=env.num_dims, num_output=env.num_choices, num_hidden=num_hidden, num_layers=1).to(device)
    model = NoisyMetaLearner(num_input=env.num_dims, num_output=env.num_choices, num_hidden=num_hidden, num_layers=1).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    losses = [] # keep track of losses
    accuracy = [] # keep track of accuracies
    
    for t in tqdm(range(int(num_episodes))):

        packed_inputs, sequence_lengths, targets = env.sample_batch()
        model_choices = model(packed_inputs, sequence_lengths)
        #true_choices = targets.view(-1).float()
        #model_choices = model_choices.view(-1).float()
        #import ipdb ; ipdb.set_trace()
        model_choices = torch.concat([model_choices[i, :seq_len] for i, seq_len in enumerate(sequence_lengths)], axis=0).squeeze().float()
        true_choices = targets.reshape(-1).float() if synthetic else torch.concat(targets, axis=0).float()
        
        # gradient step
        loss = model.compute_loss(model_choices,true_choices)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # logging
        losses.append(loss.item())
        
        if (not t % print_every):
            writer.add_scalar('Loss', loss, t)
            

        if (not t % save_every):
            torch.save([t, model], save_dir)
            #TODO: eval on experiment='synthetic' if synthetic else 'categorisation'
            acc = evaluate_1d(env_name=env_name, model_path=save_dir, mode='val', policy='greedy')
            accuracy.append(acc)
            writer.add_scalar('Val. Acc.', acc, t)
        
    return losses, accuracy

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='meta-learning for categorisation')
    parser.add_argument('--num-episodes', type=int, default=1e6, help='number of trajectories for training')
    parser.add_argument('--max-steps', type=int, default=8, help='number of points per episode')
    parser.add_argument('--print-every', type=int, default=100, help='how often to print')
    parser.add_argument('--save-every', type=int, default=100, help='how often to save')
    parser.add_argument('--runs', type=int, default=1, help='total number of runs')
    parser.add_argument('--first-run-id', type=int, default=0, help='id of the first run')
    parser.add_argument('--num_hidden', type=int, default=128, help='number of hidden units')
    parser.add_argument('--lr', type=float, default=3e-4, help='learning rate')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--env-name', default='llama_generated_tasks_params65B_dim3_data8_tasks14500', help='name of the environment')
    parser.add_argument('--env-dir', default='raven/u/ajagadish/vanilla-llama/categorisation/data', help='name of the environment')
    parser.add_argument('--save-dir', default='trained_models/', help='directory to save models')
    parser.add_argument('--test', action='store_true', default=False, help='test runs')
    parser.add_argument('--synthetic', action='store_true', default=False, help='train models on synthetic data')
    parser.add_argument('--noise', type=float, default=0., help='noise level')
    parser.add_argument('--shuffle', action='store_true', default=False, help='shuffle trials')
    # parser.add_argument('--eval', default='categorisation', help='what to eval your meta-learner on')

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = "cpu" #torch.device("cuda" if use_cuda else "cpu")
    env_name = f'/{args.env_dir}/{args.env_name}.csv'# if args.env_name is None else args.env_name

    for i in range(args.runs):
        if args.test:
            save_dir = f'{args.save_dir}env={args.env_name}_num_episodes{str(args.num_episodes)}_num_hidden={str(args.num_hidden)}_lr{str(args.lr)}_run={str(args.first_run_id + i)}_test.pt'
        else:
            save_dir = f'{args.save_dir}env={args.env_name}_num_episodes{str(args.num_episodes)}_num_hidden={str(args.num_hidden)}_lr{str(args.lr)}_noise{str(args.noise)}_shuffle{str(args.shuffle)}_run={str(args.first_run_id + i)}.pt'
        if args.synthetic:
            save_dir = save_dir.replace('.pt', '_synthetic.pt')
        run(env_name, args.num_episodes, args.synthetic, args.max_steps, args.noise, args.shuffle, args.print_every, args.save_every, args.num_hidden, save_dir, device, args.lr)