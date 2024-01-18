from envs import SyntheticCategorisationTask, RMCTask
import argparse
import torch

def simulate(num_tasks, rmc, nonlinear, num_dims, max_steps, noise, shuffle, device, batch_size=64):

    if rmc:
        assert num_dims == 3, 'RMC only supports 3 dimensions'
        env = RMCTask(num_dims=num_dims, max_steps=max_steps, batch_size=batch_size, noise=noise, shuffle_trials=shuffle, device=device).to(device)
    else:
        env = SyntheticCategorisationTask(nonlinear=nonlinear, num_dims=num_dims, max_steps=max_steps, batch_size=batch_size, noise=noise, shuffle_trials=shuffle, device=device).to(device)
    
    env.save_synthetic_data(num_tasks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='simulate and save synthetic data')
    parser.add_argument('--num-tasks', type=int, default=100, help='number of tasks')
    parser.add_argument('--num-dims', type=int, default=3, help='number of dimensions')
    parser.add_argument('--max-steps', type=int, default=8, help='number of data points per task')
    parser.add_argument('--batch-size', type=int, default=64, help='batch size')
    parser.add_argument('--noise', type=float, default=0.0, help='noise level')
    parser.add_argument('--shuffle', type=bool, default=False, help='shuffle trials')
    parser.add_argument('--nonlinear', action='store_true', default=False, help='simulate nonlinear synthetic data')
    parser.add_argument('--rmc', action='store_true', default=False, help='simulate rmc data')

    args = parser.parse_args()
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    simulate(args.num_tasks, args.rmc, args.nonlinear, args.num_dims, args.max_steps, args.noise, args.shuffle, device, args.batch_size)

