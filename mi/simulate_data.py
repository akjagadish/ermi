from envs import SyntheticDecisionmakingTask
import argparse
import torch


def simulate_synthetic_decisionmaking_tasks(num_tasks, nonlinear, num_dims, max_steps, noise, device, batch_size=64):

    env = SyntheticDecisionmakingTask(num_dims=num_dims, max_steps=max_steps, synthesize_tasks=True,
                                      batch_size=batch_size, noise=noise, device=device).to(device)

    env.save_synthetic_data(num_tasks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='simulate and save synthetic data')
    parser.add_argument('--num-tasks', type=int,
                        default=10000, help='number of tasks')
    parser.add_argument('--num-dims', type=int, default=3,
                        help='number of dimensions')
    parser.add_argument('--max-steps', type=int, default=8,
                        help='number of data points per task')
    parser.add_argument('--batch-size', type=int,
                        default=100, help='batch size')
    parser.add_argument('--noise', type=float, default=0.0, help='noise level')

    parser.add_argument('--nonlinear', action='store_true',
                        default=False, help='simulate nonlinear synthetic data')

    args = parser.parse_args()
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    simulate_synthetic_decisionmaking_tasks(args.num_tasks, args.nonlinear, args.num_dims,
                                            args.max_steps, args.noise, device, args.batch_size)
