import pandas as pd
from gcm import GeneralizedContextModel
import sys
import numpy as np
import argparse
import sys
SYS_PATH = '/u/ajagadish/ermi'
sys.path.append(f'{SYS_PATH}/categorisation/data')


def fit_gcm_to_humans(num_runs, num_blocks, num_iter, num_tasks, num_features, opt_method, loss, task_name):

    if task_name == 'devraj2022':
        df = pd.read_csv('../data/human/devraj2022rational.csv')
        df = df[df['condition'] == 'control']  # only pass 'control' condition
        NUM_TASKS, NUM_FEATURES = 1, 6
    elif task_name == 'badham2017':
        df = pd.read_csv('../data/human/badham2017deficits.csv')
        NUM_TASKS, NUM_FEATURES = 1, 3
    else:
        raise NotImplementedError

    lls, r2s, params_list = [], [], []
    for idx in range(num_runs):
        gcm = GeneralizedContextModel(num_features=NUM_FEATURES, distance_measure=1,
                                      num_iterations=num_iter, opt_method=opt_method, loss=loss)
        ll, r2, params = gcm.fit_participants(
            df, num_blocks=num_blocks, reduce='sum')
        params_list.append(params)
        lls.append(ll)
        r2s.append(r2)
        print(f'mean fit across blocks: {lls[idx].mean()} \n')
        print(f'mean pseudo-r2 across blocks: {r2s[idx].mean()}')

    # save the r2 and ll values
    lls = np.array(lls)
    r2s = np.array(r2s)
    np.savez(f'{SYS_PATH}/categorisation/data/model_comparison/{task_name}_gcm_runs={num_runs}_iters={num_iter}_blocks={num_blocks}_loss={loss}',
             r2s=r2s, lls=lls, params=np.stack(params_list), opt_method=opt_method)


def fit_gcm_to_metalearner(beta, num_runs, num_blocks, num_iter, num_tasks, num_features, opt_method, loss):

    df = pd.read_csv(f'{SYS_PATH}/categorisation/data/meta_learner/smith_categorisation__tasks12910_pversion5_stage2_model=transformer_num_episodes500000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8_noise0.0_shuffleTrue_run=1_beta={beta}_num_trials=616_num_runs=10.csv')
    NUM_TASKS, NUM_FEATURES = 1, 6
    lls, r2s, params_list = [], [], []
    for idx in range(num_runs):
        gcm = GeneralizedContextModel(num_features=NUM_FEATURES, distance_measure=1,
                                      num_iterations=num_iter, opt_method=opt_method, loss=loss)
        ll, r2, params = gcm.fit_metalearner(
            df, num_blocks=num_blocks, reduce='sum')
        params_list.append(params)
        lls.append(ll)
        r2s.append(r2)
        print(lls[idx], r2s[idx])
        print(f'mean mse across blocks: {lls[idx].mean()} \n')
        print(f'mean pseudo-r2 across blocks: {r2s[idx].mean()}')

    # save the r2 and ll values
    lls = np.array(lls)
    r2s = np.array(r2s)
    np.savez(f'{SYS_PATH}/categorisation/data/meta_learner/gcm_metalearner_devrajstask_beta={beta}_runs={num_runs}_iters={num_iter}_blocks={num_blocks}_tasks={NUM_TASKS}',
             r2s=r2s, lls=lls, params=np.stack(params_list), opt_method=opt_method)


def fit_gcm_to_fitted_simulations(num_runs, num_blocks, num_iter, num_tasks, num_features, opt_method, loss, task_name, model_name, method):

    if task_name == 'devraj2022':
        NUM_TASKS, NUM_FEATURES = 1, 6
    elif task_name == 'badham2017':
        NUM_TASKS, NUM_FEATURES = 1, 3
    else:
        raise NotImplementedError

    file_name = f"{SYS_PATH}/categorisation/data/fitted_simulation/{task_name}_{model_name}_{method}.csv"
    df = pd.read_csv(file_name)

    lls, r2s, params_list = [], [], []
    for idx in range(num_runs):
        gcm = GeneralizedContextModel(num_features=NUM_FEATURES, distance_measure=1,
                                      num_iterations=num_iter, opt_method=opt_method, loss=loss)
        ll, r2, params = gcm.fit_participants(
            df, num_blocks=num_blocks, reduce='sum')
        params_list.append(params)
        lls.append(ll)
        r2s.append(r2)
        print(f'mean fit across blocks: {lls[idx].mean()} \n')
        print(f'mean pseudo-r2 across blocks: {r2s[idx].mean()}')

    if 'syntheticnonlinear' in model_name:
        model_name = 'syntheticnonlinear'
    elif 'synthetic' in model_name:
        model_name = 'synthetic'
    else:
        model_name = 'ermi'

    # save the r2 and ll values
    lls = np.array(lls)
    r2s = np.array(r2s)
    np.savez(f'{SYS_PATH}/categorisation/data/fitted_simulation/{task_name}_gcm_runs={num_runs}_iters={num_iter}_blocks={num_blocks}_loss={loss}_model={model_name}',
             r2s=r2s, lls=lls, params=np.stack(params_list), opt_method=opt_method)


def fit_gcm_to_llm(num_runs, num_blocks, num_iter, opt_method, loss, task_name):

    if task_name == 'devraj2022':
        NUM_TASKS, NUM_FEATURES = 1, 6
        file_name = f"{SYS_PATH}/categorisation/data/llm/{task_name}rational_llm_choiceshuman.csv"
        df = pd.read_csv(file_name)
        df = df[df['condition'] == 'control']  # only pass 'control' condition
    elif task_name == 'badham2017':
        NUM_TASKS, NUM_FEATURES = 1, 3
        file_name = f"{SYS_PATH}/categorisation/data/llm/{task_name}deficits_llm_choiceshuman.csv"
        df = pd.read_csv(file_name)
    else:
        raise NotImplementedError

    lls, r2s, params_list = [], [], []
    for idx in range(num_runs):
        gcm = GeneralizedContextModel(num_features=NUM_FEATURES, distance_measure=1,
                                      num_iterations=num_iter, opt_method=opt_method, loss=loss)
        ll, r2, params = gcm.fit_llm(df, num_blocks=num_blocks, reduce='sum')
        params_list.append(params)
        lls.append(ll)
        r2s.append(r2)
        print(f'mean fit across blocks: {lls[idx].mean()} \n')
        print(f'mean pseudo-r2 across blocks: {r2s[idx].mean()}')

    # save the r2 and ll values
    lls = np.array(lls)
    r2s = np.array(r2s)
    np.savez(f'{SYS_PATH}/categorisation/data/fitted_simulation/{task_name}_gcm_runs={num_runs}_iters={num_iter}_blocks={num_blocks}_loss={loss}_model=llm',
             r2s=r2s, lls=lls, params=np.stack(params_list), opt_method=opt_method)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='fit gcm to meta-learner choices')
    parser.add_argument('--beta', type=float, required=False,
                        default=None, help='beta value')
    parser.add_argument('--num-iter', type=int, required=True,
                        default=1, help='number of iterations')
    parser.add_argument('--num-runs', type=int, required=False,
                        default=1, help='number of runs')
    parser.add_argument('--num-blocks', type=int,
                        required=False, default=11, help='number of blocks')
    parser.add_argument('--num-tasks', type=int,
                        required=False, default=1, help='number of tasks')
    parser.add_argument('--num-features', type=int,
                        required=False, default=6, help='number of features')
    parser.add_argument('--opt-method', type=str, required=False,
                        default='minimize', help='optimization method')
    parser.add_argument('--loss', type=str, required=False,
                        default='mse_transfer', help='loss function')
    parser.add_argument('--fit-human-data', action='store_true',
                        help='fit gcm to human choices')
    parser.add_argument('--fit-llm', action='store_true',
                        help='fit gcm to llm choices')
    parser.add_argument('--task-name', type=str, required=False,
                        default='devraj2022', help='task name')
    parser.add_argument('--model-name', type=str, required=False,
                        default='transformer', help='model name')
    parser.add_argument('--method', type=str, required=False, default='soft_sigmoid',
                        help='method for computing model choice probabilities')

    args = parser.parse_args()

    if args.fit_human_data:
        fit_gcm_to_humans(num_runs=args.num_runs, num_blocks=args.num_blocks, num_iter=args.num_iter,
                          num_tasks=args.num_tasks, num_features=args.num_features,
                          opt_method=args.opt_method, loss=args.loss, task_name=args.task_name)
    elif args.fit_llm:
        fit_gcm_to_llm(num_runs=args.num_runs, num_blocks=args.num_blocks, num_iter=args.num_iter,
                       opt_method=args.opt_method, loss=args.loss, task_name=args.task_name)
    else:
        fit_gcm_to_fitted_simulations(num_runs=args.num_runs, num_blocks=args.num_blocks, num_iter=args.num_iter,
                                      num_tasks=args.num_tasks, num_features=args.num_features,
                                      opt_method=args.opt_method, loss=args.loss, task_name=args.task_name, model_name=args.model_name, method=args.method)
        # assert args.beta is not None, 'beta value not provided'
        # fit_gcm_to_metalearner(beta=args.beta, num_runs=args.num_runs, num_blocks=args.num_blocks, num_iter=args.num_iter, num_tasks=args.num_tasks, num_features=args.num_features, opt_method=args.opt_method, loss=args.loss)
