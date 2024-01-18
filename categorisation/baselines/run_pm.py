import pandas as pd
from pm import PrototypeModel
import numpy as np
import argparse
import sys
SYS_PATH = '/u/ajagadish/vanilla-llama' #'/raven/u/ajagadish/vanilla-llama/'
sys.path.append(f'{SYS_PATH}/categorisation/data')

# df = pd.read_csv('exp1.csv')
# pm = PrototypeModel(num_features=3, distance_measure=1, num_iterations=1)
# ll, r2 = pm.fit_participants(df)

# print(f'mean log-likelihood across participants: {ll.mean()} \n')
# print(f'mean pseudo-r2 across participants: {r2.mean()}')

## devraj et al. 2022
#TODO: in devraj every participant does only one condition so sometimes the dataframe below can be empty
# df = pd.read_csv('../data/human/devraj2022rational.csv')
# df = df[df['condition'] == 'control'] # only pass 'control' condition
# ##### REMOVE THIS LINE AFTER TESTING #####
# # df = df[df['participant'] == 0] # keep only first participant for testing
# ##### REMOVE ABOVE LINE AFTER TESTING #####
# num_runs, num_blocks, num_iter = 1, 11, 10
# loss = 'mse_transfer'
# opt_method = 'minimize'
# NUM_TASKS, NUM_FEATURES = 1, 6
# lls, r2s, params_list = [], [], []
# for idx in range(num_runs):
#     pm = PrototypeModel(num_features=NUM_FEATURES, distance_measure=1, num_iterations=num_iter, learn_prototypes=False, prototypes='from_data', loss=loss)
#     ll, r2, params = pm.fit_participants(df, num_blocks=num_blocks)
#     params_list.append(params)
#     lls.append(ll)
#     r2s.append(r2)
#     print(lls[idx], r2s[idx])
#     print(f'mean mse across blocks: {lls[idx].mean()} \n')
#     print(f'mean pseudo-r2 across blocks: {r2s[idx].mean()}')

# # save the r2 and ll values
# lls = np.array(lls)
# r2s = np.array(r2s)
# np.savez(f'../data/baselines/pm_humans_devrajstask_runs={num_runs}_iters={num_iter}_blocks={num_blocks}_tasks={NUM_TASKS}'\
#          , r2s=r2s, lls=lls, params=np.stack(params_list), opt_method=opt_method)
 

## benchmarking pm model
# df_train = pd.read_csv('../data/human/akshay-benchmark-across-languages-train.csv')
# df_transfer = pd.read_csv('../data/human/akshay-benchmark-across-languages-transfer.csv')
# pm = PrototypeModel(num_features=2, distance_measure=1, num_iterations=1, prototypes=[[4.766967, 2.243946],[2.292030, 4.593165]])
# params = pm.benchmark(df_train, df_transfer)
# print('fitted parameters: c {}, bias {}, w1 {}, w2 {}'.format(*params))
# true_params = pd.read_csv('../data/human/akshay-benchmark-across-languages-params.csv')
# print('true parameters: ', true_params)

## fit pm model to meta-learning model choices
# shpeards task
# df = pd.read_csv('../data/meta_learner/shepard_categorisation_env=claude_generated_tasks_paramsNA_dim3_data100_tasks11518_pversion4_model=transformer_num_episodes500000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8_noise0.0_shuffleTrue_run=0_beta=0.3_num_trials=96_num_runs=1.csv') 
# pm =  PrototypeModel(num_features=3, distance_measure=1, num_iterations=1, learn_prototypes=True, prototypes=None)# prototypes='from_data'
# # pm.burn_in = True
# ll, r2 = pm.fit_metalearner(df, num_blocks=6)
# print(ll, r2)
# print(f'mean log-likelihood across participants: {ll.mean()} \n')
# print(f'mean pseudo-r2 across participants: {r2.mean()}')

# smiths task
# df = pd.read_csv('../data/meta_learner/smithstask_env=claude_generated_tasks_paramsNA_dim6_data500_tasks12911_pversion5_stage1_model=transformer_num_episodes500000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8_noise0.0_shuffleTrue_run=1_beta=0.3_num_trials=300_num_runs=1.csv')
# opt_method = 'minimize'
# num_runs, num_blocks, NUM_TASKS = 5, 6, 2
# lls, r2s = np.zeros((num_runs, NUM_TASKS, num_blocks)), np.zeros((num_runs, NUM_TASKS, num_blocks))
# params_list = []
# for idx in range(num_runs):
#     pm =  PrototypeModel(num_features=6, distance_measure=1, num_iterations=1, learn_prototypes=False, prototypes='from_data')# prototypes='from_data'
#     lls[idx], r2s[idx], params = pm.fit_metalearner(df, num_blocks=num_blocks)
#     params_list.append(params)
#     print(lls[idx], r2s[idx])
#     print(f'mean log-likelihood across blocks: {lls[idx].mean()} \n')
#     print(f'mean pseudo-r2 across blocks: {r2s[idx].mean()}')

# # save the r2 and ll values
# np.savez(f'../data/meta_learner/pm_simulations_smithstask_runs={num_runs}_blocks={num_blocks}_tasks={NUM_TASKS}'\
#          , r2s=r2s, lls=lls, params=np.stack(params_list), opt_method=opt_method)

def fit_pm_to_humans(num_runs, num_blocks, num_iter, num_tasks, num_features, opt_method, loss, learn_prototypes, prototypes, task_name):
    
    if task_name == 'devraj2022':
        df = pd.read_csv('../data/human/devraj2022rational.csv')
        df = df[df['condition'] == 'control'] # only pass 'control' condition
        NUM_TASKS, NUM_FEATURES = 1, 6
    elif task_name == 'badham2017':
        df = pd.read_csv('../data/human/badham2017deficits.csv')
        NUM_TASKS, NUM_FEATURES = 1, 3
    else:
        raise NotImplementedError
    
    # shuffle the choice column in df
    # df['choice'] = np.random.permutation(df['choice'].values)
    # df['correct_choice'] = np.random.permutation(df['correct_choice'].values)
    # # shuffle values taken by prototype_feature1 to prototype_feature5 columns
    # for col in df.columns:
    #     if col.startswith('prototype_feature'):
    #         df[col] = np.random.permutation(df[col].values)

    # set choice to correct_choice
    # df['choice'] = df['correct_choice']
    
    lls, r2s, params_list = [], [], []
    for idx in range(num_runs):
        pm = PrototypeModel(num_features=NUM_FEATURES, distance_measure=1, num_iterations=num_iter, learn_prototypes=learn_prototypes, prototypes=prototypes, loss=loss)
        ll, r2, params = pm.fit_participants(df, num_blocks=num_blocks)
        params_list.append(params)
        lls.append(ll)
        r2s.append(r2)
        print(f'mean fit across blocks: {lls[idx].mean()} \n')
        print(f'mean pseudo-r2 across blocks: {r2s[idx].mean()}')

    # save the r2 and ll values
    lls = np.array(lls)
    r2s = np.array(r2s)
    np.savez(f'{SYS_PATH}/categorisation/data/model_comparison/{task_name}_pm_runs={num_runs}_iters={num_iter}_blocks={num_blocks}_loss={loss}'\
             , r2s=r2s, lls=lls, params=np.stack(params_list), opt_method=opt_method)
    

def fit_pm_to_metalearner(beta, num_runs, num_blocks, num_iter, num_tasks, num_features, opt_method, loss):
    # beta=0.1
    df = pd.read_csv(f'{SYS_PATH}/categorisation/data/meta_learner/smith_categorisation__tasks12910_pversion5_stage2_model=transformer_num_episodes500000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8_noise0.0_shuffleTrue_run=1_beta={beta}_num_trials=616_num_runs=10.csv')
    # num_runs, num_blocks, num_iter = 1, 11, 1
    # loss = 'mse_transfer'
    # opt_method = 'minimize'
    NUM_TASKS, NUM_FEATURES = 1, 6
    lls, r2s, params_list = [], [], []
    for idx in range(num_runs):
        pm = PrototypeModel(num_features=NUM_FEATURES, distance_measure=1, num_iterations=num_iter, learn_prototypes=False, prototypes='from_data', loss=loss)
        ll, r2, params = pm.fit_metalearner(df, num_blocks=num_blocks)
        params_list.append(params)
        lls.append(ll)
        r2s.append(r2)
        print(f'mean fit across blocks: {lls[idx].mean()} \n')
        print(f'mean pseudo-r2 across blocks: {r2s[idx].mean()}')

    # save the r2 and ll values
    lls = np.array(lls)
    r2s = np.array(r2s)
    np.savez(f'{SYS_PATH}/categorisation/data/meta_learner/pm_metalearner_devrajstask_beta={beta}_runs={num_runs}_iters={num_iter}_blocks={num_blocks}_tasks={NUM_TASKS}'\
            , r2s=r2s, lls=lls, params=np.stack(params_list), opt_method=opt_method)
    

def fit_pm_to_fitted_simulations(num_runs, num_blocks, num_iter, num_tasks, num_features, opt_method, loss, task_name, model_name, method, learn_prototypes, prototypes):
    
    if task_name == 'devraj2022':
        NUM_TASKS, NUM_FEATURES = 1, 6
    elif task_name == 'badham2017':
        NUM_TASKS, NUM_FEATURES = 1, 3
    else:
        raise NotImplementedError
    
    file_name =  f"{SYS_PATH}/categorisation/data/fitted_simulation/{task_name}_{model_name}_{method}.csv"
    df = pd.read_csv(file_name)

    ## shuffle the choice column in df
    # df['choice'] = np.random.permutation(df['choice'].values)
    # df['correct_choice'] = np.random.permutation(df['correct_choice'].values)
    
    ## set the choice to correct_choice 
    # df['choice'] = df['correct_choice']

    lls, r2s, params_list = [], [], []
    for idx in range(num_runs):
        pm = PrototypeModel(num_features=NUM_FEATURES, distance_measure=1, num_iterations=num_iter, learn_prototypes=learn_prototypes, prototypes=prototypes, loss=loss)
        ll, r2, params = pm.fit_participants(df, num_blocks=num_blocks)
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
    np.savez(f'{SYS_PATH}/categorisation/data/fitted_simulation/{task_name}_pm_runs={num_runs}_iters={num_iter}_blocks={num_blocks}_loss={loss}_model={model_name}'\
             , r2s=r2s, lls=lls, params=np.stack(params_list), opt_method=opt_method)
 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='fit pm to meta-learner choices')
    parser.add_argument('--beta', type=float, required=False, help='beta value')
    parser.add_argument('--num-iter', type=int, required=True, default=1, help='number of iterations')
    parser.add_argument('--num-runs', type=int, required=False,  default=1, help='number of runs')
    parser.add_argument('--num-blocks', type=int,required=False, default=11, help='number of blocks')
    parser.add_argument('--num-tasks', type=int, required=False, default=1, help='number of tasks')
    parser.add_argument('--num-features', type=int, required=False, default=6, help='number of features')
    parser.add_argument('--opt-method', type=str, required=False, default='minimize', help='optimization method')
    parser.add_argument('--loss', type=str, required=False, default='mse_transfer', help='loss function')
    parser.add_argument('--learn-prototypes', action='store_true', help='learn prototypes')
    parser.add_argument('--prototypes', type=str, required=False, default=None, help='prototypes')
    parser.add_argument('--fit-human-data', action='store_true', help='fit pm to human choices')
    parser.add_argument('--task-name', type=str, required=False, default='devraj2022', help='task name')
    parser.add_argument('--model-name', type=str, required=False, default='transformer', help='model name')
    parser.add_argument('--method', type=str, required=False, default='soft_sigmoid', help='method for computing model choice probabilities')
    args = parser.parse_args()

    if args.fit_human_data:
        fit_pm_to_humans(num_runs=args.num_runs, num_blocks=args.num_blocks, num_iter=args.num_iter, num_tasks=args.num_tasks, num_features=args.num_features, opt_method=args.opt_method, loss=args.loss, learn_prototypes=args.learn_prototypes, prototypes=args.prototypes, task_name=args.task_name)

    else:   
        fit_pm_to_fitted_simulations(num_runs=args.num_runs, num_blocks=args.num_blocks, num_iter=args.num_iter,\
                            num_tasks=args.num_tasks, num_features=args.num_features, \
                                opt_method=args.opt_method, loss=args.loss, task_name=args.task_name, model_name=args.model_name, method=args.method,\
                                    learn_prototypes=args.learn_prototypes, prototypes=args.prototypes)
         
        # assert args.beta is not None, 'beta value not provided'
        # fit_pm_to_metalearner(beta=args.beta, num_runs=args.num_runs, num_blocks=args.num_blocks, num_iter=args.num_iter, num_tasks=args.num_tasks, num_features=args.num_features, opt_method=args.opt_method, loss=args.loss)

    