from scipy.spatial.distance import euclidean
import numpy as np
import json
import pandas as pd
SYS_PATH = '/u/ajagadish/ermi'


def compute_distance_transfer_stimulus(idx=5):
    A_array = np.array([[0, 0, 0, 1],
                        [0, 1, 0, 1],
                        [0, 1, 0, 0],
                        [0, 0, 1, 0],
                        [1, 0, 0, 0]])

    B_array = np.array([[0, 0, 1, 1],
                        [1, 0, 0, 1],
                        [1, 1, 1, 0],
                        [1, 1, 1, 1]])

    T_array = np.array([[0, 1, 1, 0],
                       [0, 1, 1, 1],
                        [0, 0, 0, 0],
                        [1, 1, 0, 1],
                        [1, 0, 1, 0],
                        [1, 1, 0, 0],
                        [1, 0, 1, 1]])

    # Extracting the T5 value
    T5_value = T_array[idx-1]

    # Computing the mean element-by-element Euclidean distance of T5 with elements of A and B
    mean_distance_A = np.mean([euclidean(T5_value, A) for A in A_array])
    mean_distance_B = np.mean([euclidean(T5_value, B) for B in B_array])

    print(mean_distance_A, mean_distance_B)


def grid_search_beta(model='ermi', task_block=32):

    with open(f'{SYS_PATH}/categorisation/data/human/johanssen2002.json') as f:
        human_data = json.load(f)

    # human data procesing
    human_data_dict = {}
    for i, stimulus_id in enumerate(human_data['x']):
        human_data_dict[stimulus_id] = human_data['y'][i]
    human_data_df = pd.DataFrame.from_dict(
        human_data_dict, orient='index', columns=['human_choice'])
    human_data_df['stimulus_id'] = human_data_df.index
    human_data_df = human_data_df[human_data_df['stimulus_id'].str.contains(
        'T')]
    human_generalisation = human_data_df['human_choice'].sort_values()

    num_runs = 1
    min_distance = np.inf
    for beta in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        if model == 'mi':
            data = pd.read_csv(
                f'{SYS_PATH}/categorisation/data/meta_learner/johanssen_categorisation_500000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8_noise0.0_shuffleTrue_run=1_synthetic_beta={beta}_num_trials=288_num_runs={num_runs}.csv')
        elif model == 'ermi':
            data = pd.read_csv(f'{SYS_PATH}/categorisation/data/meta_learner/johanssen_categorisation__tasks8950_pversion5_stage1_model=transformer_num_episodes500000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8_noise0.0_shuffleTrue_run=1_beta={beta}_num_trials=288_num_runs={num_runs}.csv')
        elif model == 'pfn':
            data = pd.read_csv(f'{SYS_PATH}/categorisation/data/meta_learner/johanssen_categorisation_500000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8_noise0.0_shuffleTrue_run=1_syntheticnonlinear_beta={beta}_num_trials=288_num_runs={num_runs}.csv')
        transfer_stimulus_ids = data[data['stimulus_id'].str.contains(
            'T')]['stimulus_id']
        transfer_data = data[data['stimulus_id'].isin(transfer_stimulus_ids)]
        # subselect stimulus_id whose values have letter 'T' in them
        transfer_stimulus_ids = data[data['stimulus_id'].str.contains(
            'T')]['stimulus_id']
        transfer_data = data[data['stimulus_id'].isin(transfer_stimulus_ids)]
        over_blocks = [task_block]  # transfer_data.task_feature.unique()
        for task_block in over_blocks:

            # choose a subset of the transfer_data dataframe where the task_feature is equal to 1
            transfer_data = transfer_data[transfer_data['task_feature'] == task_block]
            if transfer_data.shape[0] == 0:
                continue
            # get the mean choice for each stimulus_id and sort the values in the order of the magnitude of the mean for both the transfer_data and human_data_df
            meta_learner_generalisation = (
                1-transfer_data.groupby('stimulus_id')['choice'].mean())
            meta_learner_generalisation = meta_learner_generalisation[human_generalisation.index]

            # compute distance between human_generalisation and meta_learner_generalisation
            distance = np.linalg.norm(
                human_generalisation.values - meta_learner_generalisation.values)
            if distance < min_distance:
                min_distance = distance
                best_beta = beta
                best_human_generalisation = human_generalisation
                best_meta_learner_generalisation = meta_learner_generalisation
                best_task_block = task_block

    print(f'{model}: best_beta={best_beta}, best_task_block={best_task_block}, min_distance={min_distance}')

    # save the best_beta, best_task_block, min_distance in numpy array
    np.save(f'{SYS_PATH}/categorisation/data/meta_learner/johanssen_categorisation_{model}_{task_block}_best_beta.npy', best_beta)


if __name__ == '__main__':
    # compute_distance_transfer_stimulus(idx=5)
    # grid_search_beta(model='mi', task_block=32)
    grid_search_beta(model='ermi', task_block=32)
    # grid_search_beta(model='pfn', task_block=32)
