import pickle
import re
import pandas as pd
import numpy as np
import torch
import sys
SYS_PATH = '/u/ajagadish/ermi/'
sys.path.append(f"{SYS_PATH}/taskgeneration")


def parse_generated_tasks(path, file_name, gpt, num_datapoints=8, num_dim=3, last_task_id=0, use_generated_tasklabels=False, prompt_version=None):

    # load llama generated tasks which were successfully regex parsed
    with open(f"{path}/{file_name}.txt", "rb") as fp:
        datasets = pickle.load(fp)

    # regular expression pattern to extract input values from the stored inputs and targets
    pattern = r'((?: )?\s*[\d.]+)(?:,|;|\s)?\s*([\d.]+)(?:,|;|\s)?\s*([\d.]+)'

    # make a pandas dataframe for the parsed data
    df = None
    task_id = last_task_id

    # load task labels if use_generated_tasklabels is True
    if use_generated_tasklabels:
        with open(f"{path}/{file_name}_taskids.txt", "rb") as fp:
            task_label = pickle.load(fp)

    # parse the list using regex
    for task, data in enumerate(datasets):
        # initialize lists to store parsed values
        inputs, targets = [], []
        # TODO: per data make the target into A or B

        # load each input-target pair
        for item in data:
            if gpt == 'gpt3':

                inputs.append([float(item[0]), float(item[1]), float(item[2])])
                targets.append(item[3])

            elif gpt == 'gpt4':

                inputs.append([float(item[0]), float(item[1]), float(item[2])])
                targets.append(item[3][1] if len(item[3]) > 1 else item[3])

            elif gpt == 'claude':

                if num_dim == 3:
                    if prompt_version == 3:
                        inputs.append([item[0], item[1], item[2]])
                    elif prompt_version == 4:
                        try:
                            inputs.append(
                                [float(item[0]), float(item[1]), float(item[2])])
                        except:
                            print(f'{task} not parsable as float')
                            continue

                elif num_dim == 6:
                    if prompt_version == 5:
                        try:
                            inputs.append([float(item[1]), float(item[2]), float(
                                item[3]), float(item[4]), float(item[5]), float(item[6])])
                        except:
                            print(f'{task} not parsable as float')
                            continue
                    else:
                        raise NotImplementedError

                elif num_dim == 4:

                    if prompt_version == 5:
                        try:
                            inputs.append([float(item[1]), float(
                                item[2]), float(item[3]), float(item[4])])
                        except:
                            print(f'{task} not parsable as float')
                            continue
                    else:
                        raise NotImplementedError

                else:
                    raise NotImplementedError

                if use_generated_tasklabels:
                    targets.append(item[num_dim+1])
                else:
                    targets.append(item[3][1] if len(item[3]) > 1 else item[3])

            else:
                match = re.match(pattern, item[0])

                if match:
                    try:
                        inputs.append([float(match.group(1)), float(
                            match.group(2)), float(match.group(3))])
                        targets.append(item[1])
                    except:
                        print(f'no match')
                else:
                    print(f'error parsing {task, item[0]}')

        # if the number of datapoints is equal to the number of inputs, add to dataframe
        if gpt == 'gpt3' or gpt == 'gpt4' or gpt == 'claude' or ((gpt == 'llama') and (len(inputs) == num_datapoints)):
            print(f'{task} has inputs of length {len(inputs)}')
            use_task_index = task_label[task] if use_generated_tasklabels else task_id
            df = pd.DataFrame({'input': inputs, 'target': targets, 'trial_id': np.arange(len(inputs)), 'task_id': np.ones((len(inputs),))*(use_task_index)}) if df is None else pd.concat([df,
                                                                                                                                                                                           pd.DataFrame({'input': inputs, 'target': targets, 'trial_id': np.arange(len(inputs)), 'task_id': np.ones((len(inputs),))*(use_task_index)})], ignore_index=True)
            task_id += 1
        else:
            print(
                f'dataset did not have {num_datapoints} datapoints but instead had {len(inputs)} datapoints')

    # save data frame to csv
    if df is not None:
        df.to_csv(f'{path}/{file_name}.csv')
    else:
        print(f'no datasets were successfully parsed')

    return task_id


def return_generated_task(path, gpt, model, num_dim, num_data, num_tasks, run, proc_id, prompt_version, stage):
    filename = f'{gpt}_generated_tasks_params{model}_dim{num_dim}_data{num_data}_tasks{num_tasks}_run{run}_procid{proc_id}_pversion{prompt_version}'
    if stage >= 1:
        filename = f'{filename}_stage{stage}'
    return pd.read_csv(f"{path}/{filename}.csv")


def find_counts(inputs, dim, xx_min, xx_max):
    return (inputs[:, dim] < xx_max)*(inputs[:, dim] > xx_min)


def pool_generated_tasks(path, models, dims, data, tasks, runs, proc_ids):
    ''' 
    Pool the generated tasks from different processes into a single dataframe
    Args:
        path: path to the folder containing the generated tasks
        models: list of models used to generate the tasks
        dims: list of dimensions used to generate the tasks
        data: list of number of datapoints used to generate the tasks
        tasks: list of number of tasks used to generate the tasks
        runs: list of runs used to generate the tasks
        proc_ids: dictionary of process ids used to generate the tasks
    Returns:
        df: dataframe containing the pooled generated tasks
    '''

    df = None
    total_tasks = 0
    for model in models:
        for dim in dims:
            for num_data in data:
                for num_tasks in tasks:
                    for run in runs:
                        for proc_id in proc_ids[num_tasks]:
                            total_tasks += num_tasks
                            df = return_generated_task(path, model, dim, num_data, num_tasks, run, proc_id) if df is None else pd.concat([df,
                                                                                                                                          return_generated_task(path, model, dim, num_data, num_tasks, run, proc_id)], ignore_index=True)

                # save the pooled dataframe to csv
                df = df.query('target == "A" or target == "B"')
                # df['task_id'] = np.int64(np.arange(len(df))/num_data) #+ 1
                df.to_csv(
                    f"{path}/llama_generated_tasks_params{model}_dim{dim}_data{num_data}_tasks{total_tasks}.csv")

    return df


def pool_tasklabels(path_to_dir, run_gpt, model, num_dim, num_tasks, num_runs, proc_id, prompt_version, num_categories=2):
    df, last_task_id = None, 0
    for run_id in range(num_runs):
        data = None
        try:
            filename = f'{run_gpt}_generated_tasklabels_params{model}_dim{num_dim}_tasks{num_tasks}_run{run_id}_procid{proc_id}_pversion{prompt_version}'
            data = pd.read_csv(f'{path_to_dir}/{filename}.csv')
        except:
            print(f'error loading {filename}')
        if data is not None:
            # does number of features match the number of dimensions
            features = [eval(feature) for feature in data.feature_names.values]
            features_match = np.array([len(feature)
                                      for feature in features]) == num_dim
            # does number of categories match the number of dimensions
            categories = [eval(category)
                          for category in data.category_names.values]
            categories_match = np.array(
                [len(category) for category in categories]) == num_categories
            # if both match, add to dataframe
            both_match = features_match*categories_match
            processed_data = pd.DataFrame({'feature_names': data.feature_names.values[both_match], 'category_names': data.category_names.values[both_match], 'task_id': np.arange(
                len(data.task_id.values[both_match])) + last_task_id})
            df = processed_data if df is None else pd.concat(
                [df, processed_data], ignore_index=True)
            last_task_id = df.task_id.values[-1] + 1

    num_tasks = df.task_id.max()+1
    # df.feature_names = df['feature_names'].apply(lambda x: eval(x))
    # df.category_names = df['category_names'].apply(lambda x: eval(x))
    df.to_csv(f'{path_to_dir}/{run_gpt}_generated_tasklabels_params{model}_dim{num_dim}_tasks{num_tasks}_pversion{prompt_version}.csv')


def pool_synthesisedproblems(path_to_dir, run_gpt, model, num_dim, num_tasks, num_runs, proc_id, prompt_version, num_targets=1, file_name=None):
    df, last_task_id = None, 0
    for run_id in range(num_runs):
        data = None
        try:
            # filename = f'{run_gpt}_generated_tasklabels_params{model}_dim{num_dim}_tasks{num_tasks}_run{run_id}_procid{proc_id}_pversion{prompt_version}'
            filename = f'{run_gpt}_synthesized_functionlearning_problems_params{model}_dim{num_dim}_tasks{num_tasks}_run{run_id}_procid{proc_id}_pversion{prompt_version}'
            data = pd.read_csv(f'{path_to_dir}/{filename}.csv')
        except:
            print(f'error loading {filename}')

        if data is not None:
            # does number of features match the number of dimensions
            features = [eval(feature) for feature in data.feature_names.values]
            features_match = np.array([len(feature)
                                      for feature in features]) == num_dim
            # does number of categories match the number of dimensions
            targets = [target
                       for target in data.target_names.values]
            targets_match = np.array(
                [len([target]) for target in targets]) == num_targets
            # if both match, add to dataframe
            both_match = features_match*targets_match
            processed_data = pd.DataFrame({'feature_names': data.feature_names.values[both_match], 'target_names': data.target_names.values[both_match], 'task_id': np.arange(
                len(data.task_id.values[both_match])) + last_task_id})
            df = processed_data if df is None else pd.concat(
                [df, processed_data], ignore_index=True)
            last_task_id = df.task_id.values[-1] + 1

    num_tasks = df.task_id.max()+1
    # df.feature_names = df['feature_names'].apply(lambda x: eval(x))
    # df.category_names = df['category_names'].apply(lambda x: eval(x))
    df.to_csv(f'{path_to_dir}/{run_gpt}_synthesized_functionlearning_problems_params{model}_dim{num_dim}_tasks{num_tasks}_pversion{prompt_version}.csv')


def retrieve_features_and_categories(path, file_name, task_id):

    df = pd.read_csv(f'{path}/{file_name}.csv')
    task_id = df.task_id[task_id]
    df = df[df.task_id == df.task_id[task_id]]
    features = eval(df.feature_names.values[0])
    categories = eval(df.category_names.values[0])
    return features, categories, task_id


def retrieve_features_and_targets(path, file_name, task_id):
    df = pd.read_csv(f'{path}/{file_name}.csv')
    task_id = df.task_id[task_id]
    df = df[df.task_id == df.task_id[task_id]]
    features = eval(df.feature_names.values[0])
    # features from tuple to a list
    # features = [list(feature) for feature in features]
    targets = df.target_names.values[0]
    return features, targets, task_id


def get_regex_patterns(num_dim, use_generated_tasklabels, prompt_version):
    ''' 
    Generate regex patterns to parse the generated tasks
    Args:
        num_dim: number of dimensions
        use_generated_tasklabels: whether to use the generated tasklabels or not
        prompt_version: version of the prompt used to generate the tasks
    Returns:
        patterns: list of regex patterns
    '''
    if use_generated_tasklabels is False:
        og_regex_expressions = [r'x=\[(.*?)\][;,]?\s*y\s*=?\s*([AB])',
                                r"x=\[(.*?)\][^\n]*?y=(\w)",
                                r"x=\[([\d\.]+),\s*([\d\.]+),\s*([\d\.]+)\][^\n]*[y|&]=\s*(A|B)",
                                r"x=\[?\s*([\d\.]+),\s*([\d\.]+),\s*([\d\.]+)\]?\s*(?:,|;|&|->| -> |---)?\s*[y|Y]\s*=\s*(A|B)",
                                r"x=(\[.*?\])\s*---\s*y\s*=\s*([A-Z])",
                                r"x=(\[.*?\])\s*->\s*([A-Z])",
                                r"x=(\[.*?\]),\s*([A-Z])",
                                r"^([0-9]\.[0-9]{2}),(0\.[0-9]{2}),(0\.[0-9]{2}),(A|B)$",
                                r"\[([0-9]\.[0-9]{2}),(0\.[0-9]{2}),(0\.[0-9]{2})\],(A|B)",
                                r"\[\[([0-9]\.[0-9]{2}),(0\.[0-9]{2}),(0\.[0-9]{2})\],(A|B)\]",
                                r"n[0-9]+\.\[\[([0-9]\.[0-9]{2}),(0\.[0-9]{2}),(0\.[0-9]{2})\],(\'A\'|\'B\')\]",
                                r"\[\[([0-9]\.[0-9]{2}),(0\.[0-9]{2}),(0\.[0-9]{2})\],(\'A\'|\'B\')\]",
                                r"\[([0-9]\.[0-9]{2}),(0\.[0-9]{2}),(0\.[0-9]{2})\],(A|B)",
                                r"(\d+\.\d+),(\d+\.\d+),(\d+\.\d+),([A-Z])"
                                ]
    elif num_dim == 3 and prompt_version == 4:
        regex_expressions = [r'([\d.]+),([\d.]+),([\d.]+),([\w]+)',
                             r'([\w\-]+),([\w\-]+),([\w\-]+),([\w]+)',
                             r'([-\w\d,.]+),([-\w\d,.]+),([-\w\d,.]+),([-\w\d,.]+)',
                             r'([^,]+),([^,]+),([^,]+),([^,]+)',
                             r'([^,\n]+),([^,\n]+),([^,\n]+),([^,\n]+)',
                             r'(?:.*?:)?([^,-]+),([^,-]+),([^,-]+),([^,-]+)',
                             r'([^,-]+),([^,-]+),([^,-]+),([^,-]+)',]

    elif num_dim == 6 and prompt_version == 5:
        regex_expressions = [r'^(\d+):([\d.]+),([\d.]+),([\d.]+),([\d.]+),([\d.]+),([\d.]+),([\w]+)',
                             r'^(\d+):([\w\-]+),([\w\-]+),([\w\-]+),([\w\-]+),([\w\-]+),([\w\-]+),([\w]+)',
                             r'^(\d+):([-\w\d,.]+),([-\w\d,.]+),([-\w\d,.]+),([-\w\d,.]+),([-\w\d,.]+),([-\w\d,.]+),([-\w\d,.]+)',
                             r'^(\d+):([^,]+),([^,]+),([^,]+),([^,]+),([^,]+),([^,]+),([^,]+)',
                             r'^(\d+):([^,\n]+),([^,\n]+),([^,\n]+),([^,\n]+),([^,\n]+),([^,\n]+),([^,\n]+)',
                             r'^(\d+):(?:.*?:)?([^,-]+),([^,-]+),([^,-]+),([^,-]+),([^,-]+),([^,-]+),([^,-]+)',
                             r'^(\d+):([^,-]+),([^,-]+),([^,-]+),([^,-]+),([^,-]+),([^,-]+),([^,-]+)',]

    elif num_dim == 4 and prompt_version == 5:
        regex_expressions = [r'^(\d+):([\d.]+),([\d.]+),([\d.]+),([\d.]+),([\w]+)',
                             r'^(\d+):([\w\-]+),([\w\-]+),([\w\-]+),([\w\-]+),([\w]+)',
                             r'^(\d+):([-\w\d,.]+),([-\w\d,.]+),([-\w\d,.]+),([-\w\d,.]+),([-\w\d,.]+)',
                             r'^(\d+):([^,]+),([^,]+),([^,]+),([^,]+),([^,]+)',
                             r'^(\d+):([^,\n]+),([^,\n]+),([^,\n]+),([^,\n]+),([^,\n]+)',
                             r'^(\d+):(?:.*?:)?([^,-]+),([^,-]+),([^,-]+),([^,-]+),([^,-]+)',
                             r'^(\d+):([^,-]+),([^,-]+),([^,-]+),([^,-]+),([^,-]+)']

    patterns = regex_expressions if use_generated_tasklabels else og_regex_expressions

    return patterns


def get_all_regex_patterns(num_dim, prompt_version, task_name):
    ''' 
    Generate regex patterns to parse the generated tasks
    Args:
        num_dim: number of dimensions
        use_generated_tasklabels: whether to use the generated tasklabels or not
        prompt_version: version of the prompt used to generate the tasks
    Returns:
        patterns: list of regex patterns
    '''

    if task_name == 'functionlearning':
        assert (prompt_version == 2), 'only prompt version 2 is supported'
        regex_expressions = [r'([\d.]+),' * num_dim + r'([\d.]+)']
    else:
        raise NotImplementedError
    patterns = regex_expressions

    return patterns
