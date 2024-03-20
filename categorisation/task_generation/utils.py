import pickle
import re
import pandas as pd
import numpy as np
import torch
import sys
SYS_PATH = '/u/ajagadish/ermi/categorisation'
sys.path.append(f"{SYS_PATH}/categorisation/rl2")
sys.path.append(f"{SYS_PATH}/categorisation/data")
sys.path.append(f"{SYS_PATH}/categorisation/")

def return_generated_task(path, gpt, model, num_dim, num_data, num_tasks, run, proc_id, prompt_version, stage):
    filename = f'{gpt}_generated_tasks_params{model}_dim{num_dim}_data{num_data}_tasks{num_tasks}_run{run}_procid{proc_id}_pversion{prompt_version}'
    if stage>=1:
        filename = f'{filename}_stage{stage}'
    return pd.read_csv(f"{path}/{filename}.csv")

def find_counts(inputs, dim, xx_min, xx_max):
    return (inputs[:, dim]<xx_max)*(inputs[:, dim]>xx_min)

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
            features_match = np.array([len(feature) for feature in features])==num_dim 
            # does number of categories match the number of dimensions
            categories = [eval(category) for category in data.category_names.values]
            categories_match = np.array([len(category) for category in categories])==num_categories
            # if both match, add to dataframe
            both_match = features_match*categories_match
            processed_data = pd.DataFrame({'feature_names': data.feature_names.values[both_match], 'category_names': data.category_names.values[both_match], 'task_id': np.arange(len(data.task_id.values[both_match])) + last_task_id})
            df = processed_data if df is None else pd.concat([df, processed_data], ignore_index=True)
            last_task_id = df.task_id.values[-1] + 1


    num_tasks = df.task_id.max()+1
    # df.feature_names = df['feature_names'].apply(lambda x: eval(x))
    # df.category_names = df['category_names'].apply(lambda x: eval(x))
    df.to_csv(f'{path_to_dir}/{run_gpt}_generated_tasklabels_params{model}_dim{num_dim}_tasks{num_tasks}_pversion{prompt_version}.csv')             


def retrieve_features_and_categories(path, file_name, task_id):
 
    df = pd.read_csv(f'{path}/{file_name}.csv')
    task_id = df.task_id[task_id]
    df = df[df.task_id==df.task_id[task_id]]
    features = eval(df.feature_names.values[0])
    categories = eval(df.category_names.values[0])
    return features, categories, task_id

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