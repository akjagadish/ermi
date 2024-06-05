from os.path import join
from os import getenv
import openai
import gym
import time
import pandas as pd
import numpy as np
import torch
import argparse
import sys
from llms import get_llm
from prompts import retrieve_prompt, generate_data_functionlearning_problems, generate_data_decisionmaking_problems
from utils import retrieve_features_and_categories, get_all_regex_patterns, retrieve_features_and_targets
import ipdb
import pickle
import re
import os
from dotenv import load_dotenv
import anthropic
import json
load_dotenv()  # load environment variables from .env
TOKEN_COUNTER = 0

SYS_PATH = getenv('BERMI_DIR')
# generate action using LLaMA or GPT-3


def act(text=None, run_gpt='llama-3', temperature=1., max_length=300, llm=None):
    """
    Generate text using different GPT models based on the specified parameters.

    Args:
        text(str): The input text to generate a response for .
        run_gpt(str): The GPT model to use for generating the response. Possible values are 'llama', 'gpt4', 'gpt3', 'claude', 'claude_2.1'.
        temperature(float): The temperature parameter for controlling the randomness of the generated text. Higher values result in more random output.
        max_length(int): The maximum length of the generated text.

    Returns:
        str: The generated response text.

    Raises:
        NotImplementedError: If the specified GPT model is not implemented.

    """

    global TOKEN_COUNTER

    if run_gpt == 'llama-3':
        if llm.instruct:
            prompt = [
                {"role": "system", 
                "content": "You are an oracle that can generate data whose statistics match real-world data."},
                {"role": "user", "content": text},
                                ]
            
            text = llm.tokenizer.apply_chat_template(
                prompt,
                add_generation_prompt=True,
                return_tensors="pt"
            ).to("cuda")
        response = llm.generate(text)
        #.replace(' ', '').replace('.', '').replace(',', '').replace('\n', '').replace(':', '')

        return response

    elif run_gpt == 'gpt4':

        openai.api_key = os.getenv("OPENAI_API_KEY_GPT4")  # load key from env
        text = [{"role": "system", "content": "Do not generate any text other than the list of objects with their feature values and their corresponding category label in the format specified by the user."},
                {"role": "user", "content": text}]
        engine = 'gpt-4'
        try:
            response = openai.ChatCompletion.create(
                model=engine,
                messages=text,
                max_tokens=max_length,
                temperature=temperature,
            )
            TOKEN_COUNTER += response['usage']['total_tokens']
            return response.choices[0].message.content  # .replace(' ', '')
        except:
            print("Error, trying again...ratelimiterror")
            exc_type, exc_value, exc_traceback = sys.exc_info()
            print(exc_value)

    elif run_gpt == 'gpt3':

        openai.api_key = os.getenv("OPENAI_API_KEY")  # load key from env
        engine = "text-davinci-003"
        try:
            response = openai.Completion.create(
                engine=engine,
                prompt=text,
                max_tokens=max_length,
                temperature=temperature,
            )
            TOKEN_COUNTER += response['usage']['total_tokens']
            return response.choices[0].text.strip()  # .replace(' ', '')
        except:
            print("Error")
            exc_type, exc_value, exc_traceback = sys.exc_info()
            print(exc_value)
            # time.sleep(3**iter)

    elif run_gpt == 'claude':

        try:
            client = anthropic.Anthropic()
            response = client.completions.create(
                prompt=anthropic.HUMAN_PROMPT + text + anthropic.AI_PROMPT,
                # stop_sequences=[anthropic.HUMAN_PROMPT],
                model="claude-2",
                temperature=temperature,
                max_tokens_to_sample=max_length,
            ).completion  # .replace(' ', '')

        except:
            print("Error")
            exc_type, exc_value, exc_traceback = sys.exc_info()
            print(exc_value)
            time.sleep(3**iter)
        return response

    elif run_gpt == 'claude_2.1':

        client = anthropic.Anthropic()
        response = client.completions.create(
            prompt=anthropic.HUMAN_PROMPT + text + anthropic.AI_PROMPT,
            # stop_sequences=[anthropic.HUMAN_PROMPT],
            model="claude-2.1",
            temperature=temperature,
            max_tokens_to_sample=max_length,
        ).completion  # .replace(' ', '')

        return response

    else:

        return NotImplementedError

# check if action is parsable and if yes, return the matches


def check_if_parsable(action, patterns):
    for pattern in patterns:
        matches = re.findall(pattern, action.replace(' ', ''), re.MULTILINE)
        if len(matches) > 0:
            return matches
    return None


if __name__ == "__main__":
    models = ["7B", "8B", "13B", "30B", "65B", "NA"]
    parser = argparse.ArgumentParser()
    parser.add_argument("--llama-path", type=str, required=False, default=None)
    parser.add_argument("--model", type=str, required=False, choices=models)
    parser.add_argument("--task", type=str, required=False,
                        default='categorisation')
    parser.add_argument("--run-gpt", type=str, required=True,
                        choices=['llama-3', 'gpt3', 'gpt4', 'claude', 'claude_2.1'])
    parser.add_argument("--num-tasks", type=int, required=True, default=1000)
    parser.add_argument("--num-dim", type=int, required=True, default=3)
    parser.add_argument("--num-data", type=int, required=True, default=8)
    parser.add_argument("--temperature", type=float,
                        required=False, default=1.0)
    parser.add_argument("--max-length", type=int, required=False, default=300)
    parser.add_argument("--proc-id", type=int, required=False, default=0)
    parser.add_argument("--num-runs", type=int, required=False, default=1)
    parser.add_argument("--prompt-version", type=str,
                        required=False, default=None)
    parser.add_argument("--path-tasklabels", type=str, required=False,
                        default='/raven/u/ajagadish/vanilla-llama/categorisation/data/tasklabels')
    parser.add_argument("--file-name-tasklabels", type=str,
                        required=False, default=None)
    parser.add_argument("--start-task-id", type=int, required=False, default=0)
    parser.add_argument("--end-task-id", type=int,
                        required=False, default=None)
    parser.add_argument('--stage', type=int, default=0,
                        help='stage of prompt generation')

    args = parser.parse_args()
    start_loading = time.time()
    run_gpt = args.run_gpt
    assert args.model == 'NA' if args.run_gpt == 'gpt3' or args.run_gpt == 'gpt4' or args.run_gpt == 'claude' or args.run_gpt == 'claude_2.1' else args.model, "Only NA model is supported for GPT3"
    # model parameters
    temperature = args.temperature
    max_length = args.max_length
    # instruction parameters
    start_task_id = args.start_task_id
    num_tasks = args.num_tasks
    num_data = args.num_data
    num_dim = args.num_dim
    # runtime parameters
    proc_id = args.proc_id
    num_runs = args.num_runs
    prompt_version = args.prompt_version
    num_categories = 2
    llm = None

    # get regex patterns
    patterns = get_all_regex_patterns(
        num_dim=num_dim, prompt_version=prompt_version, task_name=args.task)

    # load LLaMA model and instructions
    if run_gpt == 'llama-3':
       llm, Q_, A_ = get_llm(run_gpt, max_tokens=max_length, temp=temperature)
    # load GPT-3 specific instructions
    elif run_gpt == 'gpt3':
        instructions = retrieve_prompt(
            'gpt3', version='v1', num_dim=num_dim, num_data=num_data)

    # load GPT-4 specific instructions
    elif run_gpt == 'gpt4':
        instructions = retrieve_prompt(
            'gpt4', version='v3', num_dim=num_dim, num_data=num_data)

    # load Claude specific instructions
    # elif run_gpt == 'claude':
    #     instructions = retrieve_prompt('claude', version=f'v{prompt_version}', num_dim=num_dim, num_data=num_data)

    # run gpt models
    for run in range(num_runs):

        data, unparsable_data, raw_data, task_ids = [], [], [], []
        end_task_id = start_task_id+num_tasks if args.end_task_id is None else args.end_task_id

        # filename for dataframe
        filename = f'{run_gpt}_generated_{args.task}tasks_params{args.model}_dim{num_dim}_data{num_data}_tasks{num_tasks}_run{run}_procid{proc_id}_pversion{prompt_version}'
        filename += f"_stage{str(args.stage)}" if args.stage > 0 else ""
        df = None

        for idx, t in enumerate(range(start_task_id, end_task_id)):
            # check if dataframe exists
            if os.path.exists(f"{SYS_PATH}/data/{filename}.csv"):
                df = pd.read_csv(f"{SYS_PATH}/data/{filename}.csv")
                if df.task_id.isin([t]).any():
                    print(
                        f'task {t}: task_id {t} already exists in the dataframe')
                    continue

            # LLM acts
            if (run_gpt == 'claude' or run_gpt == 'claude_2.1'):
                assert args.file_name_tasklabels is not None, "Please provide a file name for the task labels"

                if args.task == 'categorisation':
                    features, categories, task_id = retrieve_features_and_categories(path=args.path_tasklabels,
                                                                                     file_name=args.file_name_tasklabels,
                                                                                     task_id=t)
                    assert len(
                        features) == num_dim, "Number of features does not match the number of dimensions"
                    assert len(
                        categories) == num_categories, "Number of categories does not match the number of categories"
                    instructions = retrieve_prompt(
                        'claude', version=f'v{prompt_version}', num_dim=num_dim, num_data=num_data, features=features, categories=categories)
                elif args.task == 'functionlearning':

                    features, target, task_id = retrieve_features_and_targets(path=args.path_tasklabels,
                                                                              file_name=args.file_name_tasklabels,
                                                                              task_id=t)

                    assert len(
                        features) == num_dim, "Number of features does not match the number of dimensions"
                    instructions = generate_data_functionlearning_problems(
                        'claude', version=f'v{prompt_version}', num_dim=num_dim, num_data=num_data, features=features, target=target)
            
            elif run_gpt == 'llama-3':
                if args.task == 'functionlearning':

                    features, target, task_id = retrieve_features_and_targets(path=args.path_tasklabels,
                                                                              file_name=args.file_name_tasklabels,
                                                                              task_id=t)

                    assert len(
                        features) == num_dim, "Number of features does not match the number of dimensions"
                    instructions = generate_data_functionlearning_problems(
                        'llama-3', version=f'v{prompt_version}', num_dim=num_dim, num_data=num_data, features=features, target=target)
                    
                elif args.task == 'categorisation':
                    features, categories, task_id = retrieve_features_and_categories(path=args.path_tasklabels,
                                                                                     file_name=args.file_name_tasklabels,
                                                                                     task_id=t)
                    assert len(
                        features) == num_dim, "Number of features does not match the number of dimensions"
                    assert len(
                        categories) == num_categories, "Number of categories does not match the number of categories"
                    instructions = retrieve_prompt(
                        'llama-3', version=f'v{prompt_version}', num_dim=num_dim, num_data=num_data, features=features, categories=categories)

            # generate tasks in one or two stages
            if args.stage == 2:
                with open(f"{SYS_PATH}/data/raw_data/{run_gpt}_generated_tasks_params{args.model}_dim{num_dim}_data{num_data}_tasks{num_tasks}_run{run}_procid{proc_id}_pversion{prompt_version}_stage1_starttaskid{start_task_id}_raw.txt", "rb") as fp:
                    stage1_action = pickle.load(fp)[idx]
                matches = check_if_parsable(stage1_action, patterns)
                if matches is not None:  # the original few points are getting changed
                    stage2_action = act(
                        instructions + stage1_action[:int(len(stage1_action)/3)], run_gpt, temperature, max_length)
                    # action already contains the first part of the action
                    action = stage2_action + '\n' + \
                        stage1_action[int(len(stage1_action)/3):]
            else:
                action = act(instructions, run_gpt, temperature, max_length, llm)

            raw_data.append(action)
            matches = check_if_parsable(action, patterns)
            if matches is not None:

                for trial_id, data in enumerate(matches):
                    inputs, targets = data[:-1], data[-1]
                    try:
                        inputs = str([float(i) for i in inputs])
                        dataframe = pd.DataFrame({'input': inputs, 'target': targets, 'trial_id': trial_id, 'task_id': task_id}, index=[
                                            0], columns=['input', 'target', 'trial_id', 'task_id'])
                        # only append if the task_id is not already present in the dataframe
                        df = dataframe if df is None else pd.concat(
                            [df, dataframe], ignore_index=True)
                    except:
                        print(data)
                        print("Error")
                        exc_type, exc_value, exc_traceback = sys.exc_info()
                            # print(exc_value)
                df.to_csv(f"{SYS_PATH}/{args.task}/data/generated_tasks/{filename}.csv", index=False)

            elif matches is None:

                unparsable_data.append(action)

            print(
                f'task {t}: no matches found' if matches is None else f'task {t}: match found')

    if run_gpt == 'gpt4':
        print(f'total tokens used: {TOKEN_COUNTER}')
