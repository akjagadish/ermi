from utils import pool_tasklabels, pool_synthesisedproblems
import anthropic
from dotenv import load_dotenv
import os
import re
import pickle
import ipdb
from prompts import synthesize_functionlearning_problems, synthesize_decisionmaking_problems, synthesize_categorisation_problems
import openai
import gym
import time
import pandas as pd
import numpy as np
import torch
import argparse
import sys
from llms import get_llm
sys.path.append("..")
sys.path.insert(1, '/u/ajagadish/ermi/taskgeneration/')
# from inference import LLaMAInference
load_dotenv()  # load environment variables from .env
TOKEN_COUNTER = 0


def act(text=None, run_gpt='llama', temperature=1., max_length=300, llm=None):

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

        return response

    elif run_gpt == 'gpt4':

        openai.api_key = os.getenv("OPENAI_API_KEY_GPT4")  # load key from env
        text = [{"role": "system", "content": "Do not generate any text other than the list of feature names for the stimuli and their corresponding category label in the format specified by the user."},
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
            return response.choices[0].message.content.replace(' ', '')
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
            return response.choices[0].text.strip().replace(' ', '')
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
            ).completion.replace(' ', '')
        except:
            print("Error")
            exc_type, exc_value, exc_traceback = sys.exc_info()
            print(exc_value)
            time.sleep(3**iter)
        return response

    else:

        return NotImplementedError


if __name__ == "__main__":
    models = ["8B", "70B", "NA"]
    parser = argparse.ArgumentParser()
    parser.add_argument("--llama-path", type=str, required=False, default=None)
    parser.add_argument("--task", type=str, required=False,
                        default='functionlearning')
    parser.add_argument("--model", type=str, required=False, choices=models)
    parser.add_argument("--run-gpt", type=str, required=True,
                        choices=['llama-3', 'gpt3', 'gpt4', 'claude'])
    parser.add_argument("--num-tasks", type=int, required=True, default=1000)
    parser.add_argument("--num-dim", type=int, required=True, default=3)
    parser.add_argument("--temperature", type=float,
                        required=False, default=1.0)
    parser.add_argument("--max-length", type=int, required=False, default=300)
    parser.add_argument("--proc-id", type=int, required=False, default=0)
    parser.add_argument("--num-runs", type=int, required=False, default=1)
    parser.add_argument('--first-run-id', type=int,
                        default=0, help='id of the first run')
    parser.add_argument("--prompt-version", type=str,
                        required=False, default=None)
    parser.add_argument("--path", type=str, required=False,
                        default='/u/ajagadish/ermi/categorisation/data/synthesize_problems')
    parser.add_argument("--pool", action='store_true',
                        required=False, default=False)

    args = parser.parse_args()
    start_loading = time.time()
    run_gpt = args.run_gpt  # True
    assert args.model == 'NA'if (args.run_gpt == 'gpt3' or args.run_gpt == 'gpt4' or args.run_gpt == 'claude') else True, "Only NA model is supported for GPT3"
    # model parameters
    temperature = args.temperature
    max_length = args.max_length
    # instruction parameters
    num_tasks = args.num_tasks
    num_dim = args.num_dim
    # runtime parameters
    proc_id = args.proc_id
    num_runs = args.num_runs
    first_run_id = args.first_run_id
    prompt_version = args.prompt_version

    if args.pool:

        pool_synthesisedproblems(args.path, args.task, args.run_gpt, args.model, args.num_dim,
                                 args.num_tasks, args.num_runs, args.proc_id, args.prompt_version)

    else:

        # load LLaMA model and instructions
        if run_gpt == 'llama-3':
            llm, Q_, A_ = get_llm(run_gpt, max_tokens=max_length, temp=temperature)

        # load task specific instructions
        if args.task == 'functionlearning':
            instructions = synthesize_functionlearning_problems(
                run_gpt, version=f'v{prompt_version}', num_dim=num_dim, num_tasks=num_tasks)
            regex_patterns = [
                        r'([A-Za-z&]+),([A-Za-z&]+)',
                        r'([A-Za-z&]+),([A-Za-z&]+),([A-Za-z&]+)',
                        None,
                        r'([A-Za-z&]+),([A-Za-z&]+),([A-Za-z&]+),([A-Za-z&]+),([A-Za-z&]+)',
                        ]
            regex = regex_patterns[num_dim-1]
        elif args.task == 'categorisation':
            instructions = synthesize_categorisation_problems(
                run_gpt, version=f'v{prompt_version}', num_dim=num_dim, num_tasks=num_tasks)
            regex_patterns = [
                        r'\d+\.(.+?)\n',
                        ]
            regex = regex_patterns[0]
        else:
            raise NotImplementedError
            
        # run gpt models
        for run in range(first_run_id, first_run_id+num_runs):
            stimulus_dimensions, targets = [], []
            # LLM acts
            # print(instructions)
            action = act(instructions, run_gpt, temperature, max_length, llm=llm)
            # print(action)
            matches = re.findall(regex, action, re.MULTILINE)
            if len(matches) > 0:
                for match in matches:
                    # last generated token are the targets
                    targets.append(match[-1] if args.task == 'functionlearning' else match.split(',')[-2:])
                    # rest are feature names
                    stimulus_dimensions.append(match[:-1] if args.task == 'functionlearning' else match.split(',')[:-2])
                # save data
                df = pd.DataFrame({'feature_names': stimulus_dimensions, 'target_names': targets,
                                  'task_id': np.arange(len(stimulus_dimensions)), })
                file_name = f'{run_gpt}_synthesized_{args.task}_problems_params{args.model}_dim{num_dim}_tasks{num_tasks}_run{run}_procid{proc_id}_pversion{prompt_version}'
                df.to_csv(f'{args.path}/{file_name}.csv')
            else:
                print(f'no tasks were successfully parsed')

            if run_gpt == 'gpt3' or run_gpt == 'gpt4':
                print(f'total tokens used: {TOKEN_COUNTER}')
