from llm import LLMCategoryLearning
import sys
import time
import numpy as np
import pandas as pd
import ipdb
import anthropic
from dotenv import load_dotenv
import argparse
load_dotenv()  # load environment variables from .env
SYS_PATH = '/u/ajagadish/ermi'
sys.path.append(f'{SYS_PATH}/categorisation/data')


def randomized_choice_options(num_choices):
    choice_options = list(map(chr, range(65, 91)))
    return np.random.choice(choice_options, num_choices, replace=False)


def call_claude(query, model="claude-2", temperature=0., max_tokens=1):
    # anthropic api call
    client = anthropic.Anthropic()
    response = client.completions.create(
        prompt=query,
        model=model,
        temperature=temperature,
        max_tokens_to_sample=max_tokens,
    ).completion.replace(' ', '')

    return response


def run_llm_on_badham2017(mode='llm', model='claude-2', start_participant=0):
    datasets = ["data/human/badham2017deficits.csv"]
    all_prompts = []
    # categories = {'j': 'A', 'f': 'B'}

    for dataset in datasets:
        df = pd.read_csv(dataset)
        # add new column to df to store the llm predicted category
        df['llm_category'], df['true_category'] = np.nan, np.nan
        num_participants = df.participant.max() + 1
        num_tasks = df.task.max() + 1
        num_blocks = df.block.max() + 1

        for participant in range(start_participant, num_participants):
            df_participant = df[(df['participant'] == participant)]
            # participant specific number of trials
            num_trials = df_participant.trial.max() + 1
            num_features = 3
            choice_options = randomized_choice_options(num_choices=2)
            categories = {'j': choice_options[0], 'f': choice_options[1]}

            # instructions
            instructions = 'In this experiment, you will be shown examples of geometric objects. \n'\
                f'Each object has {num_features} different features: size, color, and shape. \n'\
                'Your job is to learn a rule based on the object features that allows you to tell whether each example \n'\
                f'belongs in the {str(choice_options[0])} or {str(choice_options[1])} category. As you are shown each \n'\
                'example, you will be asked to make a category judgment and then you \n'\
                'will receive feedback. At first you will have to guess, but you will \n'\
                'gain experience as you go along. Try your best to gain mastery of the \n'\
                f'{str(choice_options[0])} and {str(choice_options[1])} categories. \n\n'\

            for task in range(num_tasks):
                df_task = df_participant[(df_participant['task'] == task)]

                for block in range(num_blocks):
                    df_block = df_task[(df_task['block'] == block)]
                    num_trials_block = df_block.trial.max() + 1  # block specific number of trials
                    # + f'In this block {block+1}, you will be shown {num_trials_block} examples of geometric objects. \n'
                    block_instructions = instructions

                    for t_idx, trial in enumerate(df_block.trial.values[:num_trials_block]):
                        df_trial = df_block[(df_block['trial'] == trial)]
                        t = categories[df_trial.correct_choice.item()]
                        object_name = df_trial.object.item()
                        human_response = categories[df_trial.choice.item()]

                        # anthropic prompt
                        Q_ = anthropic.HUMAN_PROMPT if model == 'claude-2' else 'Q:'
                        A_ = anthropic.AI_PROMPT if model == 'claude-2' else 'A:'
                        question = f'{Q_} What category would a ' + object_name + ' belong to? (Give the answer in the form \"Category <your answer>\").'\
                            f'{A_} Category'
                        query = block_instructions + question
                        # print(query)
                        count = 0
                        while count < 10:
                            try:
                                time.sleep(3**count - 1)
                                llm_response = call_claude(query)
                                break
                            except Exception as e:
                                print(f'Error in anthropic {e}. Retrying...')
                                count += 1
                                continue

                        # check if response is valid
                        assert llm_response in [str(choice_options[0]), str(
                            choice_options[1])], 'Invalid response. Please try again.'
                        # while response not in ['A', 'B']:
                        #     print('Invalid response. Please try again.')
                        #     response = call_claude(query)

                        # add llm predicted category and true category to df
                        df.loc[(df['participant'] == participant) & (df['task'] == task) & (
                            df['trial'] == trial) & (df['block'] == block), 'llm_category'] = llm_response
                        df.loc[(df['participant'] == participant) & (df['task'] == task) & (
                            df['trial'] == trial) & (df['block'] == block), 'true_category'] = str(t)

                        # add to block instructions
                        if mode == 'match_ermi':
                            block_instructions += '- In trial ' + \
                                str(t_idx+1) + ', you saw ' + object_name + \
                                ' which belonged to category ' + str(t) + '.\n'
                        else:
                            # set response to llm response if mode is llm
                            response = llm_response if mode == 'llm' else human_response
                            block_instructions += '- In trial ' + str(t_idx+1) + ', you picked category ' + str(
                                response) + ' for ' + object_name + ' and category ' + str(t) + ' was correct.\n'

            # save df with llm predicted category and true category
            df.to_csv(dataset.replace(
                '.csv', f'llm_choices{mode}_{start_participant}.csv'), index=False)


def run_llm_on_devraj2022(mode='llm', model='claude-2', start_participant=0):
    # TODO: feature names, task instruction and load values from dataset
    datasets = ["data/human/devraj2022rational.csv"]
    features_to_words = {
        (0, 0, 0, 0, 0, 0): 'gafuzi',
        (1, 0, 0, 0, 0, 0): 'wafuzi',
        (0, 1, 0, 0, 0, 0): 'gyfuzi',
        (0, 0, 1, 0, 0, 0): 'gasuzi',
        (0, 0, 0, 0, 1, 0): 'gafuri',
        (0, 0, 0, 0, 0, 1): 'gafuzo',
        (1, 1, 1, 1, 0, 1): 'wysezo',
        (1, 1, 1, 1, 1, 1): 'wysero',
        (0, 1, 1, 1, 1, 1): 'gysero',
        (1, 0, 1, 1, 1, 1): 'wasero',
        (1, 1, 0, 1, 1, 1): 'wyfero',
        (1, 1, 1, 0, 1, 1): 'wysuro',
        (1, 1, 1, 1, 1, 0): 'wyseri',
        (0, 0, 0, 1, 0, 0): 'gafezi'
    }

    all_prompts = []
    # categories = {'j': 'A', 'f': 'B'}

    for dataset in datasets:
        df = pd.read_csv(dataset)
        df = df[df['condition'] == 'control']  # only pass 'control' condition
        # add new column to df to store the llm predicted category
        df['llm_category'], df['true_category'] = np.nan, np.nan
        num_participants = df.participant.max() + 1
        num_tasks = df.task.max() + 1
        num_blocks = 1  # df.block.max() + 1

        for participant in range(start_participant, num_participants):
            df_participant = df[(df['participant'] == participant)]
            # participant specific number of trials
            num_trials = df_participant.trial.max() + 1
            num_features = 6
            choice_options = randomized_choice_options(num_choices=2)
            categories = {'0': choice_options[0], '1': choice_options[1]}

            # instructions
            instructions = 'In this experiment, you will be shown examples of nonsense word stimuli. \n'\
                f'Look carefully at each word and decide if it belongs to group {str(choice_options[0])} or group {str(choice_options[1])}. \n'\
                f'Respond with {str(choice_options[0])} if you think it is a group {str(choice_options[0])} word and a'\
                f' {str(choice_options[1])} if you think it is a group {str(choice_options[1])} word. \n'\
                'You will recieve feedback about the correct group after each of your response. \n'\
                'At first, the task will seem quite difficult,'\
                ' but with time and practice, you should be able to answer correctly. \n\n'

            for task in range(num_tasks):
                df_task = df_participant[(df_participant['task'] == task)]

                for block in range(num_blocks):
                    df_block = df_task  # [(df_task['block'] == block)]
                    num_trials_block = df_block.trial.max() + 1  # block specific number of trials
                    # + f'In this block {block+1}, you will be shown {num_trials_block} examples of geometric objects. \n'
                    block_instructions = instructions

                    for t_idx, trial in enumerate(df_block.trial.values[:num_trials_block]):
                        df_trial = df_block[(df_block['trial'] == trial)]
                        t = categories[str(df_trial.correct_choice.item())]

                        feature_barcode = tuple(
                            eval(df_trial.all_features.values[0]))
                        object_name = features_to_words[feature_barcode]
                        human_response = categories[str(
                            df_trial.choice.item())]

                        # anthropic prompt
                        Q_ = anthropic.HUMAN_PROMPT if model == 'claude-2' else 'Q:'
                        A_ = anthropic.AI_PROMPT if model == 'claude-2' else 'A:'
                        question = f'{Q_} What group would the word ' + object_name + ' belong to? (Give the answer in the form \"Group <your answer>\").'\
                            f'{A_} Group'
                        query = block_instructions + question
                        # print(query)
                        # llm_response = call_claude(query)
                        count = 0
                        while count < 10:
                            try:
                                time.sleep(3**count - 1)
                                llm_response = call_claude(query)
                                break
                            except Exception as e:
                                print(f'Error in anthropic {e}. Retrying...')
                                count += 1
                                continue

                        # check if response is valid
                        assert llm_response in [str(choice_options[0]), str(
                            choice_options[1])], 'Invalid response. Please try again.'
                        # while response not in ['A', 'B']:
                        #     print('Invalid response. Please try again.')
                        #     response = call_claude(query)

                        # add llm predicted category and true category to df
                        df.loc[(df['participant'] == participant) & (df['task'] == task) & (
                            df['trial'] == trial), 'llm_category'] = llm_response
                        df.loc[(df['participant'] == participant) & (df['task'] == task) & (
                            df['trial'] == trial), 'true_category'] = str(t)

                        # set response to llm response if mode is llm
                        response = llm_response if mode == 'llm' else human_response

                        # add to block instructions
                        if mode == 'match_ermi':
                            block_instructions += '- In trial ' + \
                                str(t_idx+1) + ', you saw ' + object_name + \
                                ' which belonged to group ' + str(t) + '.\n'
                        else:
                            # set response to llm response if mode is llm
                            response = llm_response if mode == 'llm' else human_response
                            block_instructions += '- In trial ' + str(t_idx+1) + ', you picked group ' + str(
                                response) + ' for ' + object_name + ' and group ' + str(t) + ' was correct.\n'
                    # print(block_instructions)

            # save df with llm predicted category and true category
            df.to_csv(dataset.replace(
                '.csv', f'llm_choices{mode}_{start_participant}.csv'), index=False)


def fit_llm_to_humans(num_runs, num_blocks, num_iter, opt_method, loss, task_name):

    if task_name == 'devraj2022':
        df = pd.read_csv(
            f'{SYS_PATH}/categorisation/data/llm/devraj2022rational_llm_choiceshuman.csv')
        df = df[df['condition'] == 'control']  # only pass 'control' condition
        df['category'] = df['category'].astype(int)-1
        NUM_TASKS, NUM_FEATURES = 1, 6
    elif task_name == 'badham2017':
        df = pd.read_csv(
            f'{SYS_PATH}/categorisation/data/llm/badham2017deficits_llm_choiceshuman.csv')  # match_ermi
        NUM_TASKS, NUM_FEATURES = 1, 3
    else:
        raise NotImplementedError

    # comment out for testing
    # df = df[df['participant'] == 0]
    # df['llm_category'] = df['true_category']
    # df['choice'] = df['correct_choice']

    lls, r2s, params_list = [], [], []
    for idx in range(num_runs):
        llm = LLMCategoryLearning(
            num_features=NUM_FEATURES, num_iterations=num_iter, opt_method=opt_method, loss=loss)
        ll, r2, params = llm.fit_participants(
            df, num_blocks=num_blocks, reduce='sum')
        params_list.append(params)
        lls.append(ll)
        r2s.append(r2)
        print(f'mean fit across blocks: {lls[idx].mean()} \n')
        print(f'mean pseudo-r2 across blocks: {r2s[idx].mean()}')

    # save the r2 and ll values
    lls = np.array(lls)
    r2s = np.array(r2s)
    np.savez(f'{SYS_PATH}/categorisation/data/model_comparison/{task_name}_llm_runs={num_runs}_iters={num_iter}_blocks={num_blocks}_loss={loss}',
             r2s=r2s, lls=lls, params=np.stack(params_list), opt_method=opt_method)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='llm',
                        help='llm or human or match_ermi')
    parser.add_argument('--model', type=str,
                        default='claude-2', help='claude-2 or claude-1')
    parser.add_argument('--dataset', type=str,
                        default='badham2017', help='badham2017 or devraj2022')
    parser.add_argument('--start-participant', type=int,
                        default=0, help='start participant number')
    parser.add_argument('--fit-human-data',
                        action='store_true', help='fit llm to humans')
    parser.add_argument('--num-runs', type=int, required=False,
                        default=1, help='number of runs')
    parser.add_argument('--num-iter', type=int, required=False,
                        default=1, help='number of iterations')
    parser.add_argument('--num-blocks', type=int,
                        required=False, default=1, help='number of blocks')
    parser.add_argument('--opt-method', type=str, required=False,
                        default='minimize', help='optimization method')
    parser.add_argument('--loss', type=str, required=False,
                        default='nll', help='loss function')

    args = parser.parse_args()

    if args.fit_human_data:
        fit_llm_to_humans(num_runs=args.num_runs, num_blocks=args.num_blocks, num_iter=args.num_iter,
                          opt_method=args.opt_method, loss=args.loss, task_name=args.dataset)
    else:
        if args.dataset == 'badham2017':
            run_llm_on_badham2017(
                mode=args.mode, model=args.model, start_participant=args.start_participant)
        elif args.dataset == 'devraj2022':
            run_llm_on_devraj2022(
                mode=args.mode, model=args.model, start_participant=args.start_participant)
        else:
            raise ValueError(
                'Invalid dataset. Please provide a valid dataset.')
